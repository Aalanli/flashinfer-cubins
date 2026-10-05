"""DSA indexer (DeepSeek-V3.2, top-k excluded) as two single-kernel workloads.

The indexer's logits path is DeepGEMM's paged FP8 MQA logits at revision
78b6900 (see ``impls/dsa_indexer/compiler.py`` for the kernel inventory and
which kernels are live):

``dsa_indexer_metadata``
    ``sm100_paged_mqa_logits_metadata``: splits the total work
    (``sum(ceil(seq_len / 256))`` KV splits) evenly over 148 scheduling
    partitions and writes each partition's ``(q_token, kv_split)`` start,
    ``int32[149, 2]``.
``dsa_indexer_logits``
    ``sm100_paged_mqa_logits``: ``logits[b, t] = scale[t] * sum_h
    relu(q[b, h] . k[t]) * weights[b, h]`` over the paged FP8 key cache.

``num_scheduling_partitions`` (DeepGEMM's ``num_sms``) is a template argument
of the metadata kernel, fixed at compile time to 148 (B200). The logits kernel
is launched with one CTA per partition, i.e. ``schedule_meta.shape[0] - 1``;
on a GPU with a different SM count the schedule stays valid (it only changes
how evenly the persistent CTAs fill the machine).

Logits layout, as DeepGEMM's ``fp8_fp4_paged_mqa_logits``
(``csrc/apis/attention.hpp``): the block table keeps its own row stride
(``block_table_stride = block_table.stride(0)``, any width), and the FP32
logits rows are ``align(max_context_len, 256)`` wide (1024-byte aligned
rows). ``max_context_len`` is ``block_table.shape[1] * 64``, as DeepGEMM's
test passes it (``max_model_len``); every ``seq_len`` must fit in it. The
output is the whole aligned ``[B, align(width * 64, 256)]`` buffer (upstream
returns its first ``max_context_len`` columns as a view).

Written region of the logits kernel: DeepGEMM computes whole 256-token KV
splits, so for each row it writes exactly the columns
``[0, ceil(seq_len / 256) * 256)`` (nothing for an empty sequence) and leaves
the rest of the row untouched (paged ``clean_logits`` is unsupported
upstream). Inside the last split, columns at or after ``seq_len`` are scored
against the keys the kernel actually loads: the remaining tokens of the
sequence's last page, and physical page 0 for page slots at or beyond
``ceil(seq_len / 64)`` (the kernel's out-of-range page coordinate). The
reference reproduces all written columns and marks unwritten ones as NaN;
``validate`` compares the written columns only.

Cases (shared by both kernels): skewed batches with empty, length-1,
page- and split-boundary lengths, unused block-table slots, shared pages,
more than 1024 requests (the metadata kernel's multi-pass scan) and enough
256-token splits for persistent CTAs to wrap the 5-stage KV pipeline;
official inventory rows (skewed lengths, longest = ``max_num_pages * 64``,
the recorded block-table width); DeepSeek-V3.2 long-context decode; and
DeepGEMM's own ``test_paged_mqa_logits`` parametrizations this kernel serves
(checked against the pinned test source by ``tests/test_dsa_indexer.py``).

All launch preparation happens in Python: arguments are ctypes values, the
four distinct ``CUtensorMap``s are encoded with ``cuTensorMapEncodeTiled``
exactly as DeepGEMM's ``make_tma_2d_desc``/``make_tma_3d_desc`` do, and the
grid/block/shared-memory sizes follow ``csrc/jit_kernels/impls/
sm100_mqa_logits.hpp``. DeepGEMM's optional PDL launch attribute is not used;
both kernels' ``cudaGridDependencySynchronize`` is then a no-op.
"""

from __future__ import annotations

import ctypes
import math
import random
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from .. import cuda_driver
from ..registry import register
from ..throughput import model_case, skewed_lengths, synthetic, trace, upstream_case
from ..workload import CaseSpec, Workload

NUM_HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
SPLIT_KV = 256
NUM_SCHEDULING_PARTITIONS = 148  # template argument of the metadata cubin
PAGE_BYTES = PAGE_SIZE * (HEAD_DIM + 4)  # 8192 FP8 bytes, then 64 FP32 scales
SM100_SMEM_CAPACITY = 232448  # DeepGEMM SM100ArchSpec::smem_capacity

# Raw CUtensorMap* enum values from cuda.h.
SWIZZLE_NONE, SWIZZLE_128B = 0, 3
L2_PROMOTION_256B = 3
OOB_FILL_NONE = 0
INTERLEAVE_NONE = 0

# DeepGEMM sm100_paged_mqa_logits launch configuration (FP8, 64 heads).
NUM_SPECIALIZED_THREADS, NUM_MATH_THREADS = 128, 256
NUM_Q_STAGES, NUM_KV_STAGES, NUM_TMEM_STAGES = 3, 5, 3
BLOCK_Q = 128 // NUM_HEADS
METADATA_THREADS = 1024


class LaunchSpec(NamedTuple):
    grid: int
    block: int
    args: list[Any]
    shared_mem: int


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def logits_smem_size() -> int:
    """DeepGEMM get_mqa_logits_smem_size for FP8 Q/KV, FP32 weights, no MX SF."""
    swizzle_alignment = 8 * HEAD_DIM
    umma_n = _align(BLOCK_Q * NUM_HEADS, 8)
    size = 0

    def region(nbytes: int, alignment: int) -> None:
        nonlocal size
        size = _align(size, alignment) + nbytes

    region(NUM_Q_STAGES * umma_n * HEAD_DIM, swizzle_alignment)
    region(NUM_KV_STAGES * SPLIT_KV * HEAD_DIM, swizzle_alignment)
    region(NUM_Q_STAGES * 1 * 4, 128)  # num_sf_q = 1 without MX scales
    region(NUM_KV_STAGES * SPLIT_KV * 4, 128)
    weight_row = _align(NUM_HEADS * 4, 16)
    region(NUM_Q_STAGES * _align(BLOCK_Q * weight_row, 128), 128)
    barrier = 8  # cutlass::arch::ClusterTransactionBarrier
    region(2 * (NUM_Q_STAGES + NUM_KV_STAGES + NUM_TMEM_STAGES) * barrier, barrier)
    region(4, 4)
    return _align(size, swizzle_alignment)


LOGITS_SMEM = logits_smem_size()
assert LOGITS_SMEM == 220160  # static_assert in kernels/fp8_paged_mqa_logits.cu


def logits_width(table_width: int) -> int:
    """DeepGEMM's logits row stride: ``align(align(max_context_len, 256),
    1024 / sizeof(float))`` with ``max_context_len = table_width * 64``."""
    return _align(_align(table_width * PAGE_SIZE, SPLIT_KV), 1024 // 4)


def paged_mqa_logits_metadata(
    seq_lens: torch.Tensor, partitions: int = NUM_SCHEDULING_PARTITIONS
) -> torch.Tensor:
    """Torch reimplementation of sm100_paged_mqa_logits_metadata<1, true,
    false, 256, partitions> (next_n = 1, not varlen): int32[partitions + 1, 2]."""
    lens = seq_lens.reshape(-1).to(torch.int64)
    n = lens.numel()
    prefix = torch.cumsum((lens + SPLIT_KV - 1) // SPLIT_KV, 0)  # inclusive
    total = int(prefix[-1]) if n else 0
    quotient, remainder = divmod(total, partitions)
    sm = torch.arange(partitions + 1, device=lens.device)
    work = sm * quotient + sm.clamp(max=remainder)
    # The kernel's binary search finds the first request whose prefix > work.
    request = torch.searchsorted(prefix, work, right=True)
    valid = request < n
    before = torch.cat((prefix.new_zeros(1), prefix))[request]  # exclusive prefix
    q_token = torch.where(valid, request, n)
    kv_split = torch.where(valid, work - before, 0)
    return torch.stack((q_token, kv_split), dim=1).to(torch.int32)


class _DSAIndexer(Workload):
    """Shared cases and input generation (official DSA indexer layouts)."""

    package = "dsa_indexer"

    def get_cases(self) -> list[CaseSpec]:
        return smoke_cases() + upstream_cases() + throughput_cases()

    def seq_lens(self, case: CaseSpec) -> torch.Tensor:
        return torch.tensor(
            case.params["lengths"], device=self.device, dtype=torch.int32
        )

    def indexer_inputs(self, case: CaseSpec) -> tuple[torch.Tensor, ...]:
        """(q, k_index_cache, weights, seq_lens, block_table) of the official
        definition: FP8 ``q[B, 64, 128]``; int8 cache ``[pages, 64, 1, 132]``
        holding per page 8192 FP8 key bytes then 64 FP32 scales; FP32
        ``weights[B, 64]``; int32 ``seq_lens[B]``; int32 ``block_table[B,
        width]``.

        Case parameters: ``lengths``; ``width`` (block-table width, default
        the longest sequence's page count); ``pages`` (pool size, default the
        pages the sequences use); ``shared`` (page IDs drawn with repetition,
        i.e. pages shared between rows and slots). Otherwise every used slot
        gets a distinct page; slots past a sequence's last page (never read)
        hold random valid pages.
        """
        g, p = self.generator(case), case.params
        lengths = p["lengths"]
        b = len(lengths)
        used = [math.ceil(n / PAGE_SIZE) for n in lengths]
        width = p.get("width", max(1, max(used)))
        if max(used) > width:
            raise ValueError("block table must cover every sequence")
        pages = p.get("pages", max(1, sum(used)))
        q = self.randn((b, NUM_HEADS, HEAD_DIM), g, torch.float8_e4m3fn)
        keys = torch.empty(
            (pages, PAGE_SIZE * HEAD_DIM), device=self.device, dtype=torch.uint8
        )
        for start in range(0, pages, 1 << 16):  # bounded FP32 temporaries
            chunk = keys[start : start + (1 << 16)]
            chunk.view(torch.float8_e4m3fn).copy_(
                torch.randn(chunk.shape, device=self.device, generator=g)
            )
        scales = torch.rand((pages, PAGE_SIZE), device=self.device, generator=g)
        scales = scales * 0.1 + 0.01
        packed = torch.cat((keys, scales.view(torch.uint8)), dim=1)
        table = torch.randint(pages, (b, width), generator=g, device=self.device)
        if not p.get("shared"):
            if pages < sum(used):
                raise ValueError("page pool must hold the non-shared block table")
            order = torch.randperm(pages, generator=g, device=self.device)
            slots = torch.arange(width, device=self.device)
            mask = slots[None] < torch.tensor(used, device=self.device)[:, None]
            table[mask] = order[: sum(used)]
        return (
            q,
            packed.view(torch.int8).reshape(pages, PAGE_SIZE, 1, HEAD_DIM + 4),
            self.randn((b, NUM_HEADS), g, torch.float32),
            self.seq_lens(case),
            table.int(),
        )

    @abstractmethod
    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        """Allocate outputs and build the complete launch; no kernel runs."""

    def run(self, inputs: tuple) -> tuple:
        spec, outputs = self.configure_launch(inputs)
        self.launch(spec.grid, spec.block, spec.args, shared_mem=spec.shared_mem)
        return outputs

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple | None]:
        """Marshal once; the timed callable only launches into reused outputs."""
        spec, outputs = self.configure_launch(inputs)

        def launch() -> tuple:
            self.launch(spec.grid, spec.block, spec.args, shared_mem=spec.shared_mem)
            return outputs

        return launch, outputs

    @staticmethod
    def check_output(ref: tuple, impl: tuple) -> None:
        if not (isinstance(ref, tuple) and isinstance(impl, tuple)):
            raise AssertionError("outputs must be tuples")
        if len(ref) != 1 or len(impl) != 1:
            raise AssertionError(f"expected one output, got {len(ref)} and {len(impl)}")
        expected, actual = ref[0], impl[0]
        if not isinstance(actual, torch.Tensor):
            raise AssertionError("output must be a torch tensor")
        if (actual.shape, actual.dtype, actual.device) != (
            expected.shape,
            expected.dtype,
            expected.device,
        ):
            raise AssertionError(
                f"output {tuple(actual.shape)} {actual.dtype} {actual.device} != "
                f"{tuple(expected.shape)} {expected.dtype} {expected.device}"
            )


@register(name="dsa_indexer_metadata", supported_arches=("sm_100a",))
class DSAIndexerMetadata(_DSAIndexer):
    """sm100_paged_mqa_logits_metadata<1, true, false, 256, 148>.

    Input: int32 ``seq_lens[B]`` (DeepGEMM's 2D ``context_lens[B, 1]`` has the
    same memory). Output: int32 ``schedule_meta[149, 2]``, every element
    written. Integer output: exact match.
    """

    def get_inputs(self, case: CaseSpec) -> tuple:
        return (self.seq_lens(case),)

    def get_reference(self, inputs: tuple) -> tuple:
        return (paged_mqa_logits_metadata(inputs[0]),)

    def configure(self, function: cuda_driver.Function) -> None:
        static = function.get_attribute(cuda_driver.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SM100_SMEM_CAPACITY - static,
        )

    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        (seq_lens,) = inputs
        if seq_lens.dtype != torch.int32 or not seq_lens.is_contiguous():
            raise ValueError("seq_lens must be contiguous int32")
        if seq_lens.device.type != "cuda":
            raise ValueError("native launch needs CUDA tensors")
        batch = seq_lens.numel()
        # Request work prefix [B], warp sums [32] and the block total, as int32.
        shared_mem = (batch + METADATA_THREADS // 32 + 1) * 4
        if not batch or shared_mem > SM100_SMEM_CAPACITY:
            raise ValueError(f"unsupported batch size {batch}")
        schedule = torch.empty(
            (NUM_SCHEDULING_PARTITIONS + 1, 2),
            dtype=torch.int32,
            device=seq_lens.device,
        )
        args = [
            ctypes.c_uint32(batch),  # num_requests
            ctypes.c_uint32(batch),  # num_q_tokens_total (next_n = 1)
            seq_lens,  # context_lens
            ctypes.c_void_p(None),  # indices (not varlen)
            schedule,  # schedule_meta
        ]
        return LaunchSpec(1, METADATA_THREADS, args, shared_mem), (schedule,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        self.check_output(ref, impl)
        self.assert_close(ref, impl, rtol=0, atol=0)


@register(name="dsa_indexer_logits", supported_arches=("sm_100a",))
class DSAIndexerLogits(_DSAIndexer):
    """sm100_paged_mqa_logits (FP8, 64 heads, head_dim 128, page 64).

    Inputs: the official ``(q, k_index_cache, weights, seq_lens, block_table)``
    plus ``schedule_meta`` from :func:`paged_mqa_logits_metadata` (the exact
    output of ``dsa_indexer_metadata``). Output: FP32 ``logits[B,
    align(width * 64, 256)]`` with ``width = block_table.shape[1]``; see the
    module docstring for the layout and the written region.

    Tolerance: FP8 x FP8 products are exact in FP32 and the kernel accumulates
    and reduces in FP32, while the reference uses FP64. For the generated
    logits (magnitudes up to ~10) the expected difference is ~1e-6, so
    rtol 1e-3 / atol 2e-4 leaves a wide margin for summation order and
    cancellation without hiding a wrong key, scale, weight or page.
    """

    rtol, atol = 1e-3, 2e-4
    # Replaceable for argument-building checks on GPUs without TMA (sm_86
    # drivers reject cuTensorMapEncodeTiled with CUDA_ERROR_NOT_SUPPORTED).
    tensor_map_encoder = staticmethod(cuda_driver.encode_tensor_map_tiled)

    def get_inputs(self, case: CaseSpec) -> tuple:
        inputs = self.indexer_inputs(case)
        return inputs + (paged_mqa_logits_metadata(inputs[3]),)

    def get_reference(self, inputs: tuple) -> tuple:
        q, cache, weights, seq_lens, table, _ = inputs
        pages = cache.shape[0]
        flat = cache.view(torch.uint8).reshape(pages, PAGE_BYTES)
        out = torch.full(
            (q.shape[0], logits_width(table.shape[1])),
            torch.nan,
            dtype=torch.float32,
            device=q.device,
        )
        for b, length in enumerate(seq_lens.tolist()):
            columns = math.ceil(length / SPLIT_KV) * SPLIT_KV
            if not columns:
                continue
            # Page slots past the sequence (possibly past the table's width)
            # load physical page 0, as the kernel.
            used = math.ceil(length / PAGE_SIZE)
            page_ids = torch.zeros(
                columns // PAGE_SIZE, dtype=torch.long, device=q.device
            )
            page_ids[:used] = table[b, :used]
            selected = flat[page_ids]
            keys = (
                selected[:, : PAGE_SIZE * HEAD_DIM]
                .contiguous()
                .view(torch.float8_e4m3fn)
                .reshape(columns, HEAD_DIM)
                .double()
            )
            scale = (
                selected[:, PAGE_SIZE * HEAD_DIM :]
                .contiguous()
                .view(torch.float32)
                .reshape(columns)
                .double()
            )
            scores = (q[b].double() @ keys.T).relu()  # [heads, columns]
            out[b, :columns] = (
                (scores * weights[b, :, None].double()).sum(0) * scale
            ).float()
        return (out,)

    def configure(self, function: cuda_driver.Function) -> None:
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, LOGITS_SMEM
        )

    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        q, cache, weights, seq_lens, table, schedule = inputs
        if any(t.device.type != "cuda" for t in inputs):
            raise ValueError("native launch needs CUDA tensors")
        batch = q.shape[0]
        pages = cache.shape[0]
        if (
            q.dtype != torch.float8_e4m3fn
            or q.shape[1:] != (NUM_HEADS, HEAD_DIM)
            or not q.is_contiguous()
        ):
            raise ValueError("q must be contiguous float8_e4m3fn [B, 64, 128]")
        if (
            cache.dtype not in (torch.int8, torch.uint8)
            or cache.shape[1:] != (PAGE_SIZE, 1, HEAD_DIM + 4)
            or cache.stride()[1:] != (HEAD_DIM + 4, HEAD_DIM + 4, 1)
            or cache.stride(0) % 16
        ):
            raise ValueError("k_index_cache must be [pages, 64, 1, 132] bytes")
        if weights.dtype != torch.float32 or weights.shape != (batch, NUM_HEADS):
            raise ValueError("weights must be float32 [B, 64]")
        if weights.stride(1) != 1 or weights.stride(0) * 4 % 16:
            raise ValueError("weights rows must be unit-stride and 16-byte aligned")
        if seq_lens.dtype != torch.int32 or seq_lens.shape != (batch,):
            raise ValueError("seq_lens must be int32 [B]")
        if (
            table.dtype != torch.int32
            or table.dim() != 2
            or table.shape[0] != batch
            or table.stride(1) != 1
        ):
            raise ValueError("block_table must be int32 [B, width] with unit stride")
        if schedule.dtype != torch.int32 or schedule.shape[1:] != (2,):
            raise ValueError("schedule_meta must be int32 [partitions + 1, 2]")
        if not (seq_lens.is_contiguous() and schedule.is_contiguous()):
            raise ValueError("seq_lens and schedule_meta must be contiguous")
        longest = int(seq_lens.max()) if batch else 0
        if not batch or longest > table.shape[1] * PAGE_SIZE:
            raise ValueError("block_table must cover the longest sequence")
        width = logits_width(table.shape[1])
        for tensor in (q, cache, weights):
            if tensor.data_ptr() % 16:
                raise ValueError("TMA sources must be 16-byte aligned")

        logits = torch.empty((batch, width), dtype=torch.float32, device=q.device)
        page_stride = cache.stride(0)  # bytes
        address = cache.data_ptr()
        u8 = cuda_driver.CU_TENSOR_MAP_DATA_TYPE[torch.uint8]
        f32 = cuda_driver.CU_TENSOR_MAP_DATA_TYPE[torch.float32]

        def tma(dtype, ptr, dims, strides, box, swizzle):
            return self.tensor_map_encoder(
                dtype,
                ptr,
                dims,
                strides,
                box,
                [1] * len(dims),
                interleave=INTERLEAVE_NONE,
                swizzle=swizzle,
                l2_promotion=L2_PROMOTION_256B,
                oob_fill=OOB_FILL_NONE,
            )

        # make_tma_2d_desc(q, 128, B * 64, 128, BLOCK_Q * 64, q.stride(2), 128):
        # the swizzle mode (128 B) sets the inner box to 128 FP8 elements.
        tensor_map_q = tma(
            u8,
            q.data_ptr(),
            [HEAD_DIM, batch * NUM_HEADS],
            [q.stride(1)],
            [HEAD_DIM, BLOCK_Q * NUM_HEADS],
            SWIZZLE_128B,
        )
        # make_tma_3d_desc(kv, 128, 64, pages, 128, 64, 1, 128, page_stride, 128)
        tensor_map_kv = tma(
            u8,
            address,
            [HEAD_DIM, PAGE_SIZE, pages],
            [HEAD_DIM, page_stride],
            [HEAD_DIM, PAGE_SIZE, 1],
            SWIZZLE_128B,
        )
        # make_tma_2d_desc(kv_sf, 64, pages, 64, 1, page_stride / 4, 0): the
        # FP32 scales follow the 64 * 128 key bytes of each page.
        tensor_map_sf_kv = tma(
            f32,
            address + PAGE_SIZE * HEAD_DIM,
            [PAGE_SIZE, pages],
            [page_stride],
            [PAGE_SIZE, 1],
            SWIZZLE_NONE,
        )
        # FP8 (non-MX) leaves the sf_q slot unused; DeepGEMM passes sf_kv.
        tensor_map_sf_q = cuda_driver.TensorMap.from_buffer_copy(tensor_map_sf_kv)
        # make_tma_2d_desc(weights, 64, B, align(64 * 4, 16) / 4, BLOCK_Q,
        #                  weights.stride(0), 0)
        tensor_map_weights = tma(
            f32,
            weights.data_ptr(),
            [NUM_HEADS, batch],
            [weights.stride(0) * 4],
            [_align(NUM_HEADS * 4, 16) // 4, BLOCK_Q],
            SWIZZLE_NONE,
        )
        args = [
            ctypes.c_uint32(batch),  # num_q_tokens_total (next_n = 1)
            ctypes.c_uint32(logits.stride(0)),  # logits_stride
            ctypes.c_uint32(table.stride(0)),  # block_table_stride
            seq_lens,  # context_lens
            logits,
            table,
            ctypes.c_void_p(None),  # indices (not varlen)
            schedule,  # schedule_meta
            tensor_map_q,
            tensor_map_sf_q,
            tensor_map_kv,
            tensor_map_sf_kv,
            tensor_map_weights,
        ]
        grid = schedule.shape[0] - 1  # one persistent CTA per partition
        block = NUM_SPECIALIZED_THREADS + NUM_MATH_THREADS
        return LaunchSpec(grid, block, args, LOGITS_SMEM), (logits,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        """Compare the columns the kernel writes (non-NaN in the reference)."""
        self.check_output(ref, impl)
        written = ~torch.isnan(ref[0])
        self.assert_close(
            (ref[0][written],), (impl[0][written],), rtol=self.rtol, atol=self.atol
        )


# --- cases ---------------------------------------------------------------------------

THROUGHPUT = "dsa_indexer"
MODEL = "deepseek_v3"  # DeepSeek-V3.2 indexer (index_n_heads 64, head dim 128)
MODEL_LAYER = "DSA lightning indexer (64 heads, head dim 128, FP8 keys)"
DEEPGEMM_REVISION = "78b69000794d0937b47ae3387eff7663410264d1"
DEEPGEMM_TEST = "tests/test_attention.py::test_paged_mqa_logits"
# DeepGEMM test_paged_mqa_logits (enumerate_paged_mqa_logits on SM100) in the
# configuration these kernels serve: non-varlen FP8, FP32 logits and weights,
# block_kv 64, 2D context lengths, no clean_logits, next_n 1, 64 heads, head
# dim 128: (batch_size, avg_kv) with batch * avg_kv <= 32Mi pool tokens.
UPSTREAM_PAGED = ((256, 8192), (256, 65536), (4096, 8192))
# Official inventory rows (batch_size, max_num_pages): the latency rows of
# every batch size plus the widest rows (num_pages = 11923 for all).
TRACE_ROWS = (
    (1, 1),
    (1, 3),
    (2, 8),
    (4, 45),
    (8, 45),
    (12, 82),
    (15, 91),
    (16, 43),
    (30, 91),
    (31, 43),
)


def bounded_lengths(batch: int, longest: int, seed: int) -> list[int]:
    """Seeded long-tailed lengths in ``[1, longest]`` whose maximum is
    exactly ``longest`` (an inventory row records only the table width)."""
    if batch == 1:
        return [longest]
    rest = skewed_lengths(batch * longest // 3, batch - 1, seed=seed, minimum=1)
    lengths = [min(n, longest) for n in rest] + [longest]
    random.Random(seed).shuffle(lengths)
    return lengths


def smoke_cases() -> list[CaseSpec]:
    """Every hard path of the metadata and logits kernels, moderately sized."""
    skewed = skewed_lengths(300000, 12, seed=5, sigma=1.0, minimum=1)
    many = skewed_lengths(150000, 1100, seed=6, sigma=1.5, minimum=0)
    return [
        CaseSpec("short_and_empty", {"lengths": [0, 71]}, 1),
        CaseSpec("page_boundary", {"lengths": [2048, 2053]}, 2),
        CaseSpec("long", {"lengths": [4096]}, 3),
        # Official 43-page block table: the logits stride (2816) is not the
        # table width * 64 (2752); lengths skewed, longest = 43 * 64.
        CaseSpec(
            "official_width43",
            {"lengths": bounded_lengths(16, 43 * 64, seed=4), "width": 43},
            4,
        ),
        # ~1200 splits: persistent CTAs take ~8 splits each (5 KV stages
        # wrap), partition starts fall inside requests; empty, single-token,
        # page/split-boundary rows; unused block-table slots.
        CaseSpec(
            "skewed_wrap",
            {"lengths": [*skewed, 0, 1, 64, 256, 257], "width": 1400},
            5,
        ),
        # Pages shared between rows and slots (prefix sharing).
        CaseSpec(
            "shared_pages",
            {"lengths": [5000, 5000, 3000, 64, 1, 0], "pages": 90, "shared": True},
            6,
        ),
        # More than 1024 requests: the metadata kernel's per-thread multi-item
        # prefix scan and request loop; some empty rows.
        CaseSpec("requests_1100", {"lengths": many}, 7),
    ]


def upstream_cases() -> list[CaseSpec]:
    cases = []
    for b, avg in UPSTREAM_PAGED:
        rng = torch.Generator().manual_seed(b * 7 + avg)
        # randint(0.7 * avg_kv, 1.3 * avg_kv) per request, as upstream.
        lengths = torch.randint(int(0.7 * avg), int(1.3 * avg), (b,), generator=rng)
        cases.append(
            upstream_case(
                f"upstream_paged_b{b}_avg{avg}",
                {"lengths": lengths.tolist()},
                f"{DEEPGEMM_TEST}[is_varlen=False,fmt=fp8,logits=float32,"
                f"weights=float32,block_kv=64,batch_size={b},next_n=1,"
                f"num_heads=64,head_dim=128,avg_kv={avg}]",
                suite="throughput",
                seed=b + avg,
                revision=DEEPGEMM_REVISION,
            )
        )
    return cases


def throughput_cases():
    """Throughput (benchmark) cases of this package."""
    name = THROUGHPUT
    result = [
        trace(
            name,
            f"batch{b}_pages{width}",
            {
                "lengths": bounded_lengths(b, width * PAGE_SIZE, seed=b * 100 + width),
                "pages": 11923,
                "width": width,
            },
            dict(batch_size=b, max_num_pages=width, num_pages=11923),
        )
        for b, width in TRACE_ROWS
    ]
    # DeepSeek-V3.2 long-context decode: each request scores 64k-128k keys.
    for b, context in ((4, 65536), (8, 131072)):
        result.append(
            model_case(
                f"batch{b}_context{context}",
                {"lengths": [context] * b},
                MODEL,
                MODEL_LAYER,
            )
        )
    result.append(
        synthetic(
            "batch128_context8192_skewed",
            {"lengths": skewed_lengths(128 * 8192, 128, seed=8192, minimum=1)},
            "1M scored cache positions over 128 long-tailed requests.",
        )
    )
    return result

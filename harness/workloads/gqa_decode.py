"""GQA paged decode as single-kernel workloads (one cubin, one kernel each).

Official definition: ``gqa_paged_decode_h32_kv4_d128_ps1``
(``resources/gqa_decode.json``): 32 query heads sharing 4 KV heads (GQA group
8), head dim 128, page size 1, BF16 ``q [B, 32, 128]``, ``k_cache``/``v_cache
[pages, 1, 4, 128]``, ``kv_indptr [B+1]``, ``kv_indices [nnz]`` (int32),
``sm_scale`` (f32 scalar); outputs ``(output [B, 32, 128] bf16, lse [B, 32] f32
base 2)``.

The previous native package ran one kernel on sm_86 and four on sm_100a. Each
is now its own workload (``impls/gqa_decode/compiler.py`` documents which
compiled kernels are live and why):

=========================  =======  ==============================================
workload                   arch     kernel
=========================  =======  ==============================================
gqa_decode_batch_decode    sm_86    FlashInfer ``BatchDecodeWithPagedKVCacheKernel``
gqa_decode_plan            sm_100a  ``gqa_plan`` (project adapter)
gqa_decode_gather          sm_100a  ``gqa_gather`` (project adapter)
gqa_decode_fmha            sm_100a  CUTLASS ``Sm100FmhaFwdKernelTmaWarpspecialized``
gqa_decode_finish          sm_100a  ``gqa_finish`` (project adapter)
=========================  =======  ==============================================

Every workload is self-contained: ``get_inputs`` generates the official inputs
and, for later stages, runs the PyTorch references of the preceding stages to
produce that kernel's exact inputs. Kernel arguments, including the by-value
``Params`` structs of the two upstream kernels and the FMHA's TMA descriptors,
are built in Python from the layouts the compile stage recorded next to each
cubin.

Cases (shared by every stage; a stage drops cases over its memory budget,
:data:`CASE_BUDGET`):

* own smoke cases: ragged and long-tailed (skewed) lengths, empty and
  length-1 requests (leading, trailing, consecutive), random, shared and
  identity page mappings, peaked attention (scaled ``q``) and a non-default
  ``sm_scale``, ``batch * 32`` work items well above the SM count;
* upstream parametrizations (``upstream_case``): FlashInfer's paged decode
  tests restricted to this definition (page size 1, 32/4 heads, head dim
  128, no RoPE/soft cap), its #4450 extreme-negative-logits fixture, and the
  decode-shaped (``qo_len = 1``, non-causal, head dim 128) CUTLASS Blackwell
  FMHA tests; ``tests/test_gqa_decode.py`` extracts them from the pinned
  upstream sources and requires each in ``get_cases()``;
* throughput: official inventory rows (all batch-1 latency rows, low/middle/
  high batch-16 and batch-64 rows, skewed lengths with the recorded totals),
  Qwen3-30B-A3B long-context decode (B = 1-4 at 32k-128k) and stress batches
  (B = 128-256 at 8k-32k).

Case parameters: ``lengths`` (KV length per request), optional ``pages``
(page pool size), ``sm_scale``, ``q_scale`` (multiplies ``q``), ``fill``
(constant ``q``/``k``/``v`` values), ``indices`` (explicit page IDs),
``contiguous`` (identity page table, as upstream tests) or ``shared``
(page IDs drawn with repetition, i.e. pages shared between requests).
"""

from __future__ import annotations

import ctypes
import json
import math
import struct
from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import torch

from .. import cuda_driver
from ..cutlass_host import cutlass_small_tensor, driver_encode
from ..registry import register
from ..throughput import inventory, model_case, skewed_lengths, trace, upstream_case
from ..workload import CaseSpec, Workload

HEADS = 32
KV_HEADS = 4
GROUP = HEADS // KV_HEADS
DIM = 128
KV_ROW = KV_HEADS * DIM  # elements of one packed K/V row (one token, all KV heads)
B200_SMS = 148  # SM count used for planning when no CUDA device is present
INT32_MAX = 2**31 - 1


class LaunchSpec(NamedTuple):
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[Any]
    shared_mem: int = 0
    cluster: tuple[int, int, int] | None = None
    # Device buffers referenced only from inside a by-value struct; kept alive
    # for as long as the launch description (e.g. the prepared callable) lives.
    keep: tuple[torch.Tensor, ...] = ()


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


# --- PyTorch references --------------------------------------------------------


def _attention(q_rows, keys, values, scale):
    """FP32 GQA attention of one query token over ``keys``/``values`` [L, 4, 128].

    Query head ``h`` reads KV head ``h // 8``; the 8 heads of a group share
    one matmul, so no repeated K/V copies are materialized. Returns ``(out
    [32, 128] f32, base-2 lse [32] f32)``.
    """
    q = q_rows.float().reshape(KV_HEADS, GROUP, DIM)
    k = keys.float().transpose(0, 1)  # [4, L, D]
    v = values.float().transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)) * scale  # [4, 8, L]
    out = scores.softmax(-1) @ v
    lse = scores.logsumexp(-1) / math.log(2)
    return out.reshape(HEADS, DIM), lse.reshape(HEADS)


def gqa_reference(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    """The official reference: empty requests give output 0 and LSE -inf."""
    out = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], -torch.inf, dtype=torch.float32, device=q.device)
    bounds = kv_indptr.cpu().tolist()
    for b, (start, end) in enumerate(zip(bounds, bounds[1:])):
        if start == end:
            continue
        idx = kv_indices[start:end].long()
        o, s = _attention(q[b], k_cache[idx, 0], v_cache[idx, 0], float(sm_scale))
        out[b], lse[b] = o.to(q.dtype), s
    return out, lse


def batch_decode_reference(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    """FlashInfer decode: the official result; empty requests are undefined (NaN)."""
    out, lse = gqa_reference(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)
    empty = kv_indptr[1:] == kv_indptr[:-1]
    out[empty] = torch.nan
    lse[empty] = torch.nan
    return out, lse


def plan_reference(kv_indptr: torch.Tensor, sms: int) -> tuple[torch.Tensor, ...]:
    """``gqa_plan``: (qo_offsets [B+1], kv_offsets [B+1], work_indptr [sms+1],
    qo_tile_indices, qo_head_indices, batch_indices [32 B])."""
    device, i32 = kv_indptr.device, torch.int32
    batch = kv_indptr.numel() - 1
    lengths = (kv_indptr[1:] - kv_indptr[:-1]).clamp(min=1)
    kv = torch.zeros(batch + 1, dtype=i32, device=device)
    kv[1:] = lengths.cumsum(0, dtype=i32)
    qo = torch.arange(batch + 1, dtype=i32, device=device)
    work = torch.tensor(
        [i * batch * HEADS // sms for i in range(sms + 1)], dtype=i32, device=device
    )
    items = torch.arange(batch * HEADS, dtype=i32, device=device)
    return qo, kv, work, torch.zeros_like(items), items % HEADS, items // HEADS


def gather_reference(k_cache, v_cache, kv_indptr, kv_indices, kv_offsets):
    """``gqa_gather`` into zero-initialized packed ``[nnz + B, 4, 128]`` K and V:
    request b's rows at ``kv_offsets[b]``, one zero row for an empty request."""
    batch, nnz = kv_indptr.numel() - 1, kv_indices.numel()
    packed_k = k_cache.new_zeros(nnz + batch, KV_HEADS, DIM)
    packed_v = v_cache.new_zeros(nnz + batch, KV_HEADS, DIM)
    bounds, starts = kv_indptr.cpu().tolist(), kv_offsets.cpu().tolist()
    for b, (start, end) in enumerate(zip(bounds, bounds[1:])):
        if start == end:
            continue  # the kernel writes one zero row: already zero
        idx = kv_indices[start:end].long()
        packed_k[starts[b] : starts[b] + end - start] = k_cache[idx, 0]
        packed_v[starts[b] : starts[b] + end - start] = v_cache[idx, 0]
    return packed_k, packed_v


def fmha_reference(q, packed_k, packed_v, qo_offsets, kv_offsets, sm_scale):
    """CUTLASS FMHA over ragged packed K/V (one query token per request):
    ``(output [B, 32, 128] bf16, base-2 lse [B, 32] f32)``."""
    batch = q.shape[0]
    out = torch.empty_like(q)
    lse = torch.empty(batch, HEADS, dtype=torch.float32, device=q.device)
    qo, kv = qo_offsets.cpu().tolist(), kv_offsets.cpu().tolist()
    for b in range(batch):
        if qo[b + 1] - qo[b] != 1:
            raise ValueError("each request must have exactly one query token")
        o, s = _attention(
            q[qo[b]], packed_k[kv[b] : kv[b + 1]], packed_v[kv[b] : kv[b + 1]], sm_scale
        )
        out[b], lse[b] = o.to(q.dtype), s
    return out, lse


def finish_reference(fmha_out, kv_indptr, fmha_lse):
    empty = kv_indptr[1:] == kv_indptr[:-1]
    out, lse = fmha_out.clone(), fmha_lse.clone()
    out[empty] = 0
    lse[empty] = -torch.inf
    return out, lse


# --- by-value parameter structs and TMA descriptors -------------------------------


class ParamLayout:
    """A struct layout recorded by an ``impls/gqa_decode`` probe."""

    _formats = {
        "ptr": "<Q",
        "i32": "<i",
        "u32": "<I",
        "i64": "<q",
        "f32": "<f",
        "bool": "<?",
    }

    def __init__(self, layout: dict[str, Any]):
        self.raw = layout
        self.size: int = layout["param_size"]
        self.fields: dict[str, dict[str, Any]] = layout["fields"]
        self.constants: dict[str, Any] = layout["constants"]

    @classmethod
    def load(cls, path: Path) -> ParamLayout:
        return cls(json.loads(path.read_text())["param_layout"])

    def pack(self, values: dict[str, Any]) -> bytes:
        """Every recorded field must be given; padding stays zero."""
        missing = set(self.fields) - set(values)
        extra = set(values) - set(self.fields)
        if missing or extra:
            raise ValueError(f"param fields missing {missing}, unknown {extra}")
        buffer = bytearray(self.size)
        for name, value in values.items():
            field = self.fields[name]
            offset, size, kind = field["offset"], field["size"], field["kind"]
            if kind in ("bytes", "tma"):
                data = bytes(value)
                if len(data) != size:
                    raise ValueError(f"{name}: {len(data)} bytes, expected {size}")
                buffer[offset : offset + size] = data
                continue
            fmt = self._formats[kind]
            if struct.calcsize(fmt) != size:
                raise ValueError(f"{name}: {kind} is not {size} bytes")
            struct.pack_into(fmt, buffer, offset, value)
        return bytes(buffer)

    def covered(self) -> bytearray:
        """1 for every byte inside a recorded field (the rest is padding)."""
        mask = bytearray(self.size)
        for field in self.fields.values():
            start = field["offset"]
            mask[start : start + field["size"]] = b"\x01" * field["size"]
        return mask

    def sentinels(self) -> dict[str, int]:
        stride = self.raw["sentinel_stride"]
        return {
            role: stride * (i + 1) for i, role in enumerate(self.raw["pointer_roles"])
        }


# Encodes one CUtensorMap: (data_type, address, dims, strides_bytes, box,
# element_strides, interleave, swizzle, l2_promotion, oob_fill) -> 128 bytes.
TensorMapEncoder = Callable[..., bytes]


def fake_encode(
    data_type: int,
    address: int,
    dims: Sequence[int],
    strides: Sequence[int],
    box: Sequence[int],
    element_strides: Sequence[int],
    interleave: int,
    swizzle: int,
    l2_promotion: int,
    oob_fill: int,
) -> bytes:
    """The FMHA probe's deterministic CUtensorMap stand-in (its interposed
    ``cuTensorMapEncodeTiled`` in ``impls/gqa_decode/kernels/
    probe_sm100a_fmha_params.cu``); for tests only."""
    out = bytearray(128)
    out[0:6] = bytes(
        (data_type, len(dims), interleave, swizzle, l2_promotion, oob_fill)
    )
    struct.pack_into("<Q", out, 8, address & 0xFFFFFFFFFFFFFFFF)
    struct.pack_into(f"<{len(dims)}Q", out, 16, *dims)
    struct.pack_into(f"<{len(strides)}Q", out, 56, *strides)
    struct.pack_into(f"<{len(box)}I", out, 88, *box)
    struct.pack_into(f"<{len(element_strides)}I", out, 108, *element_strides)
    return bytes(out)


_ELEMENT_BYTES = {0: 1, 1: 2, 3: 4, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2}


def encode_descriptor(
    spec: dict[str, Any],
    pointers: dict[str, int],
    sizes: dict[str, int],
    encode: TensorMapEncoder,
    driver: int,
) -> bytes:
    """Encode one recorded TMA descriptor like CUTLASS ``make_tma_copy_desc``.

    CUTLASS clears bit 21 of word 1 for tensors it sizes under 128 KiB
    (``cutlass_small_tensor``) when the driver
    is at most 13.1 (a driver workaround in ``copy_traits_sm90_tma.hpp``);
    mirrored here. The tensor's cosize follows from the encode arguments.
    """

    def resolve(value):
        if isinstance(value, dict):
            return sizes[value["var"]] * value["scale"] + value["offset"]
        return value

    dims = [resolve(v) for v in spec["global_dims"]]
    strides = [resolve(v) for v in spec["global_strides"]]
    address = pointers[spec["address"]["pointer"]] + spec["address"]["offset"]
    data = bytearray(
        encode(
            spec["data_type"],
            address,
            dims,
            strides,
            spec["box_dims"],
            spec["element_strides"],
            spec["interleave"],
            spec["swizzle"],
            spec["l2_promotion"],
            spec["oob_fill"],
        )
    )
    element = _ELEMENT_BYTES[spec["data_type"]]
    cosize = element * (
        dims[0] + sum((d - 1) * s // element for d, s in zip(dims[1:], strides))
    )
    if driver <= 13010 and cutlass_small_tensor(cosize):
        word = int.from_bytes(data[8:16], "little") & ~(1 << 21)
        data[8:16] = word.to_bytes(8, "little")
    return bytes(data)


# --- workloads --------------------------------------------------------------------


class _GQADecodeStage(Workload):
    """Shared cases and official inputs; one subclass per kernel."""

    package = "gqa_decode"
    rtol = 0.0
    atol = 0.0

    def get_cases(self) -> list[CaseSpec]:
        """Every package case this stage serves within its memory budget."""
        packed = self.supported_arches != ("sm_86",)
        budget = CASE_BUDGET[self.supported_arches[0]]
        return [c for c in all_cases() if case_bytes(c.params, packed) <= budget]

    def official_inputs(self, case: CaseSpec) -> tuple[torch.Tensor, ...]:
        """(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale) of ``case``
        (parameters in the module docstring)."""
        g, p = self.generator(case), case.params
        lengths = p["lengths"]
        total, batch = sum(lengths), len(lengths)
        pages = p.get("pages", total)
        shape = (pages, 1, KV_HEADS, DIM)
        if "fill" in p:
            fill = p["fill"]
            q = torch.full((batch, HEADS, DIM), fill["q"], device=self.device)
            q = q.to(torch.bfloat16)
            k, v = (
                torch.full(shape, fill[n], device=self.device, dtype=torch.bfloat16)
                for n in "kv"
            )
        else:
            q = self.randn((batch, HEADS, DIM), g, scale=p.get("q_scale", 1.0))
            # Generated directly in BF16: no FP32 copy of multi-GiB caches.
            k, v = (
                torch.randn(shape, generator=g, device=self.device, dtype=q.dtype)
                for _ in range(2)
            )
        if "indices" in p:
            indices = torch.tensor(p["indices"], device=self.device)
        elif p.get("contiguous"):
            indices = torch.arange(total, device=self.device)
        elif p.get("shared"):
            indices = torch.randint(pages, (total,), generator=g, device=self.device)
        else:
            if pages < total:
                raise ValueError("page pool must hold the non-shared K/V indices")
            indices = torch.randperm(pages, generator=g, device=self.device)[:total]
        if indices.numel() != total or (total and int(indices.max()) >= pages):
            raise ValueError("page IDs must index the page pool, one per KV token")
        return (
            q,
            k,
            v,
            torch.tensor([0, *lengths], device=self.device, dtype=torch.int32).cumsum(
                0, dtype=torch.int32
            ),
            indices.to(torch.int32),
            self.scalar(p.get("sm_scale", 1 / math.sqrt(DIM))),
        )

    def plan_sms(self) -> int:
        """The SM count ``gqa_plan`` splits work over (the FMHA grid size)."""
        if self.device.type == "cuda":
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            return torch.cuda.get_device_properties(index).multi_processor_count
        return B200_SMS

    # Each subclass: outputs + launch description for given inputs.
    def setup_launch(self, inputs: tuple) -> tuple[tuple, LaunchSpec]:
        raise NotImplementedError

    def _launch(self, spec: LaunchSpec) -> None:
        self.launch(
            spec.grid,
            spec.block,
            spec.args,
            shared_mem=spec.shared_mem,
            cluster=spec.cluster,
        )

    def run(self, inputs: tuple) -> tuple:
        outputs, spec = self.setup_launch(inputs)
        self._launch(spec)
        return outputs

    def prepare(self, inputs: tuple):
        outputs, spec = self.setup_launch(inputs)

        def launch() -> tuple:
            self._launch(spec)
            return outputs

        return launch, outputs

    def validate(self, ref: tuple, impl: tuple) -> None:
        self.assert_close(ref, impl, rtol=self.rtol, atol=self.atol)

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    def _check_tensor(self, name, tensor, shape, dtype) -> None:
        self._require(
            isinstance(tensor, torch.Tensor)
            and tuple(tensor.shape) == tuple(shape)
            and tensor.dtype == dtype
            and tensor.is_contiguous()
            and tensor.device.type == "cuda",
            f"{self.name}: {name} must be a contiguous CUDA {dtype} tensor of shape "
            f"{tuple(shape)}",
        )

    def _check_official(self, q, k_cache, v_cache, kv_indptr, kv_indices) -> int:
        batch, pages = q.shape[0], k_cache.shape[0]
        self._require(batch >= 1, "batch must be positive")
        for name, tensor, shape, dtype in (
            ("q", q, (batch, HEADS, DIM), torch.bfloat16),
            ("k_cache", k_cache, (pages, 1, KV_HEADS, DIM), torch.bfloat16),
            ("v_cache", v_cache, (pages, 1, KV_HEADS, DIM), torch.bfloat16),
            ("kv_indptr", kv_indptr, (batch + 1,), torch.int32),
            ("kv_indices", kv_indices, (kv_indices.shape[0],), torch.int32),
        ):
            self._check_tensor(name, tensor, shape, dtype)
        return batch

    @cached_property
    def param_layout(self) -> ParamLayout:
        if self.arch is None:
            raise RuntimeError("reference-only workload has no parameter layout")
        layout = ParamLayout.load(self.cubin_path(self.arch).with_suffix(".json"))
        size = sum(p.size for p in self.kernel_params)
        if layout.size != size:
            raise ValueError(
                f"{self.name}: recorded Params {layout.size} != cubin {size}"
            )
        return layout


def _validate_defined(workload: _GQADecodeStage, ref: tuple, impl: tuple) -> None:
    """Compare (output, lse) on requests whose reference is defined (no NaN)."""
    if not isinstance(ref, tuple) or len(ref) != 2:
        raise AssertionError("reference must be (output, lse)")
    if not isinstance(impl, tuple) or len(impl) != 2:
        raise AssertionError("outputs must be (output, lse)")
    for r, i in zip(ref, impl):
        if (
            not isinstance(i, torch.Tensor)
            or i.shape != r.shape
            or i.dtype != r.dtype
            or i.device != r.device
        ):
            raise AssertionError("output shape/dtype/device mismatch")
    defined = ~torch.isnan(ref[1]).any(-1)
    workload.assert_close(
        (ref[0][defined], ref[1][defined]),
        (impl[0][defined], impl[1][defined]),
        workload.rtol,
        workload.atol,
    )


# sm_86 --------------------------------------------------------------------------


@register(name="gqa_decode_batch_decode", supported_arches=("sm_86",))
class GQADecodeBatchDecode(_GQADecodeStage):
    """FlashInfer ``BatchDecodeWithPagedKVCacheKernel<kNone, 2, 1, 8, 16, 8, 1>``.

    Inputs: the official ``(q, k_cache, v_cache, kv_indptr, kv_indices,
    sm_scale)``; outputs ``(output, lse)`` with base-2 LSE. Requests with no
    KV page are undefined in the kernel (it reads no KV and guards its page
    prefetch with ``indptr[batch]``); smoke cases include them and validation
    excludes them.

    The by-value ``BatchDecodeParams<bf16, bf16, bf16, int>`` (232 bytes) is
    built as the previous host code did: ``paged_kv_t(num_heads=4,
    page_size=1, head_dim=128, batch, NHD, k, v, indices, indptr,
    last_page_len)`` (page stride 512, entry stride 512, head stride 128 for K
    and V), ``Params(q, q_rope_offset=nullptr, paged_kv, o, lse,
    alibi=nullptr, num_qo_heads=32, q_stride_n=4096, q_stride_h=128,
    window_left=-1, logits_soft_cap=0, sm_scale, rope_scale=1,
    rope_theta=1e4)`` (``rope_rcp_* = 1/x``), then ``padded_batch_size =
    batch`` and the metadata pointers. The metadata the old ``setup`` uploaded
    once is created here per launch (outside the timed callable in
    ``prepare``): ``request_indices = arange(batch)``, ``kv_tile_indices =
    0``, ``last_page_len = 1`` and ``kv_chunk_size = 0`` (unused with
    ``partition_kv = false``). Launch: grid (batch, 4), block (16, 8), 9216
    bytes of dynamic smem.
    """

    rtol = atol = 1e-2  # BF16 inputs/outputs, FP32 accumulation

    def get_inputs(self, case):
        return self.official_inputs(case)

    def get_reference(self, inputs):
        return batch_decode_reference(*inputs)

    @staticmethod
    def build_params(
        layout: ParamLayout, pointers: dict[str, int], batch: int, sm_scale: float
    ) -> bytes:
        page_size = 1
        stride_page = KV_HEADS * page_size * DIM
        return layout.pack(
            {
                "q": pointers["q"],
                "q_rope_offset": 0,
                "paged_kv.page_size": bytes.fromhex(
                    layout.constants["page_size_fastdiv"]
                ),
                "paged_kv.num_heads": KV_HEADS,
                "paged_kv.head_dim": DIM,
                "paged_kv.batch_size": batch,
                "paged_kv.stride_page": stride_page,
                "paged_kv.stride_n": KV_HEADS * DIM,  # NHD
                "paged_kv.stride_h": DIM,
                "paged_kv.v_stride_page": stride_page,
                "paged_kv.v_stride_n": KV_HEADS * DIM,
                "paged_kv.v_stride_h": DIM,
                "paged_kv.k_data": pointers["k_cache"],
                "paged_kv.v_data": pointers["v_cache"],
                "paged_kv.indices": pointers["kv_indices"],
                "paged_kv.indptr": pointers["kv_indptr"],
                "paged_kv.last_page_len": pointers["last_page_len"],
                "paged_kv.rope_pos_offset": 0,
                "o": pointers["o"],
                "lse": pointers["lse"],
                "maybe_alibi_slopes": 0,
                "padded_batch_size": batch,
                "num_qo_heads": HEADS,
                "q_stride_n": HEADS * DIM,
                "q_stride_h": DIM,
                "window_left": -1,
                "logits_soft_cap": 0.0,
                "sm_scale": sm_scale,
                "rope_rcp_scale": _f32(1.0 / 1.0),
                "rope_rcp_theta": _f32(1.0 / _f32(1e4)),
                "request_indices": pointers["request_indices"],
                "kv_tile_indices": pointers["kv_tile_indices"],
                "o_indptr": 0,
                "kv_chunk_size_ptr": pointers["kv_chunk_size"],
                "block_valid_mask": 0,
                "partition_kv": False,
            }
        )

    def setup_launch(self, inputs):
        q, k_cache, v_cache, kv_indptr, kv_indices, scale = inputs
        batch = self._check_official(q, k_cache, v_cache, kv_indptr, kv_indices)
        device, i32 = q.device, torch.int32
        output = torch.empty_like(q)
        lse = torch.empty(batch, HEADS, dtype=torch.float32, device=device)
        request_indices = torch.arange(batch, dtype=i32, device=device)
        kv_tile_indices = torch.zeros(batch, dtype=i32, device=device)
        last_page_len = torch.ones(batch, dtype=i32, device=device)
        kv_chunk_size = torch.zeros(1, dtype=i32, device=device)
        pointers = {
            "q": q.data_ptr(),
            "k_cache": k_cache.data_ptr(),
            "v_cache": v_cache.data_ptr(),
            "kv_indices": kv_indices.data_ptr(),
            "kv_indptr": kv_indptr.data_ptr(),
            "last_page_len": last_page_len.data_ptr(),
            "o": output.data_ptr(),
            "lse": lse.data_ptr(),
            "request_indices": request_indices.data_ptr(),
            "kv_tile_indices": kv_tile_indices.data_ptr(),
            "kv_chunk_size": kv_chunk_size.data_ptr(),
        }
        layout = self.param_layout
        params = self.build_params(layout, pointers, batch, _f32(float(scale)))
        spec = LaunchSpec(
            (batch, layout.constants["grid_y"], 1),
            tuple(layout.constants["block"]),
            [params],
            layout.constants["shared_mem"],
            keep=(request_indices, kv_tile_indices, last_page_len, kv_chunk_size),
        )
        return (output, lse), spec

    def validate(self, ref, impl):
        _validate_defined(self, ref, impl)


# sm_100a ------------------------------------------------------------------------


@register(name="gqa_decode_plan", supported_arches=("sm_100a",))
class GQADecodePlan(_GQADecodeStage):
    """``gqa_plan<<<1, 1>>>(kv_indptr, qo, kv, work, tiles, heads, batches,
    batch, sms)``: varlen offsets and host-precomputed scheduler work lists.

    Inputs ``(kv_indptr [B+1] i32, sms)`` with ``sms`` a CPU int32 scalar (the
    device's SM count, as the previous launcher queried; 148 without a GPU).
    Outputs ``(qo_offsets [B+1], kv_offsets [B+1], work_indptr [sms+1],
    qo_tile_indices [32B], qo_head_indices [32B], batch_indices [32B])``.
    """

    def get_inputs(self, case):
        kv_indptr = self.official_inputs(case)[3]
        return kv_indptr, self.scalar(self.plan_sms(), torch.int32)

    def get_reference(self, inputs):
        kv_indptr, sms = inputs
        return plan_reference(kv_indptr, int(sms))

    def setup_launch(self, inputs):
        kv_indptr, sms_tensor = inputs
        batch, sms = kv_indptr.shape[0] - 1, int(sms_tensor)
        self._require(batch >= 1 and sms >= 1, "batch and sms must be positive")
        self._require(batch * HEADS <= INT32_MAX, "batch * 32 must fit in int32")
        self._check_tensor("kv_indptr", kv_indptr, (batch + 1,), torch.int32)
        device, i32 = kv_indptr.device, torch.int32
        outputs = (
            torch.empty(batch + 1, dtype=i32, device=device),
            torch.empty(batch + 1, dtype=i32, device=device),
            torch.empty(sms + 1, dtype=i32, device=device),
            torch.empty(batch * HEADS, dtype=i32, device=device),
            torch.empty(batch * HEADS, dtype=i32, device=device),
            torch.empty(batch * HEADS, dtype=i32, device=device),
        )
        args = [kv_indptr, *outputs, ctypes.c_int(batch), ctypes.c_int(sms)]
        return outputs, LaunchSpec((1, 1, 1), (1, 1, 1), args)


class _SM100PlannedInputs(_GQADecodeStage):
    def planned_inputs(self, case):
        q, k, v, indptr, indices, scale = self.official_inputs(case)
        plan = plan_reference(indptr, self.plan_sms())
        return (q, k, v, indptr, indices, scale), plan


@register(name="gqa_decode_gather", supported_arches=("sm_100a",))
class GQADecodeGather(_SM100PlannedInputs):
    """``gqa_gather<<<batch, 256>>>``: page-size-1 K/V -> ragged packed rows.

    Inputs ``(k_cache, v_cache, kv_indptr, kv_indices, kv_offsets)`` with
    ``kv_offsets`` as written by ``gqa_plan``; outputs ``(packed_k, packed_v
    [nnz + B, 4, 128])``, the previous arena's K/V capacity. ``run``
    zero-initializes them (the arena was uninitialized), so rows past
    ``kv_offsets[B]`` are zero and every row is compared exactly.
    """

    def get_inputs(self, case):
        (_, k, v, indptr, indices, _), plan = self.planned_inputs(case)
        return k, v, indptr, indices, plan[1]

    def get_reference(self, inputs):
        return gather_reference(*inputs)

    def setup_launch(self, inputs):
        k_cache, v_cache, kv_indptr, kv_indices, kv_offsets = inputs
        batch, pages, nnz = (
            kv_indptr.shape[0] - 1,
            k_cache.shape[0],
            kv_indices.shape[0],
        )
        self._require(batch >= 1, "batch must be positive")
        for name, tensor, shape, dtype in (
            ("k_cache", k_cache, (pages, 1, KV_HEADS, DIM), torch.bfloat16),
            ("v_cache", v_cache, (pages, 1, KV_HEADS, DIM), torch.bfloat16),
            ("kv_indptr", kv_indptr, (batch + 1,), torch.int32),
            ("kv_indices", kv_indices, (nnz,), torch.int32),
            ("kv_offsets", kv_offsets, (batch + 1,), torch.int32),
        ):
            self._check_tensor(name, tensor, shape, dtype)
        packed_k = k_cache.new_zeros(nnz + batch, KV_HEADS, DIM)
        packed_v = v_cache.new_zeros(nnz + batch, KV_HEADS, DIM)
        args = [k_cache, v_cache, kv_indptr, kv_indices, kv_offsets, packed_k, packed_v]
        return (packed_k, packed_v), LaunchSpec((batch, 1, 1), (256, 1, 1), args)


class _SM100FmhaInputs(_SM100PlannedInputs):
    def fmha_inputs(self, case):
        (q, k, v, indptr, indices, scale), plan = self.planned_inputs(case)
        packed_k, packed_v = gather_reference(k, v, indptr, indices, plan[1])
        return (q, packed_k, packed_v, *plan, scale), indptr


@register(name="gqa_decode_fmha", supported_arches=("sm_100a",))
class GQADecodeFMHA(_SM100FmhaInputs):
    """CUTLASS ``Sm100FmhaFwdKernelTmaWarpspecialized`` (FlashInfer
    ``FwdRunner<bf16, bf16, int, 256x128x128, 256x128x128, ResidualMask>``:
    TMA loads, tcgen05 QK/PV UMMA, host-precomputed tile scheduler).

    Inputs ``(q [B,32,128], packed_k, packed_v [R,4,128], qo_offsets,
    kv_offsets, work_indptr, qo_tile_indices, qo_head_indices, batch_indices,
    sm_scale)`` as written by ``gqa_plan``/``gqa_gather``; outputs ``(output
    [B,32,128] bf16, lse [B,32] f32 base 2)`` (an empty request attends its
    one zero row: output 0, LSE 0; ``gqa_finish`` normalizes it).

    ``Params`` (1920 bytes) mirrors ``FwdRunner::run(..., num_qo_heads=32,
    num_kv_heads=4, head_dim 128/128, q strides (4096, 128), K/V strides
    (512, 128), batch_size=B, total_qo_len=B, total_kv_len=R, max_qo_len=1)``
    -> ``FMHA::initialize`` -> ``Kernel::to_underlying_arguments``:

    * problem shape ``(Varlen(qo_offsets), Varlen(kv_offsets), 128,
      ((8, 4), B))``;
    * mainloop: TMA Q/K/V descriptors and layouts Q ``(B, 128, (8, 4)):(4096,
      1, (128, 1024))``, K ``(R, 128, (8, 4)):(512, 1, (0, 128))``, V ``(128,
      R, (8, 4)):(1, 512, (0, 128))``; ``scale_softmax = sm_scale``,
      ``scale_softmax_log2 = log2(e) * sm_scale`` (float), ``scale_output =
      1``;
    * epilogue: TMA store of O from one row before ``o`` (layout ``(1, 128,
      ((8, 4), B + 1)):(4096, 1, ((128, 1024), 4096))``), LSE layout ``(B,
      (8, 4)):(32, (1, 8))``, ``max_qo_len = 1``;
    * tile scheduler: the four work lists and ``num_sm`` (= ``len(
      work_indptr) - 1``, the plan's SM count).

    The output buffer has B + 1 rows; row 0 is the upstream "o - max_qo_len"
    origin and never written; ``output`` is rows 1..B. Launch: grid
    ``num_sm``, block 512, cluster (1, 1, 1), dynamic smem
    ``SharedStorageSize`` (196864; the max-dynamic-smem attribute is set once).
    Upstream additionally requests programmatic dependent launch, which only
    allows overlap with the preceding kernel; ``Workload.launch`` does not
    expose that attribute, so it is not set (results are unaffected).
    """

    rtol = atol = 1e-2  # BF16 inputs/outputs, FP32 accumulation
    # Replaceable for argument-building checks on GPUs without TMA (sm_86
    # drivers reject cuTensorMapEncodeTiled with CUDA_ERROR_NOT_SUPPORTED).
    tensor_map_encoder: TensorMapEncoder = staticmethod(driver_encode)

    def get_inputs(self, case):
        return self.fmha_inputs(case)[0]

    def get_reference(self, inputs):
        q, packed_k, packed_v, qo, kv, *_, scale = inputs
        return fmha_reference(q, packed_k, packed_v, qo, kv, float(scale))

    def configure(self, function):
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            self.param_layout.constants["shared_mem"],
        )

    @staticmethod
    def build_params(
        layout: ParamLayout,
        pointers: dict[str, int],
        *,
        batch: int,
        total_kv: int,
        sm_count: int,
        sm_scale: float,
        encode: TensorMapEncoder,
        driver: int,
    ) -> tuple[bytes, int]:
        """(Params bytes, grid size). ``sm_scale`` must be a float32 value."""
        max_qo_len = 1
        log2_e = _f32(layout.constants["log2_e"])
        values: dict[str, Any] = {
            "problem_shape.0": pointers["qo_offsets"],
            "problem_shape.1": pointers["kv_offsets"],
            "problem_shape.2": DIM,
            "problem_shape.3.0.0": GROUP,
            "problem_shape.3.0.1": KV_HEADS,
            "problem_shape.3.1": batch,
            "mainloop.load.layout_Q.shape.0": batch,
            "mainloop.load.layout_Q.shape.1": DIM,
            "mainloop.load.layout_Q.shape.2.0": GROUP,
            "mainloop.load.layout_Q.shape.2.1": KV_HEADS,
            "mainloop.load.layout_Q.stride.0": HEADS * DIM,
            "mainloop.load.layout_Q.stride.2.0": DIM,
            "mainloop.load.layout_Q.stride.2.1": GROUP * DIM,
            "mainloop.load.layout_K.shape.0": total_kv,
            "mainloop.load.layout_K.shape.1": DIM,
            "mainloop.load.layout_K.shape.2.0": GROUP,
            "mainloop.load.layout_K.shape.2.1": KV_HEADS,
            "mainloop.load.layout_K.stride.0": KV_ROW,
            "mainloop.load.layout_K.stride.2.1": DIM,
            "mainloop.load.layout_V.shape.0": DIM,
            "mainloop.load.layout_V.shape.1": total_kv,
            "mainloop.load.layout_V.shape.2.0": GROUP,
            "mainloop.load.layout_V.shape.2.1": KV_HEADS,
            "mainloop.load.layout_V.stride.1": KV_ROW,
            "mainloop.load.layout_V.stride.2.1": DIM,
            # scale_q * scale_k * scale_softmax (all 1 but the softmax scale)
            "mainloop.scale_softmax": sm_scale,
            "mainloop.scale_softmax_log2": _f32(log2_e * sm_scale),
            "mainloop.scale_output": 1.0,
            "epilogue.layout_O.shape.0": max_qo_len,
            "epilogue.layout_O.shape.1": DIM,
            "epilogue.layout_O.shape.2.0.0": GROUP,
            "epilogue.layout_O.shape.2.0.1": KV_HEADS,
            "epilogue.layout_O.shape.2.1": max_qo_len + batch,
            "epilogue.layout_O.stride.0": HEADS * DIM,
            "epilogue.layout_O.stride.2.0.0": DIM,
            "epilogue.layout_O.stride.2.0.1": GROUP * DIM,
            "epilogue.layout_O.stride.2.1": HEADS * DIM,
            "epilogue.ptr_LSE": pointers["lse"],
            "epilogue.layout_LSE.shape.0": batch,
            "epilogue.layout_LSE.shape.1.0": GROUP,
            "epilogue.layout_LSE.shape.1.1": KV_HEADS,
            "epilogue.layout_LSE.stride.0": HEADS,
            "epilogue.layout_LSE.stride.1.1": GROUP,
            "epilogue.max_qo_len": max_qo_len,
            "tile_scheduler.work_indptr": pointers["work_indptr"],
            "tile_scheduler.qo_tile_indices": pointers["qo_tile_indices"],
            "tile_scheduler.qo_head_indices": pointers["qo_head_indices"],
            "tile_scheduler.batch_indices": pointers["batch_indices"],
            "tile_scheduler.num_sm": sm_count,
        }
        sizes = {"batch": batch, "total_kv": total_kv}
        for name, spec in layout.raw["tensor_maps"].items():
            values[name] = encode_descriptor(spec, pointers, sizes, encode, driver)
        return layout.pack(values), sm_count  # HostPrecomputedTileScheduler grid

    def setup_launch(self, inputs, *, driver: int | None = None):
        q, packed_k, packed_v, qo, kv, work, tiles, heads, batches, scale = inputs
        batch, total_kv, sms = q.shape[0], packed_k.shape[0], work.shape[0] - 1
        self._require(batch >= 1 and sms >= 1, "batch and SM count must be positive")
        self._require(
            total_kv * KV_ROW <= INT32_MAX and batch * HEADS * DIM <= INT32_MAX,
            "problem too large for int32 layouts",
        )
        i32 = torch.int32
        for name, tensor, shape, dtype in (
            ("q", q, (batch, HEADS, DIM), torch.bfloat16),
            ("packed_k", packed_k, (total_kv, KV_HEADS, DIM), torch.bfloat16),
            ("packed_v", packed_v, (total_kv, KV_HEADS, DIM), torch.bfloat16),
            ("qo_offsets", qo, (batch + 1,), i32),
            ("kv_offsets", kv, (batch + 1,), i32),
            ("work_indptr", work, (sms + 1,), i32),
            ("qo_tile_indices", tiles, (tiles.shape[0],), i32),
            ("qo_head_indices", heads, (tiles.shape[0],), i32),
            ("batch_indices", batches, (tiles.shape[0],), i32),
        ):
            self._check_tensor(name, tensor, shape, dtype)
        padded = q.new_empty(batch + 1, HEADS, DIM)
        output = padded[1:]
        lse = torch.empty(batch, HEADS, dtype=torch.float32, device=q.device)
        pointers = {
            "q": q.data_ptr(),
            "packed_k": packed_k.data_ptr(),
            "packed_v": packed_v.data_ptr(),
            "qo_offsets": qo.data_ptr(),
            "kv_offsets": kv.data_ptr(),
            "work_indptr": work.data_ptr(),
            "qo_tile_indices": tiles.data_ptr(),
            "qo_head_indices": heads.data_ptr(),
            "batch_indices": batches.data_ptr(),
            "o": output.data_ptr(),
            "lse": lse.data_ptr(),
        }
        for role in ("q", "packed_k", "packed_v", "o"):
            self._require(pointers[role] % 16 == 0, f"{role} must be 16-byte aligned")
        layout = self.param_layout
        if self.tensor_map_encoder is driver_encode:
            cuda_driver.ensure_context(q.device)  # cuTensorMapEncodeTiled needs one
        params, grid = self.build_params(
            layout,
            pointers,
            batch=batch,
            total_kv=total_kv,
            sm_count=sms,
            sm_scale=_f32(float(scale)),
            encode=self.tensor_map_encoder,
            driver=cuda_driver.driver_version() if driver is None else driver,
        )
        spec = LaunchSpec(
            (grid, 1, 1),
            tuple(layout.constants["block"]),
            [params],
            layout.constants["shared_mem"],
            tuple(layout.constants["cluster"]),
        )
        return (output, lse), spec


@register(name="gqa_decode_finish", supported_arches=("sm_100a",))
class GQADecodeFinish(_SM100FmhaInputs):
    """``gqa_finish<<<batch, 256>>>(fmha_output, kv_indptr, output, lse)``.

    Inputs ``(fmha_output [B,32,128], kv_indptr, fmha_lse [B,32])`` as the
    FMHA stage wrote them; outputs the official ``(output, lse)``. The kernel
    updates ``lse`` in place, so ``run`` copies ``fmha_lse`` into a fresh
    output first (as the FMHA wrote straight into the final LSE before).
    """

    def get_inputs(self, case):
        inputs, indptr = self.fmha_inputs(case)
        q, packed_k, packed_v, qo, kv, *_, scale = inputs
        out, lse = fmha_reference(q, packed_k, packed_v, qo, kv, float(scale))
        return out, indptr, lse

    def get_reference(self, inputs):
        return finish_reference(*inputs)

    def setup_launch(self, inputs):
        fmha_out, kv_indptr, fmha_lse = inputs
        batch = fmha_out.shape[0]
        self._require(batch >= 1, "batch must be positive")
        self._check_tensor("fmha_output", fmha_out, (batch, HEADS, DIM), torch.bfloat16)
        self._check_tensor("kv_indptr", kv_indptr, (batch + 1,), torch.int32)
        self._check_tensor("fmha_lse", fmha_lse, (batch, HEADS), torch.float32)
        output = torch.empty_like(fmha_out)
        lse = fmha_lse.clone()
        spec = LaunchSpec(
            (batch, 1, 1), (256, 1, 1), [fmha_out, kv_indptr, output, lse]
        )
        return (output, lse), spec


def check_param_layouts(arch: str | None = None) -> list[str]:
    """Rebuild every probe example with the Python builders and compare bytes.

    Needs no GPU: FMHA descriptors use ``fake_encode`` (the probe's packing of
    the encode arguments, after CUTLASS's driver fix-up), so every encode
    argument is compared too. Padding bytes outside every recorded field are
    indeterminate in C++ and ignored. Returns the checked ``workload/arch``
    names; raises on any mismatch.
    """
    checked = []
    for cls in (GQADecodeBatchDecode, GQADecodeFMHA):
        for target in cls.supported_arches:
            if arch is not None and target != arch:
                continue
            layout = ParamLayout.load(cls.cubin_path(target).with_suffix(".json"))
            pointers = layout.sentinels()
            for example in layout.raw["examples"]:
                expected = bytes.fromhex(example["bytes"])
                if cls is GQADecodeBatchDecode:
                    built = GQADecodeBatchDecode.build_params(
                        layout, pointers, example["batch"], _f32(example["sm_scale"])
                    )
                else:
                    built, grid = GQADecodeFMHA.build_params(
                        layout,
                        pointers,
                        batch=example["batch"],
                        total_kv=example["total_kv"],
                        sm_count=example["sm_count"],
                        sm_scale=_f32(example["sm_scale"]),
                        encode=fake_encode,
                        driver=example["driver_version"],
                    )
                    if [grid, 1, 1] != example["grid"]:
                        raise AssertionError(
                            f"{cls.name}: grid {grid} != {example['grid']}"
                        )
                covered = layout.covered()
                diff = [
                    i
                    for i, (a, b) in enumerate(zip(built, expected))
                    if a != b and covered[i]
                ]
                if len(built) != len(expected) or diff:
                    raise AssertionError(
                        f"{cls.name}/{target}: bytes differ at {diff[:16]}"
                    )
            checked.append(f"{cls.name}/{target}")
    return checked


# --- cases ---------------------------------------------------------------------------

THROUGHPUT = "gqa_decode"
MODEL = "qwen3_30b_a3b"  # the definition's source model (32/4 heads, d128)
MODEL_LAYER = "attention decode (32 query heads, 4 KV heads, head dim 128)"
MAX_CONTEXT = 131072  # Qwen3-30B-A3B with YaRN (4 x 32768)
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
DECODE_TEST = "tests/attention/test_batch_decode_kernels.py"
FMHA_TEST = "tests/attention/test_blackwell_fmha.py"
# FlashInfer paged decode tests (test_batch_decode_with_paged_kv_cache and its
# fast-plan / tuple-cache / CUDA-graph variants) restricted to this
# definition: page_size 1, 4 KV / 32 query heads, head dim 128, NHD, no RoPE,
# no soft cap. Every request has kv_len tokens, identity page table.
UPSTREAM_DECODE = tuple(
    (b, kv) for b in (12, 17, 128) for kv in (54, 97, 512, 2048, 16384)
)
# CUTLASS Blackwell FMHA tests (test_blackwell_cutlass_fmha) in the decode
# shape this pipeline serves: qo_len 1, non-causal, head dim 128 (heads are
# the definition's 32/4).
UPSTREAM_FMHA = tuple(
    (b, kv) for b in (1, 2, 3, 9, 17) for kv in (1, 17, 544, 977, 1999)
)
# Per-case memory budget (inputs, outputs, reference temporaries) by arch.
CASE_BUDGET = {"sm_86": 6e9, "sm_100a": 40e9}
INT32_ROWS = INT32_MAX // KV_ROW  # FMHA int32 layouts: packed rows (nnz + B)


def case_bytes(params: dict[str, Any], packed: bool) -> int:
    """Device bytes a case needs: caches, ``q``/outputs, the sm_100a packed
    K/V copy (``packed``) and the reference's per-request FP32 K/V."""
    lengths = params["lengths"]
    total, batch = sum(lengths), len(lengths)
    row = KV_ROW * 2 * 2  # K and V, BF16
    size = params.get("pages", total) * row + batch * HEADS * DIM * 2 * 3
    if packed:
        size += (total + batch) * row
    return size + max(lengths, default=0) * row * 2


def smoke_cases() -> list[CaseSpec]:
    """Own smoke cases: every hard path of the decode/FMHA/adapter kernels."""
    skewed = skewed_lengths(20000, 300, seed=7, sigma=1.5, minimum=0)
    tail = skewed_lengths(1500, 63, seed=8, sigma=1.0, minimum=1)
    return [
        CaseSpec("single", {"lengths": [1]}, 1),
        CaseSpec("ragged", {"lengths": [7, 33, 129]}, 2),
        CaseSpec("long", {"lengths": [2048, 511]}, 3),
        # Leading, consecutive and trailing empty requests, length 1.
        CaseSpec("with_empty", {"lengths": [0, 5, 0, 0, 17, 1, 0]}, 9),
        CaseSpec("one_nonempty", {"lengths": [0, 0, 300, 0]}, 10),
        # One long request among many short ones (scheduler tail), peaked
        # attention, pages shared between requests.
        CaseSpec(
            "one_long_many_short",
            {"lengths": [6000, *tail], "pages": 4000, "shared": True, "q_scale": 4.0},
            11,
        ),
        # 300 skewed requests (some empty): 9600 work items > SM count, so
        # persistent CTAs take several tiles; non-default sm_scale.
        CaseSpec(
            "skewed_300",
            {"lengths": skewed, "pages": 24000, "sm_scale": 0.3},
            12,
        ),
        # Long KV loop (many pipeline wraps), peaked attention.
        CaseSpec("peaked_long", {"lengths": [12000, 1, 3001], "q_scale": 8.0}, 13),
    ]


def upstream_cases() -> list[CaseSpec]:
    """Upstream test parametrizations this definition's kernels serve."""
    rev = FLASHINFER_REVISION
    cases = [
        upstream_case(
            f"upstream_decode_b{b}_kv{kv}",
            {"lengths": [kv] * b, "contiguous": True},
            f"{DECODE_TEST}::test_batch_decode_with_paged_kv_cache"
            f"[batch_size={b},kv_len={kv},page_size=1,num_kv_heads=4,"
            "num_qo_heads=32,head_dim=128,pos_encoding_mode=NONE]",
            # 4 GiB of K/V: benchmark-sized, served by the throughput suite.
            suite="throughput" if b * kv > 1 << 20 else "smoke",
            seed=b * 100000 + kv,
            revision=rev,
        )
        for b, kv in UPSTREAM_DECODE
    ]
    cases.append(
        upstream_case(
            "upstream_decode_extreme_negative_logits",
            {
                "lengths": [2],
                "pages": 17,
                "indices": [15, 16],
                "fill": {"q": 64.0, "k": -64.0, "v": 1.0},
            },
            f"{DECODE_TEST}::test_paged_decode_extreme_negative_logits"
            "[dtype=bfloat16]",
            revision=rev,
        )
    )
    cases += [
        upstream_case(
            f"upstream_fmha_b{b}_kv{kv}",
            {"lengths": [kv] * b, "contiguous": True},
            f"{FMHA_TEST}::test_blackwell_cutlass_fmha[batch_size={b},qo_len=1,"
            f"kv_len={kv},num_qo_heads=32,head_dim_qk=128,head_dim_vo=128,"
            "causal=False,dtype=bfloat16]",
            seed=b * 10000 + kv,
            revision=rev,
        )
        for b, kv in UPSTREAM_FMHA
    ]
    return cases


def capped_lengths(total: int, batch: int, cap: int, seed: int) -> list[int]:
    """``skewed_lengths`` with no request above ``cap`` (the model's context
    window): the excess is spread over the shorter requests."""
    lengths = skewed_lengths(total, batch, seed=seed, minimum=1)
    while max(lengths) > cap:
        excess = sum(max(0, n - cap) for n in lengths)
        lengths = [min(n, cap) for n in lengths]
        short = [i for i, n in enumerate(lengths) if n < cap]
        for j, i in enumerate(short):
            lengths[i] += excess // len(short) + (j < excess % len(short))
    return lengths


def _row_scale(axes: dict[str, int]) -> float:
    rows, _ = inventory(THROUGHPUT)
    row = next(r for r in rows if r["axes"] == axes)
    return row["inputs"]["sm_scale"]["value"]


def throughput_cases():
    """Throughput (benchmark) cases of this package."""
    name = THROUGHPUT
    rows = [(1, kv, pages) for kv, pages in BATCH1_ROWS] + [
        (16, 1020, 1070),
        (16, 8636, 8686),
        (16, 15463, 24732),
        (16, 23079, 32348),
        (64, 28815, 28831),
        (64, 44047, 44063),
        (64, 61519, 61535),
    ]
    result = []
    for b, total, pages in rows:
        axes = dict(
            batch_size=b, len_indptr=b + 1, num_kv_indices=total, num_pages=pages
        )
        lengths = skewed_lengths(total, b, seed=total, sigma=1.0, minimum=1)
        params = {"lengths": lengths, "pages": pages, "sm_scale": _row_scale(axes)}
        result.append(trace(name, f"batch{b}_kv{total}", params, axes))
    # Qwen3-30B-A3B: long-context low-batch decode (latency), then large
    # batches far above L2 (stress); skewed lengths keep the batch totals.
    for b, context in ((1, 32768), (1, 131072), (4, 65536)):
        params = {"lengths": [context] * b}
        result.append(
            model_case(f"batch{b}_context{context}", params, MODEL, MODEL_LAYER)
        )
    for b, context in ((128, 8192), (256, 8192), (128, 32000), (256, 16000)):
        lengths = capped_lengths(b * context, b, MAX_CONTEXT, seed=b + context)
        result.append(
            model_case(
                f"batch{b}_context{context}_skewed",
                {"lengths": lengths},
                MODEL,
                MODEL_LAYER,
            )
        )
    return result


# The 16 batch-1 inventory rows (num_kv_indices, num_pages): decode latency.
BATCH1_ROWS = (
    (7, 8),
    (362, 412),
    (9, 10),
    (141, 191),
    (11, 12),
    (436, 486),
    (14, 15),
    (2, 17),
    (64, 81),
    (102, 9347),
    (72, 9317),
    (40, 57),
    (50, 67),
    (87, 9332),
    (252, 302),
    (546, 596),
)


def all_cases() -> list[CaseSpec]:
    return smoke_cases() + upstream_cases() + throughput_cases()

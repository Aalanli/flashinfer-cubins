"""DSA sparse attention as single-kernel workloads (one cubin, one kernel each).

Official definition: ``dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64``
(``resources/dsa_attention.json``): 16 query heads, 512 latent + 64 RoPE dims,
2048 sparse token IDs per query (-1 = unused), 64-token KV pages, BF16 inputs,
outputs ``(output [T, 16, 512] bf16, lse [T, 16] f32 base 2)``.

The previous native package ran three kernels per architecture. Each is now its
own workload (``impls/dsa_attention/compiler.py`` documents which compiled
kernels are live and why):

=================================  =======  ======================================
workload                           arch     kernel
=================================  =======  ======================================
dsa_attention_pack_sparse          sm_86    ``pack_sparse`` (project adapter)
dsa_attention_decode               sm_86    FlashInfer
                                            ``BatchDecodeWithPagedKVCacheKernelMLA``
dsa_attention_fix_empty            sm_86    ``fix_empty`` (project adapter)
dsa_attention_pack                 sm_100a  ``dsa_pack`` (project adapter)
dsa_attention_mla                  sm_100a  CUTLASS ``Sm100FmhaMlaKernel...``
dsa_attention_unpack               sm_100a  ``dsa_unpack`` (project adapter)
=================================  =======  ======================================

Every workload is self-contained: ``get_inputs`` generates the official sparse
inputs and, for later stages, runs the PyTorch references of the preceding
stages to produce that kernel's exact inputs. Kernel arguments (including the
by-value ``Params`` structs of the two upstream kernels) are built in Python
from the layouts that the compile stage recorded next to each cubin.

Cases (shared; a stage drops cases it cannot serve: ``pack_sparse`` holds at
most 1024 tokens, and each arch has a memory budget, :data:`CASE_BUDGET`):

* smoke: -1 padding as a suffix and scattered between valid IDs, all-invalid
  rows (leading/trailing/consecutive), single valid IDs, duplicate IDs,
  valid counts that are not multiples of the 128-token KV tile, full top-k
  rows (16 KV tiles, pipeline wrap-around), more tokens than half the SM
  count (persistent MLA CTAs take several tiles), peaked attention (scaled
  queries) and both the definition's ``1/sqrt(192)`` and DeepSeek-V3.2's
  YaRN-scaled ``sm_scale`` (0.13523, every inventory row);
* upstream: FlashInfer ``test_cutlass_mla`` parametrizations this pipeline
  serves (page size 1, BF16, ``max_seq_len <= 2048``; queries x 100,
  ``randint`` page tables), checked against the pinned test source by
  ``tests/test_dsa_attention.py``;
* throughput: every distinct official row (1-8 tokens, 8462 pages, the row's
  ``sm_scale``), DeepSeek-V3.2 decode/prefill token counts (64-2048) at the
  official cache size and larger caches.

Case parameters: ``tokens``, ``pages``, ``valid`` (valid IDs per row, an int
or a list), ``pattern`` (``suffix`` -1 padding, default, or ``scattered``),
``duplicates`` (IDs drawn with repetition), ``sm_scale``, ``q_scale``.
"""

from __future__ import annotations

import ctypes
import json
import math
import struct
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple

import torch

from .. import cuda_driver
from ..cutlass_host import cutlass_small_tensor, fast_divmod
from ..registry import register
from ..throughput import inventory, model_case, trace, upstream_case
from ..workload import CaseSpec, Workload

HEADS = 16
PADDED_HEADS = 128  # CUTLASS SM100 MLA tile (TileShapeH)
CKV = 512
KPE = 64
TOPK = 2048
PAGE = 64
PACK_CAPACITY = 1024  # pack_sparse's compile-time `capacity`
PACK_MEM_INTS = PACK_CAPACITY * 2053 + 2  # scratch allocated by the old setup()
LOG2E = 1.4426950408889634


class LaunchSpec(NamedTuple):
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[Any]
    shared_mem: int = 0
    cluster: tuple[int, int, int] | None = None


# --- PyTorch references of the six stages ------------------------------------


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _sparse_attention(q_lat, q_rope, ckv_rows, kpe_rows, ids, mask, scale):
    """FP32 softmax attention of each query row over its gathered tokens.

    Returns ``(out [T, H, 512] f32, natural-log lse [T, H] f32)``; rows whose
    mask is empty get NaN output and -inf lse.
    """
    tokens, heads, _ = q_lat.shape
    out = torch.empty(tokens, heads, CKV, dtype=torch.float32, device=q_lat.device)
    lse = torch.empty(tokens, heads, dtype=torch.float32, device=q_lat.device)
    chunk = max(1, (1 << 25) // max(1, ids.shape[1] * CKV))
    for s in range(0, tokens, chunk):
        e = min(tokens, s + chunk)
        kv = ckv_rows[ids[s:e]].float()
        kp = kpe_rows[ids[s:e]].float()
        logits = torch.einsum("thd,tkd->thk", q_lat[s:e].float(), kv)
        logits = logits + torch.einsum("thd,tkd->thk", q_rope[s:e].float(), kp)
        logits = (logits * scale).masked_fill(~mask[s:e, None, :], -torch.inf)
        lse[s:e] = torch.logsumexp(logits, -1)
        out[s:e] = torch.einsum("thk,tkd->thd", torch.softmax(logits, -1), kv)
    return out, lse


def pack_sparse_reference(indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """CSR size-1 pages: (indptr, last_page_len, request_indices,
    kv_tile_indices, kv_chunk_size, indices[nnz])."""
    valid = indices >= 0
    counts = valid.sum(1, dtype=torch.int32)
    indptr = torch.zeros(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    indptr[1:] = counts.cumsum(0, dtype=torch.int32)
    rows = torch.arange(indices.shape[0], dtype=torch.int32, device=indices.device)
    return (
        indptr,
        (counts > 0).to(torch.int32),
        rows,
        torch.zeros_like(rows),
        torch.full((1,), TOPK, dtype=torch.int32, device=indices.device),
        indices[valid].to(torch.int32),  # row-major: per-row order preserved
    )


def decode_reference(q, qp, ckv, kpe, indptr, indices, last, requests, sm_scale):
    """FlashInfer MLA decode, page size 1, ``partition_kv = false``.

    Row ``t`` attends query ``requests[t]`` over the pages
    ``indices[indptr[r] : indptr[r] + len]`` with ``len = 0`` for an empty
    request, otherwise ``indptr[r + 1] - indptr[r] - 1 + last[r]``. LSE is base 2.
    Empty rows are undefined in the kernel and NaN here.
    """
    r = requests.long()
    n = (indptr[1:] - indptr[:-1]).long()[r]
    length = torch.where(n > 0, n - 1 + last.long()[r], torch.zeros_like(n))
    width = max(1, int(length.max())) if length.numel() else 1
    offsets = torch.arange(width, device=q.device)
    mask = offsets[None] < length[:, None]
    pos = torch.where(mask, indptr.long()[r][:, None] + offsets[None], 0)
    flat = indices.long() if indices.numel() else torch.zeros(1, device=q.device).long()
    ids = torch.where(mask, flat[pos.clamp(max=flat.numel() - 1)], 0)
    out, lse = _sparse_attention(
        q[r], qp[r], ckv.reshape(-1, CKV), kpe.reshape(-1, KPE), ids, mask, sm_scale
    )
    empty = length == 0
    out[empty] = torch.nan
    lse = lse / math.log(2)
    lse[empty] = torch.nan
    return out.to(q.dtype), lse


def fix_empty_reference(indptr, output, lse):
    empty = indptr[1:] == indptr[:-1]
    output, lse = output.clone(), lse.clone()
    output[empty] = 0
    lse[empty] = -torch.inf
    return output, lse


def pack_reference(q, qp, indices):
    """Pad heads to 128 and compact valid IDs (zero-filled; empty -> length 1)."""
    tokens = q.shape[0]
    padded_q = q.new_zeros(tokens, PADDED_HEADS, CKV)
    padded_q[:, :HEADS] = q
    padded_qp = qp.new_zeros(tokens, PADDED_HEADS, KPE)
    padded_qp[:, :HEADS] = qp
    valid = indices >= 0
    order = torch.sort((~valid).to(torch.int8), dim=1, stable=True).indices
    table = torch.where(
        torch.gather(valid, 1, order), torch.gather(indices, 1, order), 0
    ).to(torch.int32)
    lengths = valid.sum(1, dtype=torch.int32).clamp(min=1)
    return padded_q, padded_qp, table, lengths


def mla_reference(padded_q, padded_qp, ckv, kpe, table, lengths, sm_scale):
    """CUTLASS MLA, page size 1: all 128 heads, natural-log LSE."""
    mask = torch.arange(table.shape[1], device=table.device)[None] < lengths[:, None]
    ids = torch.where(mask, table.long(), 0)
    out, lse = _sparse_attention(
        padded_q,
        padded_qp,
        ckv.reshape(-1, CKV),
        kpe.reshape(-1, KPE),
        ids,
        mask,
        sm_scale,
    )
    return out.to(padded_q.dtype), lse


def unpack_reference(padded_out, padded_lse, indices):
    nonempty = (indices >= 0).any(1)
    out = padded_out[:, :HEADS].clone()
    out[~nonempty] = 0
    log2e = torch.tensor(LOG2E, dtype=torch.float32, device=padded_lse.device)
    lse = padded_lse[:, :HEADS] * log2e
    lse[~nonempty] = -torch.inf
    return out, lse


# --- by-value parameter structs ------------------------------------------------


class ParamLayout:
    """A struct layout recorded by an ``impls/dsa_attention`` probe."""

    _formats = {
        "ptr": "<Q",
        "i32": "<i",
        "u32": "<I",
        "i64": "<q",
        "f32": "<f",
        "bool": "<?",
        "fast_divmod": "<iII",
        "dim3": "<III",
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
            if kind in ("bytes", "tma_atom"):
                data = bytes(value)
                if len(data) > size:
                    raise ValueError(f"{name}: {len(data)} bytes exceed {size}")
                buffer[offset : offset + len(data)] = data
                continue
            fmt = self._formats[kind]
            if struct.calcsize(fmt) != size:
                raise ValueError(f"{name}: {kind} is not {size} bytes")
            if kind == "fast_divmod":
                value = fast_divmod(int(value))
            elif not isinstance(value, tuple):
                value = (value,)
            struct.pack_into(fmt, buffer, offset, *value)
        return bytes(buffer)

    def sentinels(self) -> dict[str, int]:
        stride = self.raw["sentinel_stride"]
        return {
            role: stride * (i + 1) for i, role in enumerate(self.raw["pointer_roles"])
        }


driver_version = cuda_driver.driver_version


_ELEMENT_BYTES = {0: 1, 1: 2, 3: 4, 5: 8, 6: 2, 7: 4, 8: 8, 9: 2}


def encode_tensor_map(
    spec: dict[str, Any], pointers: dict[str, int], sizes: dict[str, int]
):
    """Encode one recorded TMA descriptor like CUTLASS ``make_tma_copy_desc``.

    Returns 128 descriptor bytes. CUTLASS clears bit 21 of word 1 for tensors
    it sizes under 128 KiB (``cutlass_small_tensor``) when the driver is at most 13.1 (a driver workaround in
    ``copy_traits_sm90_tma.hpp``); mirrored here.
    """

    def resolve(value):
        if isinstance(value, dict):
            return sizes[value["var"]] * value["scale"]
        return value

    dims = [resolve(v) for v in spec["global_dims"]]
    strides = [resolve(v) for v in spec["global_strides"]]
    address = pointers[spec["address"]["pointer"]] + spec["address"]["offset"]
    tensor_map = cuda_driver.encode_tensor_map_tiled(
        spec["data_type"],
        address,
        dims,
        strides,
        spec["box_dims"],
        spec["element_strides"],
        interleave=spec["interleave"],
        swizzle=spec["swizzle"],
        l2_promotion=spec["l2_promotion"],
        oob_fill=spec["oob_fill"],
    )
    data = bytearray(bytes(tensor_map))
    element = _ELEMENT_BYTES[spec["data_type"]]
    cosize = element * (
        dims[0] + sum((d - 1) * s // element for d, s in zip(dims[1:], strides))
    )
    if driver_version() <= 13010 and cutlass_small_tensor(cosize):
        word = int.from_bytes(data[8:16], "little") & ~(1 << 21)
        data[8:16] = word.to_bytes(8, "little")
    return bytes(data)


# --- workloads --------------------------------------------------------------------


def _valid_counts(params: dict[str, Any]) -> list[int]:
    valid = params["valid"]
    return (
        list(valid) if isinstance(valid, (list, tuple)) else [valid] * params["tokens"]
    )


class _DSAAttentionStage(Workload):
    """Shared cases and official sparse inputs; one subclass per kernel."""

    package = "dsa_attention"
    rtol = 0.0
    atol = 0.0

    # Most tokens a stage serves (pack_sparse's compile-time capacity on sm_86).
    max_tokens: int | None = None

    def get_cases(self) -> list[CaseSpec]:
        """Every package case this stage serves within its memory budget."""
        padded = self.supported_arches != ("sm_86",)
        budget = CASE_BUDGET[self.supported_arches[0]]
        return [
            c
            for c in all_cases()
            if case_bytes(c.params, padded) <= budget
            and (self.max_tokens is None or c.params["tokens"] <= self.max_tokens)
        ]

    def sparse_inputs(self, case: CaseSpec) -> tuple[torch.Tensor, ...]:
        """Official inputs: (q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices,
        sm_scale); parameters in the module docstring."""
        g, p = self.generator(case), case.params
        tokens, pages = p["tokens"], p["pages"]
        scattered = p.get("pattern", "suffix") == "scattered"
        indices = torch.full((tokens, TOPK), -1, dtype=torch.int32, device=self.device)
        for row, valid in enumerate(_valid_counts(p)):
            if p.get("duplicates"):
                ids = torch.randint(
                    pages * PAGE, (valid,), generator=g, device=self.device
                )
            else:
                ids = torch.randperm(pages * PAGE, generator=g, device=self.device)
                ids = ids[:valid]
            if scattered:
                slots = torch.randperm(TOPK, generator=g, device=self.device)[:valid]
                indices[row, slots.sort().values] = ids.int()
            else:
                indices[row, :valid] = ids.int()
        q_scale = p.get("q_scale", 1.0)
        return (
            self.randn((tokens, HEADS, CKV), g, scale=q_scale),
            self.randn((tokens, HEADS, KPE), g, scale=q_scale),
            torch.randn(
                (pages, PAGE, CKV),
                generator=g,
                device=self.device,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                (pages, PAGE, KPE),
                generator=g,
                device=self.device,
                dtype=torch.bfloat16,
            ),
            indices,
            self.scalar(p.get("sm_scale", 1 / math.sqrt(192))),
        )

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


# sm_86 --------------------------------------------------------------------------


@register(name="dsa_attention_pack_sparse", supported_arches=("sm_86",))
class DSAAttentionPackSparse(_DSAAttentionStage):
    """``pack_sparse<<<1, 1>>>(sparse_indices, mem, batch)``: CSR page metadata.

    Inputs ``(sparse_indices [T, 2048] i32,)``. Outputs are views of the
    kernel's ``mem`` scratch: ``(indptr [T+1], last_page_len [T],
    request_indices [T], kv_tile_indices [T], kv_chunk_size [1],
    indices [1024 * 2048])``; only the first ``indptr[T]`` indices are written.
    """

    max_tokens = PACK_CAPACITY

    def get_inputs(self, case):
        return (self.sparse_inputs(case)[4],)

    def get_reference(self, inputs):
        return pack_sparse_reference(inputs[0])

    def setup_launch(self, inputs):
        (indices,) = inputs
        tokens = indices.shape[0]
        self._require(1 <= tokens <= PACK_CAPACITY, f"batch must be 1..{PACK_CAPACITY}")
        self._check_tensor("sparse_indices", indices, (tokens, TOPK), torch.int32)
        mem = torch.empty(PACK_MEM_INTS, dtype=torch.int32, device=indices.device)
        c = PACK_CAPACITY
        indices_start = 4 * c + 2
        outputs = (
            mem[: tokens + 1],
            mem[c + 1 : c + 1 + tokens],
            mem[2 * c + 1 : 2 * c + 1 + tokens],
            mem[3 * c + 1 : 3 * c + 1 + tokens],
            mem[4 * c + 1 : 4 * c + 2],
            mem[indices_start : indices_start + c * TOPK],
        )
        return outputs, LaunchSpec(
            (1, 1, 1), (1, 1, 1), [indices, mem, ctypes.c_int(tokens)]
        )

    def validate(self, ref, impl):
        if not isinstance(impl, tuple) or len(impl) != len(ref):
            raise AssertionError("outputs must be a 6-tuple")
        nnz = ref[5].numel()
        if not isinstance(impl[5], torch.Tensor) or impl[5].numel() < nnz:
            raise AssertionError("packed indices are too short")
        self.assert_close(ref[:5] + (ref[5][:nnz],), impl[:5] + (impl[5][:nnz],), 0, 0)


class _SM86DecodeInputs(_DSAAttentionStage):
    max_tokens = PACK_CAPACITY  # inputs come from pack_sparse

    def decode_inputs(self, case):
        q, qp, ckv, kpe, indices, scale = self.sparse_inputs(case)
        indptr, last, requests, tiles, chunk, packed = pack_sparse_reference(indices)
        if not packed.numel():  # the kernel never reads it; keep a valid pointer
            packed = torch.zeros(1, dtype=torch.int32, device=indices.device)
        return q, qp, ckv, kpe, indptr, packed, last, requests, tiles, chunk, scale


@register(name="dsa_attention_decode", supported_arches=("sm_86",))
class DSAAttentionDecode(_SM86DecodeInputs):
    """FlashInfer ``BatchDecodeWithPagedKVCacheKernelMLA<2,16,2,32,8,1,2>``.

    Inputs ``(q_nope, q_pe, ckv_cache, kpe_cache, indptr, indices,
    last_page_len, request_indices, kv_tile_indices, kv_chunk_size, sm_scale)``
    (CSR metadata as written by ``pack_sparse``), outputs ``(output, lse)``
    with base-2 LSE. Rows with no sparse token are undefined (fixed up by
    ``fix_empty``) and excluded from validation.

    The by-value ``BatchDecodeParamsMLA`` (216 bytes) is built as the previous
    host code did: ``paged_kv_mla_t(page_size=1, 512, 64, batch, ckv, kpe,
    indices, indptr, last_page_len)`` (so page strides 512/64, entry strides
    512/64), ``Params(q_nope, q_pe, q_rope_offset=nullptr, paged_kv, o, lse,
    num_qo_heads=16, window_left=-1, logits_soft_cap=0, sm_scale,
    rope_scale=1, rope_theta=1e4)`` (``rope_rcp_* = 1/x``), then
    ``padded_batch_size = batch`` and the request/tile/chunk pointers;
    ``o_indptr``/``block_valid_mask`` stay null and ``partition_kv`` false.
    Launch: grid (batch), block (32, 8), 22528 bytes of dynamic smem.
    """

    rtol = atol = 1e-2  # BF16 inputs/outputs, FP32 accumulation

    def get_inputs(self, case):
        return self.decode_inputs(case)

    def get_reference(self, inputs):
        q, qp, ckv, kpe, indptr, indices, last, requests, _, _, scale = inputs
        return decode_reference(
            q, qp, ckv, kpe, indptr, indices, last, requests, float(scale)
        )

    @staticmethod
    def build_params(
        layout: ParamLayout, pointers: dict[str, int], batch: int, sm_scale: float
    ) -> bytes:
        page_size = 1
        return layout.pack(
            {
                "q_nope": pointers["q_nope"],
                "q_pe": pointers["q_pe"],
                "o": pointers["o"],
                "lse": pointers["lse"],
                "sm_scale": sm_scale,
                "q_rope_offset": 0,
                "paged_kv.page_size": bytes.fromhex(
                    layout.constants["page_size_fastdiv_1"]
                ),
                "paged_kv.head_dim_ckv": CKV,
                "paged_kv.head_dim_kpe": KPE,
                "paged_kv.batch_size": batch,
                "paged_kv.stride_page_ckv": page_size * CKV,
                "paged_kv.stride_page_kpe": page_size * KPE,
                "paged_kv.stride_n_ckv": CKV,
                "paged_kv.stride_n_kpe": KPE,
                "paged_kv.ckv_data": pointers["ckv"],
                "paged_kv.kpe_data": pointers["kpe"],
                "paged_kv.indices": pointers["indices"],
                "paged_kv.indptr": pointers["indptr"],
                "paged_kv.last_page_len": pointers["last_page_len"],
                "paged_kv.rope_pos_offset": 0,
                "padded_batch_size": batch,
                "num_qo_heads": HEADS,
                "window_left": -1,
                "logits_soft_cap": 0.0,
                "rope_rcp_scale": _f32(1.0 / 1.0),
                "rope_rcp_theta": _f32(1.0 / 1e4),
                "request_indices": pointers["request_indices"],
                "kv_tile_indices": pointers["kv_tile_indices"],
                "o_indptr": 0,
                "kv_chunk_size_ptr": pointers["kv_chunk_size"],
                "block_valid_mask": 0,
                "partition_kv": False,
            }
        )

    def setup_launch(self, inputs):
        q, qp, ckv, kpe, indptr, indices, last, requests, tiles, chunk, scale = inputs
        tokens, pages = q.shape[0], ckv.shape[0]
        self._require(tokens >= 1, "batch must be positive")
        i32 = torch.int32
        for name, tensor, shape, dtype in (
            ("q_nope", q, (tokens, HEADS, CKV), torch.bfloat16),
            ("q_pe", qp, (tokens, HEADS, KPE), torch.bfloat16),
            ("ckv_cache", ckv, (pages, PAGE, CKV), torch.bfloat16),
            ("kpe_cache", kpe, (pages, PAGE, KPE), torch.bfloat16),
            ("indptr", indptr, (tokens + 1,), i32),
            ("indices", indices, (indices.shape[0],), i32),
            ("last_page_len", last, (tokens,), i32),
            ("request_indices", requests, (tokens,), i32),
            ("kv_tile_indices", tiles, (tokens,), i32),
            ("kv_chunk_size", chunk, (1,), i32),
        ):
            self._check_tensor(name, tensor, shape, dtype)
        output = torch.empty_like(q)
        lse = torch.empty(tokens, HEADS, dtype=torch.float32, device=q.device)
        pointers = {
            "q_nope": q.data_ptr(),
            "q_pe": qp.data_ptr(),
            "ckv": ckv.data_ptr(),
            "kpe": kpe.data_ptr(),
            "o": output.data_ptr(),
            "lse": lse.data_ptr(),
            "indices": indices.data_ptr(),
            "indptr": indptr.data_ptr(),
            "last_page_len": last.data_ptr(),
            "request_indices": requests.data_ptr(),
            "kv_tile_indices": tiles.data_ptr(),
            "kv_chunk_size": chunk.data_ptr(),
        }
        layout = self.param_layout
        params = self.build_params(layout, pointers, tokens, float(scale))
        block = tuple(layout.constants["block"])
        spec = LaunchSpec(
            (tokens, 1, 1), block, [params], layout.constants["shared_mem"]
        )
        return (output, lse), spec

    def validate(self, ref, impl):
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
        self.assert_close(
            (ref[0][defined], ref[1][defined]),
            (impl[0][defined], impl[1][defined]),
            self.rtol,
            self.atol,
        )


@register(name="dsa_attention_fix_empty", supported_arches=("sm_86",))
class DSAAttentionFixEmpty(_SM86DecodeInputs):
    """``fix_empty<<<batch, 256>>>(indptr, output, lse)``, in place.

    Inputs ``(indptr, output, lse)``: decode outputs whose empty rows hold
    undefined values (NaN here). ``run`` copies them into fresh outputs and
    normalizes empty rows there (zero output, base-2 LSE -inf).
    """

    def get_inputs(self, case):
        q, qp, ckv, kpe, indptr, indices, last, requests, _, _, scale = (
            self.decode_inputs(case)
        )
        output, lse = decode_reference(
            q, qp, ckv, kpe, indptr, indices, last, requests, float(scale)
        )
        return indptr, output, lse

    def get_reference(self, inputs):
        return fix_empty_reference(*inputs)

    def setup_launch(self, inputs):
        indptr, output, lse = inputs
        tokens = output.shape[0]
        self._require(tokens >= 1, "batch must be positive")
        self._check_tensor("indptr", indptr, (tokens + 1,), torch.int32)
        self._check_tensor("output", output, (tokens, HEADS, CKV), torch.bfloat16)
        self._check_tensor("lse", lse, (tokens, HEADS), torch.float32)
        fixed_output, fixed_lse = output.clone(), lse.clone()
        spec = LaunchSpec(
            (tokens, 1, 1), (256, 1, 1), [indptr, fixed_output, fixed_lse]
        )
        return (fixed_output, fixed_lse), spec


# sm_100a ------------------------------------------------------------------------


@register(name="dsa_attention_pack", supported_arches=("sm_100a",))
class DSAAttentionPack(_DSAAttentionStage):
    """``dsa_pack<<<batch, 256>>>``: pad 16 heads to 128, compact the IDs.

    Inputs ``(q_nope, q_pe, sparse_indices)``; outputs ``(padded_q
    [T, 128, 512], padded_q_pe [T, 128, 64], page_table [T, 2048] i32,
    lengths [T] i32)``: valid IDs first in order, zero-filled, and length 1
    (dummy token 0) for an empty row.
    """

    def get_inputs(self, case):
        q, qp, _, _, indices, _ = self.sparse_inputs(case)
        return q, qp, indices

    def get_reference(self, inputs):
        return pack_reference(*inputs)

    def setup_launch(self, inputs):
        q, qp, indices = inputs
        tokens = q.shape[0]
        self._require(tokens >= 1, "batch must be positive")
        self._check_tensor("q_nope", q, (tokens, HEADS, CKV), torch.bfloat16)
        self._check_tensor("q_pe", qp, (tokens, HEADS, KPE), torch.bfloat16)
        self._check_tensor("sparse_indices", indices, (tokens, TOPK), torch.int32)
        padded_q = q.new_empty(tokens, PADDED_HEADS, CKV)
        padded_qp = qp.new_empty(tokens, PADDED_HEADS, KPE)
        table = torch.empty(tokens, TOPK, dtype=torch.int32, device=q.device)
        lengths = torch.empty(tokens, dtype=torch.int32, device=q.device)
        outputs = (padded_q, padded_qp, table, lengths)
        spec = LaunchSpec((tokens, 1, 1), (256, 1, 1), [q, qp, indices, *outputs])
        return outputs, spec


class _SM100MlaInputs(_DSAAttentionStage):
    def mla_inputs(self, case):
        q, qp, ckv, kpe, indices, scale = self.sparse_inputs(case)
        padded_q, padded_qp, table, lengths = pack_reference(q, qp, indices)
        return padded_q, padded_qp, ckv, kpe, table, lengths, scale, indices


@register(name="dsa_attention_mla", supported_arches=("sm_100a",))
class DSAAttentionMLA(_SM100MlaInputs):
    """CUTLASS ``Sm100FmhaMlaKernelTmaWarpspecialized`` (FlashInfer
    ``MlaSm100<bf16>``: 128x128x(512+64) tile, 2-SM cluster, persistent
    scheduler, cp.async page-table loads).

    Inputs ``(padded_q [T,128,512], padded_q_pe [T,128,64], ckv_cache,
    kpe_cache, page_table [T,2048] i32, lengths [T] i32, sm_scale)`` as written
    by ``dsa_pack``; outputs ``(output [T,128,512] bf16, lse [T,128] f32)``
    with natural-log LSE over all 128 heads (padded heads have zero queries).

    ``Params`` (1664 bytes) mirrors what the previous host code produced via
    ``args_from_options(out, lse, q, ckv, lengths, table, batch,
    page_count_per_seq=2048, page_count_total=pages*64, page_size=1, device)``
    with the DSA overrides (softmax_scale, separate latent/RoPE pointers and
    strides, ``split_kv = 1``), then ``MLA::initialize`` ->
    ``Kernel::to_underlying_arguments``:

    * problem shape ``(128, K = 1 * 2048, (512, 64), B = batch)``;
    * mainloop: strides Q latent (512, 1, 128*512), Q rope (64, 1, 128*64),
      C latent (512, 1, 512), K rope (64, 1, 64); page table stride
      (1, 2048); ``page_count = pages * 64``, ``page_size = 1``;
    * epilogue: O stride (512, 1, 128*512), LSE stride (1, 128),
      ``output_scale = 1``; split-KV accumulators null (``split_kv = 1``,
      so no workspace);
    * five TMA atoms: descriptors encoded at run time with the arguments
      CUTLASS passes (recorded by the compile-stage probe). The cp.async
      kernel never reads them (its SASS has no TMA loads and no constant-bank
      reads in that range), but they are filled as upstream does
      (``tests/test_dsa_attention.py`` checks their encode inputs with a
      recording encoder);
    * persistent tile scheduler: ``num_blocks = 2 * batch * split_kv``,
      ``FastDivmod(2), FastDivmod(batch), FastDivmod(split_kv)``,
      ``hw_info = {device_id, sm_count, 0, 0, 0}``.

    Launch (``MLA::run`` / ``ClusterLauncher::launch``): grid
    ``min(num_blocks, sm_count)``, block 256, cluster (2, 1, 1), dynamic smem
    ``SharedStorageSize``; the function gets the max-dynamic-smem and
    non-portable-cluster attributes once. The split-KV reduction kernel is
    never launched for ``split_kv = 1``.
    """

    rtol = atol = 1e-2  # BF16 inputs/outputs, FP32 accumulation
    # Replaceable for argument-building checks on GPUs without TMA (sm_86
    # drivers reject cuTensorMapEncodeTiled with CUDA_ERROR_NOT_SUPPORTED).
    tensor_map_encoder = staticmethod(encode_tensor_map)

    def get_inputs(self, case):
        return self.mla_inputs(case)[:7]

    def get_reference(self, inputs):
        padded_q, padded_qp, ckv, kpe, table, lengths, scale = inputs
        return mla_reference(
            padded_q, padded_qp, ckv, kpe, table, lengths, float(scale)
        )

    def configure(self, function):
        layout = self.param_layout
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            layout.constants["shared_mem"],
        )
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1
        )

    @staticmethod
    def build_params(
        layout: ParamLayout,
        pointers: dict[str, int],
        *,
        batch: int,
        page_count: int,
        sm_count: int,
        device_id: int,
        softmax_scale: float,
        encode=encode_tensor_map,
    ) -> tuple[bytes, int]:
        """(Params bytes, grid size). ``encode(spec, pointers, sizes)`` returns
        each 128-byte TMA descriptor."""
        heads = layout.constants["heads"]
        cluster_m = layout.constants["cluster"][0]
        split_kv = 1
        max_seq_len = 1 * TOPK  # page_size * page_count_per_seq
        sizes = {"batch": batch, "page_count": page_count}
        values: dict[str, Any] = {
            "problem_shape.K": max_seq_len,
            "problem_shape.B": batch,
            "mainloop.softmax_scale": softmax_scale,
            "mainloop.ptr_q_latent": pointers["q_latent"],
            "mainloop.stride_q_latent.0": CKV,
            "mainloop.stride_q_latent.2": heads * CKV,
            "mainloop.ptr_q_rope": pointers["q_rope"],
            "mainloop.stride_q_rope.0": KPE,
            "mainloop.stride_q_rope.2": heads * KPE,
            "mainloop.ptr_c_latent": pointers["ckv"],
            "mainloop.stride_c_latent.0": CKV,
            "mainloop.stride_c_latent.2": CKV,
            "mainloop.ptr_k_rope": pointers["kpe"],
            "mainloop.stride_k_rope.0": KPE,
            "mainloop.stride_k_rope.2": KPE,
            "mainloop.ptr_seq": pointers["lengths"],
            "mainloop.ptr_page_table": pointers["page_table"],
            "mainloop.stride_page_table.1": TOPK,
            "mainloop.page_count": page_count,
            "mainloop.page_size": 1,
            "epilogue.ptr_o": pointers["out"],
            "epilogue.ptr_o_acc": 0,
            "epilogue.stride_o.0": CKV,
            "epilogue.stride_o.2": heads * CKV,
            "epilogue.stride_o_acc.0": 0,
            "epilogue.stride_o_acc.2": 0,
            "epilogue.ptr_lse": pointers["lse"],
            "epilogue.ptr_lse_acc": 0,
            "epilogue.stride_lse.1": heads,
            "epilogue.stride_lse_acc.1": 0,
            "epilogue.output_scale": 1.0,
            "tile_scheduler.num_blocks": cluster_m * batch * split_kv,
            "tile_scheduler.divmod_m_block": cluster_m,
            "tile_scheduler.divmod_b": batch,
            "tile_scheduler.divmod_split_kv": split_kv,
            "tile_scheduler.hw_info.device_id": device_id,
            "tile_scheduler.hw_info.sm_count": sm_count,
            "tile_scheduler.hw_info.max_active_clusters": 0,
            "tile_scheduler.hw_info.cluster_shape": (0, 0, 0),
            "tile_scheduler.hw_info.cluster_shape_fallback": (0, 0, 0),
            "split_kv": split_kv,
            "ptr_split_kv": 0,
        }
        for name, tensor_map in layout.raw["tensor_maps"].items():
            descriptor = encode(tensor_map["encode"], pointers, sizes)
            offset = tensor_map["descriptor_offset"]
            values[name] = bytes(offset) + descriptor
        grid = min(cluster_m * batch * split_kv, sm_count)
        return layout.pack(values), grid

    def setup_launch(self, inputs):
        padded_q, padded_qp, ckv, kpe, table, lengths, scale = inputs
        tokens, pages = padded_q.shape[0], ckv.shape[0]
        self._require(tokens >= 1, "batch must be positive")
        for name, tensor, shape, dtype in (
            ("padded_q", padded_q, (tokens, PADDED_HEADS, CKV), torch.bfloat16),
            ("padded_q_pe", padded_qp, (tokens, PADDED_HEADS, KPE), torch.bfloat16),
            ("ckv_cache", ckv, (pages, PAGE, CKV), torch.bfloat16),
            ("kpe_cache", kpe, (pages, PAGE, KPE), torch.bfloat16),
            ("page_table", table, (tokens, TOPK), torch.int32),
            ("lengths", lengths, (tokens,), torch.int32),
        ):
            self._check_tensor(name, tensor, shape, dtype)
        output = torch.empty_like(padded_q)
        lse = torch.empty(
            tokens, PADDED_HEADS, dtype=torch.float32, device=padded_q.device
        )
        device = padded_q.device
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        sm_count = torch.cuda.get_device_properties(index).multi_processor_count
        layout = self.param_layout
        cluster = tuple(layout.constants["cluster"])
        cuda_driver.ensure_context(device)  # cuTensorMapEncodeTiled needs a context
        params, grid = self.build_params(
            layout,
            {
                "q_latent": padded_q.data_ptr(),
                "q_rope": padded_qp.data_ptr(),
                "ckv": ckv.data_ptr(),
                "kpe": kpe.data_ptr(),
                "lengths": lengths.data_ptr(),
                "page_table": table.data_ptr(),
                "out": output.data_ptr(),
                "lse": lse.data_ptr(),
            },
            batch=tokens,
            page_count=pages * PAGE,
            sm_count=sm_count,
            device_id=index,
            softmax_scale=float(scale),
            encode=self.tensor_map_encoder,
        )
        self._require(grid % cluster[0] == 0, "grid must be a multiple of the cluster")
        spec = LaunchSpec(
            (grid, 1, 1),
            tuple(layout.constants["block"]),
            [params],
            layout.constants["shared_mem"],
            cluster,
        )
        return (output, lse), spec


@register(name="dsa_attention_unpack", supported_arches=("sm_100a",))
class DSAAttentionUnpack(_SM100MlaInputs):
    """``dsa_unpack<<<batch, 256>>>``: 16 live heads, base-2 LSE, empty rows.

    Inputs ``(padded_output [T,128,512] bf16, padded_lse [T,128] f32 natural,
    sparse_indices)`` as produced by the MLA stage; outputs the official
    ``(output [T,16,512] bf16, lse [T,16] f32 base 2)``.
    """

    def get_inputs(self, case):
        padded_q, padded_qp, ckv, kpe, table, lengths, scale, indices = self.mla_inputs(
            case
        )
        out, lse = mla_reference(
            padded_q, padded_qp, ckv, kpe, table, lengths, float(scale)
        )
        return out, lse, indices

    def get_reference(self, inputs):
        return unpack_reference(*inputs)

    def setup_launch(self, inputs):
        padded, padded_lse, indices = inputs
        tokens = padded.shape[0]
        self._require(tokens >= 1, "batch must be positive")
        self._check_tensor(
            "padded_output", padded, (tokens, PADDED_HEADS, CKV), torch.bfloat16
        )
        self._check_tensor(
            "padded_lse", padded_lse, (tokens, PADDED_HEADS), torch.float32
        )
        self._check_tensor("sparse_indices", indices, (tokens, TOPK), torch.int32)
        output = padded.new_empty(tokens, HEADS, CKV)
        lse = torch.empty(tokens, HEADS, dtype=torch.float32, device=padded.device)
        spec = LaunchSpec(
            (tokens, 1, 1), (256, 1, 1), [padded, padded_lse, indices, output, lse]
        )
        return (output, lse), spec


def check_param_layouts(arch: str | None = None) -> list[str]:
    """Rebuild every probe example with the Python builders and compare bytes.

    Needs no GPU: TMA descriptors are compared as the probe's markers (the
    encode arguments themselves come from the same probe runs); padding bytes
    outside every recorded field are ignored. Returns the
    checked ``workload/arch`` names; raises on any mismatch.
    """
    checked = []
    for cls in (DSAAttentionDecode, DSAAttentionMLA):
        for target in cls.supported_arches:
            if arch is not None and target != arch:
                continue
            layout = ParamLayout.load(cls.cubin_path(target).with_suffix(".json"))
            pointers = layout.sentinels()
            for example in layout.raw["examples"]:
                expected = bytearray.fromhex(example["bytes"])
                if cls is DSAAttentionDecode:
                    built = bytearray(
                        DSAAttentionDecode.build_params(
                            layout,
                            pointers,
                            example["batch"],
                            _f32(example["sm_scale"]),
                        )
                    )
                else:
                    built_bytes, grid = DSAAttentionMLA.build_params(
                        layout,
                        pointers,
                        batch=example["batch"],
                        page_count=example["page_count"],
                        sm_count=example["sm_count"],
                        device_id=example["device_id"],
                        softmax_scale=_f32(example["softmax_scale"]),
                        encode=lambda spec, p, s: bytes(128),
                    )
                    if [grid, 1, 1] != example["grid"]:
                        raise AssertionError(
                            f"{cls.name}: grid {grid} != {example['grid']}"
                        )
                    built = bytearray(built_bytes)
                    for name in layout.raw["tensor_maps"]:
                        offset = layout.fields[name]["offset"]
                        marker = int.from_bytes(expected[offset : offset + 8], "little")
                        if marker >> 16 != 0x7E5A00000000:
                            raise AssertionError(f"{name}: no probe marker")
                        expected[offset : offset + 128] = bytes(128)
                # Struct padding is indeterminate in C++ (copied from temporaries);
                # every byte covered by a recorded field must match exactly.
                covered = bytearray(layout.size)
                for field in layout.fields.values():
                    start = field["offset"]
                    covered[start : start + field["size"]] = b"\x01" * field["size"]
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

THROUGHPUT = "dsa_attention"
MODEL = "deepseek_v3"  # DeepSeek-V3.2 MLA (16 of 128 heads per TP-8 shard)
MODEL_LAYER = "DSA sparse MLA, TP 8 (16 heads, 512 latent + 64 RoPE, top-k 2048)"
# DeepSeek-V3.2 softmax scale: 1/sqrt(192) * mscale^2 with YaRN mscale
# 0.1 * ln(40) + 1; the value every official inventory row records.
YARN_SM_SCALE = 0.1352337788608801
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
MLA_TEST = "tests/attention/test_deepseek_mla.py::test_cutlass_mla"
# FlashInfer test_cutlass_mla parametrizations this pipeline serves: page
# size 1 (sparse token IDs), BF16, max_seq_len within top-k 2048.
UPSTREAM_MLA = tuple((b, n) for b in (1, 2, 4) for n in (128, 1024))
OFFICIAL_PAGES = 8462
# Per-case memory budget (inputs, outputs, reference temporaries) by arch.
CASE_BUDGET = {"sm_86": 6e9, "sm_100a": 40e9}


def case_bytes(params: dict[str, Any], padded: bool) -> int:
    """Device bytes a case needs: caches, queries/indices/outputs, the
    sm_100a 128-head padded copies (``padded``), reference temporaries."""
    tokens = params["tokens"]
    row = (CKV + KPE) * 2
    size = params["pages"] * PAGE * row + tokens * (HEADS * row * 3 + TOPK * 8)
    if padded:
        size += tokens * PADDED_HEADS * row * 2
    return size + (1 << 25) * 4 * 2  # one gathered FP32 chunk + logits


def smoke_cases() -> list[CaseSpec]:
    """Every hard path of the six kernels, moderately sized."""
    many = [((i * 389) % 2049) for i in range(100)]  # 0 .. 2048, scattered
    many[3], many[50], many[99] = 0, 2048, 1
    return [
        CaseSpec("padding", {"tokens": 2, "pages": 2, "valid": 31}, 1),
        CaseSpec("full_topk", {"tokens": 3, "pages": 40, "valid": 2048}, 2),
        CaseSpec("empty", {"tokens": 1, "pages": 1, "valid": 0}, 3),
        CaseSpec(
            "mixed_empty",
            {"tokens": 4, "pages": 33, "valid": [100, 0, 2048, 1]},
            4,
        ),
        # -1 between valid IDs; leading/consecutive/trailing all-invalid rows;
        # YaRN-scaled sm_scale as in DeepSeek-V3.2.
        CaseSpec(
            "scattered",
            {
                "tokens": 7,
                "pages": 40,
                "valid": [0, 1500, 0, 0, 2047, 129, 0],
                "pattern": "scattered",
                "sm_scale": YARN_SM_SCALE,
            },
            5,
        ),
        CaseSpec(
            "all_invalid",
            {"tokens": 3, "pages": 4, "valid": 0, "pattern": "scattered"},
            6,
        ),
        # 100 tokens (200 MLA tiles > SMs): persistent CTAs take several
        # tiles; ragged valid counts, scattered padding, duplicate IDs, peaked
        # attention.
        CaseSpec(
            "many_tokens",
            {
                "tokens": 100,
                "pages": 300,
                "valid": many,
                "pattern": "scattered",
                "duplicates": True,
                "q_scale": 3.0,
                "sm_scale": YARN_SM_SCALE,
            },
            7,
        ),
    ]


def upstream_cases() -> list[CaseSpec]:
    return [
        upstream_case(
            f"upstream_cutlass_mla_b{b}_seq{n}",
            {
                "tokens": b,
                # test_cutlass_mla: total_page_num 8192 one-token pages.
                "pages": 8192 // PAGE,
                "valid": n,
                "duplicates": True,  # torch.randint page table
                "q_scale": 100.0,  # "use larger scale to detect bugs"
            },
            f"{MLA_TEST}[batch_size={b},max_seq_len={n},page_size=1,dtype=bfloat16]",
            seed=b * 10000 + n,
            revision=FLASHINFER_REVISION,
        )
        for b, n in UPSTREAM_MLA
    ]


def _row_scale(axes: dict[str, int]) -> float:
    rows, _ = inventory(THROUGHPUT)
    row = next(r for r in rows if r["axes"] == axes)
    return row["inputs"]["sm_scale"]["value"]


def throughput_cases():
    """Throughput (benchmark) cases of this package."""
    name = THROUGHPUT
    result = []
    for tokens in (1, 2, 6, 7, 8):  # every distinct official row
        axes = dict(num_tokens=tokens, num_pages=OFFICIAL_PAGES)
        params = {
            "tokens": tokens,
            "pages": OFFICIAL_PAGES,
            "valid": TOPK,
            "sm_scale": _row_scale(axes),
        }
        result.append(
            trace(name, f"tokens{tokens}_pages{OFFICIAL_PAGES}", params, axes)
        )
    # DeepSeek-V3.2: decode batches and prefill chunks at the official cache
    # size, a 2M-token cache (16 x 128k contexts) and an 8M-token cache.
    for tokens, pages in (
        (64, OFFICIAL_PAGES),
        (512, OFFICIAL_PAGES),
        (2048, OFFICIAL_PAGES),
        (128, 32768),
        (1024, 131072),
    ):
        params = {
            "tokens": tokens,
            "pages": pages,
            "valid": TOPK,
            "sm_scale": YARN_SM_SCALE,
        }
        result.append(
            model_case(f"tokens{tokens}_pages{pages}", params, MODEL, MODEL_LAYER)
        )
    return result


def all_cases() -> list[CaseSpec]:
    return smoke_cases() + upstream_cases() + throughput_cases()

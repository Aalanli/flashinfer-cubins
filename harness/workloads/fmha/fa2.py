"""FlashInfer FA2 kernels (sm_86) as single-kernel workloads.

The kernels come from the FlashInfer 0.7.0 sm80 JIT cache (sm_80 SASS, run on
sm_86); ``impls/fmha/fa2_build.py`` strips every live kernel into its own
cubin and writes the compact index ``impls/fmha/variants/sm_86.json`` that
this module registers from (one class per kernel family, one registered
variant per kernel; no cubin is parsed at import). The build module's
docstring has the inventory and the live/dead table.

Families (upstream host code mirrored here, FlashInfer 0.7.0):

* ``fmha_fa2_prefill_{paged,ragged}_<q>_<kv>_hd<D>_<mask>[_swa][_softcap]_cta<T>``:
  ``BatchPrefillWith{Paged,Ragged}KVCacheKernel``.
  ``csrc/batch_prefill{.cu,_paged.cuh}`` (``*Run``) fill ``PagedParams`` /
  ``RaggedParams``; ``PrefillPlan`` (``scheduler.cuh``, ported exactly:
  ``disable_split_kv``, the default binary-searched KV split,
  ``fixed_split_size``, CUDA-graph plans with and without ``uniform_q_len``)
  produces the work items. A non-split plan makes the launch write the final
  output (and LSE unless ``return_lse`` is off); a split plan launches with
  ``partition_kv`` and the kernel writes one partial state per (row, KV
  chunk), which ``run`` returns (the merge is ``fmha_fa2_merge_varlen``). The
  plan's ``cta_tile_q`` (``FA2DetermineCtaTileQ`` on sm_86) selects the
  kernel; every case is a shape whose plan selects exactly this kernel's
  ``CTA_TILE_Q``. Masks: ``none``, ``causal``, ``custom`` (bit-packed
  per-request masks with ``mask_indptr``) and, paged only, ``multiitem``
  (multi-item scoring: shared prefix, then items that see the prefix and
  their own earlier tokens). ``_swa`` modules take ``window_left >= 0`` (or
  -1) and ``_softcap`` modules ``logits_soft_cap > 0``, as
  ``BatchPrefillWith*Wrapper.plan`` selects them.
* ``fmha_fa2_sink_{paged,ragged}_<q>_<q>_hd64_{none,causal}_cta<T>``: the
  same prefill kernels instantiated with the ``AttentionSink`` variant
  (``BatchAttentionWithAttentionSinkWrapper``): one extra per-head logit in
  the normalisation of every row of KV chunk 0, ``sink`` and ``sm_scale`` as
  the only additional params, independent paged K/V strides, and
  ``window_left`` applied at run time (the ``use_swa`` module compiles to the
  identical kernel, which the build removes as a duplicate).
* ``fmha_fa2_persistent_<q>_<kv>_hd64_{none,causal}[_softcap]``:
  ``flashinfer.BatchAttention``'s ``PersistentKernelTemplate`` (both
  attention runners and the split-KV reduction, ``grid.sync()`` in between)
  launched cooperatively with ``TwoStageHolisticPlan``'s work queues
  (``csrc/batch_attention.cu``), ported exactly (including the libstdc++
  heap order of ``MinHeap``); ``v_scale`` scales the output.
* ``fmha_fa2_mla_<dtype>_{none,causal}``: ``BatchMLAPagedAttentionKernel``
  (``BatchMLAPagedAttentionWrapper``, fa2 backend): DeepSeek MLA over a
  compressed 512 + 64 latent KV cache, cooperative launch planned by
  ``MLAPlan``.
* ``fmha_fa2_decode_<q>_<kv>_hd<D>_g<G>[_swa][_softcap]``:
  ``BatchDecodeWithPagedKVCacheKernel`` (``csrc/batch_decode.cu``) planned by
  ``DecodePlan``: no split when the batch fills the occupancy-limited grid,
  else (and always under CUDA graphs) a split launch writing partial states.
  The grid comes from the build's occupancy model (``blocks_per_sm``), which
  every native launch checks against the driver's occupancy query.
* ``fmha_fa2_merge_varlen_<dtype>_hd<D>``: ``PersistentVariableLengthMerge
  StatesKernel`` (``VariableLengthMergeStates``, the merge after a split-KV
  prefill/decode, with or without LSE and with the CUDA-graph row-count
  pointer); persistent grid from the occupancy query.
* ``fmha_fa2_merge_state{,_in_place}_<dtype>_vec<V>``,
  ``fmha_fa2_merge_states_<dtype>_vec<V>``,
  ``fmha_fa2_merge_states_large_<dtype>_hd<D>``: ``cascade.cuh`` kernels
  behind ``flashinfer.cascade.merge_state{,_in_place}`` / ``merge_states``.

Cases: see the ``-- cases`` section (own smoke cases, the upstream test
regimes of ``fixtures/sm_86_upstream.json`` and model-shape throughput cases)
and ``regime`` (what upstream coverage compares).

LSEs are base 2 (FlashInfer's convention). PDL is off: upstream enables it
only on sm_90+ (``device_support_pdl``).
"""

from __future__ import annotations

import functools
import itertools
import json
import math
import struct
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, NamedTuple

import torch

from ... import cuda_driver
from ...models import MODELS
from ...registry import register_variant
from ...throughput import model_case, skewed_lengths, synthetic, upstream_case
from ...workload import IMPLS, CaseSpec, Workload

PACKAGE = IMPLS / "fmha"
INDEX = PACKAGE / "variants" / "sm_86.json"
FIXTURES = PACKAGE / "fixtures" / "sm_86.json"
ARCH = "sm_86"
LOG2E = 1.4426950408889634
MAX_SMEM_PER_BLOCK_OPTIN = 101376  # sm_86 (fa2_probe_common.h)
NUM_SMS = 82  # RTX 3090: the probes' device model
INT64_MAX = 2**63 - 1
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "e4m3": torch.float8_e4m3fn}
NHD, HND = 0, 1
MASK_IDS = {"none": 0, "causal": 1, "custom": 2, "multiitem": 3}
SENTINEL_STRIDE = 0x100000000000
# Pointer roles of the probes, in enum order (sentinel = stride * (index + 1)).
PREFILL_ROLES = (
    "q",
    "k",
    "v",
    "q_indptr",
    "kv_indptr",
    "kv_indices",
    "last_page_len",
    "o",
    "lse",
    "custom_mask",
    "mask_indptr",
    "prefix_len",
    "token_pos",
    "max_item_len",
    "request_indices",
    "qo_tile_indices",
    "kv_tile_indices",
    "o_indptr",
    "kv_chunk_size",
    "sink",
    "merge_indptr",
    "block_valid_mask",
    "total_num_rows",
)
DECODE_ROLES = (
    "q",
    "k",
    "v",
    "kv_indices",
    "kv_indptr",
    "last_page_len",
    "o",
    "lse",
    "request_indices",
    "kv_tile_indices",
    "o_indptr",
    "kv_chunk_size",
    "block_valid_mask",
)


@functools.cache
def load_index() -> dict[str, Any]:
    if not INDEX.is_file():
        return {"layouts": {}, "kernels": {}}
    return json.loads(INDEX.read_text())


# -- by-value structs ------------------------------------------------------------


def fastdiv(divisor: int) -> bytes:
    """``flashinfer::uint_fastdiv(divisor)``: ``cuda::fast_mod_div<uint32_t>``
    {divisor, multiplier, add, shift} followed by the raw divisor."""
    d = divisor if divisor else 1
    shift = d.bit_length() - 1
    if d & (d - 1) == 0:
        multiplier, add = 0, 0
    else:
        k = 32 + shift
        multiplier = ((1 << k) + (1 << shift)) // d
        add = int((1 << k) // d == multiplier)
    return struct.pack("<IIIiI", d, multiplier & 0xFFFFFFFF, add, shift, divisor)


_FORMATS = {
    "ptr": "<Q",
    "u32": "<I",
    "i32": "<i",
    "i64": "<q",
    "f32": "<f",
    "f64": "<d",
    "bool": "<?",
}


def pack(layout: dict[str, Any], values: dict[str, Any]) -> bytes:
    """A probe-recorded struct; every field must be given, padding stays 0."""
    fields = layout["fields"]
    missing, extra = set(fields) - set(values), set(values) - set(fields)
    if missing or extra:
        raise ValueError(
            f"Params fields missing {sorted(missing)}, unknown {sorted(extra)}"
        )
    out = bytearray(layout["param_size"])
    for name, value in values.items():
        f = fields[name]
        if f["kind"] == "fastdiv":
            data = fastdiv(int(value))
            if len(data) != f["size"]:
                raise ValueError(f"{name}: uint_fastdiv is {f['size']} bytes")
            out[f["offset"] : f["offset"] + f["size"]] = data
            continue
        fmt = _FORMATS[f["kind"]]
        if struct.calcsize(fmt) != f["size"]:
            raise ValueError(f"{name}: {f['kind']} is not {f['size']} bytes")
        struct.pack_into(fmt, out, f["offset"], value)
    return bytes(out)


def covered(layout: dict[str, Any]) -> bytes:
    mask = bytearray(layout["param_size"])
    for f in layout["fields"].values():
        mask[f["offset"] : f["offset"] + f["size"]] = b"\x01" * f["size"]
    return bytes(mask)


def paged_kv_fields(
    *,
    num_heads: int,
    page_size: int,
    head_dim: int,
    batch: int,
    kv_layout: int,
    k: int,
    v: int,
    k_strides: Sequence[int],
    v_strides: Sequence[int],
    indices: int,
    indptr: int,
    last_page_len: int,
) -> dict[str, Any]:
    """``paged_kv_t(num_heads, page_size, head_dim, batch, layout, k, v,
    k_strides, v_strides, indices, indptr, last_page_len)`` (page.cuh)."""
    hnd = kv_layout == HND
    return {
        "paged_kv.page_size": page_size,
        "paged_kv.num_heads": num_heads,
        "paged_kv.head_dim": head_dim,
        "paged_kv.batch_size": batch,
        "paged_kv.stride_page": k_strides[0],
        "paged_kv.stride_n": k_strides[2] if hnd else k_strides[1],
        "paged_kv.stride_h": k_strides[1] if hnd else k_strides[2],
        "paged_kv.v_stride_page": v_strides[0],
        "paged_kv.v_stride_n": v_strides[2] if hnd else v_strides[1],
        "paged_kv.v_stride_h": v_strides[1] if hnd else v_strides[2],
        "paged_kv.k_data": k,
        "paged_kv.v_data": v,
        "paged_kv.indices": indices,
        "paged_kv.indptr": indptr,
        "paged_kv.last_page_len": last_page_len,
        "paged_kv.rope_pos_offset": 0,
    }


def _additional(ptr: dict[str, int], s: dict[str, Any]) -> dict[str, Any]:
    """The prefill module's ADDITIONAL_PARAMS_SETTER (fa2 backend); the
    AttentionSink modules take the per-head sink logits and ``sm_scale``."""
    if s.get("sink"):
        return {"sink": ptr["sink"], "sm_scale": float(s["sm_scale"])}
    return {
        "maybe_custom_mask": ptr.get("custom_mask", 0),
        "maybe_mask_indptr": ptr.get("mask_indptr", 0),
        "maybe_alibi_slopes": 0,
        "maybe_prefix_len_ptr": ptr.get("prefix_len", 0),
        "maybe_token_pos_in_items_ptr": ptr.get("token_pos", 0),
        "maybe_max_item_len_ptr": ptr.get("max_item_len", 0),
        "maybe_k_cache_sf": 0,
        "maybe_v_cache_sf": 0,
        "logits_soft_cap": float(s["logits_soft_cap"]),
        "sm_scale": float(s["sm_scale"]),
        "rope_rcp_scale": 1.0,
        "rope_rcp_theta": 1.0 / 1e4,
        "token_pos_in_items_len": int(s.get("token_pos_in_items_len", 0)),
    }


def _plan_fields(
    ptr: dict[str, int], padded_batch_size: int, total_num_rows: int, split: bool
) -> dict:
    """Plan-dependent fields: a split-KV plan launches with ``partition_kv``
    and ``merge_indptr``; ``block_valid_mask`` (split) and the device
    ``total_num_rows`` come only with a CUDA-graph plan (pointers absent from
    ``ptr`` are null)."""
    return {
        "request_indices": ptr["request_indices"],
        "qo_tile_indices": ptr["qo_tile_indices"],
        "kv_tile_indices": ptr["kv_tile_indices"],
        "merge_indptr": ptr.get("merge_indptr", 0) if split else 0,
        "o_indptr": ptr["o_indptr"],
        "kv_chunk_size_ptr": ptr["kv_chunk_size"],
        "block_valid_mask": ptr.get("block_valid_mask", 0) if split else 0,
        "max_total_num_rows": total_num_rows,
        "total_num_rows": ptr.get("total_num_rows", 0),
        "padded_batch_size": padded_batch_size,
        "partition_kv": split,
    }


_SF_STRIDES = {
    f"{side}_sf_stride_{dim}": 0 for side in "kv" for dim in ("page", "n", "h")
}


def paged_params(
    layout: dict[str, Any], ptr: dict[str, int], s: dict[str, Any]
) -> bytes:
    """BatchPrefillWithPagedKVCacheRun's PagedParams."""
    values = {
        "q": ptr["q"],
        **paged_kv_fields(
            num_heads=s["num_kv_heads"],
            page_size=s["page_size"],
            head_dim=s["head_dim"],
            batch=s["batch"],
            kv_layout=s["kv_layout"],
            k=ptr["k"],
            v=ptr["v"],
            k_strides=s["k_strides"],
            v_strides=s["v_strides"],
            indices=ptr["kv_indices"],
            indptr=ptr["kv_indptr"],
            last_page_len=ptr["last_page_len"],
        ),
        "q_indptr": ptr["q_indptr"],
        "o": ptr["o"],
        "lse": ptr.get("lse", 0),
        "group_size": s["num_qo_heads"] // s["num_kv_heads"],
        **_additional(ptr, s),
        "num_qo_heads": s["num_qo_heads"],
        "q_stride_n": s["q_strides"][0],
        "q_stride_h": s["q_strides"][1],
        **_SF_STRIDES,
        "window_left": s["window_left"],
        **_plan_fields(
            ptr, s["padded_batch_size"], s["total_num_rows"], s.get("split", False)
        ),
    }
    return pack(layout, values)


def ragged_params(
    layout: dict[str, Any], ptr: dict[str, int], s: dict[str, Any]
) -> bytes:
    """BatchPrefillWithRaggedKVCacheRun's RaggedParams; k/v strides are
    (n, h) element strides of the NHD or HND tensors."""
    values = {
        "q": ptr["q"],
        "k": ptr["k"],
        "v": ptr["v"],
        "q_indptr": ptr["q_indptr"],
        "kv_indptr": ptr["kv_indptr"],
        "o": ptr["o"],
        "lse": ptr.get("lse", 0),
        "group_size": s["num_qo_heads"] // s["num_kv_heads"],
        **_additional(ptr, s),
        "num_qo_heads": s["num_qo_heads"],
        "num_kv_heads": s["num_kv_heads"],
        "q_stride_n": s["q_strides"][0],
        "q_stride_h": s["q_strides"][1],
        "k_stride_n": s["k_strides"][0],
        "k_stride_h": s["k_strides"][1],
        "v_stride_n": s["v_strides"][0],
        "v_stride_h": s["v_strides"][1],
        **_SF_STRIDES,
        "window_left": s["window_left"],
        **_plan_fields(
            ptr, s["padded_batch_size"], s["total_num_rows"], s.get("split", False)
        ),
    }
    return pack(layout, values)


def decode_params(
    layout: dict[str, Any], ptr: dict[str, int], s: dict[str, Any]
) -> bytes:
    """BatchDecodeWithPagedKVCacheRun's Params as launched (a split plan sets
    ``partition_kv``; ``block_valid_mask`` comes with CUDA-graph split plans).
    ``enable_pdl`` is never assigned upstream (the kernel does not read it);
    zero here."""
    values = {
        "q": ptr["q"],
        **paged_kv_fields(
            num_heads=s["num_kv_heads"],
            page_size=s["page_size"],
            head_dim=s["head_dim"],
            batch=s["batch"],
            kv_layout=s["kv_layout"],
            k=ptr["k"],
            v=ptr["v"],
            k_strides=s["k_strides"],
            v_strides=s["v_strides"],
            indices=ptr["kv_indices"],
            indptr=ptr["kv_indptr"],
            last_page_len=ptr["last_page_len"],
        ),
        "o": ptr["o"],
        "lse": ptr.get("lse", 0),
        "maybe_alibi_slopes": 0,
        "logits_soft_cap": float(s["logits_soft_cap"]),
        "sm_scale": float(s["sm_scale"]),
        "rope_rcp_scale": 1.0,
        "rope_rcp_theta": 1.0 / 1e4,
        "padded_batch_size": s["padded_batch_size"],
        "num_qo_heads": s["num_qo_heads"],
        "q_stride_n": s["q_strides"][0],
        "q_stride_h": s["q_strides"][1],
        "window_left": s["window_left"],
        "enable_pdl": False,
        "request_indices": ptr["request_indices"],
        "kv_tile_indices": ptr["kv_tile_indices"],
        "o_indptr": ptr["o_indptr"],
        "kv_chunk_size_ptr": ptr["kv_chunk_size"],
        "block_valid_mask": ptr.get("block_valid_mask", 0) if s.get("split") else 0,
        "partition_kv": bool(s.get("split")),
    }
    return pack(layout, values)


# -- planners (scheduler.cuh) ---------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def fa2_cta_tile_q(avg_packed_qo_len: int, head_dim: int, kv_bytes: int) -> int:
    """FA2DetermineCtaTileQ (utils.cuh) on sm_86 for head dims <= 256."""
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    if avg_packed_qo_len > 16:
        return 64
    q_tile_smem = 16 * head_dim * 2
    kv_step_smem = 2 * head_dim * 16 * 4 * kv_bytes
    return 64 if q_tile_smem + kv_step_smem > MAX_SMEM_PER_BLOCK_OPTIN else 16


def prefill_plan(
    qo_indptr: Sequence[int],
    kv_indptr: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_bytes: int,
    *,
    num_sms: int = NUM_SMS,
    window_left: int = -1,
    cuda_graph: bool = False,
    disable_split_kv: bool = True,
    fixed_split_size: int = -1,
    uniform_q_len: int = 0,
    total_num_rows: int | None = None,
) -> dict[str, Any]:
    """PrefillPlanImpl -> PrefillSplitQOKVIndptr (scheduler.cuh) for a
    symmetric head dim: kernel tile (``cta_tile_q``), work items, KV chunk
    size (tokens; the int64 value stored into the int32 workspace slot) and,
    when the plan splits KV, ``merge_indptr`` and the CUDA-graph
    ``block_valid_mask``. ``kv_indptr`` counts pages (tokens for ragged KV,
    with ``page_size`` 1); ``total_num_rows`` is the planned row count
    (``max_total_num_rows`` of a CUDA-graph wrapper; default: the batch's)."""
    batch = len(qo_indptr) - 1
    group = num_qo_heads // num_kv_heads
    if num_qo_heads % num_kv_heads:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    rows = qo_indptr[-1] if total_num_rows is None else total_num_rows
    packed = [(qo_indptr[i + 1] - qo_indptr[i]) * group for i in range(batch)]
    kv_len = [kv_indptr[i + 1] - kv_indptr[i] for i in range(batch)]
    if min(packed + kv_len, default=0) < 0:
        raise ValueError("indptr must be non-decreasing")
    max_batch_if_split = (2 * num_sms) // num_kv_heads  # num_blocks_per_sm = 2
    min_kv_chunk = max(128 // page_size, 1)
    if cuda_graph:
        if uniform_q_len > 0:
            if any(n != uniform_q_len * group for n in packed):
                raise ValueError("uniform_q_len does not match qo_indptr")
            cta = fa2_cta_tile_q(uniform_q_len * group, head_dim, kv_bytes)
            total_tiles = batch * _ceil_div(uniform_q_len * group, cta)
        else:
            cta = fa2_cta_tile_q((rows - batch + 1) * group, head_dim, kv_bytes)
            total_tiles = _ceil_div(rows * group, cta) + batch - 1
    else:
        cta = fa2_cta_tile_q(sum(packed) // batch, head_dim, kv_bytes)
        total_tiles = sum(_ceil_div(n, cta) for n in packed)
    effective = [
        min(_ceil_div(window_left + cta, page_size) if window_left >= 0 else n, n)
        for n in kv_len
    ]
    split = False
    if disable_split_kv:
        chunk = INT64_MAX
    elif fixed_split_size > 0:
        chunk = fixed_split_size
    else:  # PrefillBinarySearchKVChunkSize
        max_kv = max([1, *effective])
        low, high = min_kv_chunk, max_kv
        while low < high:
            mid = (low + high) // 2
            work = sum(
                _ceil_div(n, cta) * _ceil_div(max(e, 1), mid)
                for n, e in zip(packed, effective)
            )
            if work > max_batch_if_split:
                low = mid + 1
            else:
                high = mid
        split, chunk = cuda_graph or low < max_kv, low
    request: list[int] = []
    qo_tile: list[int] = []
    kv_tile: list[int] = []
    merge_indptr, o_indptr = [0], [0]
    for i in range(batch):
        chunks = 1 if disable_split_kv else _ceil_div(max(effective[i], 1), chunk)
        if fixed_split_size > 0 and not disable_split_kv:
            split = split or chunks > 1
        for t in range(_ceil_div(packed[i], cta)):
            for c in range(chunks):
                request.append(i)
                qo_tile.append(t)
                kv_tile.append(c)
        qo_len = packed[i] // group
        last = merge_indptr[-1]
        merge_indptr.extend(last + chunks * (r + 1) for r in range(qo_len))
        o_indptr.append(o_indptr[-1] + qo_len * chunks)
    new_batch = len(request)
    padded = max(max_batch_if_split, total_tiles) if cuda_graph else new_batch
    if new_batch > padded:
        raise ValueError("new batch size exceeds the padded batch size")
    chunk_tokens = (chunk * page_size) & 0xFFFFFFFFFFFFFFFF
    return {
        "split_kv": split,
        "cta_tile_q": cta,
        "new_batch_size": new_batch,
        "padded_batch_size": padded,
        "request_indices": request,
        "qo_tile_indices": qo_tile,
        "kv_tile_indices": kv_tile,
        "merge_indptr": merge_indptr,
        "o_indptr": o_indptr,
        "kv_chunk_size": _int32(chunk_tokens),
        "block_valid_mask": (
            [i < new_batch for i in range(padded)] if split and cuda_graph else None
        ),
        "total_num_rows": rows,
        "cuda_graph": cuda_graph,
    }


def decode_plan(
    page_counts: Sequence[int],
    num_kv_heads: int,
    page_size: int,
    max_grid_size: int,
    cuda_graph: bool = False,
) -> dict[str, Any]:
    """DecodePlanImpl with BatchDecodeWithPagedKVCacheWorkEstimationDispatched
    and DecodeSplitKVIndptr (scheduler.cuh). ``max_grid_size`` is the
    occupancy x SM count."""
    batch = len(page_counts)
    if batch * num_kv_heads >= max_grid_size:
        split, chunk_pages, new_batch = False, max([1, *page_counts]), batch
    else:  # PartitionPagedKVCacheBinarySearchMinNumPagePerBatch
        low = max(128 // page_size, 1)
        high = max([0, *page_counts])
        while low < high:
            mid = (low + high) // 2
            if (
                sum(_ceil_div(n, mid) for n in page_counts) * num_kv_heads
                > max_grid_size
            ):
                low = mid + 1
            else:
                high = mid
        chunk_pages = low
        new_batch = sum(_ceil_div(max(n, 1), low) for n in page_counts)
        split = new_batch != batch or cuda_graph
    if cuda_graph:
        padded = max_grid_size // num_kv_heads if split else batch
    else:
        padded = new_batch
    request: list[int] = []
    kv_tile: list[int] = []
    o_indptr = [0]
    for i, n in enumerate(page_counts):
        chunks = _ceil_div(max(n, 1), chunk_pages)
        request.extend([i] * chunks)
        kv_tile.extend(range(chunks))
        o_indptr.append(o_indptr[-1] + chunks)
    return {
        "split_kv": split,
        "new_batch_size": new_batch,
        "padded_batch_size": padded,
        "kv_chunk_size": chunk_pages * page_size,
        "request_indices": request,
        "kv_tile_indices": kv_tile,
        "o_indptr": o_indptr,
        # The Run function passes the mask only to CUDA-graph launches.
        "block_valid_mask": (
            [i < new_batch for i in range(padded)] if split and cuda_graph else None
        ),
        "cuda_graph": cuda_graph,
    }


# -- references ----------------------------------------------------------------

# Bytes of float32 scores one reference step may hold (query rows are chunked).
SCORE_BUDGET = 1 << 28


def attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep: torch.Tensor,
    sm_scale: float,
    soft_cap: float,
    sink: torch.Tensor | None = None,
    v_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q [Lq, Hq, D], k/v [Lk, Hkv, D] (float32), keep [Lq, Lk] or
    [Hq, Lq, Lk] -> (o [Lq, Hq, D], base-2 lse [Lq, Hq]). ``soft_cap`` > 0
    applies ``cap * tanh(s / cap)``; ``sink`` [Hq] adds one natural-log logit
    per head to every row's normalisation (AttentionSink) without a value;
    ``v_scale`` scales the output. A row without keys (and without a sink)
    gives o = 0 and lse = -inf, as the kernels write it (``d_rcp`` 0)."""
    lq, hq, d = q.shape
    lk, hkv = k.shape[0], k.shape[1]
    group = hq // hkv
    keep = keep.expand(hq, lq, lk)
    o = q.new_zeros((lq, hq, v.shape[2]))
    lse = q.new_full((lq, hq), float("-inf"))
    if lq == 0:
        return o, lse
    step = max(1, SCORE_BUDGET // max(1, 4 * hq * max(lk, 1)))
    kg = k.permute(1, 0, 2)  # [Hkv, Lk, D]
    vg = v.permute(1, 0, 2)
    for lo in range(0, lq, step):
        hi = min(lq, lo + step)
        qg = q[lo:hi].reshape(hi - lo, hkv, group, d).permute(1, 2, 0, 3)
        s = torch.matmul(qg, kg.unsqueeze(1).transpose(-1, -2)) * sm_scale
        s = s.reshape(hq, hi - lo, lk)
        if soft_cap > 0:
            s = soft_cap * torch.tanh(s / soft_cap)
        s = s.masked_fill(~keep[:, lo:hi], float("-inf"))
        if sink is not None:
            column = sink.float().view(-1, 1, 1).expand(hq, hi - lo, 1)
            s = torch.cat([s, column], dim=-1)
        row = torch.logsumexp(s, dim=-1)
        shift = torch.where(torch.isfinite(row), row, torch.zeros_like(row))
        probs = torch.exp(s - shift.unsqueeze(-1))[..., :lk]
        pg = probs.reshape(hkv, group, hi - lo, lk)
        out = torch.matmul(pg, vg.unsqueeze(1))  # [Hkv, G, Lq, D]
        o[lo:hi] = out.reshape(hq, hi - lo, -1).transpose(0, 1) * v_scale
        lse[lo:hi] = (row * LOG2E).transpose(0, 1)
    return o, lse


def merge(v: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention states over dim 0: v [n, ..., D], s [n, ...] (base-2
    LSEs) -> (v_merged, s_merged); n == 0 gives zeros and -inf."""
    if v.shape[0] == 0:
        return torch.zeros(v.shape[1:], device=v.device), torch.full(
            s.shape[1:], float("-inf"), device=s.device
        )
    s_max = s.max(dim=0).values
    w = torch.exp2(s - s_max)
    total = w.sum(dim=0)
    merged = (w.unsqueeze(-1) * v).sum(dim=0) / total.unsqueeze(-1)
    return merged, s_max + torch.log2(total)


def pack_bits(mask: torch.Tensor) -> torch.Tensor:
    """Little-endian bit packing of a flat bool tensor (``segment_packbits``)."""
    flat = mask.reshape(-1).to(torch.uint8)
    flat = torch.cat([flat, flat.new_zeros((-flat.numel()) % 8)])
    weights = torch.tensor(
        [1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=mask.device
    )
    return (flat.reshape(-1, 8) * weights).sum(dim=1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(-1)[:count].bool()


def lse_tolerance(dtype_q: str, keys: int) -> float:
    """|dLSE| bound of the FA2 tensor-core kernels (prefill, persistent, MLA):
    the softmax denominator is an MMA row sum of probabilities (<= 1 against
    the running max) already rounded to the 16-bit Q type. bf16: 2^-9
    relative per term -> log2(1 + 2^-8) = 5.6e-3. f16: 2^-11 relative (2^-10
    with margin) plus the 2^-25 absolute rounding of f16's subnormal range
    (p < 2^-14) per key -> log2(1 + 2^-10 + keys * 2^-25), so long rows get
    more room (measured: 2.1e-3 over 92906 keys, bound 5.4e-3; 2.3e-4 over
    the smoke cases' <= 3000 keys, bound 1.5e-3)."""
    if dtype_q == "bf16":
        return math.log2(1 + 2**-8)
    return math.log2(1 + 2**-10 + keys * 2**-25)


def _keys(lse: torch.Tensor, keys: int) -> torch.Tensor:
    """Tag a reference LSE with the most keys any of its rows attends to (the
    f16 bound of ``lse_tolerance`` grows with it)."""
    lse.fa2_keys = keys  # type: ignore[attr-defined]
    return lse


def _cumsum(values: Sequence[int]) -> list[int]:
    out = [0]
    for n in values:
        out.append(out[-1] + n)
    return out


# -- shared workload plumbing ------------------------------------------------------

# Case knobs every attention family accepts (``_Fa2.defaults``):
#  values      "peaked" (q ~ N(0, logit_std^2) so q.k * sm_scale has std logit_std:
#              a few keys dominate, outputs are O(1) and errors visible), "randn"
#              (unit q, as the upstream tests), "extreme" (q = 64, k = -64,
#              v = 1: the upstream extreme-negative-logit tests) or "constant"
#              (q = k = 0, v = 1: the upstream fully-masked-row test);
#  pages       paged KV page tables: "random" (a permutation of a larger pool),
#              "identity" (arange, as upstream), "offset" (arange + 3) or "shared"
#              (every request starts with the same physical pages);
#  kv_storage  "interleaved" (one [pages, 2, ...] tensor, K/V views), "separate"
#              (a (K, V) tuple), "padded" (K/V are head slices of head-padded
#              caches: non-contiguous token and page strides) or, ragged only,
#              "packed" (q/k/v column slices of one packed qkv tensor);
#  q_storage   "contiguous" or "packed" (q is a column slice of a packed qkv row);
#  poison      NaN in every page slot no request covers (past last_page_len and
#              in unreferenced pages): only covered slots may influence outputs.
COMMON_DEFAULTS: dict[str, Any] = {
    "values": "peaked",
    "logit_std": 2.5,
    "pages": "random",
    "kv_storage": "interleaved",
    "q_storage": "contiguous",
    "poison": False,
}


class _Fa2(Workload):
    """A FlashInfer FA2 kernel of the sm80 JIT cache; ``meta`` is its index
    entry (``variants/sm_86.json``)."""

    package: ClassVar[str | None] = "fmha"
    meta: ClassVar[dict[str, Any]] = {}
    defaults: ClassVar[dict[str, Any]] = {}

    def configure(self, function: cuda_driver.Function) -> None:
        smem = self.meta.get("shared_mem", 0)
        if smem:
            # Upstream sets the attribute unconditionally (cudaFuncSetAttribute).
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )

    def num_sms(self) -> int:
        """SM count the planners see: the device's, or the probes' sm_86 model
        (82, an RTX 3090) for CPU tensors."""
        if self.device.type != "cuda":
            return NUM_SMS
        return torch.cuda.get_device_properties(self.device).multi_processor_count

    def route(self, case_params: dict[str, Any]) -> dict[str, Any]:
        """The upstream plan of a case (attention families)."""
        raise NotImplementedError(f"{type(self).__name__} has no planner")

    def occupancy(self, block: Sequence[int], shared_mem: int) -> int:
        return cuda_driver.max_active_blocks_per_sm(
            self.function, math.prod(block), shared_mem
        )

    def run(self, inputs: tuple) -> tuple:
        launch, _ = self.prepare(inputs)
        return launch()

    @classmethod
    def params(cls, case_params: dict[str, Any]) -> dict[str, Any]:
        """A case's parameters with every default filled in."""
        return {**COMMON_DEFAULTS, **cls.defaults, **case_params}

    @staticmethod
    def ints(values: Sequence[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(list(values), dtype=torch.int32, device=device)

    def values(
        self, shape: Sequence[int], dtype: str, g: torch.Generator, scale: float = 1.0
    ) -> torch.Tensor:
        """N(0, scale^2) data, drawn in float32 slabs of at most 2^26 elements
        so that multi-GB caches do not need a float32 copy of themselves."""
        shape = tuple(shape)
        row = math.prod(shape[1:])
        step = max(1, (1 << 26) // max(row, 1))
        if not shape or shape[0] <= step:
            x = torch.randn(shape, generator=g, device=self.device)
            return (x * scale).to(DTYPES[dtype])
        out = torch.empty(shape, dtype=DTYPES[dtype], device=self.device)
        for lo in range(0, shape[0], step):
            n = min(step, shape[0] - lo)
            x = torch.randn((n, *shape[1:]), generator=g, device=self.device)
            out[lo : lo + n] = (x * scale).to(out.dtype)
        return out

    def qkv_values(
        self, role: str, shape: Sequence[int], dtype: str, g: torch.Generator, p: dict
    ) -> torch.Tensor:
        """Q, K or V data for the case's ``values`` distribution."""
        mode = p["values"]
        if mode == "extreme":
            fill = {"q": 64.0, "k": -64.0, "v": 1.0}[role]
            return torch.full(tuple(shape), fill, device=self.device).to(DTYPES[dtype])
        if mode == "constant":
            fill = {"q": 0.0, "k": 0.0, "v": 1.0}[role]
            return torch.full(tuple(shape), fill, device=self.device).to(DTYPES[dtype])
        scale = p["logit_std"] if role == "q" and mode == "peaked" else 1.0
        return self.values(shape, dtype, g, scale)

    def packed_rows(
        self,
        rows: int,
        widths: Sequence[int],
        dtypes: Sequence[str],
        roles: Sequence[str],
        heads: Sequence[int],
        g: torch.Generator,
        p: dict,
    ) -> list[torch.Tensor]:
        """Column slices [rows, H, D] of one packed row buffer per dtype group
        (upstream's packed-qkv tests): non-contiguous token strides."""
        total = sum(widths)
        out = []
        start = 0
        buffers: dict[str, torch.Tensor] = {}
        for width, dtype, role, h in zip(widths, dtypes, roles, heads):
            if dtype not in buffers:
                buffers[dtype] = torch.empty(
                    (rows, total), dtype=DTYPES[dtype], device=self.device
                )
            buf = buffers[dtype]
            buf[:, start : start + width] = self.qkv_values(
                role, (rows, width), dtype, g, p
            )
            out.append(buf[:, start : start + width].view(rows, h, width // h))
            start += width
        return out

    def paged_cache(
        self,
        g: torch.Generator,
        kv_lens: Sequence[int],
        page: int,
        hkv: int,
        d: int,
        dtype: str,
        p: dict,
    ) -> dict[str, Any]:
        """Paged K/V for ``kv_lens``: K/V views ([pages, P, H, D] NHD or
        [pages, H, P, D] HND), page indptr, indices and last-page lengths."""
        dev = self.device
        counts = [_ceil_div(n, page) for n in kv_lens]
        mode = p["pages"]
        shared = min(counts, default=0) // 2 if mode == "shared" else 0
        unique = sum(c - shared for c in counts) + shared
        spare = {"random": 3, "offset": 3}.get(mode, 0)
        num_pages = max(unique + spare, 1)
        if mode == "random":
            pool = torch.randperm(num_pages, generator=g, device=dev).tolist()
        elif mode == "offset":
            pool = list(range(spare, num_pages)) + list(range(spare))
        else:
            pool = list(range(num_pages))
        indices: list[int] = []
        prefix, cursor = pool[:shared], shared
        for c in counts:
            indices += prefix[: min(c, shared)]
            rest = c - min(c, shared)
            indices += pool[cursor : cursor + rest]
            cursor += rest
        nhd = p["kv_layout"] == NHD
        shape = (num_pages, page, hkv, d) if nhd else (num_pages, hkv, page, d)
        if p["kv_storage"] == "k_only":  # MLA: one latent cache, V = K
            k = v = self.qkv_values("k", shape, dtype, g, p)
        elif p["kv_storage"] == "padded":  # head slices of head-padded parents
            wide = list(shape)
            wide[2 if nhd else 1] += 2
            k = self.qkv_values("k", wide, dtype, g, p)
            v = self.qkv_values("v", wide, dtype, g, p)
            k = k[:, :, :hkv] if nhd else k[:, :hkv]
            v = v[:, :, :hkv] if nhd else v[:, :hkv]
        elif p["kv_storage"] == "separate":
            k = self.qkv_values("k", shape, dtype, g, p)
            v = self.qkv_values("v", shape, dtype, g, p)
        else:
            cache = torch.empty(
                (num_pages, 2, *shape[1:]), dtype=DTYPES[dtype], device=dev
            )
            step = max(1, (1 << 26) // math.prod(shape[1:]))  # bounded temporaries
            for lo in range(0, num_pages, step):
                part = (min(step, num_pages - lo), *shape[1:])
                cache[lo : lo + part[0], 0] = self.qkv_values("k", part, dtype, g, p)
                cache[lo : lo + part[0], 1] = self.qkv_values("v", part, dtype, g, p)
            k, v = cache[:, 0], cache[:, 1]
        last = [n - (c - 1) * page if c else 0 for n, c in zip(kv_lens, counts)]
        if p["poison"]:
            covered = torch.zeros((num_pages, page), dtype=torch.bool)
            pos = 0
            for n, c in zip(kv_lens, counts):
                for j, idx in enumerate(indices[pos : pos + c]):
                    covered[idx, : min(page, n - j * page)] = True
                pos += c
            holes = ~covered.to(dev)
            for t in (k, v) if v is not k else (k,):
                view = t if nhd else t.transpose(1, 2)
                if dtype == "e4m3":  # index_put has no float8 kernel; 0x7f is NaN
                    view.view(torch.uint8)[holes] = 0x7F
                else:
                    view[holes] = float("nan")
        return {
            "k": k,
            "v": v,
            "kv_indptr": self.ints(_cumsum(counts), dev),
            "kv_indices": self.ints(indices, dev),
            "last_page_len": self.ints(last, dev),
        }

    @staticmethod
    def gather_paged(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        b: int,
        length: int,
        kv_layout: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Request ``b``'s K and V as float32 [Lk, Hkv, D]."""
        pages = kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].long()
        kk, vv = k[pages].float(), v[pages].float()
        if kv_layout == HND:  # [pages, H, P, D] -> [pages, P, H, D]
            kk, vv = kk.transpose(1, 2), vv.transpose(1, 2)
        kk = kk.reshape(-1, kk.shape[2], kk.shape[3])[:length]
        vv = vv.reshape(-1, vv.shape[2], vv.shape[3])[:length]
        return kk, vv

    @staticmethod
    def _tolerance(dtype: str) -> tuple[float, float]:
        """Output tolerances: the kernels accumulate in FP32 but feed the
        probabilities to the second MMA (and store the output) in the
        16-bit output type; a few output ulps at O(1) magnitude."""
        return (2e-2, 2e-2) if dtype == "bf16" else (4e-3, 4e-3)


# -- prefill -----------------------------------------------------------------------


class PrefillInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor | None
    last_page_len: torch.Tensor | None
    custom_mask: torch.Tensor | None
    mask_indptr: torch.Tensor | None
    prefix_len: torch.Tensor | None
    token_pos: torch.Tensor | None
    max_item_len: torch.Tensor | None
    sink: torch.Tensor | None
    params: dict[str, Any]


def prefill_route(meta: dict[str, Any], p: dict[str, Any], num_sms: int) -> dict:
    """The plan ``BatchPrefillWith*Wrapper.plan`` makes for case params
    ``p`` (filled by ``Fa2Prefill.params``); its ``cta_tile_q`` selects the
    kernel. Paged KV is planned in pages, ragged KV in tokens."""
    page = p["page_size"] if meta["kind"] == "paged" else 1
    counts = [_ceil_div(n, page) for n in p["kv_lens"]]
    return prefill_plan(
        _cumsum(p["q_lens"]),
        _cumsum(counts),
        p["hq"],
        p["hkv"],
        meta["head_dim"],
        page,
        1 if meta["dtype_kv"] == "e4m3" else 2,
        num_sms=num_sms,
        window_left=p["window_left"],
        cuda_graph=p["cuda_graph"],
        disable_split_kv=p["disable_split_kv"],
        fixed_split_size=p["fixed_split_size"],
        uniform_q_len=p["uniform_q_len"],
        total_num_rows=p["graph_rows"],
    )


def causal_mask_skipped(meta: dict[str, Any], p: dict[str, Any], plan: dict) -> bool:
    """Whether the causal kernel leaves keys unmasked that the causal mask
    removes, a defect of ``BatchPrefillWith{Paged,Ragged}KVCacheKernel`` for
    requests with more queries than keys: with ``kv_len + tile_start / group
    < qo_len`` the uint32 ``mask_iteration`` bound wraps and the first
    ``chunk_size / CTA_TILE_KV`` KV iterations skip ``logits_mask`` unless
    the window bound happens to cover them. Evaluated per work item exactly
    as the kernel computes its bounds (checked against native runs)."""
    if meta["mask"] != MASK_IDS["causal"]:
        return False
    tkv = 16 * meta["num_mma_kv"] * meta["block"][2]
    cta = meta["cta_tile_q"]
    group = p["hq"] // p["hkv"]
    wrap = 1 << 32

    def sub(x: int, y: int) -> int:  # sub_if_greater_or_zero on uint32 values
        x, y = x % wrap, y % wrap
        return x - y if x > y else 0

    split, chunk = plan["split_kv"], plan["kv_chunk_size"]
    for i in range(plan["new_batch_size"]):
        b, t, c = (
            plan[k][i]
            for k in ("request_indices", "qo_tile_indices", "kv_tile_indices")
        )
        lq, lk = p["q_lens"][b], p["kv_lens"][b]
        if lq <= lk:
            continue
        wl = p["window_left"] if p["window_left"] >= 0 else lk
        start = sub(lk + t * cta // group, lq + wl)
        if split:
            first = min(c * chunk + start, lk)
            end = min((c + 1) * chunk + start, lk)
        else:
            first, end = start, lk
        size = end - first
        hi, lo = _ceil_div((t + 1) * cta, group), _ceil_div(t * cta, group)
        iterations = _ceil_div(min(size, sub(lk - lq + hi, first)), tkv)
        window_it = _ceil_div(sub(lk + hi, lq + wl + first), tkv)
        mask_it = min(size, sub(lk + lo - lq, first)) // tkv
        top_row = t * cta // group  # the tile's row with the fewest keys
        for it in range(iterations):
            if it >= mask_it or it < window_it:
                continue  # logits_mask applies
            last_key = min(first + (it + 1) * tkv, end) - 1
            if last_key > lk - lq + top_row:
                return True
    return False


class Fa2Prefill(_Fa2):
    """``BatchPrefillWith{Paged,Ragged}KVCacheKernel`` (one mask mode, one
    ``CTA_TILE_Q``, the sm_86 ``NUM_MMA_KV``).

    Case parameters (besides ``COMMON_DEFAULTS``): ``hq``/``hkv`` heads,
    ``q_lens``/``kv_lens`` per request, ``page_size``, ``kv_layout``,
    ``window_left``, ``logits_soft_cap``, ``sm_scale``; the plan options
    ``disable_split_kv`` (default: the harness's single-launch plan),
    ``fixed_split_size``, ``cuda_graph`` (+ ``uniform_q_len``,
    ``graph_rows`` = the wrapper's ``max_total_num_rows``); ``return_lse``;
    ``mask`` for custom masks (``random``: density 0.6, every row keeps its
    diagonal; ``holes``: some rows fully masked; ``tril``: upstream's
    causal-equivalent ``tril(diagonal=kv-qo)``); ``sink_scale``.

    A split-KV plan makes the kernel write one partial state per (query row,
    KV chunk) into ``tmp_v``/``tmp_s`` (``partition_kv``; the merge is the
    separate ``fmha_fa2_merge_varlen`` kernel): ``run`` then returns those
    partial states, and the reference computes each chunk's normalised
    attention exactly as the kernel bounds it (``kv_start_idx`` of the Q tile,
    ``kv_chunk_size``). A CUDA-graph plan pads the grid; the padded CTAs
    return on ``block_valid_mask`` (split plans) or, without a mask, redo
    work item 0 of the zero-filled index arrays (identical values)."""

    defaults: ClassVar[dict[str, Any]] = {
        "page_size": 1,
        "kv_layout": NHD,
        "window_left": -1,
        "logits_soft_cap": 0.0,
        "return_lse": True,
        "disable_split_kv": True,
        "cuda_graph": False,
        "fixed_split_size": -1,
        "uniform_q_len": 0,
        "graph_rows": None,
        "mask": "random",
        "sink_scale": 2.0,
    }

    @classmethod
    def params(cls, case_params: dict[str, Any]) -> dict[str, Any]:
        p = super().params(case_params)
        m = cls.meta
        p.setdefault("sm_scale", 1.0 / math.sqrt(m["head_dim"]))
        if m["kind"] == "ragged":
            p["page_size"] = 1
        # Modules without use_sliding_window still offset kv_start_idx by a
        # window_left >= 0 without masking; upstream passes -1 to them.
        if p["window_left"] >= 0 and not (m["swa"] or m.get("sink")):
            raise ValueError("window_left needs a sliding-window kernel")
        if (p["logits_soft_cap"] > 0) != m["softcap"]:
            raise ValueError("logits_soft_cap > 0 exactly for soft-cap kernels")
        return p

    def route(self, case_params: dict[str, Any]) -> dict[str, Any]:
        return prefill_route(self.meta, self.params(case_params), self.num_sms())

    def get_cases(self) -> list[CaseSpec]:
        cases = prefill_cases(self.meta)
        for case in cases:
            plan = self.route(case.params)
            if plan["cta_tile_q"] != self.meta["cta_tile_q"]:
                raise AssertionError(
                    f"{self.name}/{case.name}: plan picks {plan['cta_tile_q']}"
                )
            if causal_mask_skipped(self.meta, self.params(case.params), plan):
                raise AssertionError(f"{self.name}/{case.name}: hits the mask defect")
        return cases

    # -- inputs ----------------------------------------------------------------

    def get_inputs(self, case: CaseSpec) -> tuple:
        m, p = self.meta, self.params(case.params)
        g = self.generator(case)
        dev = self.device
        d, hq, hkv = m["head_dim"], p["hq"], p["hkv"]
        dq, dkv = m["dtype_q"], m["dtype_kv"]
        q_lens, kv_lens = p["q_lens"], p["kv_lens"]
        rows, total_kv = sum(q_lens), sum(kv_lens)
        p["plan"] = prefill_route(m, p, self.num_sms())
        qo_indptr = self.ints(_cumsum(q_lens), dev)
        kv_indices = last_page_len = None
        if m["kind"] == "ragged" and p["kv_storage"] == "packed":
            if q_lens != kv_lens:
                raise ValueError("packed qkv needs equal q and kv lengths")
            q, k, v = self.packed_rows(
                rows,
                (hq * d, hkv * d, hkv * d),
                (dq, dkv, dkv),
                ("q", "k", "v"),
                (hq, hkv, hkv),
                g,
                p,
            )
            kv_indptr = qo_indptr.clone()
        else:
            if p["q_storage"] == "packed":
                q, *_ = self.packed_rows(
                    rows,
                    (hq * d, hkv * d, hkv * d),
                    (dq, dq, dq),
                    ("q", "k", "v"),
                    (hq, hkv, hkv),
                    g,
                    p,
                )
            else:
                q = self.qkv_values("q", (rows, hq, d), dq, g, p)
            if m["kind"] == "paged":
                cache = self.paged_cache(g, kv_lens, p["page_size"], hkv, d, dkv, p)
                k, v = cache["k"], cache["v"]
                kv_indptr, kv_indices = cache["kv_indptr"], cache["kv_indices"]
                last_page_len = cache["last_page_len"]
            else:
                shape = (
                    (total_kv, hkv, d) if p["kv_layout"] == NHD else (hkv, total_kv, d)
                )
                k = self.qkv_values("k", shape, dkv, g, p)
                v = self.qkv_values("v", shape, dkv, g, p)
                kv_indptr = self.ints(_cumsum(kv_lens), dev)
        custom_mask = mask_indptr = prefix_len = token_pos = max_item_len = None
        if m["mask"] == MASK_IDS["custom"]:
            chunks, offsets = [], [0]
            for lq, lk in zip(q_lens, kv_lens):
                rows_q = torch.arange(lq, device=dev).unsqueeze(1)
                keys = torch.arange(lk, device=dev).unsqueeze(0)
                if p["mask"] == "tril":
                    keep = keys <= rows_q + (lk - lq)
                else:
                    keep = torch.rand((lq, lk), generator=g, device=dev) < 0.6
                    diag = torch.ones(lq, dtype=torch.bool, device=dev)
                    if p["mask"] == "holes":  # leave ~1/4 of the rows fully masked
                        empty = torch.rand(lq, generator=g, device=dev) < 0.25
                        keep[empty] = False
                        diag = ~empty
                    # Diagonal (bottom-right aligned) keys stay: inside any window.
                    r = torch.arange(lq, device=dev)
                    cols = r + lk - lq
                    ok = diag & (cols >= 0) & (cols < lk)
                    keep[r[ok], cols[ok]] = True
                packed = pack_bits(keep)
                chunks.append(packed)
                offsets.append(offsets[-1] + packed.numel())
            custom_mask = torch.cat(chunks)
            mask_indptr = self.ints(offsets, dev)
        elif m["mask"] == MASK_IDS["multiitem"]:
            prefixes, positions, maxima = [], [], []
            for lq, lk in zip(q_lens, kv_lens):
                prefix = max(1, (lk - lq) // 2)
                pos: list[int] = []
                while len(pos) < lk - prefix:
                    item = int(torch.randint(1, 9, (1,), generator=g, device=dev))
                    pos.extend(range(1, item + 1))
                pos = pos[: lk - prefix]
                prefixes.append(prefix)
                positions.append(pos)
                maxima.append(max(pos, default=0))
            width = max(len(x) for x in positions)
            p["token_pos_in_items_len"] = width
            token_pos = torch.zeros((len(q_lens), width), dtype=torch.int32, device=dev)
            for b, pos in enumerate(positions):
                token_pos[b, : len(pos)] = torch.tensor(pos, dtype=torch.int32)
            token_pos = token_pos.to(torch.uint16)
            prefix_len = torch.tensor(prefixes, dtype=torch.int32, device=dev).to(
                torch.uint32
            )
            max_item_len = torch.tensor(maxima, dtype=torch.int32, device=dev).to(
                torch.uint16
            )
        sink = None
        if m.get("sink"):
            sink = torch.randn(hq, generator=g, device=dev) * p["sink_scale"]
        return PrefillInputs(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            custom_mask,
            mask_indptr,
            prefix_len,
            token_pos,
            max_item_len,
            sink,
            p,
        )

    # -- reference --------------------------------------------------------------

    def _gather(
        self, inputs: PrefillInputs, b: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Request ``b``'s K and V as float32 [Lk, Hkv, D]."""
        p = inputs.params
        lk = p["kv_lens"][b]
        if self.meta["kind"] == "ragged":
            lo = int(inputs.kv_indptr[b])
            k, v = inputs.k, inputs.v
            if p["kv_layout"] == NHD or p["kv_storage"] == "packed":
                return k[lo : lo + lk].float(), v[lo : lo + lk].float()
            return (
                k[:, lo : lo + lk].transpose(0, 1).float(),
                v[:, lo : lo + lk].transpose(0, 1).float(),
            )
        assert inputs.kv_indices is not None
        return self.gather_paged(
            inputs.k,
            inputs.v,
            inputs.kv_indptr,
            inputs.kv_indices,
            b,
            lk,
            p["kv_layout"],
        )

    def keep_mask(self, inputs: PrefillInputs, b: int) -> torch.Tensor:
        """[Lq, Lk] keys request ``b``'s queries attend to (queries are the
        last Lq of the Lk positions)."""
        p, m = inputs.params, self.meta
        lq, lk = p["q_lens"][b], p["kv_lens"][b]
        dev = inputs.q.device
        pos = torch.arange(lq, device=dev).unsqueeze(1) + (lk - lq)
        key = torch.arange(lk, device=dev).unsqueeze(0)
        keep = torch.ones((lq, lk), dtype=torch.bool, device=dev)
        window = p["window_left"]
        if m["mask"] in (MASK_IDS["causal"], MASK_IDS["multiitem"]):
            keep &= key <= pos
        if m["mask"] == MASK_IDS["custom"]:
            assert inputs.custom_mask is not None and inputs.mask_indptr is not None
            lo = int(inputs.mask_indptr[b])
            keep &= unpack_bits(inputs.custom_mask[lo:], lq * lk).reshape(lq, lk)
        # Sink kernels apply the window at run time in every module flavour.
        if window >= 0 and (m["swa"] or m.get("sink")):
            keep &= key >= pos - window
        if m["mask"] == MASK_IDS["multiitem"]:
            assert inputs.prefix_len is not None and inputs.token_pos is not None
            prefix = int(inputs.prefix_len[b].to(torch.int64))
            item_pos = inputs.token_pos[b].to(torch.int32).long()
            in_items = (pos >= prefix).squeeze(1)
            rel = (pos.squeeze(1) - prefix).clamp(min=0)
            start = pos.squeeze(1) - item_pos[rel]  # keys after this are in the item
            item_keep = (key < prefix) | (key > start.unsqueeze(1))
            keep &= torch.where(in_items.unsqueeze(1), item_keep, torch.ones_like(keep))
        return keep

    def get_reference(self, inputs: tuple) -> tuple:
        inputs = PrefillInputs(*inputs)
        m, p = self.meta, inputs.params
        plan = p["plan"]
        hq, hkv, d = p["hq"], p["hkv"], m["head_dim"]
        group = hq // hkv
        split = plan["split_kv"]
        dev = inputs.q.device
        rows = plan["o_indptr"][-1] if split else sum(p["q_lens"])
        o_all = torch.zeros((rows, hq, d), device=dev)
        lse_all = torch.full((rows, hq), float("-inf"), device=dev)
        args = (p["sm_scale"], p["logits_soft_cap"])
        qo = _cumsum(p["q_lens"])
        cta, chunk = m["cta_tile_q"], plan["kv_chunk_size"]
        for b, (lq, lk) in enumerate(zip(p["q_lens"], p["kv_lens"])):
            if lq == 0:
                continue
            kk, vv = self._gather(inputs, b)
            keep = self.keep_mask(inputs, b)
            qb = inputs.q[qo[b] : qo[b + 1]].float()
            if not split:
                o, lse = attend(qb, kk, vv, keep, *args, inputs.sink)
                o_all[qo[b] : qo[b + 1]], lse_all[qo[b] : qo[b + 1]] = o, lse
                continue
            # partition_kv: the Q tile of (row, head) fixes kv_start_idx, KV chunk
            # c covers [start + c * chunk, start + (c + 1) * chunk) clipped to Lk.
            wl = p["window_left"] if p["window_left"] >= 0 else lk
            nch = _ceil_div(min(max(lk, 1), wl + cta), chunk)
            base = plan["o_indptr"][b]
            if plan["o_indptr"][b + 1] - base != lq * nch:
                raise AssertionError("plan and kernel disagree on the KV chunk count")
            r = torch.arange(lq, device=dev).unsqueeze(0)
            h = torch.arange(hq, device=dev).unsqueeze(1)
            tile = (r * group + h % group) // cta
            start = (lk + tile * cta // group - (lq + wl)).clamp(min=0)
            key = torch.arange(lk, device=dev)
            for c in range(nch):
                lo = (start + c * chunk).clamp(max=lk).unsqueeze(-1)
                hi = (start + (c + 1) * chunk).clamp(max=lk).unsqueeze(-1)
                kc = keep.unsqueeze(0) & (key >= lo) & (key < hi)
                o, lse = attend(qb, kk, vv, kc, *args, inputs.sink if c == 0 else None)
                idx = base + torch.arange(lq, device=dev) * nch + c
                o_all[idx], lse_all[idx] = o, lse
        o_all = o_all.to(DTYPES[m["dtype_o"]])
        keys = max(p["kv_lens"], default=0)
        if split:
            keys = min(keys, chunk)
        if split or p["return_lse"]:
            return o_all, _keys(lse_all, keys)
        return (o_all,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != len(impl):
            raise AssertionError("output count differs")
        rtol, atol = self._tolerance(self.meta["dtype_o"])
        self.assert_close(ref[:1], impl[:1], rtol=rtol, atol=atol)
        if len(ref) > 1:  # fully masked rows are -inf in both
            keys = getattr(ref[1], "fa2_keys", 0)
            atol = lse_tolerance(self.meta["dtype_q"], keys)
            self.assert_close(ref[1:], impl[1:], rtol=0.0, atol=atol)

    # -- launch ---------------------------------------------------------------

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        inputs = PrefillInputs(*inputs)
        m, p = self.meta, inputs.params
        q = inputs.q
        dev = q.device
        if dev.type != "cuda":
            raise ValueError("native launch needs CUDA tensors")
        plan = p["plan"]
        if plan["cta_tile_q"] != m["cta_tile_q"]:
            raise ValueError(
                f"plan selects CTA_TILE_Q={plan['cta_tile_q']}, not this kernel"
            )
        hq, hkv, d = p["hq"], p["hkv"], m["head_dim"]
        split = plan["split_kv"]
        rows = plan["o_indptr"][-1] if split else q.shape[0]
        o = torch.empty((rows, hq, d), dtype=DTYPES[m["dtype_o"]], device=dev)
        lse = None
        if split or p["return_lse"]:
            lse = torch.empty((rows, hq), dtype=torch.float32, device=dev)
        padded = plan["padded_batch_size"]

        def padded_ints(values: list[int]) -> torch.Tensor:
            return self.ints(values + [0] * (padded - len(values)), dev)

        meta = {
            name: padded_ints(plan[name])
            for name in ("request_indices", "qo_tile_indices", "kv_tile_indices")
        }
        meta["o_indptr"] = self.ints(plan["o_indptr"], dev)
        meta["kv_chunk_size"] = self.ints([plan["kv_chunk_size"]], dev)
        if split:
            meta["merge_indptr"] = self.ints(plan["merge_indptr"], dev)
        if plan["block_valid_mask"] is not None:
            meta["block_valid_mask"] = torch.tensor(
                plan["block_valid_mask"], dtype=torch.bool, device=dev
            )
        if plan["cuda_graph"]:
            meta["total_num_rows"] = torch.tensor(
                [plan["total_num_rows"]], dtype=torch.int32, device=dev
            )
        tensors = {
            "q": q,
            "k": inputs.k,
            "v": inputs.v,
            "q_indptr": inputs.qo_indptr,
            "kv_indptr": inputs.kv_indptr,
            "kv_indices": inputs.kv_indices,
            "last_page_len": inputs.last_page_len,
            "o": o,
            "lse": lse,
            "custom_mask": inputs.custom_mask,
            "mask_indptr": inputs.mask_indptr,
            "prefix_len": inputs.prefix_len,
            "token_pos": inputs.token_pos,
            "max_item_len": inputs.max_item_len,
            "sink": inputs.sink,
            **meta,
        }
        ptr = {name: t.data_ptr() for name, t in tensors.items() if t is not None}
        settings = {
            "batch": len(p["q_lens"]),
            "num_qo_heads": hq,
            "num_kv_heads": hkv,
            "page_size": p["page_size"],
            "head_dim": d,
            "kv_layout": p["kv_layout"],
            "q_strides": q.stride()[:2],
            "window_left": p["window_left"],
            "logits_soft_cap": p["logits_soft_cap"],
            "sm_scale": p["sm_scale"],
            "token_pos_in_items_len": p.get("token_pos_in_items_len", 0),
            "padded_batch_size": padded,
            "total_num_rows": plan["total_num_rows"],
            "sink": m.get("sink", False),
            "split": split,
        }
        layouts = load_index()["layouts"]
        suffix = "_sink" if m.get("sink") else ""
        k, v = inputs.k, inputs.v
        if m["kind"] == "paged":
            data = paged_params(
                layouts["paged" + suffix],
                ptr,
                {**settings, "k_strides": k.stride(), "v_strides": v.stride()},
            )
        else:
            nhd = p["kv_layout"] == NHD or p["kv_storage"] == "packed"
            k_strides = (
                (k.stride(0), k.stride(1)) if nhd else (k.stride(1), k.stride(0))
            )
            v_strides = (
                (v.stride(0), v.stride(1)) if nhd else (v.stride(1), v.stride(0))
            )
            data = ragged_params(
                layouts["ragged" + suffix],
                ptr,
                {**settings, "k_strides": k_strides, "v_strides": v_strides},
            )
        grid = (padded, 1, hkv)
        keep = tuple(meta.values())
        outputs = (o, lse) if lse is not None else (o,)

        def launch() -> tuple:
            if grid[0]:  # padded_batch_size == 0: upstream returns early
                self.launch(grid, m["block"], [data], shared_mem=m["shared_mem"])
            assert keep  # plan arrays live as long as the closure
            return outputs

        return launch, outputs


# -- decode ------------------------------------------------------------------------


class DecodeInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    last_page_len: torch.Tensor
    params: dict[str, Any]


def decode_route(meta: dict[str, Any], p: dict[str, Any], num_sms: int) -> dict:
    """``DecodePlan`` for case params ``p``: the occupancy-limited grid is
    ``meta["blocks_per_sm"]`` (build-time occupancy model of this kernel,
    checked against the driver at launch) times the SM count."""
    counts = [_ceil_div(n, p["page_size"]) for n in p["kv_lens"]]
    return decode_plan(
        counts,
        p["hkv"],
        p["page_size"],
        meta["blocks_per_sm"] * num_sms,
        cuda_graph=p["cuda_graph"],
    )


class Fa2Decode(_Fa2):
    """``BatchDecodeWithPagedKVCacheKernel`` for one GQA group size.

    Case parameters (besides ``COMMON_DEFAULTS``): ``hkv``, ``kv_lens``,
    ``page_size``, ``kv_layout``, ``window_left``, ``logits_soft_cap``,
    ``sm_scale``, ``return_lse`` and ``cuda_graph``. ``DecodePlan`` splits
    the KV of short batches (fewer work items than the occupancy-limited
    grid) and always under CUDA graphs; a split launch writes one partial
    state per (request, KV chunk) to ``tmp_v``/``tmp_s`` (merged by the
    separate ``fmha_fa2_merge_varlen`` kernel), which ``run`` returns."""

    defaults: ClassVar[dict[str, Any]] = {
        "page_size": 16,
        "kv_layout": NHD,
        "window_left": -1,
        "logits_soft_cap": 0.0,
        "return_lse": True,
        "cuda_graph": False,
    }

    @classmethod
    def params(cls, case_params: dict[str, Any]) -> dict[str, Any]:
        p = super().params(case_params)
        m = cls.meta
        p.setdefault("sm_scale", 1.0 / math.sqrt(m["head_dim"]))
        p["hq"] = p["hkv"] * m["group_size"]
        if p["window_left"] >= 0 and not m["swa"]:
            raise ValueError("window_left needs a sliding-window kernel")
        if (p["logits_soft_cap"] > 0) != m["softcap"]:
            raise ValueError("logits_soft_cap > 0 exactly for soft-cap kernels")
        return p

    def route(self, case_params: dict[str, Any]) -> dict[str, Any]:
        return decode_route(self.meta, self.params(case_params), self.num_sms())

    def get_cases(self) -> list[CaseSpec]:
        return decode_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        m, p = self.meta, self.params(case.params)
        g = self.generator(case)
        d, hq, hkv = m["head_dim"], p["hq"], p["hkv"]
        batch = len(p["kv_lens"])
        p["plan"] = decode_route(m, p, self.num_sms())
        if p["q_storage"] == "packed":
            q, *_ = self.packed_rows(
                batch,
                (hq * d, hkv * d, hkv * d),
                (m["dtype_q"],) * 3,
                ("q", "k", "v"),
                (hq, hkv, hkv),
                g,
                p,
            )
        else:
            q = self.qkv_values("q", (batch, hq, d), m["dtype_q"], g, p)
        cache = self.paged_cache(
            g, p["kv_lens"], p["page_size"], hkv, d, m["dtype_kv"], p
        )
        return DecodeInputs(
            q,
            cache["k"],
            cache["v"],
            cache["kv_indptr"],
            cache["kv_indices"],
            cache["last_page_len"],
            p,
        )

    def get_reference(self, inputs: tuple) -> tuple:
        inputs = DecodeInputs(*inputs)
        m, p = self.meta, inputs.params
        plan = p["plan"]
        split = plan["split_kv"]
        dev = inputs.q.device
        hq, d = p["hq"], m["head_dim"]
        rows = plan["new_batch_size"] if split else len(p["kv_lens"])
        o_all = torch.zeros((rows, hq, d), device=dev)
        lse_all = torch.full((rows, hq), float("-inf"), device=dev)
        chunk = plan["kv_chunk_size"]
        for b, lk in enumerate(p["kv_lens"]):
            kk, vv = self.gather_paged(
                inputs.k,
                inputs.v,
                inputs.kv_indptr,
                inputs.kv_indices,
                b,
                lk,
                p["kv_layout"],
            )
            key = torch.arange(lk, device=dev)
            keep = torch.ones(lk, dtype=torch.bool, device=dev)
            if p["window_left"] >= 0 and m["swa"]:
                keep &= key >= lk - 1 - p["window_left"]
            qb = inputs.q[b : b + 1].float()
            spans = (
                [
                    (c * chunk, min((c + 1) * chunk, lk))
                    for c in range(_ceil_div(max(lk, 1), chunk))
                ]
                if split
                else [(0, lk)]
            )
            base = plan["o_indptr"][b] if split else b
            if split and plan["o_indptr"][b + 1] - base != len(spans):
                raise AssertionError("plan and kernel disagree on the KV chunk count")
            for c, (lo, hi) in enumerate(spans):
                kc = (keep & (key >= lo) & (key < hi)).unsqueeze(0)
                o, lse = attend(qb, kk, vv, kc, p["sm_scale"], p["logits_soft_cap"])
                o_all[base + c], lse_all[base + c] = o[0], lse[0]
        o_all = o_all.to(DTYPES[m["dtype_o"]])
        if split or p["return_lse"]:
            return o_all, lse_all
        return (o_all,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != len(impl):
            raise AssertionError("output count differs")
        # Decode keeps P and its sum in FP32 (CUDA-core kernel); only the
        # output rounds. LSE: ex2/lg2/tanh.approx.f32 errors (measured 2e-5).
        rtol, atol = self._tolerance(self.meta["dtype_o"])
        self.assert_close(ref[:1], impl[:1], rtol=rtol, atol=atol)
        if len(ref) > 1:
            self.assert_close(ref[1:], impl[1:], rtol=1e-4, atol=1e-4)

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        inputs = DecodeInputs(*inputs)
        m, p = self.meta, inputs.params
        q = inputs.q
        dev = q.device
        if dev.type != "cuda":
            raise ValueError("native launch needs CUDA tensors")
        blocks = self.occupancy(m["block"], m["shared_mem"])
        if blocks != m["blocks_per_sm"]:
            raise AssertionError(
                f"driver occupancy {blocks} != the build's model {m['blocks_per_sm']}"
            )
        plan = p["plan"]
        hq, hkv, d = p["hq"], p["hkv"], m["head_dim"]
        split = plan["split_kv"]
        padded = plan["padded_batch_size"]
        rows = padded if split else q.shape[0]
        o = torch.empty((rows, hq, d), dtype=DTYPES[m["dtype_o"]], device=dev)
        lse = None
        if split or p["return_lse"]:
            lse = torch.empty((rows, hq), dtype=torch.float32, device=dev)

        def padded_ints(values: list[int]) -> torch.Tensor:
            return self.ints(values + [0] * (padded - len(values)), dev)

        meta = {
            "request_indices": padded_ints(plan["request_indices"]),
            "kv_tile_indices": padded_ints(plan["kv_tile_indices"]),
            "o_indptr": self.ints(plan["o_indptr"], dev),
            "kv_chunk_size": self.ints([plan["kv_chunk_size"]], dev),
        }
        if plan["block_valid_mask"] is not None:
            meta["block_valid_mask"] = torch.tensor(
                plan["block_valid_mask"], dtype=torch.bool, device=dev
            )
        tensors = {
            "q": q,
            "k": inputs.k,
            "v": inputs.v,
            "kv_indptr": inputs.kv_indptr,
            "kv_indices": inputs.kv_indices,
            "last_page_len": inputs.last_page_len,
            "o": o,
            "lse": lse,
            **meta,
        }
        data = decode_params(
            load_index()["layouts"]["decode"],
            {name: t.data_ptr() for name, t in tensors.items() if t is not None},
            {
                "batch": q.shape[0],
                "num_qo_heads": hq,
                "num_kv_heads": hkv,
                "page_size": p["page_size"],
                "head_dim": d,
                "kv_layout": p["kv_layout"],
                "k_strides": inputs.k.stride(),
                "v_strides": inputs.v.stride(),
                "q_strides": q.stride()[:2],
                "window_left": p["window_left"],
                "logits_soft_cap": p["logits_soft_cap"],
                "sm_scale": p["sm_scale"],
                "padded_batch_size": padded,
                "split": split,
            },
        )
        grid = (padded, hkv, 1)
        keep = tuple(meta.values())
        new_batch = plan["new_batch_size"]
        outputs: tuple[torch.Tensor, ...] = (o, lse) if lse is not None else (o,)
        if split:  # rows past new_batch_size belong to masked-off CTAs
            outputs = tuple(t[:new_batch] for t in outputs)

        def launch() -> tuple:
            self.launch(grid, m["block"], [data], shared_mem=m["shared_mem"])
            assert keep
            return outputs

        return launch, outputs


# -- persistent (holistic) batch attention -------------------------------------------


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


class _MinHeap:
    """flashinfer::MinHeap (heap.h): (index, float cost) pairs ordered by
    libstdc++'s ``std::push_heap``/``std::pop_heap`` with ``a.cost > b.cost``
    (ties resolve exactly as the C++ algorithms do)."""

    def __init__(self, capacity: int):
        self.heap: list[tuple[int, float]] = [(i, 0.0) for i in range(capacity)]

    @staticmethod
    def _less(a: tuple[int, float], b: tuple[int, float]) -> bool:
        return a[1] > b[1]

    def _push(self, hole: int, top: int, value: tuple[int, float]) -> None:
        h = self.heap
        parent = (hole - 1) // 2
        while hole > top and self._less(h[parent], value):
            h[hole] = h[parent]
            hole = parent
            parent = (hole - 1) // 2
        h[hole] = value

    def _adjust(self, hole: int, length: int, value: tuple[int, float]) -> None:
        h = self.heap
        top = second = hole
        while second < (length - 1) // 2:
            second = 2 * (second + 1)
            if self._less(h[second], h[second - 1]):
                second -= 1
            h[hole] = h[second]
            hole = second
        if length % 2 == 0 and second == (length - 2) // 2:
            second = 2 * (second + 1)
            h[hole] = h[second - 1]
            hole = second - 1
        self._push(hole, top, value)

    def insert(self, element: tuple[int, float]) -> None:
        self.heap.append(element)
        self._push(len(self.heap) - 1, 0, element)

    def pop(self) -> tuple[int, float]:
        h = self.heap
        if len(h) > 1:
            last = len(h) - 1
            value = h[last]
            h[last] = h[0]
            self._adjust(0, last, value)
        return h.pop()


def _packed_causal_kv_end(
    qo_len: int, kv_len: int, tile: int, num_tiles: int, cluster_tile_q: int, group: int
) -> int:
    if tile + 1 == num_tiles:
        return kv_len
    init = kv_len - qo_len
    return max(min(init + _ceil_div((tile + 1) * cluster_tile_q, group), kv_len), 0)


HOLISTIC_TASK_ARRAYS = (
    "q_indptr",
    "kv_indptr",
    "partial_indptr",
    "q_len",
    "kv_len",
    "q_start",
    "kv_start",
    "kv_end",
    "kv_head_idx",
    "work_indptr",
)
MAX_TOTAL_NUM_WORKS = 65536


def holistic_plan(
    qo_indptr: Sequence[int],
    kv_indptr: Sequence[int],
    kv_len: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    num_sms: int,
) -> dict[str, Any]:
    """TwoStageHolisticPlan (scheduler.cuh): per task (CTA_TILE_Q 128 and 16)
    the work arrays of every persistent cluster, the KV chunk lengths, the
    merge arrays of split rows and the workspace offsets (AlignedAllocator on
    16-byte aligned buffers)."""
    tiles = (128, 16)
    group = num_qo_heads // num_kv_heads
    num_clusters = num_sms * (1 if head_dim >= 256 else 2)
    requests: tuple[list[tuple[int, int, int]], ...] = ([], [])
    for i in range(len(kv_len)):
        qo_len = qo_indptr[i + 1] - qo_indptr[i]
        if qo_len < 0:
            raise ValueError("qo_indptr must be non-decreasing")
        task = 0 if qo_len * group > tiles[1] else 1
        requests[task].append((i, qo_len, kv_len[i]))
    max_num_kv_splits = 4 * num_clusters * (tiles[0] + tiles[1])
    total_kv_lens = 0
    for task in range(2):
        for _, qo_len, kvl in requests[task]:
            num_tiles = _ceil_div(qo_len * group, tiles[task])
            for t in reversed(range(num_tiles)):
                total_kv_lens += (
                    _packed_causal_kv_end(qo_len, kvl, t, num_tiles, tiles[task], group)
                    if causal
                    else kvl
                )
    heap = _MinHeap(num_clusters)
    partial_nnz = 0
    merge_indptr, merge_o_indices = [0], []
    len_kv_chunk = [0, 0]
    tasks = []
    for task in range(2):
        tile = tiles[task]
        limit = _ceil_div(total_kv_lens * num_kv_heads, num_clusters)
        limit = max(limit, 1)
        limit = 128 if limit <= 128 else _ceil_div(limit, 256) * 256
        if tile >= 64:
            limit //= min(num_kv_heads, 2)
        len_kv_chunk[task] = limit
        works: list[list[tuple[int, ...]]] = [[] for _ in range(num_clusters)]
        for i, qo_len, kvl in requests[task]:
            packed = qo_len * group
            num_tiles = _ceil_div(packed, tile)
            for t in range(num_tiles):
                remaining = (
                    _packed_causal_kv_end(qo_len, kvl, t, num_tiles, tile, group)
                    if causal
                    else kvl
                )
                kv_start = 0
                split = remaining > limit
                num_kv_tiles = _ceil_div(remaining, limit) if split else 1
                rows = min(tile, packed - t * tile)
                zero = remaining == 0
                while remaining > 0 or zero:
                    actual = min(remaining, limit)
                    for head in range(num_kv_heads):
                        cluster, cost = heap.pop()
                        heap.insert((cluster, _f32(cost + _f32(2.0 * tile + actual))))
                        works[cluster].append(
                            (
                                qo_indptr[i],
                                kv_indptr[i],
                                partial_nnz,
                                qo_len,
                                kvl,
                                t * tile,
                                kv_start,
                                kv_start + actual,
                                head,
                            )
                        )
                    remaining -= actual
                    zero = remaining == 0
                    kv_start += actual
                    if zero:
                        break
                if split:
                    for row in range(rows):
                        merge_indptr.append(merge_indptr[-1] + num_kv_tiles)
                        q, r = divmod(t * tile + row, group)
                        merge_o_indices.append(
                            (qo_indptr[i] + q) * num_kv_heads * group + r
                        )
                    partial_nnz += rows * num_kv_tiles
        work_indptr = [0]
        for cluster_works in works:
            work_indptr.append(work_indptr[-1] + len(cluster_works))
        if work_indptr[-1] > MAX_TOTAL_NUM_WORKS:
            raise ValueError(f"{work_indptr[-1]} works exceed {MAX_TOTAL_NUM_WORKS}")
        flat = [w for cluster_works in works for w in cluster_works]
        arrays = {
            name: [w[j] for w in flat]
            for j, name in enumerate(HOLISTIC_TASK_ARRAYS[:-1])
        }
        arrays["work_indptr"] = work_indptr
        tasks.append(arrays)
    if len(merge_indptr) > max_num_kv_splits:
        raise ValueError("too many KV splits for the plan's buffers")
    # Workspace offsets: every per-task array holds MAX_TOTAL_NUM_WORKS ids.
    offset = 0
    task_offsets = []
    for _ in range(2):
        offsets = {}
        for name in HOLISTIC_TASK_ARRAYS:
            offsets[name] = offset
            offset += 4 * MAX_TOTAL_NUM_WORKS
        task_offsets.append(offsets)
    shared = {"len_kv_chunk": offset}
    offset = _align(offset + 4 * 2, 16)
    shared["merge_indptr"] = offset
    offset += 4 * max_num_kv_splits
    shared["merge_o_indices"] = offset
    offset += 4 * max_num_kv_splits
    shared["num_qo_len"] = offset
    offset += 4
    partial_o_bytes = max_num_kv_splits * 2 * head_dim * num_kv_heads
    return {
        "num_blks": (1, num_clusters),
        "tasks": tasks,
        "len_kv_chunk": len_kv_chunk,
        "merge_indptr": merge_indptr,
        "merge_o_indices": merge_o_indices,
        "num_packed_qo_len": len(merge_indptr) - 1,
        "task_offsets": task_offsets,
        "shared_offsets": shared,
        "int_workspace": offset,
        "partial_o_offset": 0,
        "partial_lse_offset": _align(partial_o_bytes, 16),
        "float_workspace": _align(partial_o_bytes, 16)
        + 4 * max_num_kv_splits * num_kv_heads,
    }


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def persistent_params(
    layout: dict[str, Any],
    plan: dict[str, Any],
    task: int,
    ptr: dict[str, int],
    s: dict[str, Any],
) -> bytes:
    """One of the two PersistentParams BatchPagedAttentionRun builds."""
    ints, floats = ptr["int_workspace"], ptr["float_workspace"]
    t = plan["task_offsets"][task]
    shared = plan["shared_offsets"]
    values = {
        "q": ptr["q"],
        "k": ptr["k"],
        "v": ptr["v"],
        "o": 0,
        "partial_o": floats + plan["partial_o_offset"],
        "partial_lse": floats + plan["partial_lse_offset"],
        "final_o": ptr["o"],
        "final_lse": ptr.get("lse", 0),
        "q_indptr": ints + t["q_indptr"],
        "kv_indptr": ints + t["kv_indptr"],
        "partial_indptr": ints + t["partial_indptr"],
        "kv_indices": ptr["kv_indices"],
        "q_len": ints + t["q_len"],
        "kv_len": ints + t["kv_len"],
        "q_start": ints + t["q_start"],
        "kv_start": ints + t["kv_start"],
        "kv_end": ints + t["kv_end"],
        "kv_head_idx_arr": ints + t["kv_head_idx"],
        "work_indptr": ints + t["work_indptr"],
        "len_kv_chunk": ints + shared["len_kv_chunk"] + 4 * task,
        "merge_indptr": ints + shared["merge_indptr"],
        "merge_o_indices": ints + shared["merge_o_indices"],
        "num_packed_qo_len": ints + shared["num_qo_len"],
        "num_kv_heads": s["num_kv_heads"],
        "gqa_group_size": s["num_qo_heads"] // s["num_kv_heads"],
        "page_size": s["page_size"],
        "q_stride_n": s["q_strides"][0],
        "q_stride_h": s["q_strides"][1],
        "k_stride_page": s["k_strides"][0],
        "k_stride_h": s["k_strides"][1],
        "k_stride_n": s["k_strides"][2],
        "v_stride_page": s["v_strides"][0],
        "v_stride_h": s["v_strides"][1],
        "v_stride_n": s["v_strides"][2],
        **{f"{side}_sf_stride_{d}": 0 for side in "kv" for d in ("page", "h", "n")},
        "sm_scale": _f32(s["sm_scale"]),
        "logits_soft_cap": _f32(s["logits_soft_cap"]),
        "v_scale": _f32(s.get("v_scale", 1.0)),
        "maybe_k_cache_sf": 0,
        "maybe_v_cache_sf": 0,
    }
    return pack(layout, values)


def _paged_strides(kv_layout: int, cache: torch.Tensor) -> tuple[int, int, int]:
    """(page, head, entry) element strides of a K or V cache view, as
    BatchPagedAttentionRun reads them from the tensor."""
    if kv_layout == NHD:  # [pages, P, H, D]
        return cache.stride(0), cache.stride(2), cache.stride(1)
    return cache.stride(0), cache.stride(1), cache.stride(2)


class PersistentInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len: torch.Tensor
    params: dict[str, Any]


class Fa2Persistent(_Fa2):
    """``PersistentKernelTemplate`` of ``flashinfer.BatchAttention``: both
    attention runners (CTA_TILE_Q 128 and 16) and the state reduction in one
    cooperative launch, planned by ``TwoStageHolisticPlan``.

    Case parameters (besides ``COMMON_DEFAULTS``): ``hq``/``hkv``,
    ``q_lens``/``kv_lens``, ``page_size``, ``kv_layout``,
    ``logits_soft_cap``, ``sm_scale``, ``v_scale`` (``run(v_scale=)``, applied
    to the normalised output) and ``q_storage`` ``"chunked"`` (q is the first
    half of a [N, H, 2D] tensor, upstream's non-contiguous-q test)."""

    defaults: ClassVar[dict[str, Any]] = {
        "page_size": 16,
        "kv_layout": NHD,
        "logits_soft_cap": 0.0,
        "v_scale": 1.0,
    }

    @classmethod
    def params(cls, case_params: dict[str, Any]) -> dict[str, Any]:
        p = super().params(case_params)
        m = cls.meta
        p.setdefault("sm_scale", 1.0 / math.sqrt(m["head_dim"]))
        p["causal"] = m["mask"] == MASK_IDS["causal"]
        if (p["logits_soft_cap"] > 0) != m["softcap"]:
            raise ValueError("logits_soft_cap > 0 exactly for soft-cap kernels")
        return p

    def route(self, case_params: dict[str, Any]) -> dict[str, Any]:
        p = self.params(case_params)
        page = p["page_size"]
        counts = [_ceil_div(n, page) for n in p["kv_lens"]]
        return holistic_plan(
            _cumsum(p["q_lens"]),
            _cumsum(counts),
            p["kv_lens"],
            p["hq"],
            p["hkv"],
            self.meta["head_dim"],
            p["causal"],
            self.num_sms(),
        )

    def get_cases(self) -> list[CaseSpec]:
        return persistent_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        m, p = self.meta, self.params(case.params)
        g = self.generator(case)
        dev = self.device
        d, hq, hkv = m["head_dim"], p["hq"], p["hkv"]
        rows = sum(p["q_lens"])
        if p["q_storage"] == "chunked":
            q = self.qkv_values("q", (rows, hq, 2 * d), m["dtype_q"], g, p)[..., :d]
        else:
            q = self.qkv_values("q", (rows, hq, d), m["dtype_q"], g, p)
        cache = self.paged_cache(
            g, p["kv_lens"], p["page_size"], hkv, d, m["dtype_kv"], p
        )
        return PersistentInputs(
            q,
            cache["k"],
            cache["v"],
            self.ints(_cumsum(p["q_lens"]), dev),
            cache["kv_indptr"],
            cache["kv_indices"],
            self.ints(p["kv_lens"], dev),
            p,
        )

    def get_reference(self, inputs: tuple) -> tuple:
        inputs = PersistentInputs(*inputs)
        p = inputs.params
        dev = inputs.q.device
        outs, lses = [], []
        bounds = _cumsum(p["q_lens"])
        for b, lk in enumerate(p["kv_lens"]):
            kk, vv = self.gather_paged(
                inputs.k,
                inputs.v,
                inputs.kv_indptr,
                inputs.kv_indices,
                b,
                lk,
                p["kv_layout"],
            )
            lq = bounds[b + 1] - bounds[b]
            keep = torch.ones((lq, lk), dtype=torch.bool, device=dev)
            if p["causal"]:
                pos = torch.arange(lq, device=dev).unsqueeze(1) + (lk - lq)
                keep &= torch.arange(lk, device=dev).unsqueeze(0) <= pos
            o, lse = attend(
                inputs.q[bounds[b] : bounds[b + 1]].float(),
                kk,
                vv,
                keep,
                p["sm_scale"],
                p["logits_soft_cap"],
                v_scale=p["v_scale"],
            )
            outs.append(o)
            lses.append(lse)
        lse = _keys(torch.cat(lses), max(p["kv_lens"], default=0))
        return torch.cat(outs).to(DTYPES[self.meta["dtype_o"]]), lse

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != 2 or len(impl) != 2:
            raise AssertionError("expected (o, lse)")
        # As FA2 prefill (16-bit probabilities into the PV MMA and the row sum),
        # plus one more 16-bit rounding of split rows' partial outputs before
        # the in-kernel merge.
        rtol, atol = self._tolerance(self.meta["dtype_o"])
        self.assert_close(ref[:1], impl[:1], rtol=2 * rtol, atol=2 * atol)
        atol = lse_tolerance(self.meta["dtype_q"], getattr(ref[1], "fa2_keys", 0))
        self.assert_close(ref[1:], impl[1:], rtol=0.0, atol=atol)

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        inputs = PersistentInputs(*inputs)
        m, p = self.meta, inputs.params
        q = inputs.q
        dev = q.device
        if dev.type != "cuda":
            raise ValueError("native launch needs CUDA tensors")
        hq, hkv, d = p["hq"], p["hkv"], m["head_dim"]
        sms = self.num_sms()
        plan = holistic_plan(
            inputs.qo_indptr.tolist(),
            inputs.kv_indptr.tolist(),
            inputs.kv_len.tolist(),
            hq,
            hkv,
            d,
            p["causal"],
            sms,
        )
        grid = plan["num_blks"]
        # cudaLaunchCooperativeKernel needs the whole grid resident.
        resident = self.occupancy(m["block"], m["shared_mem"]) * sms
        if resident < grid[0] * grid[1]:
            raise ValueError(
                f"cooperative grid {grid} exceeds {resident} resident CTAs"
            )
        words = torch.zeros(plan["int_workspace"] // 4 + 1, dtype=torch.int32)
        for task, arrays in enumerate(plan["tasks"]):
            for name, values in arrays.items():
                start = plan["task_offsets"][task][name] // 4
                words[start : start + len(values)] = torch.tensor(
                    values, dtype=torch.int32
                )
        shared = plan["shared_offsets"]
        for name, values in (
            ("len_kv_chunk", plan["len_kv_chunk"]),
            ("merge_indptr", plan["merge_indptr"]),
            ("merge_o_indices", plan["merge_o_indices"]),
            ("num_qo_len", [plan["num_packed_qo_len"]]),
        ):
            start = shared[name] // 4
            words[start : start + len(values)] = torch.tensor(values, dtype=torch.int32)
        int_workspace = words.to(dev)
        float_workspace = torch.empty(
            plan["float_workspace"], dtype=torch.uint8, device=dev
        )
        o = torch.empty((q.shape[0], hq, d), dtype=DTYPES[m["dtype_o"]], device=dev)
        lse = torch.empty((q.shape[0], hq), dtype=torch.float32, device=dev)
        k, v = inputs.k, inputs.v
        ptr = {
            "q": q.data_ptr(),
            "k": k.data_ptr(),
            "v": v.data_ptr(),
            "kv_indices": inputs.kv_indices.data_ptr(),
            "o": o.data_ptr(),
            "lse": lse.data_ptr(),
            "int_workspace": int_workspace.data_ptr(),
            "float_workspace": float_workspace.data_ptr(),
        }
        settings = {
            "num_qo_heads": hq,
            "num_kv_heads": hkv,
            "page_size": p["page_size"],
            "q_strides": q.stride()[:2],
            "k_strides": _paged_strides(p["kv_layout"], k),
            "v_strides": _paged_strides(p["kv_layout"], v),
            "sm_scale": p["sm_scale"],
            "logits_soft_cap": p["logits_soft_cap"],
            "v_scale": p["v_scale"],
        }
        layout = load_index()["layouts"]["persistent"]
        args = [persistent_params(layout, plan, task, ptr, settings) for task in (0, 1)]
        keep = (int_workspace, float_workspace)
        outputs = (o, lse)

        def launch() -> tuple:
            self.launch(
                grid, m["block"], args, shared_mem=m["shared_mem"], cooperative=True
            )
            assert keep
            return outputs

        return launch, outputs


# -- MLA (BatchMLAPagedAttentionWrapper, fa2 backend) ----------------------------------

MLA_WORK_ARRAYS = (
    "q_indptr",
    "kv_indptr",
    "partial_indptr",
    "q_len",
    "kv_len",
    "q_start",
    "kv_start",
    "kv_end",
)
MLA_MERGE_ARRAYS = (
    "merge_packed_offset_start",
    "merge_packed_offset_end",
    "merge_partial_packed_offset_start",
    "merge_partial_packed_offset_end",
    "merge_partial_stride",
)
MLA_MAX_TOTAL_NUM_WORKS = 16384


def mla_plan(
    qo_indptr: Sequence[int],
    kv_indptr: Sequence[int],
    kv_len: Sequence[int],
    num_heads: int,
    head_dim_o: int,
    causal: bool,
    num_sms: int,
) -> dict[str, Any]:
    """MLAPlan (scheduler.cuh): cluster work arrays, the merge CTAs' arrays and
    the workspace offsets (AlignedAllocator on 16-byte aligned buffers)."""
    batch = len(kv_len)
    requests = []
    packed_total = 0
    for i in range(batch):
        qo_len = qo_indptr[i + 1] - qo_indptr[i]
        if qo_len < 0:
            raise ValueError("qo_indptr must be non-decreasing")
        packed_total += qo_len * num_heads
        requests.append((i, qo_len, kv_len[i]))
    cluster_size = 2 if packed_total // batch > 64 else 1
    num_clusters = num_sms // cluster_size
    cluster_tile_q = cluster_size * 64
    total_kv_lens = 0
    for _, qo_len, kvl in requests:
        tiles = _ceil_div(qo_len * num_heads, cluster_tile_q)
        for t in reversed(range(tiles)):
            total_kv_lens += (
                _packed_causal_kv_end(qo_len, kvl, t, tiles, cluster_tile_q, num_heads)
                if causal
                else kvl
            )
    x = max(_ceil_div(total_kv_lens, num_clusters), 1)
    if x <= 8:
        limit = 32
    elif x <= 16:
        limit = 64
    elif x <= 32:
        limit = 128
    elif x <= 64:
        limit = 192
    else:
        limit = _ceil_div(x, 256) * 256
    heap = _MinHeap(num_clusters)
    works: list[list[tuple[int, ...]]] = [[] for _ in range(num_clusters)]
    merge = {name: [0] * num_sms for name in MLA_MERGE_ARRAYS}
    merge_ctas = 0
    partial_nnz = 0
    for i, qo_len, kvl in requests:
        packed = qo_len * num_heads
        tiles = _ceil_div(packed, cluster_tile_q)
        for t in reversed(range(tiles)):
            remaining = (
                _packed_causal_kv_end(qo_len, kvl, t, tiles, cluster_tile_q, num_heads)
                if causal
                else kvl
            )
            kv_start = 0
            split = remaining > limit
            rows = min(cluster_tile_q, packed - t * cluster_tile_q)
            if split:
                chunks = max(remaining * cluster_size // limit, 1)
                chunk_rows = _ceil_div(rows, chunks)
                tile_end = min(cluster_tile_q, packed - t * cluster_tile_q)
                base = qo_indptr[i] * num_heads + t * cluster_tile_q
                for offset in range(0, rows, chunk_rows):
                    merge["merge_packed_offset_start"][merge_ctas] = base + offset
                    merge["merge_packed_offset_end"][merge_ctas] = base + min(
                        offset + chunk_rows, tile_end
                    )
                    merge["merge_partial_packed_offset_start"][merge_ctas] = (
                        partial_nnz + offset
                    )
                    merge["merge_partial_packed_offset_end"][merge_ctas] = (
                        partial_nnz + _ceil_div(remaining, limit) * rows
                    )
                    merge["merge_partial_stride"][merge_ctas] = rows
                    merge_ctas += 1
            zero = remaining == 0
            while remaining > 0 or zero:
                cluster, cost = heap.pop()
                actual = min(remaining, limit)
                heap.insert((cluster, _f32(cost + _f32(2.0 * cluster_tile_q + actual))))
                if split:
                    partial = partial_nnz
                    partial_nnz += rows
                else:
                    partial = -1
                works[cluster].append(
                    (
                        qo_indptr[i],
                        kv_indptr[i],
                        partial,
                        qo_len,
                        kvl,
                        t * cluster_tile_q,
                        kv_start,
                        kv_start + actual,
                    )
                )
                remaining -= actual
                kv_start += actual
                if zero:
                    break
    if merge_ctas > num_sms:
        raise ValueError("merge CTAs exceed the SM count")
    work_indptr = [0]
    for cluster_works in works:
        work_indptr.append(work_indptr[-1] + len(cluster_works))
    if work_indptr[-1] > MLA_MAX_TOTAL_NUM_WORKS:
        raise ValueError("too many works for the plan's buffers")
    flat = [w for cluster_works in works for w in cluster_works]
    arrays: dict[str, list[int]] = {
        name: [w[j] for w in flat] for j, name in enumerate(MLA_WORK_ARRAYS)
    }
    arrays.update(merge)
    arrays["work_indptr"] = work_indptr
    offsets, offset = {}, 0
    for name in (
        "q_indptr",
        "kv_indptr",
        "partial_indptr",
        *MLA_MERGE_ARRAYS,
        "q_len",
        "kv_len",
        "q_start",
        "kv_start",
        "kv_end",
        "work_indptr",
    ):
        count = num_sms if name in MLA_MERGE_ARRAYS else MLA_MAX_TOTAL_NUM_WORKS
        offset = _align(offset, 16)
        offsets[name] = offset
        offset += 4 * count
    partial_o = 2 * num_clusters * cluster_tile_q * 2 * head_dim_o
    return {
        "num_blks": (cluster_size, num_clusters),
        "arrays": arrays,
        "offsets": offsets,
        "int_workspace": offset,
        "partial_o_offset": 0,
        "partial_lse_offset": _align(partial_o, 16),
        "float_workspace": _align(partial_o, 16)
        + 2 * num_clusters * cluster_tile_q * 4,
    }


def mla_params(
    layout: dict[str, Any], plan: dict[str, Any], ptr: dict[str, int], s: dict[str, Any]
) -> bytes:
    """BatchMLAPagedAttentionRun's MLAParams (no FP8 scales, base-2 LSE)."""
    ints, floats = ptr["int_workspace"], ptr["float_workspace"]
    off = plan["offsets"]
    values = {
        "q_nope": ptr["q_nope"],
        "q_pe": ptr["q_pe"],
        "ckv": ptr["ckv"],
        "kpe": ptr["kpe"],
        "partial_o": floats + plan["partial_o_offset"],
        "partial_lse": floats + plan["partial_lse_offset"],
        "final_o": ptr["o"],
        "final_lse": ptr.get("lse", 0),
        **{name: ints + off[name] for name in (*MLA_WORK_ARRAYS, *MLA_MERGE_ARRAYS)},
        "kv_indices": ptr["kv_indices"],
        "work_indptr": ints + off["work_indptr"],
        "block_size": s["page_size"],
        "num_heads": s["num_heads"],
        **{name: s["strides"][name] for name in MLA_STRIDES},
        "sm_scale": _f32(s["sm_scale"]),
        "ckv_scale": 1.0,
        "kpe_scale": 1.0,
        "ckv_scale_arr": 0,
        "return_lse_base_on_e": False,
    }
    return pack(layout, values)


MLA_STRIDES = (
    "q_nope_stride_n",
    "q_nope_stride_h",
    "q_pe_stride_n",
    "q_pe_stride_h",
    "ckv_stride_page",
    "ckv_stride_n",
    "kpe_stride_page",
    "kpe_stride_n",
    "o_stride_n",
    "o_stride_h",
)


class MlaInputs(NamedTuple):
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    ckv: torch.Tensor
    kpe: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len: torch.Tensor
    params: dict[str, Any]


class Fa2Mla(_Fa2):
    """``BatchMLAPagedAttentionKernel`` (DeepSeek MLA, compressed KV 512 +
    RoPE 64, one shared latent KV head): attention and the split-KV merge in
    one cooperative launch, planned by ``MLAPlan``.

    Case parameters (besides ``COMMON_DEFAULTS``; ``kv_storage`` and
    ``kv_layout`` do not apply): ``heads``, ``q_lens``/``kv_lens``,
    ``page_size``, ``sm_scale`` (default DeepSeek's 1/sqrt(128 + 64));
    ``poison`` fills the uncovered slots of the latent caches with NaN, as
    upstream's out-of-bounds test does."""

    defaults: ClassVar[dict[str, Any]] = {
        "page_size": 64,
        "sm_scale": 1.0 / math.sqrt(128 + 64),
        "kv_layout": NHD,
    }

    @classmethod
    def params(cls, case_params: dict[str, Any]) -> dict[str, Any]:
        p = super().params(case_params)
        p["causal"] = cls.meta["causal"]
        return p

    def route(self, case_params: dict[str, Any]) -> dict[str, Any]:
        p = self.params(case_params)
        counts = [_ceil_div(n, p["page_size"]) for n in p["kv_lens"]]
        return mla_plan(
            _cumsum(p["q_lens"]),
            _cumsum(counts),
            p["kv_lens"],
            p["heads"],
            512,
            p["causal"],
            self.num_sms(),
        )

    def get_cases(self) -> list[CaseSpec]:
        return mla_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        m, p = self.meta, self.params(case.params)
        g = self.generator(case)
        dev = self.device
        heads, page = p["heads"], p["page_size"]
        n = sum(p["q_lens"])
        # Peaked logits over the 576-wide MQA product (see ``qkv_values``).
        scale = {**p, "logit_std": p["logit_std"] / math.sqrt(576 * p["sm_scale"] ** 2)}
        q_nope = self.qkv_values("q", (n, heads, 512), m["dtype_q"], g, scale)
        q_pe = self.qkv_values("q", (n, heads, 64), m["dtype_q"], g, scale)
        # One latent head: ckv/kpe are the K (and ckv the V) of a paged cache.
        both = self.paged_cache(
            g, p["kv_lens"], page, 1, 576, m["dtype_kv"], {**p, "kv_storage": "k_only"}
        )
        latent = both["k"].reshape(-1, page, 576)
        ckv = latent[..., :512].contiguous()
        kpe = latent[..., 512:].contiguous()
        return MlaInputs(
            q_nope,
            q_pe,
            ckv,
            kpe,
            self.ints(_cumsum(p["q_lens"]), dev),
            both["kv_indptr"],
            both["kv_indices"],
            self.ints(p["kv_lens"], dev),
            p,
        )

    def get_reference(self, inputs: tuple) -> tuple:
        inputs = MlaInputs(*inputs)
        p = inputs.params
        bounds = _cumsum(p["q_lens"])
        outs, lses = [], []
        dev = inputs.q_nope.device
        for b, lk in enumerate(p["kv_lens"]):
            pages = inputs.kv_indices[
                int(inputs.kv_indptr[b]) : int(inputs.kv_indptr[b + 1])
            ].long()
            c = inputs.ckv[pages].float().reshape(-1, 512)[:lk]
            r = inputs.kpe[pages].float().reshape(-1, 64)[:lk]
            lo, hi = bounds[b], bounds[b + 1]
            # MLA as MQA over the concatenated [ckv, kpe] key and the ckv value.
            q = torch.cat(
                [inputs.q_nope[lo:hi].float(), inputs.q_pe[lo:hi].float()], dim=-1
            )
            k = torch.cat([c, r], dim=-1).unsqueeze(1)
            keep = torch.ones((hi - lo, lk), dtype=torch.bool, device=dev)
            if p["causal"]:
                pos = torch.arange(hi - lo, device=dev).unsqueeze(1) + (lk - (hi - lo))
                keep &= torch.arange(lk, device=dev).unsqueeze(0) <= pos
            o, lse = attend(q, k, c.unsqueeze(1), keep, p["sm_scale"], 0.0)
            outs.append(o)
            lses.append(lse)
        lse = _keys(torch.cat(lses), max(p["kv_lens"], default=0))
        return torch.cat(outs).to(DTYPES[self.meta["dtype_o"]]), lse

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != 2 or len(impl) != 2:
            raise AssertionError("expected (o, lse)")
        # 16-bit probabilities into the PV MMA and the row sums (as FA2
        # prefill), plus the 16-bit partial outputs of split rows.
        rtol, atol = self._tolerance(self.meta["dtype_o"])
        self.assert_close(ref[:1], impl[:1], rtol=2 * rtol, atol=2 * atol)
        atol = lse_tolerance(self.meta["dtype_q"], getattr(ref[1], "fa2_keys", 0))
        self.assert_close(ref[1:], impl[1:], rtol=0.0, atol=atol)

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        inputs = MlaInputs(*inputs)
        m, p = self.meta, inputs.params
        q_nope, q_pe, ckv, kpe = inputs.q_nope, inputs.q_pe, inputs.ckv, inputs.kpe
        dev = q_nope.device
        if dev.type != "cuda":
            raise ValueError("native launch needs CUDA tensors")
        sms = self.num_sms()
        plan = mla_plan(
            inputs.qo_indptr.tolist(),
            inputs.kv_indptr.tolist(),
            inputs.kv_len.tolist(),
            p["heads"],
            512,
            p["causal"],
            sms,
        )
        grid = plan["num_blks"]
        resident = self.occupancy(m["block"], m["shared_mem"]) * sms
        if resident < grid[0] * grid[1]:
            raise ValueError(
                f"cooperative grid {grid} exceeds {resident} resident CTAs"
            )
        words = torch.zeros(plan["int_workspace"] // 4 + 1, dtype=torch.int32)
        for name, values in plan["arrays"].items():
            start = plan["offsets"][name] // 4
            words[start : start + len(values)] = torch.tensor(values, dtype=torch.int32)
        int_workspace = words.to(dev)
        float_workspace = torch.empty(
            plan["float_workspace"], dtype=torch.uint8, device=dev
        )
        n, heads = q_nope.shape[0], p["heads"]
        o = torch.empty((n, heads, 512), dtype=DTYPES[m["dtype_o"]], device=dev)
        lse = torch.empty((n, heads), dtype=torch.float32, device=dev)
        ptr = {
            "q_nope": q_nope.data_ptr(),
            "q_pe": q_pe.data_ptr(),
            "ckv": ckv.data_ptr(),
            "kpe": kpe.data_ptr(),
            "kv_indices": inputs.kv_indices.data_ptr(),
            "o": o.data_ptr(),
            "lse": lse.data_ptr(),
            "int_workspace": int_workspace.data_ptr(),
            "float_workspace": float_workspace.data_ptr(),
        }
        strides = {
            "q_nope_stride_n": q_nope.stride(0),
            "q_nope_stride_h": q_nope.stride(1),
            "q_pe_stride_n": q_pe.stride(0),
            "q_pe_stride_h": q_pe.stride(1),
            "ckv_stride_page": ckv.stride(0),
            "ckv_stride_n": ckv.stride(1),
            "kpe_stride_page": kpe.stride(0),
            "kpe_stride_n": kpe.stride(1),
            "o_stride_n": o.stride(0),
            "o_stride_h": o.stride(1),
        }
        data = mla_params(
            load_index()["layouts"]["mla"],
            plan,
            ptr,
            {
                "page_size": p["page_size"],
                "num_heads": heads,
                "strides": strides,
                "sm_scale": p["sm_scale"],
            },
        )
        keep = (int_workspace, float_workspace)
        outputs = (o, lse)

        def launch() -> tuple:
            self.launch(
                grid, m["block"], [data], shared_mem=m["shared_mem"], cooperative=True
            )
            assert keep
            return outputs

        return launch, outputs


# -- merges ------------------------------------------------------------------------


def _merge_tolerance(dtype: str) -> tuple[float, float]:
    """Merged values are convex combinations of 16-bit inputs computed in
    FP32 and rounded once (ex2.approx weights add ~2^-22 relative error)."""
    return (1.6e-2, 1.6e-2) if dtype == "bf16" else (2e-3, 2e-3)


class _Merge(_Fa2):
    def _states(self, shape: Sequence[int], g: torch.Generator) -> tuple:
        """States as split attention produces them: O(1) values and LSEs
        spread over +-6 (base 2), so merge weights range over 2^12."""
        v = self.values(shape, self.meta["dtype_in"], g)
        s = torch.rand(tuple(shape[:-1]), generator=g, device=self.device) * 12 - 6
        return v, s

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != len(impl):
            raise AssertionError("output count differs")
        rtol, atol = _merge_tolerance(self.meta["dtype_o"])
        self.assert_close(ref[:1], impl[:1], rtol=rtol, atol=atol)
        # log2/ex2.approx: absolute error ~1e-6 on the merged LSE.
        self.assert_close(ref[1:], impl[1:], rtol=1e-5, atol=1e-4)


class Fa2MergeVarlen(_Merge):
    """``PersistentVariableLengthMergeStatesKernel`` (VariableLengthMergeStates):
    merges the partial states of a split-KV prefill/decode. Case parameters:
    ``sets`` (partial states per row, 0 allowed), ``heads``, ``d``;
    ``return_lse`` (False: ``s_merged`` is null, as after a split prefill
    run without LSE) and ``seq_len`` (a device row count < len(sets), the
    CUDA-graph ``total_num_rows`` pointer: rows past it stay unwritten)."""

    def get_cases(self) -> list[CaseSpec]:
        return merge_varlen_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        p = {"return_lse": True, "seq_len": None, **case.params}
        g = self.generator(case)
        v, s = self._states((sum(p["sets"]), p["heads"], p["d"]), g)
        indptr = self.ints(_cumsum(p["sets"]), self.device)
        return v, s, indptr, p

    def get_reference(self, inputs: tuple) -> tuple:
        v, s, indptr, p = inputs
        bounds = indptr.tolist()
        rows = len(bounds) - 1 if p["seq_len"] is None else p["seq_len"]
        outs, lses = [], []
        for lo, hi in itertools.pairwise(bounds[: rows + 1]):
            o, lse = merge(v[lo:hi].float(), s[lo:hi])
            outs.append(o)
            lses.append(lse)
        o = torch.stack(outs).to(DTYPES[self.meta["dtype_o"]])
        return (o, torch.stack(lses)) if p["return_lse"] else (o,)

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        m = self.meta
        v, s, indptr, p = inputs
        seq, heads = indptr.numel() - 1, v.shape[1]
        out = torch.empty(
            (seq, heads, m["head_dim"]), dtype=DTYPES[m["dtype_o"]], device=v.device
        )
        lse = None
        if p["return_lse"]:
            lse = torch.empty((seq, heads), dtype=torch.float32, device=v.device)
        seq_len = None
        if p["seq_len"] is not None:
            seq_len = torch.tensor([p["seq_len"]], dtype=torch.int32, device=v.device)
        sms = self.num_sms()
        blocks = min(
            self.occupancy(m["block"], m["shared_mem"]), _ceil_div(seq * heads, sms)
        )
        args = [
            v,
            s,
            indptr,
            out,
            lse if lse is not None else ctypes_ptr(0),
            ctypes_u32(seq),
            seq_len if seq_len is not None else ctypes_ptr(0),
            ctypes_u32(heads),
        ]
        rows = seq if p["seq_len"] is None else p["seq_len"]
        outputs = (out[:rows], lse[:rows]) if lse is not None else (out[:rows],)

        def launch() -> tuple:
            self.launch(sms * blocks, m["block"], args, shared_mem=m["shared_mem"])
            assert seq_len is None or seq_len.numel()
            return outputs

        return launch, outputs


def ctypes_u32(value: int) -> Any:
    import ctypes

    return ctypes.c_uint32(value)


def ctypes_ptr(value: int) -> Any:
    import ctypes

    return ctypes.c_void_p(value)


class Fa2MergeState(_Merge):
    """``MergeStateKernel`` (``merge_state``): two states per position."""

    def get_cases(self) -> list[CaseSpec]:
        return merge_state_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        p = case.params
        g = self.generator(case)
        shape = (p["seq"], p["heads"], p["d"])
        return (*self._states(shape, g), *self._states(shape, g))

    def get_reference(self, inputs: tuple) -> tuple:
        v_a, s_a, v_b, s_b = inputs
        o, lse = merge(torch.stack([v_a.float(), v_b.float()]), torch.stack([s_a, s_b]))
        return o.to(DTYPES[self.meta["dtype_o"]]), lse

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        v_a, s_a, v_b, s_b = inputs
        seq, heads, d = v_a.shape
        out = torch.empty_like(v_a, dtype=DTYPES[self.meta["dtype_o"]])
        lse = torch.empty_like(s_a)
        args = [v_a, s_a, v_b, s_b, out, lse, ctypes_u32(heads), ctypes_u32(d)]
        block = (d // self.meta["vec_size"], heads)
        outputs = (out, lse)

        def launch() -> tuple:
            self.launch(seq, block, args)
            return outputs

        return launch, outputs


class Fa2MergeStateInPlace(_Merge):
    """``MergeStateInPlaceKernel`` (``merge_state_in_place``): ``run`` merges
    into copies of ``v``/``s`` (inputs are not mutated). ``mask`` is
    ``"none"`` (null mask pointer), ``"ones"``, ``"zeros"`` or ``"random"``.

    The kernel is not idempotent: every launch merges the other state into
    the output again, so the timed callable of ``prepare`` (re-launched by
    the benchmark) keeps accumulating; only its first launch is validated."""

    def get_cases(self) -> list[CaseSpec]:
        return merge_state_in_place_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        p = {"mask": "random", **case.params}
        g = self.generator(case)
        shape = (p["seq"], p["heads"], p["d"])
        v, s = self._states(shape, g)
        v_other, s_other = self._states(shape, g)
        dev = self.device
        mask = {
            "none": None,
            "ones": torch.ones(p["seq"], dtype=torch.uint8, device=dev),
            "zeros": torch.zeros(p["seq"], dtype=torch.uint8, device=dev),
            "random": (torch.rand(p["seq"], generator=g, device=dev) < 0.7).to(
                torch.uint8
            ),
        }[p["mask"]]
        return v, s, v_other, s_other, mask

    def get_reference(self, inputs: tuple) -> tuple:
        v, s, v_other, s_other, mask = inputs
        o, lse = merge(
            torch.stack([v.float(), v_other.float()]), torch.stack([s, s_other])
        )
        if mask is not None:
            keep = mask.bool()
            o = torch.where(keep[:, None, None], o, v.float())
            lse = torch.where(keep[:, None], lse, s)
        return o.to(v.dtype), lse

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        v, s, v_other, s_other, mask = inputs
        seq, heads, d = v.shape
        out, lse = v.clone(), s.clone()
        args = [
            out,
            lse,
            v_other,
            s_other,
            mask if mask is not None else ctypes_ptr(0),
            ctypes_u32(heads),
            ctypes_u32(d),
        ]
        block = (d // self.meta["vec_size"], heads)
        outputs = (out, lse)

        def launch() -> tuple:
            self.launch(seq, block, args)
            return outputs

        return launch, outputs


class Fa2MergeStates(_Merge):
    """``MergeStatesKernel`` (``merge_states`` with fewer index sets than
    positions) or ``MergeStatesLargeNumIndexSetsKernel`` (at least as many)."""

    @property
    def large(self) -> bool:
        return self.meta["family"] == "merge_states_large"

    def get_cases(self) -> list[CaseSpec]:
        return merge_states_cases(self.meta)

    def get_inputs(self, case: CaseSpec) -> tuple:
        p = case.params
        g = self.generator(case)
        return self._states((p["seq"], p["sets"], p["heads"], p["d"]), g)

    def get_reference(self, inputs: tuple) -> tuple:
        v, s = inputs
        o, lse = merge(v.float().transpose(0, 1), s.transpose(0, 1))
        return o.to(DTYPES[self.meta["dtype_o"]]), lse

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple]:
        m = self.meta
        v, s = inputs
        seq, sets, heads, d = v.shape
        if self.large != (sets >= seq):
            raise ValueError("merge_states dispatches this shape to the other kernel")
        out = torch.empty((seq, heads, d), dtype=DTYPES[m["dtype_o"]], device=v.device)
        lse = torch.empty((seq, heads), dtype=torch.float32, device=v.device)
        if self.large:
            args = [v, s, out, lse, ctypes_u32(sets), ctypes_u32(heads)]
            grid: tuple[int, ...] = (seq, heads)
            block, smem = m["block"], m["shared_mem"]
        else:
            args = [v, s, out, lse, ctypes_u32(sets), ctypes_u32(heads), ctypes_u32(d)]
            grid, block, smem = (seq,), (d // m["vec_size"], heads), 0
        outputs = (out, lse)

        def launch() -> tuple:
            self.launch(grid, block, args, shared_mem=smem)
            return outputs

        return launch, outputs


# -- cases -------------------------------------------------------------------------
#
# Every kernel gets (1) its own smoke cases, built so that together they reach
# every hard path of the kernel: several Q tiles and KV tiles (KV loops far
# beyond the cp.async pipeline), ragged skewed batches with empty and length-1
# requests, random page tables with partial last pages, shared pages, NaN in
# uncovered page slots, GQA groups, both KV layouts, separate and interleaved
# K/V, non-contiguous q, the split-KV, fixed-split and CUDA-graph plans,
# windows active and inactive, soft caps that bend the (peaked) logits, sinks
# and every mask flavour; (2) the representative cases of the upstream test
# regimes that dispatch to it (``fixtures/sm_86_upstream.json``, written by
# the build from FlashInfer's tests; see ``impls/fmha/fa2_upstream.py``); and
# (3) throughput cases: model shapes of ``harness.models`` at sm_86 scale (a
# few GB per case) with skewed lengths.

UPSTREAM = PACKAGE / "fixtures" / "sm_86_upstream.json"
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
# Inputs + outputs + reference temporaries of one sm_86 case stay below ~6 GB.
KV_BUDGET = 2_500_000_000


@functools.cache
def load_upstream() -> dict[str, list[dict[str, Any]]]:
    if not UPSTREAM.is_file():
        return {}
    return json.loads(UPSTREAM.read_text())["kernels"]


def upstream_cases(meta: dict[str, Any]) -> list[CaseSpec]:
    """The recorded upstream regime representatives of this kernel."""
    return [
        upstream_case(
            f"upstream_{i}",
            row["params"],
            row["test"],
            suite=row["suite"],
            seed=200 + i,
            revision=FLASHINFER_REVISION,
        )
        for i, row in enumerate(load_upstream().get(meta["name"], []))
    ]


def _cycle(values: Sequence[Any], i: int) -> Any:
    return values[i % len(values)]


def _prefill_shapes(m: dict[str, Any]) -> list[dict[str, Any]]:
    """Smoke shapes whose plan selects ``m``'s CTA_TILE_Q (FA2DetermineCtaTileQ
    of the average packed query length; the CUDA-graph plan uses the largest
    possible request instead)."""
    cta, ragged = m["cta_tile_q"], m["kind"] == "ragged"
    shapes: list[dict[str, Any]]
    search = {"disable_split_kv": False}
    graph = {"cuda_graph": True, "disable_split_kv": False}
    if cta == 16:  # decode-like: avg packed <= 16
        shapes = [
            {
                "hq": 8,
                "hkv": 2,
                "q_lens": [1, 3, 0, 2, 4, 1],
                "kv_lens": [37, 300, 5, 129, 1, 64],
            },
            {"hq": 16, "hkv": 16, "q_lens": [16, 1, 9, 2], "kv_lens": [16, 700, 33, 0]},
            {
                "hq": 4,
                "hkv": 1,
                "q_lens": [2, 1, 1, 4],
                "kv_lens": [2048, 3, 77, 513],
                **search,
            },
            {
                "hq": 6,
                "hkv": 2,
                "q_lens": [1, 2, 1],
                "kv_lens": [100, 260, 33],
                **graph,
            },
        ]
        packed = {"hq": 8, "hkv": 2, "q_lens": [1, 3, 2], "kv_lens": [1, 3, 2]}
    elif cta == 64:  # 16 < avg packed <= 64 (and > 64 at head dim 256)
        shapes = [
            {
                "hq": 8,
                "hkv": 2,
                "q_lens": [5, 9, 16, 0, 1],
                "kv_lens": [40, 900, 16, 7, 1],
            },
            {
                "hq": 6,
                "hkv": 6,
                "q_lens": [33, 64, 17, 50],
                "kv_lens": [33, 300, 50, 0],
                "disable_split_kv": False,
                "fixed_split_size": 32,
            },
            {"hq": 16, "hkv": 2, "q_lens": [5, 3], "kv_lens": [1500, 300], **search},
            {"hq": 4, "hkv": 4, "q_lens": [20, 21], "kv_lens": [400, 21], **graph},
        ]
        if m["head_dim"] == 256:
            shapes.append(
                {"hq": 3, "hkv": 3, "q_lens": [100, 130, 70], "kv_lens": [256, 130, 70]}
            )
        packed = {"hq": 4, "hkv": 4, "q_lens": [17, 30, 9], "kv_lens": [17, 30, 9]}
    else:  # avg packed > 64
        shapes = [
            {
                "hq": 8,
                "hkv": 2,
                "q_lens": [20, 60, 1, 0, 3],
                "kv_lens": [20, 150, 700, 3, 0],
            },
            {
                "hq": 3,
                "hkv": 3,
                "q_lens": [100, 130, 70],
                "kv_lens": [256, 130, 70],
                "disable_split_kv": False,
                "fixed_split_size": 48,
            },
            {"hq": 32, "hkv": 8, "q_lens": [100], "kv_lens": [3000], **search},
            {"hq": 8, "hkv": 1, "q_lens": [17, 30], "kv_lens": [17, 600], **graph},
        ]
        packed = {
            "hq": 4,
            "hkv": 4,
            "q_lens": [100, 70, 129],
            "kv_lens": [100, 70, 129],
        }
    if ragged:
        shapes.append({**packed, "kv_storage": "packed"})
    if m["mask"] == MASK_IDS["multiitem"]:  # prefix + items need kv > q
        for s in shapes:
            s["kv_lens"] = [max(k, q + 2) for q, k in zip(s["q_lens"], s["kv_lens"])]
    return shapes  # fmt: skip


def _prefill_features(m: dict[str, Any], i: int) -> dict[str, Any]:
    """Runtime features of smoke case ``i``: every value of every knob the
    kernel accepts appears in some case."""
    f: dict[str, Any] = {
        "kv_layout": _cycle((NHD, HND), i),
        "return_lse": i != 1,
    }
    if m["kind"] == "paged":
        f |= {
            "page_size": _cycle((5, 16, 1, 16, 64), i),
            "pages": _cycle(("random", "shared", "random", "identity", "offset"), i),
            "kv_storage": _cycle(("interleaved", "separate"), i),
            # Multi-item scoring loads KV past kv_len without zero-filling V:
            # finite stale values get probability 0, NaN would propagate.
            "poison": i in (0, 2, 3) and m["mask"] != MASK_IDS["multiitem"],
            "q_storage": "packed" if i == 1 else "contiguous",
        }
    if m["swa"]:
        f["window_left"] = _cycle((31, -1, 40, 100000, 37), i)
    if m.get("sink"):
        f["window_left"] = _cycle((31, -1, 128, -1, 40), i)
    if m["softcap"]:
        f["logits_soft_cap"] = _cycle((3.0, 5.0, 2.0, 4.0, 3.5), i)
    if m["mask"] == MASK_IDS["custom"]:
        f["mask"] = _cycle(("random", "holes", "tril", "random", "holes"), i)
    return f


def prefill_cases(m: dict[str, Any]) -> list[CaseSpec]:
    cases = []
    for i, shape in enumerate(_prefill_shapes(m)):
        params = {**shape, **_prefill_features(m, i)}
        if "fixed_split_size" in params:  # in pages: chunks of 64 tokens
            params["fixed_split_size"] = max(1, 64 // params.get("page_size", 1))
        cases.append(CaseSpec(f"smoke_{i}", params, 1 + i))
    return cases + upstream_cases(m) + prefill_throughput(m)


def _model_heads(head_dim: int, alt: bool = False) -> str:
    """The ``harness.models`` model whose attention a head-dim case takes."""
    return {
        64: "gpt_oss_20b",
        128: "qwen3_235b_a22b" if alt else "llama3_8b",
        256: "qwen3_next_80b_a3b" if alt else "gemma2_9b",
    }[head_dim]


def _feature_params(m: dict[str, Any], model: str, alt: bool) -> dict[str, Any]:
    """Model-level runtime features for a kernel's throughput case; sink
    kernels take gpt-oss's sliding-window layer, then its full-attention
    layer (``alt``)."""
    spec = MODELS[model]
    f: dict[str, Any] = {}
    if m.get("swa") or (m.get("sink") and not alt):
        f["window_left"] = spec.get("sliding_window") or 1024
    if m.get("softcap"):
        f["logits_soft_cap"] = spec.get("attn_logit_softcapping") or 50.0
    return f


def prefill_throughput(m: dict[str, Any]) -> list[CaseSpec]:
    """Two serving shapes per kernel in its CTA_TILE_Q regime, at sm_86 scale,
    planned with upstream's default (split-KV enabled) plan."""
    cta, d = m["cta_tile_q"], m["head_dim"]
    kv_bytes = 1 if m["dtype_kv"] == "e4m3" else 2
    out = []
    for alt in (False, True):
        model = _model_heads(d, alt)
        if m.get("swa") and d == 128:
            model = "gemma3_27b" if not alt else model
        spec = MODELS[model]
        hq, hkv = spec.heads, spec.kv_heads
        group = hq // hkv
        per_token = hkv * d * 2 * kv_bytes
        if cta == 16 or (cta == 64 and d < 256):
            # Tensor-core decode (cta 16: packed q_len <= 16) or speculative
            # verification / short chunks (cta 64: 16 < avg packed <= 64) over
            # long skewed contexts.
            batch = 64 if alt else 128
            mean = 8192 if alt else 4096
            if cta == 16:
                q = [max(1, 16 // group)] * batch
            elif alt:
                q = skewed_lengths(
                    batch * max(1, 48 // group), batch, seed=17, minimum=1
                )
                while sum(q) * group // batch <= 16:
                    q[q.index(min(q))] += 1
            else:
                q = [max(1, 32 // group)] * batch
            total = min(batch * mean, KV_BUDGET // per_token)
            kv = skewed_lengths(total, batch, seed=7 + alt, sigma=0.8, minimum=max(q))
            label = f"decode_b{batch}_kv{total}"
        else:
            # Prefill: fresh prompts (8k tokens) or chunked prefill over history.
            batch = 4 if alt else 8
            tokens = 4096 if alt else 8192
            q = skewed_lengths(tokens, batch, seed=11 + alt, sigma=0.7, minimum=80)
            history = (
                skewed_lengths(4 * tokens, batch, seed=13, sigma=0.7)
                if alt
                else [0] * batch
            )
            kv = [a + b for a, b in zip(q, history)]
            label = f"prefill_b{batch}_q{tokens}_kv{sum(kv)}"
        params = {
            "hq": hq,
            "hkv": hkv,
            "q_lens": q,
            "kv_lens": kv,
            "page_size": 16 if m["kind"] == "paged" else 1,
            "disable_split_kv": False,
            **_feature_params(m, model, alt),
        }
        if m["mask"] == MASK_IDS["multiitem"]:
            params["kv_lens"] = [max(k, a + 2) for a, k in zip(q, params["kv_lens"])]
        out.append(model_case(f"{model}_{label}", params, model, "attention"))
    return out


def decode_cases(m: dict[str, Any]) -> list[CaseSpec]:
    grid = m["blocks_per_sm"] * NUM_SMS
    big = -(-grid // 8) + 5  # batch x 8 KV heads beyond the grid: no split
    shapes = [
        {"hkv": 4, "kv_lens": [1, 17, 64, 128, 100, 0], "page_size": 1, "poison": True},
        {"hkv": 2, "kv_lens": [33, 3000, 7], "page_size": 16, "kv_layout": HND,
         "kv_storage": "separate"},
        {"hkv": 3, "kv_lens": [5, 99, 1024], "page_size": 5, "cuda_graph": True,
         "pages": "shared", "poison": True},
        {"hkv": 8, "kv_lens": skewed_lengths(40 * big, big, seed=3, minimum=1),
         "page_size": 16, "return_lse": False, "q_storage": "packed", "pages": "identity"},
    ]  # fmt: skip
    cases = []
    for i, s in enumerate(shapes):
        if m["swa"]:
            s["window_left"] = _cycle((40, -1, 100000, 63), i)
        if m["softcap"]:
            s["logits_soft_cap"] = _cycle((3.0, 5.0, 2.0, 4.0), i)
        cases.append(CaseSpec(f"smoke_{i}", s, 11 + i))
    return cases + upstream_cases(m) + decode_throughput(m)


def _decode_model(m: dict[str, Any]) -> tuple[str | None, int]:
    """(model, KV heads) of a decode kernel's group size and head dim."""
    for name in ("llama3_70b", "llama3_8b", "gemma3_27b", "gpt_oss_20b",
                 "gemma2_9b", "qwen3_next_80b_a3b", "qwen3_30b_a3b"):  # fmt: skip
        spec = MODELS[name]
        if spec.head_dim == m["head_dim"] and spec.group == m["group_size"]:
            return name, spec.kv_heads
    return None, 8


def decode_throughput(m: dict[str, Any]) -> list[CaseSpec]:
    """Batch 64 x 8k and batch 256 x 2k skewed contexts (capped at ~2.5 GB of
    KV) with the model's KV heads, or 8 KV heads where no model in
    ``harness.models`` has this group size and head dim."""
    model, hkv = _decode_model(m)
    kv_bytes = 1 if m["dtype_kv"] == "e4m3" else 2
    per_token = hkv * m["head_dim"] * 2 * kv_bytes
    out = []
    for batch, mean, seed in ((64, 8192, 5), (256, 2048, 6)):
        total = min(batch * mean, KV_BUDGET // per_token)
        params: dict[str, Any] = {
            "hkv": hkv,
            "kv_lens": skewed_lengths(total, batch, seed=seed, sigma=0.8),
            "page_size": 16,
        }
        if m["swa"]:
            params["window_left"] = (
                MODELS[model].get("sliding_window") if model else None
            ) or 1024
        if m["softcap"]:
            params["logits_soft_cap"] = 50.0
        label = f"decode_b{batch}_kv{total}"
        if model:
            out.append(model_case(f"{model}_{label}", params, model, "attention"))
        else:
            out.append(
                synthetic(
                    label,
                    params,
                    f"decode at group size {m['group_size']} (no harness.models "
                    f"model has it at head dim {m['head_dim']}): long skewed contexts",
                )
            )
    return out


def persistent_cases(m: dict[str, Any]) -> list[CaseSpec]:
    shapes = [
        # decode + prefill requests: both tasks, causal tiles, KV splits
        {"hq": 8, "hkv": 2, "q_lens": [1, 1, 37, 3, 100], "kv_lens": [33, 700, 37, 20, 260],
         "page_size": 16, "kv_layout": NHD, "poison": True},
        {"hq": 4, "hkv": 4, "q_lens": [64, 1, 2], "kv_lens": [3000, 1, 9], "page_size": 1,
         "kv_layout": HND, "kv_storage": "separate"},
        {"hq": 6, "hkv": 1, "q_lens": [5, 7], "kv_lens": [5, 2047], "page_size": 5,
         "pages": "shared"},
        {"hq": 28, "hkv": 4, "q_lens": [235, 1, 17], "kv_lens": [2, 900, 4000], "page_size": 8,
         "kv_layout": HND, "v_scale": 2.0, "q_storage": "chunked", "poison": True},
    ]  # fmt: skip
    cases = []
    for i, s in enumerate(shapes):
        if m["softcap"]:
            s["logits_soft_cap"] = _cycle((3.0, 5.0, 2.0, 4.0), i)
        cases.append(CaseSpec(f"smoke_{i}", s, 41 + i))
    # Mixed serving batches (gpt-oss attention: 64 / 8 heads of 64).
    spec = MODELS["gpt_oss_20b"]
    cap = {"logits_soft_cap": 50.0} if m["softcap"] else {}
    decode = skewed_lengths(128 * 4096, 128, seed=21, sigma=0.8, minimum=64)
    mixed = {
        "hq": spec.heads,
        "hkv": spec.kv_heads,
        "q_lens": [1] * 128 + [1024] * 4,
        "kv_lens": decode + [4096] * 4,
        "page_size": 16,
        **cap,
    }
    prompt = skewed_lengths(8192, 4, seed=22, sigma=0.6, minimum=256)
    prefill = {
        "hq": spec.heads,
        "hkv": spec.kv_heads,
        "q_lens": prompt,
        "kv_lens": [n + 4096 for n in prompt],
        "page_size": 16,
        **cap,
    }
    return (
        cases
        + upstream_cases(m)
        + [
            model_case("gpt_oss_20b_mixed_b132", mixed, "gpt_oss_20b", "attention"),
            model_case(
                "gpt_oss_20b_chunked_prefill", prefill, "gpt_oss_20b", "attention"
            ),
        ]
    )


def mla_cases(m: dict[str, Any]) -> list[CaseSpec]:
    causal = m["causal"]
    shapes = [
        {"heads": 16, "q_lens": [1, 1, 1, 1], "kv_lens": [33, 20, 600, 1 if causal else 0],
         "page_size": 16, "poison": True},
        {"heads": 128, "q_lens": [4, 1], "kv_lens": [600, 130], "page_size": 1},
        {"heads": 64, "q_lens": [17, 1, 3], "kv_lens": [17, 3000, 514], "page_size": 32,
         "pages": "shared", "poison": True},
    ]  # fmt: skip
    cases = [CaseSpec(f"smoke_{i}", s, 51 + i) for i, s in enumerate(shapes)]
    spec = MODELS["deepseek_v3"]
    tp1 = {
        "heads": spec.heads,
        "q_lens": [1] * 128,
        "kv_lens": skewed_lengths(128 * 4096, 128, seed=31, sigma=0.8),
        "page_size": 64,
    }
    tp8 = {  # 16 heads per GPU at TP 8, MTP verification with 2 tokens
        "heads": spec.heads // 8,
        "q_lens": [2] * 256,
        "kv_lens": skewed_lengths(256 * 4096, 256, seed=32, sigma=0.8, minimum=2),
        "page_size": 64,
    }
    return (
        cases
        + upstream_cases(m)
        + [
            model_case("deepseek_v3_decode_b128", tp1, "deepseek_v3", "mla"),
            model_case("deepseek_v3_tp8_mtp_b256", tp8, "deepseek_v3", "mla"),
        ]
    )


def merge_varlen_cases(m: dict[str, Any]) -> list[CaseSpec]:
    d = m["head_dim"]
    return [
        CaseSpec(
            "smoke_mixed", {"sets": [0, 1, 2, 5, 3, 1, 8], "heads": 4, "d": d}, 31
        ),
        CaseSpec("smoke_gqa", {"sets": [2] * 9, "heads": 32, "d": d}, 32),
        CaseSpec(
            "smoke_no_lse",
            {"sets": [3, 1, 4, 1, 5, 9, 2, 6], "heads": 8, "d": d, "return_lse": False},
            33,
        ),
        CaseSpec(
            "smoke_graph_rows",
            {"sets": [2, 7, 1, 1, 3, 0, 4], "heads": 6, "d": d, "seq_len": 5},
            34,
        ),
        *upstream_cases(m),
        model_case(
            "llama3_8b_split_decode_b2048",
            {"sets": [4] * 2048, "heads": 32, "d": d},
            "llama3_8b",
            "attention",
        ),
        synthetic(
            "split_prefill_rows",
            {
                "sets": skewed_lengths(3 * 8192, 8192, seed=9, sigma=0.5),
                "heads": 32,
                "d": d,
            },
            "merge after a split-KV prefill of 8192 rows (1-8 chunks per row)",
        ),
    ]


def _merge_dims(m: dict[str, Any]) -> list[int]:
    return list(m.get("head_dims") or [m["head_dim"]])


def merge_state_cases(m: dict[str, Any]) -> list[CaseSpec]:
    cases = [
        CaseSpec(f"smoke_hd{d}", {"seq": 37, "heads": 3, "d": d}, 21 + i)
        for i, d in enumerate(_merge_dims(m))
    ]
    d = _merge_dims(m)[-1]
    return (
        cases
        + upstream_cases(m)
        + [
            synthetic(
                f"hd{d}",
                {"seq": 8192, "heads": 32, "d": d},
                "merge of 8192 x 32 states",
            )
        ]
    )


def merge_state_in_place_cases(m: dict[str, Any]) -> list[CaseSpec]:
    cases = []
    for i, d in enumerate(_merge_dims(m)):
        for j, mask in enumerate(("random", "none", "ones", "zeros")):
            if j and i:
                continue  # every mask once, every head dim once
            cases.append(
                CaseSpec(
                    f"smoke_hd{d}_{mask}",
                    {"seq": 41, "heads": 5, "d": d, "mask": mask},
                    21 + 4 * i + j,
                )
            )
    d = _merge_dims(m)[-1]
    return (
        cases
        + upstream_cases(m)
        + [
            synthetic(
                f"hd{d}",
                {"seq": 8192, "heads": 32, "d": d, "mask": "random"},
                "cascade merge of 8192 x 32 states",
            )
        ]
    )


def merge_states_cases(m: dict[str, Any]) -> list[CaseSpec]:
    large = m["family"] == "merge_states_large"
    cases = []
    for i, d in enumerate(_merge_dims(m)):
        if large:  # num_index_sets >= seq_len
            params = {"seq": 3, "sets": 7, "heads": 3, "d": d}
        else:
            params = {"seq": 29, "sets": 1 + d // 64, "heads": 3, "d": d}
        cases.append(CaseSpec(f"smoke_hd{d}", params, 21 + i))
    d0 = _merge_dims(m)[0]
    if large:
        cases.append(
            CaseSpec("smoke_one_row", {"seq": 1, "sets": 300, "heads": 4, "d": d0}, 28)
        )
        big = {"seq": 64, "sets": 512, "heads": 32, "d": _merge_dims(m)[-1]}
    else:
        cases.insert(
            0, CaseSpec("smoke_one_set", {"seq": 9, "sets": 1, "heads": 2, "d": d0}, 29)
        )
        big = {"seq": 4096, "sets": 8, "heads": 32, "d": _merge_dims(m)[-1]}
    return (
        cases
        + upstream_cases(m)
        + [synthetic("many_states", big, "merge of many attention states")]
    )


# -- regimes (upstream coverage) -----------------------------------------------------
#
# A case's regime names the code paths and shape regime its launch takes on
# its kernel: the features upstream tests switch (plan kind, KV layout and
# storage, page size, GQA group, window, LSE, special values, ...) and the
# planner's decisions (split or not, tasks used). ``impls/fmha/fa2_upstream.py``
# maps every upstream test parametrization to (kernel, params); the upstream
# tests are a subset of the harness when every such regime is the regime of
# one of the kernel's cases (``tests/test_fmha.py``).


@functools.cache
def probe(name: str) -> _Fa2:
    """A reference-only CPU instance of workload ``name`` (planning only)."""
    meta = {**load_index()["kernels"][name], "name": name}
    cls = type(name, (FAMILIES[meta["family"]],), {"meta": meta})
    return cls(None, device="cpu")


def _window(p: dict[str, Any], q_lens: Sequence[int]) -> str:
    wl = p.get("window_left", -1)
    if wl < 0:
        return "off"
    active = any(lq and lk - 1 > wl for lq, lk in zip(q_lens, p["kv_lens"]))
    return "active" if active else "inactive"


def regime(name: str, case_params: dict[str, Any]) -> dict[str, Any]:
    w = probe(name)
    m = w.meta
    family = m["family"]
    if family == "prefill":
        p = w.params(case_params)
        plan = w.route(case_params)
        pairs = list(zip(p["q_lens"], p["kv_lens"]))
        bottom = m["mask"] in (MASK_IDS["causal"], MASK_IDS["multiitem"]) or (
            m["mask"] == MASK_IDS["custom"] and p["mask"] == "tril"
        )
        masked = any(lq and (lk == 0 or (bottom and lq > lk)) for lq, lk in pairs)
        if m["mask"] == MASK_IDS["custom"] and p["mask"] == "holes":
            masked = True
        if p["disable_split_kv"]:
            mode = "disable"
        else:
            mode = "fixed" if p["fixed_split_size"] > 0 else "search"
        graph = p["cuda_graph"] and ("uniform" if p["uniform_q_len"] else "ragged")
        return {
            "page_size": p["page_size"] if m["kind"] == "paged" else None,
            "kv_layout": p["kv_layout"],
            "group": p["hq"] // p["hkv"],
            "plan": mode,
            "split": plan["split_kv"],
            "cuda_graph": graph,
            "window": _window(p, p["q_lens"]),
            "lse": plan["split_kv"] or p["return_lse"],
            "q_storage": p["q_storage"],
            "kv_storage": p["kv_storage"],
            "masked_rows": masked,
            "values": p["values"] if p["values"] in ("extreme", "constant") else "",
        }
    if family == "decode":
        p = w.params(case_params)
        plan = w.route(case_params)
        return {
            "page_size": p["page_size"],
            "kv_layout": p["kv_layout"],
            "split": plan["split_kv"],
            "cuda_graph": p["cuda_graph"],
            "window": _window(p, [1] * len(p["kv_lens"])),
            "lse": plan["split_kv"] or p["return_lse"],
            "q_storage": p["q_storage"],
            "kv_storage": p["kv_storage"],
            "empty": any(n == 0 for n in p["kv_lens"]),
            "values": p["values"] if p["values"] in ("extreme", "constant") else "",
        }
    if family == "persistent":
        p = w.params(case_params)
        plan = w.route(case_params)
        pairs = list(zip(p["q_lens"], p["kv_lens"]))
        return {
            "page_size": p["page_size"],
            "kv_layout": p["kv_layout"],
            "group": p["hq"] // p["hkv"],
            "v_scale": p["v_scale"] != 1.0,
            "q_storage": p["q_storage"],
            "tasks": [bool(t["q_indptr"]) for t in plan["tasks"]],
            "split": len(plan["merge_indptr"]) > 1,
            "masked_rows": any(
                lq and (lk == 0 or (p["causal"] and lq > lk)) for lq, lk in pairs
            ),
        }
    if family == "mla":
        p = w.params(case_params)
        plan = w.route(case_params)
        ends = plan["arrays"]["merge_packed_offset_end"]
        starts = plan["arrays"]["merge_packed_offset_start"]
        return {
            "page_size": p["page_size"],
            "heads": p["heads"],
            "cluster": plan["num_blks"][0],
            "split": any(e > s for s, e in zip(starts, ends)),
            "empty": any(n == 0 for n in p["kv_lens"]),
            "poison": p["poison"],
        }
    p = case_params
    if family == "merge_varlen":
        return {
            "lse": p.get("return_lse", True),
            "graph_rows": p.get("seq_len") is not None,
            "empty_rows": 0 in p["sets"],
        }
    if family == "merge_state":
        return {"d": p["d"]}
    if family == "merge_state_in_place":
        return {"d": p["d"], "mask": p.get("mask", "random")}
    if family == "merge_states":
        return {"d": p["d"], "one_set": p["sets"] == 1}
    return {"d": p["d"], "one_row": p["seq"] == 1}  # merge_states_large


def regime_key(name: str, case_params: dict[str, Any]) -> str:
    return json.dumps(regime(name, case_params), sort_keys=True)


def case_cost(name: str, p: dict[str, Any]) -> float:
    """Work of a case (attention FLOPs plus a weight on its bytes), to pick the
    cheapest upstream parametrization of a regime."""
    m = probe(name).meta
    family = m["family"]
    if family in ("prefill", "persistent"):
        d = m["head_dim"]
        flops = 4 * d * p["hq"] * sum(a * b for a, b in zip(p["q_lens"], p["kv_lens"]))
        return flops + 64 * sum(p["kv_lens"]) * p["hkv"] * d
    if family == "decode":
        d = m["head_dim"]
        hq = p["hkv"] * m["group_size"]
        return 4 * d * hq * sum(p["kv_lens"]) + 64 * sum(p["kv_lens"]) * p["hkv"] * d
    if family == "mla":
        flops = (
            4 * 576 * p["heads"] * sum(a * b for a, b in zip(p["q_lens"], p["kv_lens"]))
        )
        return flops + 64 * sum(p["kv_lens"]) * 576
    if family == "merge_varlen":
        return float(sum(p["sets"]) * p["heads"] * p["d"])
    return float(p["seq"] * p.get("sets", 2) * p["heads"] * p["d"])


def case_suite(name: str, p: dict[str, Any]) -> str:
    """``smoke`` for upstream cases cheap enough for every test run (well
    under a second on an RTX 3090), else ``throughput``."""
    return "smoke" if case_cost(name, p) <= 4e9 else "throughput"


# -- layout checks -----------------------------------------------------------------


def _check_prefill_example(
    layouts: dict[str, Any],
    sentinel: dict[str, int],
    example: dict[str, Any],
    suffix: str,
    sink: bool,
) -> None:
    a = example["args"]
    kind = example["kind"]
    hq, hkv, d = a["num_qo_heads"], a["num_kv_heads"], a["head_dim"]
    ptr = dict(sentinel)
    split, graph = a["split"], a["cuda_graph"]
    if not a["custom"]:
        ptr.pop("custom_mask"), ptr.pop("mask_indptr")
    if not a["multi_item"]:
        for role in ("prefix_len", "token_pos", "max_item_len"):
            ptr.pop(role)
    if a["no_lse"] and not split:
        ptr.pop("lse")
    if not split:
        ptr.pop("merge_indptr")
    if not (split and graph):
        ptr.pop("block_valid_mask")
    if not graph:
        ptr.pop("total_num_rows")
    settings = {
        "batch": a["batch"],
        "num_qo_heads": hq,
        "num_kv_heads": hkv,
        "head_dim": d,
        "kv_layout": a["kv_layout"],
        "q_strides": (hq * d, d),
        "window_left": a["window_left"],
        "logits_soft_cap": a["logits_soft_cap"],
        "sm_scale": a["sm_scale"],
        "token_pos_in_items_len": a["token_pos_in_items_len"],
        "padded_batch_size": a["padded_batch_size"],
        "total_num_rows": a["total_num_rows"],
        "sink": sink,
        "split": split,
    }
    layout = layouts[kind + suffix]
    if kind == "paged":
        page = a["page_size"]
        strides = (
            (hkv * page * d, page * d, d, 1)
            if a["kv_layout"] == HND
            else (page * hkv * d, hkv * d, d, 1)
        )
        built = paged_params(
            layout,
            ptr,
            {**settings, "page_size": page, "k_strides": strides, "v_strides": strides},
        )
    else:
        total = a["total_kv"]
        nh = (hkv * d, d) if a["kv_layout"] == NHD else (d, total * d)
        built = ragged_params(
            layout, ptr, {**settings, "k_strides": nh, "v_strides": nh}
        )
    _compare(f"prefill{suffix} {kind} {a}", layout, built, example["bytes"])


def check_param_layouts() -> bool:
    """Byte-compare Params rebuilt here with the probes' reference structs
    (padding excluded, every ``uint_fastdiv`` included), the Python planners
    (prefill split-KV / fixed-split / CUDA-graph, decode, holistic, MLA) with
    the probes' plans, and every kernel's launch constants with the probes'
    dispatch records."""
    index = load_index()
    if not index["kernels"]:
        return False
    fixtures = json.loads(FIXTURES.read_text())
    layouts = index["layouts"]
    sentinel = {role: SENTINEL_STRIDE * (i + 1) for i, role in enumerate(PREFILL_ROLES)}
    sections = (("prefill", "", False), ("prefill_sink", "_sink", True))
    for section, suffix, sink in sections:
        for example in fixtures[section]["examples"]:
            _check_prefill_example(layouts, sentinel, example, suffix, sink)
    sentinel = {role: SENTINEL_STRIDE * (i + 1) for i, role in enumerate(DECODE_ROLES)}
    for example in fixtures["decode"]["examples"]:
        a = example["args"]
        ptr = dict(sentinel)
        if a["no_lse"] and not a["split"]:
            ptr.pop("lse")
        if not (a["split"] and a["cuda_graph"]):
            ptr.pop("block_valid_mask")
        hq, hkv, d, page = (
            a["num_qo_heads"],
            a["num_kv_heads"],
            a["head_dim"],
            a["page_size"],
        )
        strides = (
            (hkv * page * d, page * d, d, 1)
            if a["kv_layout"] == HND
            else (page * hkv * d, hkv * d, d, 1)
        )
        built = decode_params(
            layouts["decode"],
            ptr,
            {
                "batch": a["batch"],
                "num_qo_heads": hq,
                "num_kv_heads": hkv,
                "page_size": page,
                "head_dim": d,
                "kv_layout": a["kv_layout"],
                "k_strides": strides,
                "v_strides": strides,
                "q_strides": (hq * d, d),
                "window_left": a["window_left"],
                "logits_soft_cap": a["logits_soft_cap"],
                "sm_scale": a["sm_scale"],
                "padded_batch_size": a["padded_batch_size"],
                "split": a["split"],
            },
        )
        _compare(f"decode {a}", layouts["decode"], built, example["bytes"])
    for plan in fixtures["prefill"]["plans"]:
        got = prefill_plan(
            plan["qo_indptr"],
            plan["kv_indptr"],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            plan["head_dim"],
            plan["page_size"],
            plan["kv_dtype_bytes"],
            num_sms=NUM_SMS,
            window_left=plan["window_left"],
            cuda_graph=plan["cuda_graph"],
            disable_split_kv=plan["disable_split_kv"],
            fixed_split_size=plan["fixed_split_size"],
            uniform_q_len=plan["uniform_q_len"],
            total_num_rows=plan["total_num_rows"],
        )
        got["block_valid_mask"] = [int(x) for x in got["block_valid_mask"] or []]
        for key in (
            "split_kv",
            "cta_tile_q",
            "new_batch_size",
            "padded_batch_size",
            "kv_chunk_size",
            "request_indices",
            "qo_tile_indices",
            "kv_tile_indices",
            "merge_indptr",
            "o_indptr",
            "block_valid_mask",
        ):
            if got[key] != plan[key]:
                raise AssertionError(
                    f"plan {plan['qo_indptr']}: {key} {got[key]} != {plan[key]}"
                )
    for plan in fixtures["decode"]["plans"]:
        got = decode_plan(
            plan["num_pages"],
            plan["num_kv_heads"],
            plan["page_size"],
            plan["max_grid_size"],
            cuda_graph=plan["cuda_graph"],
        )
        got["block_valid_mask"] = [int(x) for x in got["block_valid_mask"] or []]
        for key in (
            "split_kv",
            "new_batch_size",
            "padded_batch_size",
            "kv_chunk_size",
            "request_indices",
            "kv_tile_indices",
            "o_indptr",
            "block_valid_mask",
        ):
            if got[key] != plan[key]:
                raise AssertionError(
                    f"decode plan {plan['num_pages']}: {key} {got[key]} != {plan[key]}"
                )
    # Launch constants recorded per workload agree with the probe's dispatch.
    decode = {
        (r["dtype_q"], r["dtype_kv"], r["head_dim"], r["group_size"]): r
        for r in fixtures["decode"]["dispatch"]
    }
    merges = {(r["dtype"], r["head_dim"]): r for r in fixtures["decode"]["merge"]}
    prefill = {
        (d["kind"], d["dtype_q"], d["dtype_kv"], d["head_dim"], d["cta_tile_q"]): d
        for d in fixtures["prefill"]["dispatch"]
    }
    for name, m in index["kernels"].items():
        if m["family"] == "prefill":
            d = prefill[
                (m["kind"], m["dtype_q"], m["dtype_kv"], m["head_dim"], m["cta_tile_q"])
            ]
            if (
                not d["fits"]
                or d["num_mma_kv"] != m["num_mma_kv"]
                or d["shared_mem"] != m["shared_mem"]
                or [32, d["num_warps_q"], d["num_warps_kv"]] != m["block"]
            ):
                raise AssertionError(
                    f"{name}: launch differs from the prefill dispatch"
                )
        if m["family"] == "decode":
            r = decode[(m["dtype_q"], m["dtype_kv"], m["head_dim"], m["group_size"])]
            if [r["bdx"], r["bdy"], r["bdz"]] != m["block"] or r["shared_mem"] != m[
                "shared_mem"
            ]:
                raise AssertionError(f"{name}: launch differs from the decode dispatch")
        elif m["family"] in ("merge_varlen", "merge_states_large"):
            head_dim = m.get("head_dim") or m["head_dims"][0]
            r = merges[(m["dtype_in"], head_dim)]
            if [r["bdx"], r["bdy"], 1] != m["block"] or r["shared_mem"] != m[
                "shared_mem"
            ]:
                raise AssertionError(
                    f"{name}: launch differs from VariableLengthMergeStates"
                )
    _check_holistic(layouts, fixtures)
    _check_mla(layouts, fixtures)
    return True


# Fake tensor / workspace pointers of the persistent and MLA probes (role order).
_PERSISTENT_ROLES = (
    "q",
    "k",
    "v",
    "kv_indices",
    "o",
    "lse",
    "int_workspace",
    "float_workspace",
)
_MLA_ROLES = (
    "q_nope",
    "q_pe",
    "ckv",
    "kpe",
    "kv_indices",
    "o",
    "lse",
    "int_workspace",
    "float_workspace",
)


def _check_holistic(layouts: dict[str, Any], fixtures: dict[str, Any]) -> None:
    """TwoStageHolisticPlan port (every work and merge array, the workspace
    offsets) and both task PersistentParams against the probe (sm_86 model:
    82 SMs)."""
    ptr = {role: SENTINEL_STRIDE * (i + 1) for i, role in enumerate(_PERSISTENT_ROLES)}
    for example in fixtures["persistent"]["examples"]:
        a = example["args"]
        plan = holistic_plan(
            a["qo_indptr"],
            a["kv_indptr"],
            a["kv_len"],
            a["num_qo_heads"],
            a["num_kv_heads"],
            a["head_dim"],
            a["causal"],
            82,
        )
        shared = plan["shared_offsets"]
        info = [*plan["num_blks"]]
        for offsets in plan["task_offsets"]:
            info += [offsets[name] for name in HOLISTIC_TASK_ARRAYS]
        info += [
            shared["len_kv_chunk"],
            plan["partial_o_offset"],
            plan["partial_lse_offset"],
            shared["merge_indptr"],
            shared["merge_o_indices"],
            shared["num_qo_len"],
        ]
        got = {
            "plan_info": info,
            "tasks": plan["tasks"],
            "len_kv_chunk": plan["len_kv_chunk"],
            "merge_indptr": plan["merge_indptr"],
            "merge_o_indices": plan["merge_o_indices"],
            "num_packed_qo_len": plan["num_packed_qo_len"],
        }
        for key, value in got.items():
            if value != example[key]:
                raise AssertionError(f"holistic plan {a['qo_indptr']}: {key} differs")
        hkv, page, d = a["num_kv_heads"], a["page_size"], a["head_dim"]
        strides = (
            (hkv * page * d, d, hkv * d)
            if a["kv_layout"] == NHD
            else (hkv * page * d, page * d, d)
        )
        settings = {
            "num_qo_heads": a["num_qo_heads"],
            "num_kv_heads": hkv,
            "page_size": page,
            "q_strides": (a["num_qo_heads"] * d, d),
            "k_strides": strides,
            "v_strides": strides,
            "sm_scale": a["sm_scale"],
            "logits_soft_cap": a["logits_soft_cap"],
            "v_scale": a["v_scale"],
        }
        for task in (0, 1):
            built = persistent_params(layouts["persistent"], plan, task, ptr, settings)
            _compare(
                f"persistent {a['qo_indptr']} task {task}",
                layouts["persistent"],
                built,
                example["params"][task],
            )


def _check_mla(layouts: dict[str, Any], fixtures: dict[str, Any]) -> None:
    """MLAPlan port (every work and merge array, offsets) and MLAParams
    against the probe (sm_86 model: 82 SMs)."""
    ptr = {role: SENTINEL_STRIDE * (i + 1) for i, role in enumerate(_MLA_ROLES)}
    for example in fixtures["mla"]["examples"]:
        a = example["args"]
        plan = mla_plan(
            a["qo_indptr"],
            a["kv_indptr"],
            a["kv_len"],
            a["num_heads"],
            512,
            a["causal"],
            82,
        )
        off = plan["offsets"]
        info = [
            *plan["num_blks"],
            off["q_indptr"],
            off["kv_indptr"],
            off["partial_indptr"],
            *[off[name] for name in MLA_MERGE_ARRAYS],
            off["q_len"],
            off["kv_len"],
            off["q_start"],
            off["kv_start"],
            off["kv_end"],
            off["work_indptr"],
            plan["partial_o_offset"],
            plan["partial_lse_offset"],
        ]
        if info != example["plan_info"]:
            raise AssertionError(f"MLA plan {a['qo_indptr']}: offsets differ")
        if plan["int_workspace"] != example["staged_int_workspace_bytes"]:
            raise AssertionError(f"MLA plan {a['qo_indptr']}: workspace size differs")
        for name in (*MLA_WORK_ARRAYS, *MLA_MERGE_ARRAYS, "work_indptr"):
            if plan["arrays"][name] != example[name]:
                raise AssertionError(f"MLA plan {a['qo_indptr']}: {name} differs")
        heads, page = a["num_heads"], a["page_size"]
        strides = {
            "q_nope_stride_n": heads * 512,
            "q_nope_stride_h": 512,
            "q_pe_stride_n": heads * 64,
            "q_pe_stride_h": 64,
            "ckv_stride_page": page * 512,
            "ckv_stride_n": 512,
            "kpe_stride_page": page * 64,
            "kpe_stride_n": 64,
            "o_stride_n": heads * 512,
            "o_stride_h": 512,
        }
        built = mla_params(
            layouts["mla"],
            plan,
            ptr,
            {
                "page_size": page,
                "num_heads": heads,
                "strides": strides,
                "sm_scale": a["sm_scale"],
            },
        )
        _compare(f"mla {a['qo_indptr']}", layouts["mla"], built, example["params"])


def _compare(
    what: str, layout: dict[str, Any], built: bytes, expected_hex: str
) -> None:
    expected = bytes.fromhex(expected_hex)
    mask = covered(layout)
    if len(built) != len(expected):
        raise AssertionError(f"{what}: {len(built)} bytes, probe {len(expected)}")
    for i, (x, y, c) in enumerate(zip(built, expected, mask)):
        if c and x != y:
            names = [
                n
                for n, f in layout["fields"].items()
                if f["offset"] <= i < f["offset"] + f["size"]
            ]
            raise AssertionError(f"{what}: byte {i} ({names}) differs from the probe")


# -- registration ------------------------------------------------------------------

FAMILIES: dict[str, type[_Fa2]] = {
    "prefill": Fa2Prefill,
    "decode": Fa2Decode,
    "merge_varlen": Fa2MergeVarlen,
    "merge_state": Fa2MergeState,
    "merge_state_in_place": Fa2MergeStateInPlace,
    "merge_states": Fa2MergeStates,
    "merge_states_large": Fa2MergeStates,
    "persistent": Fa2Persistent,
    "mla": Fa2Mla,
}


def _register_all() -> None:
    for name, meta in load_index()["kernels"].items():
        register_variant(
            FAMILIES[meta["family"]],
            name=name,
            supported_arches=(ARCH,),
            meta={**meta, "name": name},
        )


_register_all()

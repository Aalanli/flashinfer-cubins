"""trtllm-gen FMHA kernels (sm_100a) as single-kernel workloads.

Every cubin of ``cubins/fmha`` (flashinfer_cubin 0.6.8/0.6.9, trtllm-gen
artifact ``55bba559.../fmha/trtllm-gen``) holds one attention kernel plus a
never-launched ``<kernel>GetSmemSize`` helper, stripped at compile time
(``impls/fmha/trtllm_build.py``). One base class, :class:`TrtllmFmha`,
implements the contract from the kernel's meta-info row
(``flashInferMetaInfo.h``: dtypes, tile sizes, head dims, QKV layout, mask,
kernel type, scheduler, multi-CTA KV mode, ...); each live kernel is
registered as a variant from the compact index ``impls/fmha/variants/
sm_100a.json``.

Host code mirrored here (FlashInfer v0.6.9, the release of the cubins):

* ``csrc/trtllm_fmha_kernel_launcher.cu`` and the Python wrappers
  (``flashinfer/decode.py``, ``prefill.py``, ``mla/_core.py``): how FlashInfer
  fills ``TllmGenFmhaRunnerParams`` (paged context/decode with HND or NHD
  caches, shared or separate K/V page indices, non-contiguous queries,
  attention sinks, host or device bmm1/bmm2 scales, NVFP4 output scale-factor
  offsets; ragged prefill), extended to the generic runner fields the kernel
  table also serves (packed QKV, static context scheduler, ...);
* ``include/flashinfer/trtllm/fmha/fmhaKernels.cuh``:
  ``computeCtaAndClusterConfig`` (grid, cluster, multi-CTA KV split) for the
  selected kernel and the launch attributes;
* ``include/flashinfer/trtllm/fmha/kernelParams.h``:
  ``KernelParams::setKernelParams`` (TMA descriptors and scalar fields).

Each variant's cases are runner parameters for which upstream kernel
selection (``TllmGenFmhaKernel::run`` with the driver interposed, compile-time
probe ``impls/fmha/kernels/trtllm_probe.cu``) picks exactly this kernel; the
probe also recorded its launch configuration and ``KernelParams`` bytes, which
``check_param_layouts`` compares with the Python port byte for byte (fake
pointers, recording TMA encoder). Case sources: ``smoke`` cases chosen for
coverage, ``upstream_test`` cases mirroring the FlashInfer v0.6.9 test
parametrizations that select the kernel, ``model_shape``/``synthetic_stress``
throughput cases (see ``trtllm_build``).

Inputs make errors visible: attention is peaked (query scale chosen for a
logit spread of 1.5-2.5) and every KV head adds its own offset to V, so
outputs are O(1) and head-specific; skip-softmax cases place "beacon" keys
in every other KV tile and push the remaining tiles' logits 30+ below them,
so the kernel skips those tiles (``exp(local - running max)`` is below
``threshold / kv length``) while their exact contribution stays below 1e-13
of the output.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import struct
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import torch

from ... import cuda_driver
from ...cutlass_host import ceil_div, driver_encode
from ...registry import register_variant
from ...workload import IMPLS, CaseSpec, Workload

PACKAGE = IMPLS / "fmha"
INDEX = PACKAGE / "variants" / "sm_100a.json"
FIXTURES = PACKAGE / "fixtures" / "sm_100a.json"
ARCH = "sm_100a"

INT_MAX = 2**31 - 1
LOG2E = 1.44269504088896340736  # M_LOG2E
SENTINEL_STRIDE = 0x100000000000
# Pointer roles in the probe's order (sentinel address = stride * (index + 1)).
ROLES = (
    "q",
    "k",
    "v",
    "kv",
    "qkv",
    "kSf",
    "vSf",
    "customMask",
    "customMaskOffsets",
    "firstSparse",
    "counter",
    "seqLensKv",
    "cumSeqLensQ",
    "cumSeqLensKv",
    "pageIdx",
    "outputScale",
    "scaleSoftmaxLog2",
    "kvSfScale",
    "oSfScale",
    "scratch",
    "softmaxStats",
    "lse",
    "sinks",
    "o",
    "oSf",
)

# Data_type (include/flashinfer/trtllm/common.h) and element bits.
DATA_TYPE = {"fp16": 0, "bf16": 1, "fp32": 2, "e4m3": 5, "e2m1": 7}
BITS = {"fp16": 16, "bf16": 16, "fp32": 32, "e4m3": 8, "e2m1": 4}
# CUtensorMapDataType / CUtensorMapSwizzle (cuda.h).
TMA_UINT8, TMA_FLOAT16, TMA_BFLOAT16, TMA_16U4_ALIGN16B = 0, 6, 9, 14
SWIZZLE_NONE, SWIZZLE_32B, SWIZZLE_64B, SWIZZLE_128B = 0, 1, 2, 3
L2_PROMOTION_128B = 2

# QkvLayout, TrtllmGenAttentionMaskType, FmhaKernelType, TileScheduler,
# MultiCtasKvMode (fmhaRunnerParams.h).
SEPARATE_QKV, PACKED_QKV, PAGED_KV, CONTIGUOUS_KV = 0, 1, 2, 3
DENSE, CAUSAL, SLIDING, CUSTOM = 0, 1, 2, 3
CONTEXT, GENERATION, SWAPS_AB, KEEPS_AB = 0, 1, 2, 3
STATIC, PERSISTENT = 0, 1
# CUclusterSchedulingPolicy (cuda.h).
CLUSTER_POLICY_DEFAULT, CLUSTER_POLICY_SPREAD = 0, 1
MCTA_DISABLED, MCTA_GMEM, MCTA_GMEM_SEPARATE, MCTA_CGA = 0, 1, 2, 3

# Columns of a kernel row in the variant index.
META_COLUMNS = (
    "name",
    "sm",
    "dtq",
    "dtkv",
    "dto",
    "tileQ",
    "tileKv",
    "stepQ",
    "stepKv",
    "hdPerCtaV",
    "hdQk",
    "hdV",
    "smem",
    "threads",
    "layout",
    "tpp",
    "mask",
    "ktype",
    "sched",
    "mcta",
    "groupsHeadsQ",
    "groupsTokensHeadsQ",
    "reuseK",
    "twoCta",
    "sparse",
    "skips",
)

# Encodes one CUtensorMap from cuTensorMapEncodeTiled's arguments.
TensorMapEncoder = Callable[..., bytes]

# Runner mMultiProcessorCount of every case: the SM count of the B200 these
# kernels are selected and validated for (FlashInfer passes
# get_device_sm_count). Kernel selection (multi-CTA KV split, CGA fallback,
# GQA tile heuristics), the grid and the multi-CTA scratch size all depend on
# it; the probe uses the same value, and B200 cluster occupancy as measured.
NUM_SMS = 148
E4M3_MAX = 448.0
NVFP4_O_SCALE = 300.0  # FlashInfer's test choice for the NVFP4 output scale
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# RMS of an NVFP4 KV element of get_inputs (uniform codes, scales 0.25-1).
RMS_FP4_KV = math.sqrt(sum(v * v for v in E2M1_VALUES) / 8 * 0.46875)
# Skip-softmax data: logit levels of beacon, high-tile and low-tile keys.
BEACON_LEVEL, LOW_LEVEL = 6.0, -30.0


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
    """The probe's deterministic CUtensorMap stand-in (its interposed
    ``cuTensorMapEncodeTiled``); for tests and fixtures only."""
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


def f32(value: float) -> float:
    """``value`` rounded to IEEE single precision (a C ``float``)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def sentinel(role: str) -> int:
    return SENTINEL_STRIDE * (ROLES.index(role) + 1)


# -- index -------------------------------------------------------------------


@functools.lru_cache(None)
def load_index() -> dict[str, Any]:
    if not INDEX.is_file():
        return {"layout": None, "kernels": [], "cases": {}}
    return json.loads(INDEX.read_text())


def meta_of(row: Sequence[Any]) -> dict[str, Any]:
    return dict(zip(META_COLUMNS, row))


def workload_name(kernel: str) -> str:
    """``fmhaSm100fKernel_<config>`` -> ``fmha_trtllm_<config>``."""
    return "fmha_trtllm_" + kernel.split("_", 1)[1]


def is_mla_gen(r: dict[str, Any]) -> bool:
    """isMlaGenKernel(runner params)."""
    return r["hdqk"] == 576 and r["hdv"] == 512


# -- case layout ----------------------------------------------------------------
#
# Case fields beyond shapes (all optional, default off): ``packed`` (a
# SeparateQkv kernel served one packed QKV tensor, the generic runner's
# PackedQkv layout; same kernel code), ``kv_layout`` ("NHD": the cache is
# [pages, 2, P, Hkv, D] and the kernel gets the transposed view, as
# trtllm_batch_*_with_kv_cache(kv_layout="NHD")), ``shared_idx`` (False:
# TRT-LLM page tables [B, 2, maxPages] over an interleaved K/V page pool),
# ``q_noncontig`` (queries are [..., :D] of a [tokens, H, 2D] tensor),
# ``sinks`` (one float32 sink logit per query head), ``device_scales``
# (bmm1/bmm2 scales as device tensors), ``sf_start``/``sf_rows`` (NVFP4
# output scale-factor offset and buffer rows), ``skip_data`` (skip-softmax
# tile pattern), ``prefix_pages`` (requests 0 and 1 share their first pages),
# ``kv_tuple`` (separate K and V cache tensors, ``kv_cache=(k, v)``), and
# ``max_q``/``max_kv`` (the maxima the caller passes when they exceed the
# lengths', as FlashInfer's tests do).


def case_layout(m: dict[str, Any], case: dict[str, Any]) -> int:
    """QkvLayout the runner sees (``packed`` serves a SeparateQkv kernel a
    packed QKV tensor)."""
    return PACKED_QKV if case.get("packed") else m["layout"]


def kv_store_dim(m: dict[str, Any]) -> int:
    """Stored elements per paged K/V row (bytes for FP4: two per byte)."""
    d = max(m["hdQk"], m["hdV"])
    return d // 2 if m["dtkv"] == "e2m1" else d


def kv_cache_shape(m: dict[str, Any], case: dict[str, Any]) -> tuple[int, ...]:
    """Paged KV cache allocation (FlashInfer's layouts).

    Separate K/V caches are the two halves of ``[pages, 2, Hkv, P, D]``
    (HND) or ``[pages, 2, P, Hkv, D]`` (NHD) (``kv_cache.unbind(1)``); with
    separate page indices the same storage is the interleaved pool
    ``[2 * pages, ...]``. MLA uses one ``[pages, 1, P, Dqk]`` cache for both K
    and V. FP4 caches hold two elements per byte.
    """
    pages, page = case["num_pages"], case["page_size"]
    d = kv_store_dim(m)
    if case["shared_kv"]:
        return (pages, 1, page, d)
    rows = (
        (page, case["hkv"]) if case.get("kv_layout") == "NHD" else (case["hkv"], page)
    )
    if case.get("kv_tuple"):
        return (2, pages, *rows, d)
    return (pages, 2, *rows, d)


def kv_strides(m: dict[str, Any], case: dict[str, Any]) -> tuple[int, int, int]:
    """(keys, heads, batch) strides of the K/V views in elements, as the
    launcher reads them from the HND view (``* 2`` for FP4 storage)."""
    page, hkv = case["page_size"], case["hkv"]
    d = max(m["hdQk"], m["hdV"])
    if case["shared_kv"]:
        return d, page * d, page * d
    separate = case.get("kv_tuple") or not case.get("shared_idx", True)
    pool_factor = 1 if separate else 2
    if case.get("kv_layout") == "NHD":
        return hkv * d, d, pool_factor * page * hkv * d
    return d, page * d, pool_factor * hkv * page * d


# -- runner parameters (TllmGenFmhaRunnerParams) ------------------------------


def _i32(v: int) -> int:
    """C ``int32_t`` wrap-around."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= 1 << 31 else v


def runner_params(m: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """TllmGenFmhaRunnerParams of ``case`` as the FlashInfer launcher fills
    them (pointers as role names; strides in elements)."""
    layout, hq, hkv = case_layout(m, case), case["hq"], case["hkv"]
    q_lens, kv_lens = case["q_lens"], case["kv_lens"]
    hd_qk, hd_v = m["hdQk"], m["hdV"]
    fp4_kv = m["dtkv"] == "e2m1"
    run = case["runner"]
    device_scales = bool(case.get("device_scales"))
    r: dict[str, Any] = {
        "dtq": m["dtq"],
        "dtkv": m["dtkv"],
        "dto": m["dto"],
        "layout": layout,
        "mask": run["mask"],
        "ktype": run["ktype"],
        "sched": run["sched"],
        "mcta": run["mcta"],
        "hdqk": hd_qk,
        "hdv": hd_v,
        "hq": hq,
        "hkv": hkv,
        "batch": len(q_lens),
        "max_q": case.get("max_q", max(q_lens)),
        "max_kv": case.get("max_kv", max(kv_lens)),
        "max_cache_kv": 0,
        "sum_q": sum(q_lens),
        "sum_kv": 0,
        "window": INT_MAX if case["window_left"] < 0 else case["window_left"] + 1,
        "chunk": INT_MAX,
        "tpp": 0,
        "max_pages": 0,
        "pool": 0,
        "sms": NUM_SMS,
        "skips": int(case["skip_thr"] != 0.0),
        "skip_thr": case["skip_thr"],
        "sparse_topk": case.get("topk", 0),
        "shared_idx": int(case.get("shared_idx", True)),
        "q_st": 0,
        "q_sh": 0,
        "k_skv": 0,
        "k_sh": 0,
        "k_sb": 0,
        "v_skv": 0,
        "v_sh": 0,
        "v_sb": 0,
        "ksf_sh": 0,
        "ksf_sb": 0,
        "vsf_sh": 0,
        "vsf_sb": 0,
        # A device bmm1 scale replaces the value 1.0 (launcher default).
        "scale_log2": f32((1.0 if device_scales else case["bmm1_scale"]) * LOG2E),
        "out_scale": f32(1.0 if device_scales else case["bmm2_scale"]),
        "sf_scale_kv": 0.0,
        "sf_scale_o": 0.0,
        "sf_start": case.get("sf_start", 0),
        "pdl": 1,
        # Probe only: max active clusters (-1: as measured on a 148-SM B200).
        "occupancy": case.get("occupancy", -1),
        "ptrs": ["o"],
    }
    ptrs = r["ptrs"]
    if device_scales:
        ptrs += ["scaleSoftmaxLog2", "outputScale"]
    if case.get("sinks"):
        ptrs += ["sinks"]
    if run["ktype"] != CONTEXT:
        ptrs += ["counter", "scratch"]
    if layout == PAGED_KV:
        # trtllm_paged_attention_launcher
        page = case["page_size"]
        d = max(hd_qk, hd_v)
        q_factor = 2 if case.get("q_noncontig") else 1
        r.update(
            tpp=page,
            max_pages=case["max_pages"],
            pool=case["num_pages"] if case["shared_kv"] else 2 * case["num_pages"],
            q_st=hq * hd_qk * q_factor,
            q_sh=hd_qk * q_factor,
            sf_scale_kv=1.0,
            sf_scale_o=-1.0,
        )
        skv, sh, sb = kv_strides(m, case)
        for side in "kv":
            r[f"{side}_skv"], r[f"{side}_sh"], r[f"{side}_sb"] = skv, sh, sb
        ptrs += ["q", "k", "v", "pageIdx", "seqLensKv"]
        if fp4_kv:
            r.update(
                ksf_sh=page * d // 16,
                ksf_sb=hkv * page * d // 16,
                vsf_sh=page * d // 16,
                vsf_sb=hkv * page * d // 16,
            )
            ptrs += ["kSf", "vSf"]
        if run["ktype"] == CONTEXT:
            ptrs += ["cumSeqLensQ", "cumSeqLensKv"]
        elif case.get("cum_q"):
            ptrs += ["cumSeqLensQ"]
    elif layout == SEPARATE_QKV:
        # trtllm_ragged_attention_launcher (always allocates softmax stats);
        # the int64 batch strides (key.numel()) wrap in the int runner field.
        n_kv = sum(kv_lens)
        r.update(
            sum_kv=n_kv,
            q_st=0,
            q_sh=0,
            k_skv=hkv * hd_qk,
            k_sh=hd_qk,
            k_sb=_i32(n_kv * hkv * hd_qk),
            v_skv=hkv * hd_v,
            v_sh=hd_v,
            v_sb=_i32(n_kv * hkv * hd_v),
            sf_scale_o=-1.0,
        )
        ptrs += [
            "q",
            "k",
            "v",
            "seqLensKv",
            "cumSeqLensQ",
            "cumSeqLensKv",
            "counter",
            "softmaxStats",
            "scratch",
        ]
    elif layout == PACKED_QKV:
        # Generic runner: one [tokens, (Hq + 2 Hkv) * D] tensor.
        width = (hq + 2 * hkv) * hd_qk
        r.update(
            sum_kv=sum(kv_lens),
            k_skv=width,
            k_sh=hd_qk,
            v_skv=width,
            v_sh=hd_v,
            sf_scale_o=-1.0,
        )
        ptrs += ["qkv", "seqLensKv", "cumSeqLensQ", "cumSeqLensKv"]
    else:
        raise ValueError(f"unsupported QKV layout {layout}")
    if m["dto"] == "e2m1":
        # Every launcher passes out_scale_factor and o_sf_scale for NVFP4 output.
        r["sf_scale_o"] = NVFP4_O_SCALE
        ptrs += ["oSf"]
    r["ptrs"] = sorted(set(ptrs), key=ROLES.index)
    return r


def _num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(int(value))


def probe_line(case_id: str, r: dict[str, Any]) -> str:
    """One stdin line of the compile-time probe for runner params ``r``."""
    tokens = [f"id={case_id}"]
    for key, value in r.items():
        if key == "ptrs":
            tokens.append("ptrs=" + ",".join(value))
        elif key in ("dtq", "dtkv", "dto"):
            tokens.append(f"{key}={value}")
        else:
            tokens.append(f"{key}={_num(value)}")
    return " ".join(tokens)


# -- launch configuration (computeCtaAndClusterConfig) ------------------------


def launch_config(m: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """Grid, cluster and CTA counts upstream computes for kernel ``m``."""
    is_context = r["ktype"] == CONTEXT
    hpk = r["hq"] // r["hkv"]
    ctas_q = ceil_div(r["max_q"], m["stepQ"])
    if r["max_q"] > 1 and not is_context:
        if not m["groupsTokensHeadsQ"]:
            ctas_q = r["max_q"]
        else:
            tokens_per_cta = max(1, m["stepQ"] // hpk)
            ctas_q = ceil_div(r["max_q"], tokens_per_cta)
    heads_per_cta = min(hpk, m["stepQ"]) if m["groupsHeadsQ"] else 1
    ctas_heads = r["hq"] // heads_per_cta
    if heads_per_cta * ctas_heads != r["hq"]:
        raise ValueError("The numHeadsQ/numHeadsKv is not supported.")
    if m["hdV"] % m["hdPerCtaV"]:
        raise ValueError("The headDimPerCtaV is not supported.")
    ctas_head_dim = m["hdV"] // m["hdPerCtaV"]
    x, y, z = ctas_q, ctas_heads * ctas_head_dim, r["batch"]
    if is_mla_gen(r) and m["twoCta"]:
        if ctas_heads != 2 or ctas_head_dim != 2:
            raise ValueError("Internal error: numCtasPerHeadDim should be 2.")
        x *= 2
        y //= ctas_heads * ctas_head_dim
    ctas_kv = 1
    if m["mcta"] != MCTA_DISABLED:
        window = r["max_kv"]
        if r["sparse_topk"] > 0:
            window = min(r["max_kv"], r["sparse_topk"])
        if m["mask"] == SLIDING:
            if r["max_kv"] > r["window"]:
                window = min(r["max_kv"], r["window"] + m["stepKv"] - 1)
            else:
                window = min(r["max_kv"], r["chunk"])
        max_ctas_kv = (window + 2 * m["stepKv"] - 1) // (2 * m["stepKv"])
        ctas_kv = min(max_ctas_kv, max(1, r["sms"] // (x * y * z)))
        x *= ctas_kv
    cluster = 2 if m["twoCta"] else 1
    if m["mcta"] == MCTA_CGA:
        cluster *= ctas_kv
    return {
        "grid": [x, y, z],
        "cluster": [cluster, 1, 1],
        "max_ctas_q": ctas_q,
        "max_ctas_kv": ctas_kv,
    }


# -- KernelParams (setKernelParams) -------------------------------------------


def _tma(
    r: dict[str, Any],
    dtype: str,
    shapes: list[int],
    strides: list[int],
    tile: list[int],
    address: int,
    encode: TensorMapEncoder,
    swizzled: bool = True,
    unpack4b: bool = False,
) -> bytes:
    """KernelParams::buildNdTmaDescriptor."""
    if dtype == "e2m1":
        fmt = TMA_16U4_ALIGN16B if unpack4b else TMA_UINT8
    elif dtype == "e4m3":
        fmt = TMA_UINT8
    elif dtype == "fp16":
        fmt = TMA_FLOAT16
    elif dtype == "bf16":
        fmt = TMA_BFLOAT16
    else:
        raise ValueError(f"Unexpected dtype {dtype}")
    leading = tile[0] * BITS[dtype] // 8
    if not swizzled:
        swizzle = SWIZZLE_NONE
    elif fmt == TMA_16U4_ALIGN16B or leading % 128 == 0:
        swizzle = SWIZZLE_128B
    elif leading % 64 == 0:
        swizzle = SWIZZLE_64B
    elif leading % 32 == 0:
        swizzle = SWIZZLE_32B
    else:
        raise ValueError(f"Unexpected numBytesInLeadingDim {leading}")
    if address & 15:
        raise ValueError("TMA address must be 16-byte aligned")
    if not 2 <= len(shapes) <= 5 or any(not 1 <= s <= 1 << 32 for s in shapes):
        raise ValueError(f"invalid TMA shape {shapes}")
    if strides[0] != 1:
        raise ValueError("TMA stride 0 must be 1")
    elt = max(BITS[dtype], 8) // 8
    stride_bytes = [(s * elt) & 0xFFFFFFFFFFFFFFFF for s in strides[1:]]
    # cuTensorMapEncodeTiled reads ``rank`` box entries (the sparse MLA K
    # descriptor is rank 2 with a 4-entry tile).
    return encode(
        fmt,
        address,
        [s & 0xFFFFFFFFFFFFFFFF for s in shapes],
        stride_bytes,
        tile[: len(shapes)],
        [1] * len(shapes),
        0,
        swizzle,
        L2_PROMOTION_128B,
        0,
    )


def _u64(v: int) -> int:
    return v & 0xFFFFFFFFFFFFFFFF


def _cdiv(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def kernel_param_fields(
    m: dict[str, Any],
    r: dict[str, Any],
    pointers: dict[str, int],
    launch: dict[str, Any],
    encode: TensorMapEncoder,
) -> dict[str, Any]:
    """KernelParams::setKernelParams(options, kernelMeta, maxNumCtasQ, maxNumCtasKv)."""
    ptr = {role: pointers.get(role, 0) for role in ROLES}
    layout = r["layout"]
    paged, packed = layout == PAGED_KV, layout == PACKED_QKV
    contiguous = layout == CONTIGUOUS_KV
    dtq, dtkv = m["dtq"], m["dtkv"]
    bits_kv = BITS[dtkv]
    hd_qk, hd_v = r["hdqk"], r["hdv"]
    hq, hkv = r["hq"], r["hkv"]
    hpk = hq // hkv
    fields: dict[str, Any] = {}

    # getDevicePtrs
    q_ptr, k_ptr, v_ptr = ptr["q"], ptr["k"], ptr["v"]
    if packed:
        q_ptr = ptr["qkv"]
        k_ptr = ptr["qkv"] + hq * hd_qk * bits_kv // 8
        v_ptr = ptr["qkv"] + (hq + hkv) * hd_qk * bits_kv // 8
    elif contiguous:
        raise ValueError("ContiguousKv is not served")
    max_hd_kv = max(hd_qk, hd_v)
    pool = 0
    if paged:
        pool = r["pool"] if r["pool"] else r["max_pages"] * 2 * r["batch"]
    fields["mNumPagesInMemPool"] = pool

    # Q
    elts_128b_q = 128 * 8 // BITS[dtq]
    clamped_q = min(elts_128b_q, hd_qk)
    grouped = 1
    if m["groupsHeadsQ"]:
        grouped = min(m["tileQ"], hpk)
    heads = hq // grouped if m["groupsHeadsQ"] else hq
    if heads * grouped != hq:
        raise ValueError("internal error")
    shape_q = [hd_qk, grouped, heads, r["sum_q"]]
    stride_tokens = r["q_st"]
    if stride_tokens == 0:
        hidden_qkv = hq * hd_qk
        if packed:
            hidden_qkv += hkv * (hd_qk + hd_v)
        stride_tokens = hidden_qkv
    stride_heads = r["q_sh"] or hd_qk
    stride_grouped = r["q_sh"] or hd_qk
    if m["groupsHeadsQ"]:
        stride_heads *= grouped
    stride_q = [1, stride_grouped, stride_heads, stride_tokens]
    tile_q = [clamped_q, 1, 1, m["tileQ"]]
    tokens_per_cta_q = m["tileQ"]
    if m["groupsHeadsQ"]:
        if m["groupsTokensHeadsQ"]:
            tokens_per_cta_q = tokens_per_cta_q // grouped
        else:
            grouped = m["tileQ"]
            tokens_per_cta_q = 1
        tile_q = [clamped_q, grouped, 1, tokens_per_cta_q]
    fields["tmaQ_"] = _tma(r, dtq, shape_q, stride_q, tile_q, q_ptr, encode)

    # K/V
    keys_per_tile = min(r["tpp"], m["tileKv"]) if paged else m["tileKv"]
    elts_128b_kv = 128 * 8 // bits_kv
    clamped_kv = min(elts_128b_kv, max_hd_kv, 128)
    transforms_kv = dtkv != dtq
    swaps_ab = m["ktype"] == SWAPS_AB
    tmem_kv = dtkv == "e2m1" and dtq == "e4m3" and max_hd_kv >= 128 and swaps_ab
    swizzle_kv = tmem_kv or not transforms_kv
    can_reshape = paged and hd_qk == hd_v and not swizzle_kv
    reshape = 1
    if can_reshape:
        reshape = max(
            1,
            min(
                128 // max_hd_kv,
                _cdiv(128, max_hd_kv * bits_kv // 8),
                keys_per_tile,
            ),
        )
    if paged:
        num_keys, batch_kv = r["tpp"], pool
    else:
        num_keys, batch_kv = r["sum_kv"], 1

    def shape_stride_kv(is_k: bool) -> tuple[list[int], list[int]]:
        side = "k" if is_k else "v"
        s_keys, s_heads, s_batch = r[f"{side}_skv"], r[f"{side}_sh"], r[f"{side}_sb"]
        if not paged and not contiguous and s_batch < 0:
            s_batch = 0
        head_dim = (hd_qk if is_k else hd_v) if not paged else max_hd_kv
        div = 2 if dtkv == "e2m1" else 1
        first = head_dim if tmem_kv else _cdiv(head_dim, div)
        shape = [first * reshape, _cdiv(num_keys, reshape), hkv, batch_kv]
        stride = [
            1,
            _u64(_cdiv(s_keys, div) * reshape),
            _u64(_cdiv(s_heads, div)),
            _u64(_cdiv(s_batch, div)),
        ]
        return [_u64(s) for s in shape], stride

    shape_k, stride_k = shape_stride_kv(True)
    shape_v, stride_v = shape_stride_kv(False)
    elts_div = 2 if dtkv == "e2m1" and not tmem_kv else 1
    tile_kv = [1] * len(shape_k)
    tile_kv[0] = clamped_kv // elts_div * reshape
    tile_kv[1] = keys_per_tile // reshape
    if r["sparse_topk"] > 0:
        shape_k = [hd_qk, INT_MAX]
        stride_k = [1, hd_qk]
        tile_kv[1] = 1
    fields["tmaK_"] = _tma(
        r, dtkv, shape_k, stride_k, tile_kv, k_ptr, encode, swizzle_kv, tmem_kv
    )
    fields["tmaV_"] = _tma(
        r, dtkv, shape_v, stride_v, tile_kv, v_ptr, encode, swizzle_kv, tmem_kv
    )
    if dtkv == "e2m1":
        if not paged:
            raise ValueError("The qkvLayout is not supported.")
        reshape_sf = min(128 // (max_hd_kv // 16), keys_per_tile)
        shape_sf = [
            max_hd_kv // 16 * reshape_sf,
            num_keys // reshape_sf,
            hkv,
            batch_kv,
        ]
        stride_sf = [1, max_hd_kv // 16 * reshape_sf, r["ksf_sh"], r["ksf_sb"]]
        tile_sf = [max_hd_kv // 16 * reshape_sf, keys_per_tile // reshape_sf, 1, 1]
        fields["tmaKSf_"] = _tma(
            r, "e4m3", shape_sf, stride_sf, tile_sf, ptr["kSf"], encode, False
        )
        fields["tmaVSf_"] = _tma(
            r, "e4m3", shape_sf, stride_sf, tile_sf, ptr["vSf"], encode, False
        )

    # O
    shape_o = [hd_v, r["sum_q"], hkv, hpk, 1]
    stride_o = [1, hq * hd_v, hd_v, hkv * hd_v, 0]
    tile_o = [1] * 5
    tile_o[0] = clamped_q
    tile_o[1] = m["tileQ"]
    fields["tmaO_"] = _tma(r, dtq, shape_o, stride_o, tile_o, ptr["o"], encode)

    fields["ptrCumSeqLensQ"] = ptr["cumSeqLensQ"]
    fields["ptrCumSeqLensKv"] = ptr["cumSeqLensKv"]
    fields["ptrCustomMask"] = ptr["customMask"]
    fields["ptrCustomMaskOffsets"] = ptr["customMaskOffsets"]
    fields["ptrFirstSparseMaskOffsetsKv"] = ptr["firstSparse"]
    fields["ptrO"] = ptr["o"]
    fields["ptrSfO"] = ptr["oSf"]
    fields["ptrOutputScale"] = ptr["outputScale"]
    fields["ptrSeqLensKv"] = ptr["seqLensKv"]
    fields["ptrAttentionSinks"] = ptr["sinks"]
    partial_stats = r["sms"] * m["stepQ"]
    fields["ptrMultiCtasKvCounter"] = ptr["counter"]
    fields["ptrPartialStats"] = ptr["scratch"]
    fields["ptrPartialO"] = _u64(ptr["scratch"] + 8 * partial_stats)
    fields["ptrPageIdxKv"] = ptr["pageIdx"]
    fields["ptrScaleSoftmaxLog2"] = ptr["scaleSoftmaxLog2"]
    fields["ptrScaleSfKv"] = ptr["kvSfScale"]
    fields["ptrScaleSfO"] = ptr["oSfScale"]
    fields["mScaleSfO"] = r["sf_scale_o"]
    fields["mAttentionWindowSize"] = r["window"]
    fields["mChunkedAttentionSizeLog2"] = 0
    fields["mNumTokensPerPageLog2"] = int(math.log2(r["tpp"])) if paged else -1
    fields["mMaxSeqLenQ"] = r["max_q"]
    fields["mMaxSeqLenKv"] = r["max_kv"]
    fields["mMaxNumCtasQ"] = launch["max_ctas_q"]
    fields["mMaxNumCtasKv"] = launch["max_ctas_kv"]
    fields["mMaxNumPagesPerSeqKv"] = r["max_pages"]
    fields["mSumOfSeqLensQ"] = r["sum_q"]
    fields["mSumOfSeqLensKv"] = r["sum_kv"]
    fields["mBatchSize"] = r["batch"]
    fields["mNumHeadsQ"] = hq
    fields["mNumHeadsKv"] = hkv
    fields["mNumHeadsQPerKv"] = hpk
    fields["mNumHeadsQPerKvDivisor"] = fast_mod_div(hpk)
    fields["mNumHiddenEltsO"] = hq * hd_qk
    fields["mNumTokensPerCtaQ"] = tokens_per_cta_q
    fields["mOutputScale"] = r["out_scale"]
    fields["mScaleSoftmaxLog2"] = r["scale_log2"]
    fields["mSkipSoftmaxThresholdScaleFactor"] = r["skip_thr"]
    fields["mStartTokenIdxSfO"] = r["sf_start"]
    fields["mScaleSfKv"] = r["sf_scale_kv"]
    fields["ptrSoftmaxStats"] = ptr["softmaxStats"]
    fields["mSparseMlaTopK"] = r["sparse_topk"]
    fields["mUseBlockSparseAttention"] = False
    fields["mUsesSharedPagedKvIdx"] = bool(r["shared_idx"])
    return fields


def fast_mod_div(divisor: int) -> bytes:
    """FastModDivInt32(divisor): {mDivisor, mMultiplier, mAdd, mShift}."""
    shift = max(math.ceil(math.log2(divisor)) - 1, 0)
    multiplier = ceil_div(1 << (32 + shift), divisor) & 0xFFFFFFFF
    return struct.pack("<iIIi", divisor, multiplier, 0, shift)


_PACK = {"i32": "<i", "f32": "<f", "ptr": "<Q", "i64": "<q", "bool": "<?"}


def pack_kernel_params(layout: dict[str, Any], fields: dict[str, Any]) -> bytes:
    """The KernelParams bytes (zero-initialized like upstream's memset)."""
    out = bytearray(layout["size"])
    spec = layout["fields"]
    for name, value in fields.items():
        offset, size, kind = spec[name]
        if kind in ("tma", "fastmoddiv"):
            if len(value) != size:
                raise ValueError(f"{name}: {len(value)} bytes, expected {size}")
            out[offset : offset + size] = value
        else:
            struct.pack_into(_PACK[kind], out, offset, value)
    return bytes(out)


def build_params(
    m: dict[str, Any],
    r: dict[str, Any],
    pointers: dict[str, int],
    encode: TensorMapEncoder,
    layout: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """(KernelParams bytes, launch configuration); ``layout`` defaults to the
    variant index's (the compile stage passes the probe's)."""
    layout = layout or load_index()["layout"]
    launch = launch_config(m, r)
    fields = kernel_param_fields(m, r, pointers, launch, encode)
    return pack_kernel_params(layout, fields), launch


def check_param_layouts() -> bool:
    """Every recorded probe run: the Python port reproduces upstream's kernel
    selection inputs, launch configuration and KernelParams bytes."""
    index = load_index()
    fixtures = json.loads(FIXTURES.read_text())
    metas = {row[0]: meta_of(row) for row in index["kernels"]}
    pointers = {role: sentinel(role) for role in ROLES}
    checked = 0
    for kernel, records in fixtures["records"].items():
        m = metas[kernel]
        cases = index["cases"][kernel]
        for record in records:
            case = cases[record["case"]]
            r = runner_params(m, case)
            ptrs = {role: pointers[role] for role in r["ptrs"]}
            data, launch = build_params(m, r, ptrs, fake_encode)
            got = {
                "grid": launch["grid"],
                "cluster": launch["cluster"],
                "block": [m["threads"], 1, 1],
                "smem": m["smem"],
                "params_sha256": hashlib.sha256(data).hexdigest(),
            }
            want = {key: record[key] for key in got}
            if got != want:
                raise AssertionError(f"{kernel} case {record['case']}: {got} != {want}")
            checked += 1
    for kernel, raw in fixtures["examples"].items():
        record = fixtures["records"][kernel][0]
        if hashlib.sha256(bytes.fromhex(raw)).hexdigest() != record["params_sha256"]:
            raise AssertionError(f"{kernel}: example bytes disagree with their hash")
    return checked > 0


# -- inputs and reference -----------------------------------------------------

TORCH_DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
}
# Elements generated at a time when filling large inputs (bounds the float32
# temporaries of multi-GB KV caches).
FILL_CHUNK = 1 << 26
# Logit elements the reference materializes at a time.
REF_CHUNK = 1 << 27


def _e2m1_table(device: torch.device) -> torch.Tensor:
    values = list(E2M1_VALUES) + [-v for v in E2M1_VALUES]
    return torch.tensor(values, dtype=torch.float32, device=device)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """uint8 pairs -> float32 values (low nibble first, as cast_from_fp4)."""
    low, high = packed & 0xF, packed >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2).long()
    return _e2m1_table(packed.device)[codes]


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """float32 values already on the E2M1 grid -> codes (round to nearest)."""
    table = _e2m1_table(x.device)[:8]
    mag = (x.abs().unsqueeze(-1) - table).abs().argmin(-1)
    return mag + 8 * (x < 0).long()


def unswizzle_v_scales(sf: torch.Tensor) -> torch.Tensor:
    """Inverse of nvfp4_quantize_paged_kv_cache's V-scale swizzle (HND)."""
    pages, heads, page, dim = sf.shape
    tmp = sf.reshape(pages, heads, page // 4, 4, dim // 4, 4)
    # forward: [.., t//4, t%4, s//4, s%4] -> permute(0,1,2,4,5,3)
    return tmp.permute(0, 1, 2, 5, 3, 4).reshape(pages, heads, page, dim)


def swizzle_v_scales(sf: torch.Tensor) -> torch.Tensor:
    pages, heads, page, dim = sf.shape
    tmp = sf.reshape(pages, heads, page // 4, 4, 4, dim // 4)
    return tmp.permute(0, 1, 2, 4, 5, 3).reshape(pages, heads, page, dim).contiguous()


def recover_swizzled_scales(
    scale: torch.Tensor, m: int, n: int, start: int = 0
) -> torch.Tensor:
    """FlashInfer's 128x4-swizzled NVFP4 scale layout -> rows
    ``[start, start + m)`` of the linear ``[rows, n // 16]`` scales."""
    full_m = scale.shape[0]
    scale_n = n // 16
    rounded_n = ceil_div(scale_n, 4) * 4
    tmp = scale.reshape(1, full_m // 128, rounded_n // 4, 32, 4, 4)
    tmp = tmp.permute(0, 1, 4, 3, 2, 5)
    return tmp.reshape(full_m, rounded_n).float()[start : start + m, :scale_n]


# Relative rounding error of E4M3 (3 mantissa bits, round to nearest).
E4M3_REL_ROUNDING = 2.0**-4


def p_rounding_bound(abs_v: torch.Tensor) -> torch.Tensor:
    """Per-element error bound from the E4M3 rounding of the softmax
    probabilities that FP8-query kernels apply before P V: each weight p_k
    moves by at most 2^-4 p_k, so o_d moves by at most 2^-4 sum_k p_k |v_kd|
    (``abs_v``, scaled like the output). Measured on B200: at the elements
    beyond the flat tolerance, the kernels match a reference with P rounded to
    E4M3 (pre-scaled by 448, normalized by the exact sum) within 1e-4."""
    return E4M3_REL_ROUNDING * abs_v


def e4m3_ulp(magnitude: torch.Tensor) -> torch.Tensor:
    """Spacing of the E4M3 grid at ``magnitude`` (3 mantissa bits; below the
    smallest normal 2^-6 the subnormal spacing 2^-9)."""
    exponent = torch.floor(torch.log2(magnitude.clamp(min=2.0**-6)))
    return torch.exp2(exponent - 3)


def e2m1_half_gap(units: torch.Tensor) -> torch.Tensor:
    """Half the E2M1 grid spacing at ``units`` (|x| / block step): 0.25 below
    2, 0.5 below 4, 1 above (which also covers saturation at 6 after a scale
    rounded down by at most 1/16)."""
    return torch.where(units < 2, 0.25, torch.where(units < 4, 0.5, 1.0))


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """float32 values on the E2M1 grid -> uint8 pairs (low nibble first, the
    inverse of ``unpack_e2m1``)."""
    codes = quantize_e2m1(values).to(torch.uint8)
    pairs = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 2, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


def swizzle_scales(
    scales: torch.Tensor, rows: int, cols: int, start: int = 0
) -> torch.Tensor:
    """[m, n // 16] E4M3-representable scales at rows ``start..`` -> FlashInfer's
    128x4-swizzled uint8 layout of shape [rows, cols] (the inverse of
    ``recover_swizzled_scales``; padding zero)."""
    linear = torch.zeros((rows, cols), dtype=torch.float32, device=scales.device)
    linear[start : start + scales.shape[0], : scales.shape[1]] = scales
    raw = linear.to(torch.float8_e4m3fn).view(torch.uint8)
    tmp = raw.reshape(1, rows // 128, 4, 32, cols // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return tmp.reshape(rows, cols).contiguous()


def nvfp4_scale_shape(p: dict[str, Any], hq: int, hd_v: int) -> tuple[int, int]:
    """Rows/columns of the swizzled output scale-factor buffer."""
    sum_q = sum(p["q_lens"])
    rows = p.get("sf_rows") or ceil_div(sum_q, 128) * 128
    return rows, ceil_div(hq * hd_v // 16, 4) * 4


def counter_size(m: dict[str, Any], p: dict[str, Any]) -> int:
    """Length of the zeroed multi-CTA KV counter buffer ``run`` allocates."""
    r = runner_params(m, p)
    grid = launch_config(m, r)["grid"]
    ctas = grid[0] * grid[1] * grid[2]
    return max(1024, 2 * ctas, p["hq"] * r["batch"] * r["max_q"])


def ref_fp4_quant(x: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, ...]:
    """tests/test_helpers/utils_fp4.ref_fp4_quant (block 16, E4M3 scales)."""
    blocks = x.reshape(*x.shape[:-1], x.shape[-1] // 16, 16).float()
    vec_max = blocks.abs().amax(-1, keepdim=True)
    scale = (global_scale * (vec_max / 6.0)).to(torch.float8_e4m3fn).float()
    inv = torch.where(scale == 0, torch.zeros_like(scale), global_scale / scale)
    scaled = (blocks * inv).clamp(-6.0, 6.0).reshape(x.shape)
    grid = _e2m1_table(x.device)[quantize_e2m1(scaled)]
    return grid, scale.squeeze(-1)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    causal: bool,
    window_left: int,
    sinks: torch.Tensor | None = None,
    allowed: torch.Tensor | None = None,
    with_abs_v: bool = False,
) -> tuple[torch.Tensor, ...]:
    """One request: q [Lq, Hq, D], k/v [Lk, Hkv, D] (float32) -> (o, lse2),
    plus ``P |V|`` (the softmax weights applied to |V|) for ``with_abs_v``.

    Queries are the last Lq of the Lk positions (bottom-right causal, as
    FlashInfer); ``window_left`` keeps keys j >= pos - window_left;
    ``sinks`` [Hq] adds exp(sink) to each row's softmax denominator (in
    logit units, FlashInfer's sink_attention_unified); ``allowed`` [Lq, Lk]
    further restricts keys (sparse MLA top-k). ``lse2`` is the base-2
    log-sum-exp over the keys (without the sink). Queries are processed in
    chunks so the logits stay below ``REF_CHUNK`` elements; grouped heads
    share their KV head without copying it.
    """
    lq, hq, d = q.shape
    lk, hkv, dv = v.shape
    group = hq // hkv
    qg = q.reshape(lq, hkv, group, d)
    step = max(1, REF_CHUNK // max(1, hq * lk))
    key = torch.arange(lk, device=q.device).unsqueeze(0)
    outs, lses, abs_outs = [], [], []
    for start in range(0, lq, step):
        stop = min(lq, start + step)
        logits = torch.einsum("qhgd,khd->hgqk", qg[start:stop], k) * scale
        pos = torch.arange(start, stop, device=q.device).unsqueeze(1) + (lk - lq)
        keep = torch.ones(stop - start, lk, dtype=torch.bool, device=q.device)
        if causal:
            keep &= key <= pos
        if window_left >= 0:
            keep &= key >= pos - window_left
        if allowed is not None:
            keep &= allowed[start:stop]
        logits = logits.masked_fill(~keep, float("-inf"))
        lse = torch.logsumexp(logits, dim=-1)  # [hkv, g, q]
        total = lse
        if sinks is not None:
            sink = sinks.reshape(hkv, group, 1).expand_as(lse)
            total = torch.logaddexp(lse, sink)
        probs = torch.exp(logits - total.unsqueeze(-1))
        out = torch.einsum("hgqk,khd->qhgd", probs, v).reshape(stop - start, hq, dv)
        outs.append(out)
        lses.append((lse * LOG2E).reshape(hq, stop - start).transpose(0, 1))
        if with_abs_v:
            abs_outs.append(
                torch.einsum("hgqk,khd->qhgd", probs, v.abs()).reshape(
                    stop - start, hq, dv
                )
            )
        del logits, probs
    if with_abs_v:
        return torch.cat(outs), torch.cat(lses), torch.cat(abs_outs)
    return torch.cat(outs), torch.cat(lses)


def logit_spread(m: dict[str, Any], p: dict[str, Any]) -> float:
    """Target logit standard deviation of the peaked inputs.

    16-bit P: 2.5 (a few keys hold most of each row's weight). FP8 queries
    make the kernel quantize P to E4M3; there the spread shrinks with the KV
    length so that the many small probabilities of long rows stay
    representable (2.0 up to 2k keys, 1.5 up to 8k, 1.0 beyond).
    """
    if m["dtq"] != "e4m3":
        return 2.5
    longest = max(p["kv_lens"])
    return 2.0 if longest <= 2048 else 1.5 if longest <= 8192 else 1.0


def v_offsets(hkv: int, device: torch.device) -> torch.Tensor:
    """Per-KV-head V offset (|offset| in [0.5, 1.5], alternating signs):
    outputs are O(1) even where attention is diffuse and differ between
    heads."""
    if hkv == 1:
        return torch.full((1,), 0.75, device=device)
    signs = torch.tensor([(-1.0) ** h for h in range(hkv)], device=device)
    return torch.linspace(0.5, 1.5, hkv, device=device) * signs


def _fill(
    out: torch.Tensor, make: Callable[[int, int], torch.Tensor], dim0: int = 0
) -> None:
    """``out[i:j] = make(i, j)`` in chunks of dim 0 (float32 temporaries of
    at most ``FILL_CHUNK`` elements)."""
    n = out.shape[dim0]
    per = max(1, out[0:1].numel())
    step = max(1, FILL_CHUNK // per)
    for start in range(0, n, step):
        stop = min(n, start + step)
        out[start:stop] = make(start, stop).to(out.dtype)


def skip_levels(
    positions: torch.Tensor, tile: int, gen: torch.Generator
) -> torch.Tensor:
    """Logit level of the key at each sequence position (skip-softmax data):
    even KV tiles are high (their first and last keys beacons at
    ``BEACON_LEVEL``, the rest in [-2, 2]), odd tiles low (``LOW_LEVEL``);
    unused slots (-1) low. A causal row sees its tile's first key and a
    window of >= ``tile - 1`` keys the last key of the tile it cuts, so every
    partly visible high tile shows a beacon."""
    high = (positions >= 0) & ((positions // tile) % 2 == 0)
    level = torch.rand(positions.shape, generator=gen, device=positions.device) * 4 - 2
    level = torch.where(high, level, torch.full_like(level, LOW_LEVEL))
    edge = (positions % tile == 0) | (positions % tile == tile - 1)
    return torch.where(high & edge, torch.full_like(level, BEACON_LEVEL), level)


# -- the workload ---------------------------------------------------------------


class TrtllmFmha(Workload):
    """One trtllm-gen FMHA kernel; ``meta`` is its meta-info row."""

    package: ClassVar[str | None] = "fmha"
    meta: ClassVar[dict[str, Any]] = {}
    case_params: ClassVar[list[dict[str, Any]]] = []

    # -- cases --------------------------------------------------------------

    def get_cases(self) -> list[CaseSpec]:
        cases = []
        for i, params in enumerate(self.case_params):
            suite = params["suite"]
            name = params["label"]
            if suite != "smoke":
                name = "throughput_" + name
            source = dict(params.get("source") or {})
            if not source:
                source = (
                    {"kind": "smoke", "reason": "coverage of this kernel's paths"}
                    if suite == "smoke"
                    else {
                        "kind": "synthetic_stress",
                        "reason": "trtllm-gen FMHA shape that upstream dispatch "
                        "sends to this kernel (verified by the compile-time probe)",
                    }
                )
            seed = params.get("seed", (100 if suite == "smoke" else 1111) + i)
            cases.append(CaseSpec(name, params, seed, suite, source))
        return cases

    def get_inputs(self, case: CaseSpec) -> tuple:
        m, p = self.meta, case.params
        gen = self.generator(case)
        dev = self.device
        hq, hkv = p["hq"], p["hkv"]
        q_lens, kv_lens = p["q_lens"], p["kv_lens"]
        sum_q, n_kv = sum(q_lens), sum(kv_lens)
        dqk, dv = m["hdQk"], m["hdV"]
        layout = case_layout(m, p)
        skip = bool(p.get("skip_data"))
        offsets = v_offsets(hkv, dev)
        bmm1 = p["bmm1_scale"]

        def ints(values: Sequence[int]) -> torch.Tensor:
            return torch.tensor(list(values), dtype=torch.int32, device=dev)

        def cumsum(values: Sequence[int]) -> list[int]:
            total, out = 0, [0]
            for v in values:
                total += v
                out.append(total)
            return out

        def randn(*shape: int) -> torch.Tensor:
            return torch.randn(shape, generator=gen, device=dev)

        def cast(x: torch.Tensor, dtype: str) -> torch.Tensor:
            if dtype == "e4m3":
                return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            return x.to(TORCH_DTYPE[dtype])

        t: dict[str, Any] = {}
        t["seq_lens"] = ints(kv_lens)
        t["cum_q"] = ints(cumsum(q_lens))
        t["cum_kv"] = ints(cumsum(kv_lens))
        rms_k = RMS_FP4_KV if m["dtkv"] == "e2m1" else 1.0
        if skip:
            # Level coordinate d-1: q = 1/bmm1 there, k = the key's level;
            # the other coordinates add N(0, 1) logits.
            noise = 1.0 / (bmm1 * math.sqrt(dqk - 1) * rms_k)
        else:
            noise = logit_spread(m, p) / (bmm1 * math.sqrt(dqk) * rms_k)

        def make_q(n: int) -> torch.Tensor:
            q = randn(n, hq, dqk) * noise
            if skip:
                q[..., dqk - 1] = 1.0 / bmm1
            return q

        def kv_positions(lengths: Sequence[int]) -> torch.Tensor:
            return torch.cat(
                [torch.arange(n, device=dev) for n in lengths]
                or [torch.zeros(0, dtype=torch.long, device=dev)]
            )

        if layout == PACKED_QKV:
            # One [tokens, (Hq + 2 Hkv) * D] tensor (q_lens == kv_lens).
            q = make_q(sum_q)
            k = randn(sum_q, hkv, dqk)
            v = randn(sum_q, hkv, dv) + offsets.view(1, hkv, 1)
            if skip:
                pos = kv_positions(kv_lens).view(-1, 1).expand(-1, hkv)
                k[..., dqk - 1] = skip_levels(pos, m["tileKv"], gen)
            qkv = torch.cat(
                [q.reshape(sum_q, -1), k.reshape(sum_q, -1), v.reshape(sum_q, -1)], 1
            )
            t["qkv"] = cast(qkv, m["dtq"])
        else:
            if p.get("q_noncontig"):
                # [..., :D] of a zeroed [tokens, H, 2D] tensor (FlashInfer's
                # make_query_non_contiguous).
                base = torch.zeros(
                    (sum_q, hq, 2 * dqk), dtype=TORCH_DTYPE[m["dtq"]], device=dev
                )
                base[..., :dqk] = cast(make_q(sum_q), m["dtq"])
                t["q"] = base[..., :dqk]
            else:
                q_out = torch.empty(
                    (sum_q, hq, dqk), dtype=TORCH_DTYPE[m["dtq"]], device=dev
                )
                _fill(q_out, lambda i, j: cast(make_q(j - i), m["dtq"]).float())
                t["q"] = q_out
        if layout == SEPARATE_QKV:
            pos = kv_positions(kv_lens)
            k_out = torch.empty(
                (n_kv, hkv, dqk), dtype=TORCH_DTYPE[m["dtkv"]], device=dev
            )
            v_out = torch.empty(
                (n_kv, hkv, dv), dtype=TORCH_DTYPE[m["dtkv"]], device=dev
            )

            def make_k(i: int, j: int) -> torch.Tensor:
                k = randn(j - i, hkv, dqk)
                if skip:
                    k[..., dqk - 1] = skip_levels(
                        pos[i:j].view(-1, 1).expand(-1, hkv), m["tileKv"], gen
                    )
                return cast(k, m["dtkv"]).float()

            _fill(k_out, make_k)
            _fill(
                v_out,
                lambda i, j: cast(
                    randn(j - i, hkv, dv) + offsets.view(1, hkv, 1), m["dtkv"]
                ).float(),
            )
            t["k"], t["v"] = k_out, v_out
        elif layout == PAGED_KV:
            self._paged_inputs(p, t, gen, offsets)
        if p.get("sinks"):
            # Sink logits in [0, 5] (FlashInfer's tests): comparable to the
            # rows' peak logits, so they take a material share of the weight.
            t["sinks"] = torch.rand(hq, generator=gen, device=dev) * 5.0
        if p.get("device_scales"):
            t["scale_log2"] = torch.tensor(
                [bmm1], dtype=torch.float32, device=dev
            ) * torch.tensor(LOG2E, dtype=torch.float32, device=dev)
            t["out_scale"] = torch.tensor(
                [p["bmm2_scale"]], dtype=torch.float32, device=dev
            )
        return (p, t)

    def _paged_inputs(
        self,
        p: dict[str, Any],
        t: dict[str, Any],
        gen: torch.Generator,
        offsets: torch.Tensor,
    ) -> None:
        """KV cache, page table (random permutation of the pool, spare pages,
        optional shared prefix pages) and, for NVFP4 KV, block scales."""
        m, dev = self.meta, self.device
        hkv, page = p["hkv"], p["page_size"]
        kv_lens, q_lens = p["kv_lens"], p["q_lens"]
        batch = len(kv_lens)
        pages = p["num_pages"]
        dqk, dv = m["hdQk"], m["hdV"]
        d = max(dqk, dv)
        perm = torch.randperm(pages, generator=gen, device=dev)
        counts = [ceil_div(n, page) for n in kv_lens]
        shared = min(p.get("prefix_pages", 0), *counts[:2]) if batch > 1 else 0
        tables: list[torch.Tensor] = []
        start = 0
        for b, n in enumerate(counts):
            if b == 1 and shared:
                own = perm[start : start + n - shared]
                start += n - shared
                tables.append(torch.cat([tables[0][:shared], own]))
            else:
                tables.append(perm[start : start + n])
                start += n
        if start > pages:
            raise ValueError(f"{start} pages needed, {pages} allocated")
        # Sequence position of each page's first token (-1: unused page).
        page_pos = torch.full((pages,), -1, dtype=torch.long, device=dev)
        for row in tables:
            page_pos[row] = torch.arange(row.numel(), device=dev) * page
        skip = bool(p.get("skip_data"))
        nhd = p.get("kv_layout") == "NHD"
        shape = kv_cache_shape(m, p)
        if m["dtkv"] == "e2m1":
            cache = torch.randint(
                0, 256, shape, generator=gen, device=dev, dtype=torch.uint8
            )
            # V codes lean positive on even, negative on odd KV heads (90%):
            # a per-head offset of about +-1.1, as v_offsets for other caches.
            v_half = cache[1] if p.get("kv_tuple") else cache[:, 1]
            positive = torch.tensor(
                [0.9 if h % 2 == 0 else 0.1 for h in range(hkv)], device=dev
            ).view(1, hkv, 1, 1)

            def v_codes(i: int, j: int) -> torch.Tensor:
                rows = (j - i, hkv, page, 2 * v_half.shape[-1])
                mag = torch.randint(0, 8, rows, generator=gen, device=dev)
                neg = torch.rand(rows, generator=gen, device=dev) >= positive
                code = (mag + 8 * neg).to(torch.uint8)
                return code[..., 0::2] | (code[..., 1::2] << 4)

            _fill(v_half, v_codes)
            sf_shape = (pages, hkv, page, d // 16)
            choices = torch.tensor([0.25, 0.5, 0.75, 1.0], device=dev)
            k_sf = choices[torch.randint(0, 4, sf_shape, generator=gen, device=dev)]
            v_lin = choices[torch.randint(0, 4, sf_shape, generator=gen, device=dev)]
            k_sf = k_sf.to(torch.float8_e4m3fn)
            v_sf = swizzle_v_scales(v_lin.to(torch.float8_e4m3fn))
            if p.get("shared_idx", True):
                t["k_sf"], t["v_sf"] = k_sf, v_sf
            else:
                t["kv_sf"] = torch.stack([k_sf, v_sf], 1).reshape(
                    2 * pages, *sf_shape[1:]
                )
        else:
            dtype = TORCH_DTYPE[m["dtkv"]]
            cache = torch.empty(shape, dtype=dtype, device=dev)
            # Filled as [pages, 2 (K/V), ...] (a view for separate K/V tensors).
            fill = cache.transpose(0, 1) if p.get("kv_tuple") else cache
            fshape = tuple(fill.shape)
            heads_dim = 2 if nhd else 1  # head axis of a [pages, ...] slice

            def make(i: int, j: int) -> torch.Tensor:
                x = torch.randn((j - i, *fshape[1:]), generator=gen, device=dev)
                if p["shared_kv"]:
                    # MLA: V is K[..., :dv]; a per-cache offset shifts all
                    # logits of a row equally (softmax unchanged).
                    x[..., :dv] += offsets[0]
                    k = x[:, 0]
                else:
                    view = [1] * (len(fshape) - 1)
                    view[heads_dim] = hkv
                    x[:, 1] += offsets.view(view)
                    k = x[:, 0]
                if skip:
                    tok = torch.arange(page, device=dev)
                    pos = page_pos[i:j].view(-1, 1) + tok.view(1, -1)
                    pos = torch.where(page_pos[i:j].view(-1, 1) >= 0, pos, -1)
                    if p["shared_kv"]:
                        k[..., dqk - 1] = skip_levels(pos, m["tileKv"], gen)
                    elif nhd:
                        lv = pos.unsqueeze(-1).expand(-1, -1, hkv)
                        k[..., dqk - 1] = skip_levels(lv, m["tileKv"], gen)
                    else:
                        lv = pos.unsqueeze(1).expand(-1, hkv, -1)
                        k[..., dqk - 1] = skip_levels(lv, m["tileKv"], gen)
                if dtype == torch.float8_e4m3fn:
                    x = x.clamp(-E4M3_MAX, E4M3_MAX)
                return x

            _fill(fill, make)
        t["kv_cache"] = cache
        if p.get("topk", 0) > 0:
            # Sparse MLA: [batch, q_len, topk] token slots in the pool.
            topk = p["topk"]
            q_len = q_lens[0]
            rows = []
            for b in range(batch):
                for i in range(q_len):
                    visible = kv_lens[b] - q_len + i + 1
                    pos = torch.randperm(visible, generator=gen, device=dev)
                    pos = pos[: min(topk, visible)]
                    slots = tables[b][pos // page] * page + pos % page
                    row = torch.full((topk,), -1, dtype=torch.long, device=dev)
                    row[: slots.numel()] = slots
                    rows.append(row)
            t["page_table"] = (
                torch.stack(rows).reshape(batch, q_len, topk).to(torch.int32)
            )
            return
        table = torch.zeros((batch, p["max_pages"]), dtype=torch.int32, device=dev)
        for b, row in enumerate(tables):
            table[b, : row.numel()] = row
        if not p.get("shared_idx", True):
            if p["shared_kv"]:
                table = torch.stack([table, table], 1)
            else:
                table = torch.stack([2 * table, 2 * table + 1], 1)
        t["page_table"] = table.contiguous()

    # -- paged views (what the kernel addresses) --------------------------------

    def _paged_views(self, p: dict[str, Any], t: dict[str, Any]):
        """(K view, V view, K scales, V scales) as [pool, Hkv, P, D] views of
        the inputs, the tensors whose addresses the kernel gets."""
        cache = t["kv_cache"]
        nhd = p.get("kv_layout") == "NHD"
        if p["shared_kv"]:
            k = v = cache  # [pages, 1, P, D]
        elif p.get("kv_tuple"):
            k, v = cache[0], cache[1]
            if nhd:
                k, v = k.transpose(1, 2), v.transpose(1, 2)
        elif p.get("shared_idx", True):
            k, v = cache[:, 0], cache[:, 1]
            if nhd:
                k, v = k.transpose(1, 2), v.transpose(1, 2)
        else:
            pool = cache.reshape(2 * cache.shape[0], *cache.shape[2:])
            k = v = pool.transpose(1, 2) if nhd else pool
        if self.meta["dtkv"] != "e2m1":
            return k, v, None, None
        if p.get("shared_idx", True):
            return k, v, t["k_sf"], t["v_sf"]
        return k, v, t["kv_sf"], t["kv_sf"]

    def _request_kv(
        self, p: dict[str, Any], t: dict[str, Any], b: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Float32 K [Lk, Hkv, Dqk] and V [Lk, Hkv, Dv] of request ``b``."""
        m = self.meta
        dqk, dv = m["hdQk"], m["hdV"]
        kv_lens = p["kv_lens"]
        n = kv_lens[b]
        start = sum(kv_lens[:b])
        layout = case_layout(m, p)
        if layout == SEPARATE_QKV:
            return (
                t["k"][start : start + n].float(),
                t["v"][start : start + n].float(),
            )
        if layout == PACKED_QKV:
            hq, hkv = p["hq"], p["hkv"]
            rows = t["qkv"][start : start + n].float()
            k = rows[:, hq * dqk : (hq + hkv) * dqk].reshape(n, hkv, dqk)
            v = rows[:, (hq + hkv) * dqk :].reshape(n, hkv, dv)
            return k, v
        page = p["page_size"]
        pages = ceil_div(n, page)
        table = t["page_table"]
        if table.dim() == 3:
            k_idx, v_idx = table[b, 0, :pages].long(), table[b, 1, :pages].long()
        else:
            k_idx = v_idx = table[b, :pages].long()
        k_view, v_view, k_sf, v_sf = self._paged_views(p, t)

        def tokens(x: torch.Tensor) -> torch.Tensor:
            # [pages, H, P, D] -> [pages * P, H, D]
            return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])

        if m["dtkv"] == "e2m1":
            k_val = unpack_e2m1(k_view[k_idx])
            v_val = unpack_e2m1(v_view[v_idx])
            ks = k_sf[k_idx].float().repeat_interleave(16, dim=-1)
            vs = unswizzle_v_scales(v_sf[v_idx]).float().repeat_interleave(16, dim=-1)
            k, v = tokens(k_val * ks), tokens(v_val * vs)
        else:
            k, v = tokens(k_view[k_idx].float()), tokens(v_view[v_idx].float())
        return k[:n, :, :dqk], v[:n, :, :dv]

    def get_reference(self, inputs: tuple) -> tuple:
        """``(o as stored, o scale factors or None, multi-CTA KV counters (the
        kernels leave them zeroed), base-2 LSE or None (ragged kernels), o in
        float32 before output quantization, P |V| scaled like o or None (FP8
        queries, for p_rounding_bound))``."""
        p, t = inputs
        m = self.meta
        hq = p["hq"]
        q_lens, kv_lens = p["q_lens"], p["kv_lens"]
        dqk, dv = m["hdQk"], m["hdV"]
        layout = case_layout(m, p)
        if layout == PACKED_QKV:
            q_all = t["qkv"][:, : hq * dqk].reshape(-1, hq, dqk)
        else:
            q_all = t["q"]
        causal = p["runner"]["mask"] != DENSE
        sinks = t.get("sinks")
        sum_q = sum(q_lens)
        device = q_all.device
        out = torch.empty((sum_q, hq, dv), dtype=torch.float32, device=device)
        lse_all = torch.empty((sum_q, hq), dtype=torch.float32, device=device)
        # FP8 queries: the kernel rounds P to E4M3 (see p_rounding_bound).
        abs_v = None
        if m["dtq"] == "e4m3":
            abs_v = torch.empty_like(out)
        start = 0
        for b, (lq, lk) in enumerate(zip(q_lens, kv_lens)):
            q = q_all[start : start + lq].float()
            if p.get("topk", 0) > 0:
                res = self._sparse_request(p, t, b, q, with_abs_v=abs_v is not None)
            else:
                k, v = self._request_kv(p, t, b)
                res = attention(
                    q,
                    k,
                    v,
                    p["bmm1_scale"],
                    causal=causal,
                    window_left=p["window_left"],
                    sinks=sinks,
                    with_abs_v=abs_v is not None,
                )
                del k, v
            out[start : start + lq] = res[0]
            lse_all[start : start + lq] = res[1]
            if abs_v is not None:
                abs_v[start : start + lq] = res[2]
            start += lq
        out *= p["bmm2_scale"]
        if abs_v is not None:
            abs_v *= abs(p["bmm2_scale"])
        counter = torch.zeros(counter_size(m, p), dtype=torch.int32, device=device)
        # Ragged kernels write softmax stats; with sinks the kernel's stats
        # convention (sink in the sum or not) is unspecified: not compared.
        lse_ref = lse_all if layout == SEPARATE_QKV and sinks is None else None
        if m["dto"] == "e2m1":
            values, scales = ref_fp4_quant(out.reshape(sum_q, -1), NVFP4_O_SCALE)
            o = pack_e2m1(values).reshape(sum_q, hq, dv // 2)
            rows, cols = nvfp4_scale_shape(p, hq, dv)
            o_sf = swizzle_scales(scales, rows, cols, p.get("sf_start", 0))
            return (o, o_sf, counter, lse_ref, out, abs_v)
        if m["dto"] == "e4m3":
            fp8 = out.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
            return (fp8.view(torch.uint8), None, counter, lse_ref, out, abs_v)
        return (out.to(TORCH_DTYPE[m["dto"]]), None, counter, lse_ref, out, abs_v)

    def _sparse_request(
        self,
        p: dict[str, Any],
        t: dict[str, Any],
        b: int,
        q: torch.Tensor,
        with_abs_v: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Sparse MLA: each query row attends to its top-k token slots."""
        m = self.meta
        dqk, dv = m["hdQk"], m["hdV"]
        cache = t["kv_cache"][:, 0]  # [pages, P, D]
        flat = cache.reshape(-1, cache.shape[-1])
        slots = t["page_table"][b].long()  # [lq, topk]
        allowed = slots >= 0
        rows = []
        for i in range(q.shape[0]):
            k = flat[slots[i].clamp(min=0)].float().unsqueeze(1)  # [topk, 1, D]
            rows.append(
                attention(
                    q[i : i + 1],
                    k[..., :dqk],
                    k[..., :dv],
                    p["bmm1_scale"],
                    causal=False,
                    window_left=-1,
                    sinks=t.get("sinks"),
                    allowed=allowed[i : i + 1],
                    with_abs_v=with_abs_v,
                )
            )
        return tuple(torch.cat(parts) for parts in zip(*rows))

    # -- native launch -------------------------------------------------------

    def configure(self, function: cuda_driver.Function) -> None:
        """loadKernel: the dynamic shared memory limit for >= 48 KiB."""
        if self.meta["smem"] >= 48 * 1024:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.meta["smem"],
            )

    def _buffers(self, p: dict[str, Any]):
        m = self.meta
        dev = self.device
        sum_q = sum(p["q_lens"])
        hq = p["hq"]
        if m["dto"] == "e2m1":
            o = torch.empty((sum_q, hq, m["hdV"] // 2), dtype=torch.uint8, device=dev)
            o_sf = torch.zeros(
                nvfp4_scale_shape(p, hq, m["hdV"]), dtype=torch.uint8, device=dev
            )
        else:
            dtype = torch.uint8 if m["dto"] == "e4m3" else TORCH_DTYPE[m["dto"]]
            o = torch.empty((sum_q, hq, m["hdV"]), dtype=dtype, device=dev)
            o_sf = None
        r = runner_params(m, p)
        launch = launch_config(m, r)
        # Workspace as upstream partitions it: zeroed counters (the kernels
        # reset them) and scratch for the partial results of multi-CTA KV.
        grid = launch["grid"]
        ctas = grid[0] * grid[1] * grid[2]
        counter = torch.zeros(counter_size(m, p), dtype=torch.int32, device=dev)
        partial = 8 * NUM_SMS * m["stepQ"] + 4 * ctas * m["stepQ"] * m["hdV"] * 2
        scratch = torch.empty(partial + 1024, dtype=torch.uint8, device=dev)
        stats = None
        if "softmaxStats" in r["ptrs"]:
            stats = torch.empty((sum_q, hq, 2), dtype=torch.float32, device=dev)
        return r, launch, o, o_sf, counter, scratch, stats

    def prepare(self, inputs: tuple):
        p, t = inputs
        m = self.meta
        r, launch, o, o_sf, counter, scratch, stats = self._buffers(p)
        tensors = {
            "q": t.get("q"),
            "qkv": t.get("qkv"),
            "o": o,
            "oSf": o_sf,
            "counter": counter,
            "scratch": scratch,
            "softmaxStats": stats,
            "seqLensKv": t["seq_lens"],
            "cumSeqLensQ": t["cum_q"],
            "cumSeqLensKv": t["cum_kv"],
            "pageIdx": t.get("page_table"),
            "sinks": t.get("sinks"),
            "scaleSoftmaxLog2": t.get("scale_log2"),
            "outputScale": t.get("out_scale"),
        }
        if r["layout"] == PAGED_KV:
            k, v, k_sf, v_sf = self._paged_views(p, t)
            tensors.update(k=k, v=v, kSf=k_sf, vSf=v_sf)
        elif r["layout"] == SEPARATE_QKV:
            tensors["k"], tensors["v"] = t["k"], t["v"]
        pointers = {}
        for role in r["ptrs"]:
            tensor = tensors.get(role)
            if tensor is None:
                raise RuntimeError(f"{self.name}: no tensor for pointer {role}")
            pointers[role] = tensor.data_ptr()
        data, _ = build_params(m, r, pointers, driver_encode)
        grid, cluster = launch["grid"], launch["cluster"]
        function = self.function
        if cluster[0] > 8:  # setNonPortableClusterIfNeeded
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1
            )
        if m["mcta"] == MCTA_CGA:
            # TllmGenFmhaKernel::run falls back to GmemReduction when the
            # clusters do not fit in one wave; the cases assume they do.
            clusters = cuda_driver.max_active_clusters(
                function, grid, m["threads"], m["smem"], cluster
            )
            if clusters * cluster[0] < math.prod(grid):
                raise ValueError(
                    f"{self.name}: {clusters} active clusters of {cluster[0]} CTAs do "
                    f"not cover {math.prod(grid)} CTAs; upstream would select the "
                    "GmemReduction kernel"
                )
        # buildLaunchConfig: cluster dims, SPREAD scheduling for clusters > 1,
        # PDL (enable_pdl on sm_100).
        policy = CLUSTER_POLICY_SPREAD if cluster[0] > 1 else CLUSTER_POLICY_DEFAULT
        outputs = (o, o_sf, counter, stats, None)

        def launch_once() -> tuple:
            self.launch(
                grid,
                m["threads"],
                [data],
                shared_mem=m["smem"],
                cluster=cluster,
                programmatic_serialization=True,
                cluster_scheduling_policy=policy,
            )
            return outputs

        return launch_once, outputs

    def run(self, inputs: tuple) -> tuple:
        """``(o, o scale factors or None, counters, softmax stats (max, sum)
        or None, None)``, the prepared outputs; ``validate`` derives the
        base-2 LSE from the stats the ragged kernels write, as upstream's
        ComputeLSEFromMD does after the launch."""
        launch_once, _ = self.prepare(inputs)
        return launch_once()

    # -- validation -----------------------------------------------------------

    def validate(self, ref: tuple, impl: tuple) -> None:
        """``ref`` as ``get_reference`` returns it, ``impl`` as ``run`` (or
        the reference itself). Every stored output element is compared."""
        m = self.meta
        if len(ref) != 6 or len(impl) not in (5, 6):
            raise AssertionError(
                "expected (o, o_sf, counters, lse, o float32[, P |V|])"
            )
        stats = impl[3]
        if stats is not None and stats.dim() == 3:
            # Softmax stats (max, sum) as the kernel writes them; not compared
            # when the reference has no LSE (sinks).
            lse = None
            if ref[3] is not None:
                lse = LOG2E * stats[..., 0] + torch.log2(stats[..., 1])
            impl = (*impl[:3], lse, *impl[4:])
        names = ("o", "o_sf", "counter", "lse")
        for name, expected, actual in zip(names, ref[:4], impl[:4]):
            if (expected is None) != (actual is None):
                raise AssertionError(f"{name}: one output is missing")
            if expected is not None and (
                not isinstance(actual, torch.Tensor)
                or actual.shape != expected.shape
                or actual.dtype != expected.dtype
            ):
                raise AssertionError(
                    f"{name}: shape or dtype differs from the reference"
                )
        o_ref, sf_ref, counter_ref, lse_ref, exact, abs_v = ref
        o, o_sf, counter, lse = impl[:4]
        exact = exact.to(o.device)
        p_term = p_rounding_bound(abs_v.to(o.device)) if abs_v is not None else 0.0
        if not torch.equal(counter, counter_ref.to(counter.device)):
            raise AssertionError("multi-CTA KV counters were not reset to zero")
        rtol, atol, l2 = self.tolerance()
        if m["dto"] == "e2m1":
            self._check_nvfp4(o, o_sf, sf_ref, exact, rtol, atol, p_term)
        elif m["dto"] == "e4m3":
            actual = o.view(torch.float8_e4m3fn).float()
            # The kernel's fp32 result is within the kernel tolerance; its
            # E4M3 rounding adds at most half an ulp of its magnitude.
            bound = (
                atol
                + rtol * exact.abs()
                + p_term
                + 0.5 * e4m3_ulp(torch.maximum(actual.abs(), exact.abs()))
            )
            self._check_elements("FP8 output", actual, exact, bound)
            # Relative L2: E4M3 rounding alone contributes up to 2^-4.
            self._check_l2("FP8 output", actual, exact, l2 + 2.0**-4)
        else:
            actual = o.float()
            bound = atol + rtol * exact.abs() + p_term
            self._check_elements("output", actual, exact, bound)
            self._check_l2("output", actual, exact, l2)
        if lse_ref is not None:
            # Upstream compares the LSE at atol = rtol = 1e-3 (bf16); FP8
            # inputs quantize P, which the softmax sum may include.
            tol = 1e-3 if m["dtq"] != "e4m3" else 1e-2
            torch.testing.assert_close(
                lse.float(), lse_ref.to(lse.device).float(), rtol=tol, atol=tol
            )

    def _check_elements(
        self, what: str, actual: torch.Tensor, exact: torch.Tensor, bound
    ) -> None:
        excess = (actual - exact).abs() - bound
        bad = excess.isnan() | (excess > 0)
        # NVFP4 KV: FlashInfer's tests leave 10% of elements unchecked at
        # rtol = atol = 0.5 for the in-kernel quantization noise. Here at
        # most 1e-4 of the elements may exceed the bound, by < 2x.
        spare = int(1e-4 * excess.numel()) if self.meta["dtkv"] == "e2m1" else 0
        if int(bad.sum()) > spare or bool((excess.isnan() | (excess > bound)).any()):
            index = tuple(int(i) for i in torch.nonzero(bad)[0])
            raise AssertionError(
                f"{what}: {int(bad.sum())} of {excess.numel()} elements beyond "
                f"tolerance, first at {index}: {actual[index].item():.4g} vs "
                f"{exact[index].item():.4g}"
            )

    @staticmethod
    def _check_l2(what: str, actual: torch.Tensor, exact: torch.Tensor, tol: float):
        norm = exact.double().norm().item()
        err = (actual.double() - exact.double()).norm().item()
        if not err <= tol * norm:
            raise AssertionError(
                f"{what}: relative L2 error {err / max(norm, 1e-30):.3e} > {tol:.3e}"
            )

    def _check_nvfp4(
        self,
        o: torch.Tensor,
        o_sf: torch.Tensor,
        sf_ref: torch.Tensor,
        exact: torch.Tensor,
        rtol: float,
        atol: float,
        p_term: torch.Tensor | float = 0.0,
    ) -> None:
        """NVFP4 output against the unquantized reference (``p_term``: the
        per-element E4M3 P rounding bound of FP8-query kernels).

        Scale factors: the kernel's block scale is E4M3(300 * amax / 6) of its
        own block maximum, which is within the kernel tolerance of the
        reference's; E4M3 rounding moves it by at most 1/16. Values: each
        dequantized element is within the kernel tolerance plus half the
        E2M1 spacing at its magnitude (in units of its own block step) of the
        exact value. Scale-factor rows outside the output's rows (the
        reference's nonzero rows: O(1) outputs give every block a nonzero
        scale; ``sf_start`` offsets them) stay zero.
        """
        sum_q, hq = o.shape[0], o.shape[1]
        n = hq * self.meta["hdV"]
        rows = sf_ref.shape[0]
        ref_full = recover_swizzled_scales(sf_ref.view(torch.float8_e4m3fn), rows, n)
        used = torch.nonzero((ref_full != 0).any(1)).flatten()
        start = int(used[0]) if used.numel() else 0
        if used.numel() and int(used[-1]) >= start + sum_q:
            raise AssertionError("reference scale factors span more rows than o")
        values = unpack_e2m1(o.reshape(sum_q, -1))
        full = recover_swizzled_scales(o_sf.view(torch.float8_e4m3fn), o_sf.shape[0], n)
        scales = full[start : start + sum_q]
        outside = torch.cat([full[:start], full[start + sum_q :]])
        if bool((outside != 0).any()):
            raise AssertionError("NVFP4 output: scale factors written outside rows")
        exact = exact.reshape(sum_q, n // 16, 16)
        if isinstance(p_term, torch.Tensor):
            p_term = p_term.reshape(sum_q, n // 16, 16)
            p_block = p_term.amax(-1)
        else:
            p_block = p_term
        amax = exact.abs().amax(-1)
        slack = atol + rtol * amax + p_block
        lo = NVFP4_O_SCALE * (amax - slack).clamp(min=0) / 6.0
        hi = NVFP4_O_SCALE * (amax + slack) / 6.0
        bad = (scales * (1 + 1 / 16) < lo) | (scales * (1 - 1 / 16) > hi)
        if bool(bad.any()):
            index = tuple(int(i) for i in torch.nonzero(bad)[0])
            raise AssertionError(
                f"NVFP4 output: {int(bad.sum())} block scales outside "
                f"E4M3(300 * amax / 6), first at {index}: {scales[index].item():.4g} "
                f"for amax {amax[index].item():.4g}"
            )
        unit = (scales / NVFP4_O_SCALE).unsqueeze(-1)
        deq = values.reshape(sum_q, n // 16, 16) * unit
        units = torch.maximum(deq.abs(), exact.abs()) / unit.clamp(min=1e-30)
        bound = atol + rtol * exact.abs() + p_term + e2m1_half_gap(units) * unit
        self._check_elements("NVFP4 output", deq, exact, bound)

    def tolerance(self) -> tuple[float, float, float]:
        """(rtol, atol, relative L2) of the kernel's unquantized result.

        16-bit inputs: FlashInfer's 1e-2 (bf16 P and output rounding, < 0.4%
        each) and 1e-2 relative L2. FP8 inputs quantize the softmax
        probabilities to E4M3 inside the kernel: FlashInfer's 4e-2 / 6e-2,
        now against O(1) outputs, and 3e-2 relative L2 (simulated E4M3
        rounding of P at these logit spreads: <= 1.1e-2 when the kernel
        normalizes by the quantized sum, <= 2.6e-2 by the exact one). NVFP4 KV
        dequantizes in-kernel to E4M3: 6e-2 / 8e-2 and 5e-2. FP8 and NVFP4
        output rounding, and (FP8 queries) the E4M3 rounding of P at rows a
        few keys dominate (``p_rounding_bound``), are bounded separately in
        ``validate``.
        """
        m = self.meta
        if m["dtkv"] == "e2m1":
            return 6e-2, 8e-2, 5e-2
        if m["dtq"] == "e4m3" or m["dtkv"] == "e4m3":
            return 4e-2, 6e-2, 3e-2
        return 1e-2, 1e-2, 1e-2


def _register_all() -> None:
    index = load_index()
    for row in index["kernels"]:
        m = meta_of(row)
        register_variant(
            TrtllmFmha,
            name=workload_name(m["name"]),
            supported_arches=(ARCH,),
            meta=m,
            case_params=index["cases"][m["name"]],
        )


_register_all()

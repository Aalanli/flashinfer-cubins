"""Python mirror of the trtllm-gen host code that launches the sm_100a kernels.

Pinned sources (FlashInfer v0.6.9 and the trtllm-gen export headers of its
artifact paths, ``resources/trtllm-gen-artifacts/``):

* ``trtllmGen_bmm_export/KernelParams.h`` ``KernelParamsSetup::
  setKernelParams`` (BatchN branch: every exported batched config transposes
  the MMA output, so the weights are A and the tokens are B) and
  ``TmaDescriptor.h`` (``buildNdTmaDescriptor``, ``buildSfTmaDescriptor``);
* ``trtllmGen_bmm_export/BatchedGemmInterface.h`` ``run`` (grid, cluster,
  PDL) and ``trtllm/gen/CudaKernelLauncher.h`` ``launchKernel``;
* ``trtllmGen_gemm_export/KernelParams.h`` / ``GemmInterface.h`` for the dense
  GEMM kernels (same descriptor builders);
* ``include/flashinfer/trtllm/fused_moe/runner.h``
  (``getMaxNumCtasInBatchDim``) and the routing kernels' CTA tables
  (``RoutingKernel.cuh``).

Option dictionaries use the trtllm-gen option names (``mTileM``, ...) exactly
as the compile-time probes record them (``impls/batched_gemm/kernels``).
Everything here is plain integer arithmetic; ``tests/test_batched_gemm.py``
compares the structs and descriptor arguments it builds byte for byte with
the probes' recordings of the upstream host code.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

# --- trtllm::gen::Dtype (DtypeDecl.h: TLLM_ENCODE_DTYPE) ---------------------


def encode_dtype(block: int, signed: int, integer: int, bits: int, uid: int) -> int:
    return (block << 24) | (signed << 20) | (integer << 16) | (bits << 8) | uid


DTYPES = {
    "Bfloat16": encode_dtype(0, 1, 0, 16, 0),
    "E2m1": encode_dtype(1, 1, 0, 4, 2),
    "E4m3": encode_dtype(0, 1, 0, 8, 5),
    "Fp16": encode_dtype(0, 1, 0, 16, 7),
    "Fp32": encode_dtype(0, 1, 0, 32, 8),
    "MxE2m1": encode_dtype(1, 1, 0, 4, 12),
    "MxE4m3": encode_dtype(1, 1, 0, 8, 13),
    "MxInt4": encode_dtype(1, 1, 1, 4, 14),
    "UE8m0": encode_dtype(0, 0, 0, 8, 15),
    "UInt8": encode_dtype(0, 0, 1, 8, 16),
}
DTYPE_NAMES = {code: name for name, code in DTYPES.items()}


def dtype_bits(dtype: int) -> int:
    return (dtype >> 8) & 0xFF


def dtype_name(dtype: int) -> str:
    return DTYPE_NAMES[dtype]


# MmaKind (MmaDecl.h), SfLayout (SfLayoutDecl.h), MatrixLayout / TileScheduler /
# SplitK (Enums.h), RouteImpl (BatchedGemmEnums.h), ActType
# (GemmGatedActOptions.h), EltwiseActType (Enums.h).
MMA_KIND_MXFP8FP6FP4 = 5
SF_LAYOUT_LINEAR, SF_LAYOUT_R8C4, SF_LAYOUT_R8C16, SF_LAYOUT_R128C4 = 0, 1, 2, 3
LAYOUT_MAJOR_K, LAYOUT_MAJOR_MN, LAYOUT_BLOCK_MAJOR_K = 0, 1, 2
SCHEDULER_STATIC, SCHEDULER_PERSISTENT = 0, 1
SCHEDULER_STATIC_PERSISTENT, SCHEDULER_PERSISTENT_SM90 = 2, 3
ROUTE_NONE, ROUTE_LDGSTS, ROUTE_TMA, ROUTE_LDG_PLUS_STS = 0, 1, 2, 3
ACT_SWIGLU, ACT_GEGLU = 0, 1
ELTWISE_NONE, ELTWISE_GELU, ELTWISE_RELU2, ELTWISE_SILU = 0, 1, 2, 3
SPLIT_K_NONE, SPLIT_K_GMEM, SPLIT_K_DSMEM = 0, 1, 2

# CommonUtils.h
TMA_DIM_MAX = 1 << 31
X_LARGE_N = 1 << 35

# Raw CUtensorMap* enum values (cuda.h).
TMA_UINT8, TMA_FLOAT16, TMA_FLOAT32, TMA_BFLOAT16 = 0, 6, 7, 9
TMA_16U4_ALIGN8B, TMA_16U4_ALIGN16B = 13, 14
SWIZZLE_NONE, SWIZZLE_32B, SWIZZLE_64B, SWIZZLE_128B = 0, 1, 2, 3
L2_PROMOTION_128B = 2
U64 = (1 << 64) - 1

# CUlaunchAttributeID / CUclusterSchedulingPolicy (cuda.h)
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
CLUSTER_SCHEDULING_DEFAULT, CLUSTER_SCHEDULING_SPREAD = 0, 1

# Encodes one CUtensorMap: (data_type, address, dims, strides_bytes, box,
# element_strides, interleave, swizzle, l2_promotion, oob_fill) -> 128 bytes.
TensorMapEncoder = Callable[..., bytes]


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def div_up_mul(a: int, b: int) -> int:
    return ceil_div(a, b) * b


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
    """The probes' deterministic CUtensorMap stand-in (``fake_tensor_map`` in
    ``impls/batched_gemm/kernels/probe_common.cuh``); for tests only."""
    out = bytearray(128)
    out[0:6] = bytes(
        (data_type, len(dims), interleave, swizzle, l2_promotion, oob_fill)
    )
    struct.pack_into("<Q", out, 8, address)
    struct.pack_into(f"<{len(dims)}Q", out, 16, *dims)
    struct.pack_into(f"<{len(strides)}Q", out, 56, *strides)
    struct.pack_into(f"<{len(box)}I", out, 88, *box)
    struct.pack_into(f"<{len(element_strides)}I", out, 108, *element_strides)
    return bytes(out)


# --- TmaDescriptor.h ---------------------------------------------------------


def dtype_needs_padding(dtype: int, mma_kind: int) -> bool:
    """trtllm::gen::dtypeNeedsPadding."""
    return mma_kind == MMA_KIND_MXFP8FP6FP4 and dtype == DTYPES["MxE2m1"]


def build_nd_tma_descriptor(
    encode: TensorMapEncoder,
    dtype: int,
    shapes: Sequence[int],
    strides: Sequence[int],
    tile_shapes: Sequence[int],
    address: int,
    do_pad: bool,
    do_swizzle: bool = True,
) -> bytes:
    """gemm::buildNdTmaDescriptor."""
    pad = 1
    if dtype in (DTYPES["E4m3"], DTYPES["MxE4m3"], DTYPES["UE8m0"], DTYPES["UInt8"]):
        data_type = TMA_UINT8
    elif dtype == DTYPES["Fp16"]:
        data_type = TMA_FLOAT16
    elif dtype == DTYPES["Bfloat16"]:
        data_type = TMA_BFLOAT16
    elif dtype == DTYPES["E2m1"]:
        data_type = TMA_16U4_ALIGN8B
    elif dtype in (DTYPES["MxE2m1"], DTYPES["MxInt4"]):
        if do_pad:
            pad, data_type = 2, TMA_16U4_ALIGN16B
        else:
            data_type = TMA_16U4_ALIGN8B
    elif dtype == DTYPES["Fp32"]:
        data_type = TMA_FLOAT32
    else:
        raise ValueError(f"buildNdTmaDescriptor: unexpected dtype {dtype:#x}")
    bits = dtype_bits(dtype)
    swizzle = SWIZZLE_NONE
    fastest = tile_shapes[0] * bits * pad // 8
    if do_swizzle:
        if fastest % 128 == 0:
            swizzle = SWIZZLE_128B
        elif fastest % 64 == 0:
            swizzle = SWIZZLE_64B
        elif fastest % 32 == 0:
            swizzle = SWIZZLE_32B
        elif fastest % 16 == 0 and dtype in (
            DTYPES["UE8m0"],
            DTYPES["E4m3"],
            DTYPES["E2m1"],
            DTYPES["UInt8"],
        ):
            swizzle = SWIZZLE_NONE
        else:
            raise ValueError(f"unexpected fastest-dim tile size {fastest} B")
    if address % 16:
        raise ValueError("TMA global address must be 16-byte aligned")
    dim = len(shapes)
    if dim not in (2, 3, 4) or len(strides) != dim or strides[0] != 1:
        raise ValueError("invalid TMA shape/stride")
    if any(not 1 <= s <= 1 << 32 for s in shapes):
        raise ValueError(f"TMA shape out of range: {list(shapes)}")
    strides_bytes = [(s * bits // 8) & U64 for s in strides[1:]]
    elts_per_u32 = 4 * 8 // (bits * pad)
    box = [1] * dim
    box[0] = min(elts_per_u32 * 32, tile_shapes[0])
    for i in range(1, len(tile_shapes)):
        if tile_shapes[i] > 256:
            raise ValueError(f"boxDim too large {tile_shapes[i]}")
        box[i] = tile_shapes[i]
    return encode(
        data_type,
        address,
        [int(s) for s in shapes],
        strides_bytes,
        box,
        [1] * dim,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        swizzle,
        L2_PROMOTION_128B,
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )


def build_sf_tma_descriptor(
    encode: TensorMapEncoder,
    dtype: int,
    shapes: Sequence[int],
    strides: Sequence[int],
    tile_shapes: Sequence[int],
    address: int,
) -> bytes:
    """gemm::buildSfTmaDescriptor (no swizzle)."""
    if dtype in (DTYPES["E4m3"], DTYPES["UE8m0"]):
        data_type = TMA_UINT8
    elif dtype == DTYPES["Bfloat16"]:
        data_type = TMA_BFLOAT16
    else:
        raise ValueError(f"buildSfTmaDescriptor: unexpected dtype {dtype:#x}")
    if address % 16:
        raise ValueError("TMA global address must be 16-byte aligned")
    if len(strides) != len(shapes) or strides[0] != 1:
        raise ValueError("invalid SF TMA shape/stride")
    bits = dtype_bits(dtype)
    return encode(
        data_type,
        address,
        [int(s) for s in shapes],
        [(s * bits // 8) & U64 for s in strides[1:]],
        [int(t) for t in tile_shapes],
        [1] * len(shapes),
        0,
        SWIZZLE_NONE,
        L2_PROMOTION_128B,
        0,
    )


def sf_shape_stride(
    num_tokens: int,
    hidden: int,
    tokens_per_tile: int,
    hidden_per_tile: int,
    layout: int,
    reshape_factor: int,
    elts_per_sf: int,
) -> tuple[list[int], list[int], list[int]]:
    """makeTmaShapeStrideSfAb (batched and dense: identical arithmetic)."""
    if layout == SF_LAYOUT_R128C4:
        shape = [
            256,
            2,
            ceil_div(hidden, elts_per_sf * 4),
            ceil_div(num_tokens, 128),
        ]
        tile = [
            256,
            2,
            ceil_div(hidden_per_tile, elts_per_sf * 4),
            ceil_div(tokens_per_tile, 128),
        ]
    elif layout == SF_LAYOUT_R8C4:
        r = reshape_factor
        if r > 0 and r & (r - 1):
            raise ValueError("mSfReshapeFactor must be a power of 2")
        repeats = min(ceil_div(hidden_per_tile, elts_per_sf * 4), r)
        if ceil_div(hidden, elts_per_sf * 4) % repeats:
            raise ValueError("SF hiddenSize K must be a multiple of repeats")
        shape = [
            repeats * 32,
            ceil_div(hidden, elts_per_sf * 4 * repeats),
            ceil_div(num_tokens, 8),
        ]
        tile = [
            repeats * 32,
            ceil_div(hidden_per_tile, elts_per_sf * 4 * repeats),
            ceil_div(tokens_per_tile, 8),
        ]
    else:
        raise ValueError("Unsupported SF layout")
    stride = [1]
    for i in range(1, len(shape)):
        stride.append(shape[i - 1] * stride[i - 1])
    return shape, stride, tile


# --- batched GEMM: KernelParams.h (BatchN) -----------------------------------

MATRIX_A, MATRIX_B, MATRIX_C = 0, 1, 2


def _uses_tma_oob(o: Mapping[str, Any], matrix: int) -> bool:
    if matrix == MATRIX_A:
        return False  # BatchM only
    if matrix == MATRIX_B:
        return o["mRouteImpl"] == ROUTE_NONE and bool(o["mUseTmaOobOpt"])
    return bool(o["mUseTmaStore"]) and bool(o["mUseTmaOobOpt"])


def bmm_shape_stride_abc(
    o: Mapping[str, Any],
    size_m: int,
    size_n: int,
    size_k: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    matrix: int,
    valid_m: int = -1,
    valid_n: int = -1,
    valid_k: int = -1,
) -> tuple[list[int], list[int], list[int]]:
    """makeTmaShapeStrideAbc for transposeMmaOutput (weights = A)."""
    valid_m = size_m if valid_m < 0 else valid_m
    valid_n = size_n if valid_n < 0 else valid_n
    valid_k = size_k if valid_k < 0 else valid_k
    if not o["mTransposeMmaOutput"]:
        raise ValueError("only transposed-output batched configs are exported")
    is_weights = matrix == MATRIX_A
    oob = _uses_tma_oob(o, matrix)
    ac = matrix in (MATRIX_A, MATRIX_C)
    num_tokens = size_m if ac else size_n
    num_tokens_valid = valid_m if ac else valid_n
    cta_tile_tokens = tile_m if ac else tile_n
    tile_tokens = o["mEpilogueTileM"] if matrix == MATRIX_C else cta_tile_tokens
    hidden = size_n if matrix == MATRIX_C else size_k
    hidden_valid = valid_n if matrix == MATRIX_C else valid_k
    cta_tile_hidden = tile_n if matrix == MATRIX_C else tile_k
    tile_hidden = o["mEpilogueTileN"] if matrix == MATRIX_C else cta_tile_hidden
    if matrix == MATRIX_C:  # transposed output: swap
        num_tokens, hidden = hidden, num_tokens
        num_tokens_valid, hidden_valid = hidden_valid, num_tokens_valid
        cta_tile_tokens, cta_tile_hidden = cta_tile_hidden, cta_tile_tokens
        tile_tokens, tile_hidden = tile_hidden, tile_tokens
    if o["mFusedAct"] and matrix == MATRIX_C:
        hidden //= 2
        hidden_valid //= 2
        tile_hidden //= 2
        cta_tile_hidden //= 2
    shape = [hidden_valid, num_tokens_valid]
    stride = [1, hidden]
    if oob:
        shape = [hidden_valid, cta_tile_tokens, TMA_DIM_MAX, TMA_DIM_MAX]
        stride = [1, hidden, (X_LARGE_N - hidden) & U64, hidden]
    elif is_weights:
        shape = [hidden_valid, num_tokens_valid, o["mNumBatches"]]
        stride = [1, hidden, hidden * num_tokens]
    tile = [tile_hidden, tile_tokens]
    if matrix != MATRIX_C:
        if matrix == MATRIX_B and tile[1] > 1 and o["mClusterDimX"] >= 2:
            tile[1] //= 2
        layout = o["mLayoutA"] if matrix == MATRIX_A else o["mLayoutB"]
        if layout == LAYOUT_MAJOR_MN:
            shape[0], shape[1] = shape[1], shape[0]
            stride[1] = num_tokens
            tile = [tile[1], tile[0]]
        elif layout == LAYOUT_BLOCK_MAJOR_K:
            block_k = o["mBlockK"]
            shape = [block_k, num_tokens, size_k // block_k, o["mNumBatches"]]
            stride = [1, block_k, num_tokens * block_k, hidden * num_tokens]
            tile_block_k = min(block_k, tile_hidden)
            tile = [tile_block_k, tile_tokens, tile_hidden // tile_block_k]
            if matrix == MATRIX_B and o["mClusterDimX"] >= 2:
                tile[1] //= 2
    return shape, stride, tile


def sf_dtype(dtype: int) -> int | None:
    """Scale-factor dtype of a block-scaled element dtype (None: no SFs)."""
    if dtype == DTYPES["E2m1"]:
        return DTYPES["E4m3"]
    if dtype in (DTYPES["MxE2m1"], DTYPES["MxE4m3"]):
        return DTYPES["UE8m0"]
    if dtype == DTYPES["MxInt4"]:
        return DTYPES["Bfloat16"]
    return None


class BmmProblem(NamedTuple):
    """BatchedGemmData of TrtllmGenBatchedGemmRunner::run (transposed: the
    kernel's M is the runner's n, its N the runner's m)."""

    m: int  # kernel M: weight rows (runner n)
    n: int  # kernel N: tokens (runner m)
    k: int
    num_tokens: int
    num_batches: int
    max_num_ctas: int


BMM_POINTER_FIELDS = (
    "a",
    "b",
    "c",
    "sf_a",
    "sf_b",
    "sf_c",
    "per_token_sf_a",
    "per_token_sf_b",
    "bias",
    "scale_c",
    "scale_gate",
    "alpha",
    "beta",
    "clamp_limit",
    "route_map",
    "total_num_padded_tokens",
    "cta_idx_xy_to_batch_idx",
    "cta_idx_xy_to_mn_limit",
    "num_non_exiting_ctas",
)


def bmm_grid(o: Mapping[str, Any], p: BmmProblem, sm_count: int) -> tuple[int, ...]:
    """BatchedGemmInterface::getLaunchGrid for BatchN (dynamic batch)."""
    if o["mTileScheduler"] in (SCHEDULER_STATIC_PERSISTENT, SCHEDULER_PERSISTENT_SM90):
        xy = o["mClusterDimX"] * o["mClusterDimY"]
        ctas = sm_count // o["mNumSlicesForSplitK"] // xy * xy
        return (ctas, 1, o["mNumSlicesForSplitK"])
    if o["mIsStaticBatch"]:
        raise ValueError("static-batch configs are not served")
    if (
        not (o["mEnablesEarlyExit"] or o["mEnablesDelayedEarlyExit"])
        or not p.num_tokens
    ):
        raise ValueError("Invalid combination of options")
    batch = div_up_mul(p.max_num_ctas, o["mClusterDimY"])
    tile = div_up_mul(ceil_div(p.m, o["mTileM"]), o["mClusterDimX"])
    return (tile, batch, o["mNumSlicesForSplitK"])


def bmm_params(
    layout: Mapping[str, Any],
    o: Mapping[str, Any],
    p: BmmProblem,
    ptr: Mapping[str, int],
    encode: TensorMapEncoder,
) -> bytes:
    """KernelParamsSetup::setKernelParams for BatchN with dynamic batching
    (TrtllmGenBatchedGemmRunner::run -> BatchedGemmInterface::run).

    ``ptr`` maps ``BMM_POINTER_FIELDS`` to device addresses (0 = nullptr);
    ``o`` holds the config's options. Unset descriptors and the static CTA
    tables stay zero (upstream leaves them indeterminate).
    """
    o = {
        **o,
        "mM": p.m,
        "mN": p.n,
        "mK": p.k,
        "mNumTokens": p.num_tokens,
        "mNumBatches": p.num_batches,
    }
    if o["mIsStaticBatch"]:
        raise ValueError("static-batch configs are not served")
    if o["mSparsityA"]:
        raise ValueError("sparse A is not exported")
    if p.m % o["mTileM"]:
        raise ValueError("0 == mM % tileM")
    a, b, c = o["mDtypeA"], o["mDtypeB"], o["mDtypeC"]
    v: dict[str, Any] = {name: 0 for name in layout["fields"]}
    for name, field in layout["fields"].items():
        if field["kind"] in ("tensor_map", "i32_array"):
            v[name] = b""
    v.update(
        ptrRouteMap=ptr["route_map"],
        numTokens=p.num_tokens,
        ptrScaleC=ptr["scale_c"],
        # TrtllmGenBatchedGemmRunner::run: scaleAct = scaleGate
        ptrScaleAct=ptr["scale_gate"],
        ptrScaleGate=ptr["scale_gate"],
        ptrClampLimit=ptr["clamp_limit"],
        ptrGatedActAlpha=ptr["alpha"],
        ptrGatedActBeta=ptr["beta"],
        ptrTotalNumPaddedTokens=ptr["total_num_padded_tokens"],
        ptrCtaIdxXyToBatchIdx=ptr["cta_idx_xy_to_batch_idx"],
        ptrCtaIdxXyToMnLimit=ptr["cta_idx_xy_to_mn_limit"],
    )
    # Dynamic batch: totalNumPaddedTokens etc. come from device buffers.
    cta_offset = p.max_num_ctas  # numCtaBatch of getGridDim
    cta_offset = div_up_mul(cta_offset, o["mClusterDimY"])
    if o["mUseDeepSeekFp8"] and c == DTYPES["E4m3"]:
        v["ptrDqSfsC"] = ptr["sf_c"]
    v.update(
        ptrA=ptr["a"],
        ptrB=ptr["b"],
        strideInBytesA=p.k * dtype_bits(a) // 8,
        strideInBytesB=p.k * dtype_bits(b) // 8,
        ptrSfA=ptr["sf_a"],
        ptrSfB=ptr["sf_b"],
        ptrSfC=ptr["sf_c"],
    )
    pad_a = dtype_needs_padding(a, o["mMmaKind"])
    pad_b = dtype_needs_padding(b, o["mMmaKind"])
    v["tileStridePerBatch"] = p.m // o["mTileM"]
    v["nm"] = p.m
    shape, stride, tile = bmm_shape_stride_abc(
        o, p.m, p.n, p.k, o["mTileM"], o["mTileN"], o["mTileK"], MATRIX_A
    )
    v["tmaA"] = build_nd_tma_descriptor(
        encode, a, shape, stride, tile, ptr["a"], pad_a, True
    )
    input_tokens = cta_offset * o["mTileN"]
    if o["mRouteImpl"] != ROUTE_LDGSTS:
        route = o["mRouteImpl"] == ROUTE_TMA
        tokens = p.num_tokens if route else input_tokens
        shape, stride, tile = bmm_shape_stride_abc(
            o,
            p.m,
            tokens,
            p.k,
            o["mTileM"],
            1 if route else o["mTileN"],
            o["mTileK"],
            MATRIX_B,
            p.m,
            tokens,
            p.k,
        )
        v["tmaB"] = build_nd_tma_descriptor(
            encode, b, shape, stride, tile, ptr["b"], pad_b, True
        )
    sfa = sf_dtype(a)
    if sfa is not None:
        shape, stride, tile = sf_shape_stride(
            p.m * p.num_batches,
            p.k,
            o["mTileM"],
            o["mTileK"],
            o["mSfLayoutA"],
            o["mSfReshapeFactor"],
            o["mSfBlockSizeA"],
        )
        v["tmaSfA"] = build_sf_tma_descriptor(
            encode, sfa, shape, stride, tile, ptr["sf_a"]
        )
    sfb = sf_dtype(b) if b != DTYPES["MxInt4"] else None
    if sfb is not None:
        block = o["mSfBlockSizeB"]
        if o["mRouteSfsImpl"] == ROUTE_TMA:
            sfs_k = ceil_div(p.k // block, 16) * 16
            shape, stride, tile = bmm_shape_stride_abc(
                o,
                p.m,
                p.num_tokens,
                sfs_k,
                o["mTileM"],
                1,
                o["mTileK"] // block,
                MATRIX_B,
                p.m,
                p.num_tokens,
                sfs_k,
            )
            v["tmaSfB"] = build_nd_tma_descriptor(
                encode, sfb, shape, stride, tile, ptr["sf_b"], False, True
            )
        elif o["mRouteSfsImpl"] == ROUTE_NONE:
            shape, stride, tile = sf_shape_stride(
                input_tokens,
                p.k,
                o["mTileN"],
                o["mTileK"],
                o["mSfLayoutB"],
                o["mSfReshapeFactor"],
                block,
            )
            v["tmaSfB"] = build_sf_tma_descriptor(
                encode, sfb, shape, stride, tile, ptr["sf_b"]
            )
    if not o["mUseTmaStore"]:
        v["ptrC"] = ptr["c"]
    else:
        shape, stride, tile = bmm_shape_stride_abc(
            o,
            p.m,
            cta_offset * o["mTileN"],
            p.k,
            o["mTileM"],
            o["mTileN"],
            o["mTileK"],
            MATRIX_C,
        )
        v["tmaC"] = build_nd_tma_descriptor(
            encode, c, shape, stride, tile, ptr["c"], False
        )
    v.update(
        k=p.k,
        numBatches=p.num_batches,
        rank=0,
        tpGrpSize=1,
        ptrPartialRowMax=0,
        ptrRowMaxCompletionBars=0,
        ptrNumNonExitingCtas=ptr["num_non_exiting_ctas"],
        ptrPerTokenSfA=ptr["per_token_sf_a"],
        ptrPerTokenSfB=ptr["per_token_sf_b"],
        ptrBias=ptr["bias"],
        ptrDynamicTileCounter=0,
    )
    return pack_struct(layout, v)


# --- dense GEMM: gemm KernelParams.h -----------------------------------------


class GemmProblem(NamedTuple):
    """GemmData of the FlashInfer runners (transposed: kernel M = runner n)."""

    m: int  # kernel M (weight rows)
    n: int  # kernel N (activation rows)
    k: int


GEMM_POINTER_FIELDS = ("a", "b", "c", "sf_a", "sf_b", "sf_c", "scale_c")


def gemm_ab_shape_stride(
    o: Mapping[str, Any], p: GemmProblem, matrix: int
) -> tuple[list[int], list[int], list[int]]:
    """makeTmaShapeStrideAb (dense)."""
    if o["mSparsityA"]:
        raise ValueError("sparse A is not exported")
    tokens = p.m if matrix == MATRIX_A else p.n
    tile_mn = o["mTileM"] if matrix == MATRIX_A else o["mTileN"]
    shape = [p.k, tokens]
    stride = [1, p.k]
    tile = [o["mTileK"], tile_mn]
    if matrix == MATRIX_B and o["mClusterDimX"] >= 2:
        tile[1] //= 2
    layout = o["mLayoutA"] if matrix == MATRIX_A else o["mLayoutB"]
    if layout == LAYOUT_MAJOR_MN:
        shape = [shape[1], shape[0]]
        stride[1] = tokens
        tile = [tile[1], tile[0]]
    elif layout == LAYOUT_BLOCK_MAJOR_K:
        block_k = o["mBlockK"]
        shape = [block_k, tokens, p.k // block_k]
        stride = [1, block_k, tokens * block_k]
        tile_block_k = min(block_k, o["mTileK"])
        tile = [tile_block_k, tile_mn, o["mTileK"] // tile_block_k]
        if matrix == MATRIX_B and o["mClusterDimX"] >= 2:
            tile[1] //= 2
    return shape, stride, tile


def gemm_grid(o: Mapping[str, Any], p: GemmProblem, sm_count: int) -> tuple[int, ...]:
    """GemmInterface::run's grid (getGridSize / getFixedGridSize)."""
    if o["mTileScheduler"] in (SCHEDULER_STATIC_PERSISTENT, SCHEDULER_PERSISTENT_SM90):
        xy = o["mClusterDimX"] * o["mClusterDimY"]
        ctas = sm_count // o["mNumSlicesForSplitK"] // xy * xy
        return (ctas, 1, o["mNumSlicesForSplitK"])
    return (
        div_up_mul(ceil_div(p.m, o["mTileM"]), o["mClusterDimX"]),
        div_up_mul(ceil_div(p.n, o["mTileN"]), o["mClusterDimY"]),
        o["mNumSlicesForSplitK"],
    )


def gemm_params(
    layout: Mapping[str, Any],
    o: Mapping[str, Any],
    p: GemmProblem,
    ptr: Mapping[str, int],
    encode: TensorMapEncoder,
) -> bytes:
    """gemm::KernelParamsSetup::setKernelParams for one device (rank 0 of 1,
    no all-reduce, no global-memory split-K)."""
    if o["mSplitK"] == SPLIT_K_GMEM:
        raise ValueError("global-memory split-K is not exported")
    a, b, c = o["mDtypeA"], o["mDtypeB"], o["mDtypeC"]
    v: dict[str, Any] = {name: 0 for name in layout["fields"]}
    for name, field in layout["fields"].items():
        if field["kind"] == "tensor_map":
            v[name] = b""
    pad_a = dtype_needs_padding(a, o["mMmaKind"])
    pad_b = dtype_needs_padding(b, o["mMmaKind"])
    shape, stride, tile = gemm_ab_shape_stride(o, p, MATRIX_A)
    v["tmaA"] = build_nd_tma_descriptor(
        encode, a, shape, stride, tile, ptr["a"], pad_a, True
    )
    shape, stride, tile = gemm_ab_shape_stride(o, p, MATRIX_B)
    v["tmaB"] = build_nd_tma_descriptor(
        encode, b, shape, stride, tile, ptr["b"], pad_b, not o["mSliceK"]
    )
    sfa = sf_dtype(a)
    if sfa is not None:
        shape, stride, tile = sf_shape_stride(
            p.m,
            p.k,
            o["mTileM"],
            o["mMmaTileK"],
            o["mSfLayoutA"],
            o["mSfReshapeFactor"],
            o["mSfBlockSizeA"],
        )
        v["tmaSfA"] = build_sf_tma_descriptor(
            encode, sfa, shape, stride, tile, ptr["sf_a"]
        )
    sfb = sf_dtype(b) if b != DTYPES["MxInt4"] else None
    if sfb is not None:
        shape, stride, tile = sf_shape_stride(
            p.n,
            p.k,
            o["mTileN"],
            o["mMmaTileK"],
            o["mSfLayoutB"],
            o["mSfReshapeFactor"],
            o["mSfBlockSizeB"],
        )
        v["tmaSfB"] = build_sf_tma_descriptor(
            encode, sfb, shape, stride, tile, ptr["sf_b"]
        )
    if o["mUseTmaStore"]:
        transposed = o["mTransposeMmaOutput"]
        tokens, hidden = (p.n, p.m) if transposed else (p.m, p.n)
        out_m = o["mEpilogueTileN"] if transposed else o["mEpilogueTileM"]
        out_n = o["mEpilogueTileM"] if transposed else o["mEpilogueTileN"]
        v["tmaC"] = build_nd_tma_descriptor(
            encode, c, [hidden, tokens], [1, hidden], [out_n, out_m], ptr["c"], False
        )
    v.update(
        ptrSfA=ptr["sf_a"],
        ptrSfB=ptr["sf_b"],
        ptrC=ptr["c"],
        ptrScaleC=ptr["scale_c"],
        ptrSfC=ptr["sf_c"],
        m=p.m,
        n=p.n,
        k=p.k,
        rank=0,
        tpGrpSize=1,
    )
    return pack_struct(layout, v)


# --- struct packing ------------------------------------------------------------


def pack_struct(layout: Mapping[str, Any], values: Mapping[str, Any]) -> bytes:
    """Pack ``values`` at the probe's field offsets; padding stays zero."""
    fields = layout["fields"]
    if set(values) != set(fields):
        raise ValueError(
            f"fields missing {set(fields) - set(values)}, "
            f"unknown {set(values) - set(fields)}"
        )
    buf = bytearray(layout["size"])
    formats = {"ptr": "<Q", "u64": "<Q", "i32": "<i"}
    for name, value in values.items():
        f = fields[name]
        offset, size, kind = f["offset"], f["size"], f["kind"]
        if kind in ("tensor_map", "i32_array"):
            data = bytes(value)
            if len(data) > size:
                raise ValueError(f"{name}: {len(data)} bytes exceed {size}")
            buf[offset : offset + len(data)] = data
            continue
        fmt = formats[kind]
        if struct.calcsize(fmt) != size:
            raise ValueError(f"{name}: {kind} is not {size} bytes")
        struct.pack_into(fmt, buf, offset, value)
    return bytes(buf)


def padding_mask(layout: Mapping[str, Any]) -> list[bool]:
    """True for every byte inside a field."""
    covered = [False] * layout["size"]
    for f in layout["fields"].values():
        covered[f["offset"] : f["offset"] + f["size"]] = [True] * f["size"]
    return covered


# --- launch ----------------------------------------------------------------------


class LaunchSpec(NamedTuple):
    grid: tuple[int, ...]
    block: tuple[int, int, int]
    params: bytes
    shared_mem: int
    cluster: tuple[int, int, int]
    pdl: bool
    scheduling_policy: int


def launch_spec(
    config: Mapping[str, Any],
    grid: tuple[int, ...],
    params: bytes,
    *,
    pdl_safe_fields: Sequence[str],
    enable_pdl: bool = True,
) -> LaunchSpec:
    """trtllm::gen::launchKernel: block (threads), dynamic smem, cluster,
    spread scheduling for multi-CTA clusters, PDL when usePdl && pdlSafe."""
    o = config["options"]
    cluster = (o["mClusterDimX"], o["mClusterDimY"], o["mClusterDimZ"])
    size = cluster[0] * cluster[1] * cluster[2]
    pdl_safe = any(o[f] for f in pdl_safe_fields)
    return LaunchSpec(
        grid,
        (config["threads"], 1, 1),
        params,
        config["shared_mem"],
        cluster,
        enable_pdl and pdl_safe,
        CLUSTER_SCHEDULING_SPREAD if size > 1 else CLUSTER_SCHEDULING_DEFAULT,
    )


BMM_PDL_FIELDS = (
    "mGridWaitForPrimaryRouting",
    "mGridWaitForPrimaryEarlyExit",
    "mGridWaitForPrimaryA",
    "mGridWaitForPrimaryB",
)
GEMM_PDL_FIELDS = (
    "mGridWaitForPrimaryEarlyExit",
    "mGridWaitForPrimaryA",
    "mGridWaitForPrimaryB",
)


def recorded_attrs(spec: LaunchSpec) -> list[dict[str, Any]]:
    """The launch attributes as the probes record them."""
    return [
        {"id": CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, "value": list(spec.cluster)},
        {
            "id": CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
            "value": spec.scheduling_policy,
        },
        {
            "id": CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            "value": int(spec.pdl),
        },
    ]


# --- MoE routing (runner.h, RoutingKernel.cuh) ---------------------------------


def max_num_ctas_in_batch_dim(tokens: int, top_k: int, experts: int, tile: int) -> int:
    """Routing::getMaxNumCtasInBatchDim."""
    remaining = tokens * top_k
    filled = min(experts, remaining)
    ctas = filled
    remaining -= filled
    if remaining > 0:
        ctas += remaining // tile
    return ctas


# --- FlashInfer v0.6.9 dispatch policy ---------------------------------------------
#
# Which runner launches a config, the problem it passes and which optional
# buffers its callers pass non-null. Shared by impls/batched_gemm/compiler.py
# (probe cases) and the workloads (launches), so both always agree; the
# compile-time probe runs the MoE runners themselves on every case, and the
# KernelParams byte compare (check_param_layouts) proves that moe_problem and
# moe_buffers reproduce what PermuteGemm1/Gemm2::Runner::run pass.


def moe_role(o: Mapping[str, Any]) -> str:
    """``fc1`` (PermuteGemm1: routed activations) or ``fc2`` (Gemm2)."""
    return "fc1" if o["mRouteImpl"] != ROUTE_NONE else "fc2"


def moe_gate_factor(o: Mapping[str, Any]) -> int:
    """FC1 weight rows per intermediate channel: 2 for gated activations
    (fused, or DeepSeek FP8 whose activation kernel runs separately; the MoE
    runner always builds its DeepSeek runner with SwiGlu)."""
    if moe_role(o) != "fc1":
        return 1
    return 2 if (o["mFusedAct"] or o["mUseDeepSeekFp8"]) else 1


def moe_activation(o: Mapping[str, Any]) -> str:
    """The MoE ``ActivationType`` whose PermuteGemm1 options build this routed
    config (bmm_moe_probe.cu ``config_activation``)."""
    if o["mUseDeepSeekFp8"]:
        return "Swiglu"
    if o["mFusedAct"]:
        return "Swiglu" if o["mActType"] == ACT_SWIGLU else "Geglu"
    return "Relu2" if o["mEltwiseActType"] == ELTWISE_RELU2 else "Identity"


def moe_problem(o: Mapping[str, Any], params: Mapping[str, int]) -> BmmProblem:
    """PermuteGemm1/Gemm2::run's runner problem for MoE ``params`` (tokens,
    top_k, experts, hidden, intermediate), in the kernel's (transposed) view."""
    fc1 = moe_role(o) == "fc1"
    m = moe_gate_factor(o) * params["intermediate"] if fc1 else params["hidden"]
    k = params["hidden"] if fc1 else params["intermediate"]
    ctas = max_num_ctas_in_batch_dim(
        params["tokens"], params["top_k"], params["experts"], o["mTileN"]
    )
    return BmmProblem(m, params["tokens"], k, params["tokens"], params["experts"], ctas)


# Optional runtime features a case switches on (params key: 0/1) and which
# FlashInfer v0.6.9 launcher can pass them (trtllm_fused_moe_kernel_launcher.cu):
#   bias            gemm1_bias / gemm2_bias: FP4BlockScaleLauncher (NvFP4 and
#                   MxFP4 weights), configs with BiasType M;
#   gated_act       gemm1_alpha / gemm1_beta / gemm1_clamp_limit:
#                   FP4BlockScaleLauncher and MxInt4BlockScaleLauncher, fused
#                   gated FC1 configs (clamp only with SwiGlu here, the form
#                   KernelParamsDecl.h documents);
#   routing_scales  token_scales = expert_weights (use_routing_scales_on_input,
#                   Llama4 routing, top-1): Fp8PerTensorLauncher, FC1 configs
#                   with per-token B scales;
#   zero            all-zero hidden states (upstream zero_hidden_states).
MOE_FEATURES = ("bias", "gated_act", "routing_scales", "zero")


def moe_launcher_weights(o: Mapping[str, Any]) -> str:
    return dtype_name(o["mDtypeA"])


def moe_supports(o: Mapping[str, Any], feature: str) -> bool:
    """Whether a FlashInfer launcher can switch ``feature`` on for this config."""
    fp4 = o["mDtypeA"] in (DTYPES["E2m1"], DTYPES["MxE2m1"])
    fc1 = moe_role(o) == "fc1"
    if feature == "bias":
        return fp4 and o["mBiasType"] == 1
    if feature == "gated_act":
        return (
            fc1 and bool(o["mFusedAct"]) and (fp4 or o["mDtypeA"] == DTYPES["MxInt4"])
        )
    if feature == "routing_scales":
        return fc1 and bool(o["mUsePerTokenSfB"])
    if feature == "zero":
        return True
    raise ValueError(feature)


def moe_buffers(
    o: Mapping[str, Any], params: Mapping[str, Any] | None = None
) -> list[str]:
    """Optional buffers FlashInfer v0.6.9's MoE launchers pass non-null for
    this config's quantization mode (``trtllm_fused_moe_kernel_launcher.cu``)
    and the case's features (``MOE_FEATURES``); everything else keeps
    FlashInfer's default (None)."""
    params = params or {}
    a, b = o["mDtypeA"], o["mDtypeB"]
    ds = bool(o["mUseDeepSeekFp8"])
    fc1 = moe_role(o) == "fc1"
    out = []
    if sf_dtype(a) is not None or ds:
        out.append("sf_a")
    if sf_dtype(b) is not None or ds:
        out.append("sf_b")
    # Output scales are required for E2m1/E4m3 activations (FP4 and FP8
    # per-tensor launchers); the BF16, FP8 block-scale and MxInt4 launchers
    # never pass them, the FP4 launcher's MxE2m1-weight modes leave them None.
    scaled = b in (DTYPES["E2m1"], DTYPES["E4m3"]) and not ds
    if fc1:
        # gemm1_output_scale: DeepSeek, MxFP8 and NVFP4 outputs, and the
        # buffer the FP8 per-tensor launcher allocates (unused by its kernel).
        per_tensor = a == DTYPES["E4m3"] and b == DTYPES["E4m3"] and not ds
        if ds or b in (DTYPES["E2m1"], DTYPES["MxE4m3"]) or per_tensor:
            out.append("sf_c")
        if scaled:
            out += ["scale_c", "scale_gate"]
    elif scaled:
        out.append("scale_c")
    if params.get("bias") and moe_supports(o, "bias"):
        out.append("bias")
    if fc1 and params.get("gated_act") and moe_supports(o, "gated_act"):
        out += ["alpha", "beta"]
        if o["mActType"] == ACT_SWIGLU:
            out.append("clamp_limit")
    if fc1:
        out.append("route_map")
    if params.get("routing_scales") and moe_supports(o, "routing_scales"):
        out.append("per_token_sf_b")
    return out


def next_power_of_two(value: float) -> int:
    """nextPowerOfTwo of trtllm_fused_moe_kernel_launcher.cu."""
    n = math.ceil(value)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def tile_center(ladder: Sequence[int], tokens: int, top_k: int, experts: int) -> int:
    """computeSelectedTileN's heuristic tile: the average tokens per local
    expert rounded up to a power of two, clamped to the ladder (float32
    arithmetic as upstream)."""
    avg = float(struct.unpack("<f", struct.pack("<f", tokens * top_k / experts))[0])
    return min(max(next_power_of_two(avg), ladder[0]), ladder[-1])


def selected_tiles(
    ladder: Sequence[int], tokens: int, top_k: int, experts: int
) -> list[int]:
    """computeSelectedTileN: the center tile, the next two and the previous
    (the compiler checks this port against the probe's verbatim copy)."""
    i = list(ladder).index(tile_center(ladder, tokens, top_k, experts))
    return sorted(ladder[max(i - 1, 0) : i + 3])


B200_SMEM_PER_SM = 233472  # 228 KiB
B200_THREADS_PER_SM = 2048
SCHEDULERS_PERSISTENT = (SCHEDULER_PERSISTENT, SCHEDULER_STATIC_PERSISTENT,
                         SCHEDULER_PERSISTENT_SM90)  # fmt: skip


def resident_ctas(entry: Mapping[str, Any], sm_count: int = 148) -> int:
    """An upper bound of the CTAs a B200 keeps resident (dynamic shared memory
    plus the 1 KiB per-CTA reservation, threads; registers can only lower it)."""
    per_sm = min(
        B200_SMEM_PER_SM // (entry["shared_mem"] + 1024),
        B200_THREADS_PER_SM // entry["threads"],
    )
    return sm_count * max(per_sm, 1)


def k_stages(o: Mapping[str, Any]) -> int:
    """The mainloop's pipeline depth (smem stages of A/B)."""
    return max(o.get("mNumStagesA", 0), o.get("mNumStagesB", 0), o.get("mNumStages", 0))


def moe_regime(entry: Mapping[str, Any], params: Mapping[str, Any]) -> tuple:
    """The code-path class of a MoE case on this kernel: FlashInfer's center
    tile for the problem (how full the token tiles are), whether the K loop
    wraps the smem pipeline, whether a persistent scheduler hands some CTAs a
    second tile, and the switched-on features."""
    o = entry["options"]
    p = moe_problem(o, params)
    k_tiles = ceil_div(p.k, o["mTileK"] * o["mNumSlicesForSplitK"])
    grid = div_up_mul(ceil_div(p.m, o["mTileM"]), o["mClusterDimX"]) * div_up_mul(
        p.max_num_ctas, o["mClusterDimY"]
    )
    multi = o["mTileScheduler"] in SCHEDULERS_PERSISTENT and grid > resident_ctas(entry)
    features = tuple(f for f in MOE_FEATURES if params.get(f) and moe_supports(o, f))
    return (
        (
            tile_center(
                entry["ladder"], params["tokens"], params["top_k"], params["experts"]
            )
            if entry.get("ladder")
            else 0
        ),
        k_tiles > k_stages(o),
        multi,
        features,
    )


GEMM_BUFFERS = {
    "gemm_fp4": ("sf_a", "sf_b", "scale_c"),  # mm_fp4: alpha (global scale)
    "gemm_mxfp8": ("sf_a", "sf_b"),  # mm_mxfp8: no alpha
    "gemm_fp8_blockscale": ("sf_a", "sf_b"),  # gemm_fp8_nt_groupwise
    "gemm_low_latency": ("scale_c",),  # trtllm_low_latency_gemm(global_scale)
}

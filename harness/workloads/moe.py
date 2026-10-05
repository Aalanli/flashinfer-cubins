"""Routed FP8 block-scale MoE (DeepSeek-V3 routing) as single-kernel workloads.

Official definition: ``moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048``
(``resources/moe.json``). Upstream, FlashInfer v0.2.10's
``trtllm_fp8_block_scale_moe`` runs this pipeline (see
``impls/moe/compiler.py`` for the dispatch and the live/dead table); each
kernel is a workload here and owns exactly the token counts upstream sends to
it:

======================  =======  ====================================  ==============
workload                arch     kernel                                tokens
======================  =======  ====================================  ==============
moe_routing_main        86,100a  routingMainKernel                     all
moe_routing_cluster     100a     routingIndicesClusterKernel           <= 1024
moe_routing_coop        100a     routingIndicesCoopKernel (coop.)      1025..262144
moe_gemm1               100a     trtllm-gen FC1, static scheduler      <= 32
moe_gemm1_persistent    100a     trtllm-gen FC1, persistent            >= 33
moe_activation          86,100a  activationDeepSeekKernel              all
moe_gemm2               100a     trtllm-gen FC2, static scheduler      <= 16
moe_gemm2_persistent    100a     trtllm-gen FC2, persistent            >= 17
moe_finalize            86,100a  finalizeKernel                        <= 42
moe_finalize_vec        86,100a  finalizeKernelVecLoad                 >= 43
======================  =======  ====================================  ==============

Every workload generates the official inputs it needs (seeded) and produces
its kernel's inputs from them with PyTorch versions of the earlier stages.
Buffers are sized as upstream's launcher allocates them. Scratch and outputs
are torch allocations; parameter structs (including the trtllm-gen GEMMs'
0x4380-byte ``KernelParams`` and its TMA descriptors) are built here,
mirroring the upstream host code, and are checked byte for byte against the
compile-time probe's recordings by ``tests/test_moe.py``.

Cases vary what the definition leaves to run time: the token count (which
selects the kernel), the local expert offset and the routing distribution
(``ROUTING_MODES``: near-balanced, upstream's test distribution, model-like
expert popularity, a few hot experts spanning many tiles, no local expert,
and exact ties). Exact ties are defined: upstream's top-k reductions prefer
the lower index, and so do the references. Near-ties are not: the kernel's
sigmoid uses ``tanhf`` under ``-use_fast_math``, so values within
``TIE_MARGIN`` of each other may order differently from an FP64 reference;
rows with such a near-tie are resampled. The routing permutation within an
expert depends on atomic order and is validated as a valid permutation; later
stages use the reference's (token-ordered) permutation.

Smoke cases (``SMOKE``) cover every dispatch boundary, both ends of the local
offset range and an unaligned offset, experts spanning 1 to ~340 tiles with
partial last tiles, empty experts, a rank with no local token, exact ties, and
persistent GEMM grids several times larger than the 148 resident CTAs.
Throughput cases (``throughput_cases``) are every row of the official
inventory, DeepSeek-V3 decode/prefill token counts (the definition is
DeepSeek-V3's MoE layer at EP8) and 16384-token stress with balanced and hot
routing. ``UPSTREAM_TESTS`` maps FlashInfer's own test of this pipeline onto
smoke cases.
"""

from __future__ import annotations

import functools
import json
import struct
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, NamedTuple

import torch

from .. import cuda_driver
from ..cutlass_host import ceil_div, driver_encode
from ..registry import register
from ..throughput import inventory, model_case, synthetic, trace, upstream_case
from ..workload import CaseSpec, Workload

NUM_EXPERTS, TOP_K, N_GROUP, TOPK_GROUP = 256, 8, 8, 4
EXPERTS_PER_GROUP = NUM_EXPERTS // N_GROUP
HIDDEN, INTERMEDIATE, LOCAL_EXPERTS, TILE = 7168, 2048, 32, 8
GEMM1_N = 2 * INTERMEDIATE
BLOCK = 128
PADDING_LOG2 = 3
E4M3_MAX = 448.0
ROUTED_SCALING_FACTOR = 2.5
CLUSTER_MAX_TOKENS = 1024  # routingDeepSeek::run: useSingleCluster
COOP_BLOCKS = 128
COOP_MAX_TOKENS = COOP_BLOCKS * 256 * 64 // TOP_K
FINALIZE_THREADS = 256
HISTOGRAM_RING = 1024  # pre-zeroed coop histograms per prepared launch
TIE_MARGIN = 2e-3
WEIGHT_SEED = 0x5EED  # expert weights are model parameters: one set for all cases

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
    """The probe's deterministic CUtensorMap stand-in (``fake_tensor_map`` in
    ``impls/moe/kernels/moe_probe.cu``); for tests only."""
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


# --- upstream sizing helpers (runner.h / IntFastDiv.h) --------------------------


def max_permuted_padded_count(tokens: int, num_experts: int = NUM_EXPERTS) -> int:
    """Routing::getMaxPermutedPaddedCount(tokens, top_k, num_experts, tile)."""
    count = tokens * TOP_K + (TILE - 1) * num_experts
    return ceil_div(count, TILE) * TILE


def max_num_ctas(tokens: int, num_experts: int = NUM_EXPERTS) -> int:
    """Routing::getMaxNumCtasInBatchDim(tokens, top_k, num_experts, tile)."""
    bound = min(tokens * TOP_K, num_experts) * ceil_div(tokens, TILE)
    return min(bound, ceil_div(max_permuted_padded_count(tokens, num_experts), TILE))


def int_fast_div(divisor: int) -> tuple[int, int, int, int]:
    """trtllm::dev::IntFastDiv(divisor) as (divisor, magic_m, magic_s, add_sign)."""
    if divisor == 1:
        return 1, 0, -1, 1
    if divisor == -1:
        return -1, 0, -1, -1
    two31 = 0x80000000
    ad = abs(divisor)
    t = two31 + ((divisor & 0xFFFFFFFF) >> 31)
    anc = t - 1 - t % ad
    p = 31
    q1, r1 = two31 // anc, two31 - (two31 // anc) * anc
    q2, r2 = two31 // ad, two31 - (two31 // ad) * ad
    while True:
        p += 1
        q1, r1 = (2 * q1) & 0xFFFFFFFF, (2 * r1) & 0xFFFFFFFF
        if r1 >= anc:
            q1, r1 = (q1 + 1) & 0xFFFFFFFF, r1 - anc
        q2, r2 = (2 * q2) & 0xFFFFFFFF, (2 * r2) & 0xFFFFFFFF
        if r2 >= ad:
            q2, r2 = (q2 + 1) & 0xFFFFFFFF, r2 - ad
        delta = ad - r2
        if not (q1 < delta or (q1 == delta and r1 == 0)):
            break
    magic = (q2 + 1) & 0xFFFFFFFF
    if divisor < 0:
        magic = (-magic) & 0xFFFFFFFF
    magic_signed = magic - (1 << 32) if magic >= 1 << 31 else magic
    add_sign = (
        1
        if divisor > 0 and magic_signed < 0
        else -1 if divisor < 0 and magic_signed > 0 else 0
    )
    return divisor, magic_signed, p - 32, add_sign


# --- parameter structs --------------------------------------------------------


class Struct:
    """A by-value parameter struct laid out as the compile-time probe recorded."""

    def __init__(self, layout: dict[str, Any]):
        self.size: int = layout["size"]
        self.fields: dict[str, dict[str, Any]] = layout["fields"]

    def pack(self, values: dict[str, Any]) -> bytes:
        """Every recorded field must be given; padding stays zero."""
        missing, extra = set(self.fields) - set(values), set(values) - set(self.fields)
        if missing or extra:
            raise ValueError(f"param fields missing {missing}, unknown {extra}")
        buf = bytearray(self.size)
        for name, value in values.items():
            f = self.fields[name]
            offset, size, kind = f["offset"], f["size"], f["kind"]
            if kind in ("bytes", "tensor_map"):
                data = bytes(value)
                if len(data) > size:
                    raise ValueError(f"{name}: {len(data)} bytes exceed {size}")
                buf[offset : offset + len(data)] = data
                continue
            fmt = {
                "ptr": "<Q",
                "u64": "<Q",
                "i32": "<i",
                "f32": "<f",
                "int_fast_div": "<4i",
            }[kind]
            if struct.calcsize(fmt) != size:
                raise ValueError(f"{name}: {kind} is not {size} bytes")
            if kind == "int_fast_div":
                struct.pack_into(fmt, buf, offset, *int_fast_div(int(value)))
            else:
                struct.pack_into(fmt, buf, offset, value)
        return bytes(buf)


class LaunchSpec(NamedTuple):
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    params: bytes
    shared_mem: int = 0
    cluster: tuple[int, int, int] | None = None
    cooperative: bool = False


Pointers = dict[str, int]


def routing_params(
    layout: Struct, tokens: int, offset: int, scale: float, p: Pointers
) -> bytes:
    """routingDeepSeek::KernelParams::setKernelParams for Routing::Runner::run's
    Data (identical for the main, cluster and coop kernels of one call)."""
    counts = p["expert_count_histogram"] if tokens > CLUSTER_MAX_TOKENS else 0
    return layout.pack(
        {
            "mPtrExpertCounts": counts,  # nullptr on the single-cluster path
            "mPtrPermutedIdxSize": p["total_num_padded_tokens"],
            "mPtrExpandedIdxToPermutedIdx": p["expanded_idx_to_permuted_idx"],
            "mPtrPermutedIdxToTokenIdx": p["permuted_idx_to_token_idx"],
            "mPtrCtaIdxXyToBatchIdx": p["cta_idx_xy_to_batch_idx"],
            "mPtrCtaIdxXyToMnLimit": p["cta_idx_xy_to_mn_limit"],
            "mPtrNumNonExitingCtas": p["num_non_exiting_ctas"],
            "mPtrExpertWeights": p["expert_weights"],
            "mPtrScores": p["routing_logits"],
            "mNumTokens": tokens,
            "mNumExperts": NUM_EXPERTS,
            "mPaddingLog2": PADDING_LOG2,
            "mLocalExpertsStartIdx": offset,
            "mLocalExpertsStrideLog2": 0,
            "mNumLocalExperts": LOCAL_EXPERTS,
            "mPtrExpertIdx": p["expert_indexes"],
            "mPtrRoutingBias": p["routing_bias"],
            "mNumExpertGroups": N_GROUP,
            "mNumExpertsPerGroup": EXPERTS_PER_GROUP,
            "mNumLimitedGroups": TOPK_GROUP,
            "mTopK": TOP_K,
            "mRouteScale": scale,
        }
    )


def activation_params(layout: Struct, tokens: int, p: Pointers) -> bytes:
    """activation::KernelParams::setKernelParams (MoE::Runner::setOpsData)."""
    return layout.pack(
        {
            "inPtr": p["gemm1_output"],
            "outPtr": p["activation_output"],
            "inDqSfsPtr": p["gemm1_output_scale"],
            "outDqSfsPtr": p["activation_output_scale"],
            "innerDim": GEMM1_N,
            "numTokens": tokens,
            "topK": TOP_K,
            "expandedIdxToPermutedIdx": p["expanded_idx_to_permuted_idx"],
            "totalNumPaddedTokens": p["total_num_padded_tokens"],
        }
    )


def finalize_params(layout: Struct, tokens: int, p: Pointers) -> bytes:
    """finalize::KernelParams::setKernelParams (MoE::Runner::setOpsData; no
    gemm2/output scales, routing weights applied here)."""
    return layout.pack(
        {
            "inPtr": p["gemm2_output"],
            "expertWeightsPtr": p["expert_weights"],
            "outPtr": p["output"],
            "inDqSfsPtr": 0,
            "outDqSfsPtr": 0,
            "expandedIdxToPermutedIdx": p["expanded_idx_to_permuted_idx"],
            "hiddenDim": HIDDEN,
            "hiddenDimPadded": HIDDEN,
            "numTokens": tokens,
            "numExperts": NUM_EXPERTS,
            "topK": TOP_K,
            "totalNumPaddedTokens": p["total_num_padded_tokens"],
        }
    )


# tg::Dtype values of KernelMetaInfo.h used by these configs.
DTYPE_E4M3, DTYPE_BF16 = 1050629, 1052672
CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 0, 9
SWIZZLE = {128: 3, 64: 2, 32: 1}
L2_PROMOTION_128B = 2
ROUTE_IMPL_NO_ROUTE, ROUTE_IMPL_LDGSTS = 0, 1


def _nd_descriptor(
    encode: TensorMapEncoder,
    dtype: int,
    bits: int,
    shape: Sequence[int],
    stride: Sequence[int],
    tile: Sequence[int],
    address: int,
) -> bytes:
    """gemm::buildNdTmaDescriptor (TmaDescriptor.h) for E4m3 / Bfloat16 with
    swizzling, MmaKind without padding."""
    data_type = {
        DTYPE_E4M3: CU_TENSOR_MAP_DATA_TYPE_UINT8,
        DTYPE_BF16: CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    }[dtype]
    fastest = tile[0] * bits // 8
    swizzle = next((v for k, v in SWIZZLE.items() if fastest % k == 0), None)
    if swizzle is None:
        raise ValueError(f"unexpected fastest-dim tile size {fastest}")
    if address % 16:
        raise ValueError("TMA global address must be 16-byte aligned")
    strides_bytes = [s * bits // 8 for s in stride[1:]]
    elements_in_128b = (32 // bits) * 32
    box = [min(elements_in_128b, tile[0]), *tile[1:]]
    box += [1] * (len(shape) - len(box))
    if any(b > 256 for b in box[1:]):
        raise ValueError("TMA box dimension too large")
    return encode(
        data_type,
        address,
        list(shape),
        strides_bytes,
        box,
        [1] * len(shape),
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        swizzle,
        L2_PROMOTION_128B,
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )


def gemm_params(
    layout: Struct,
    options: dict[str, Any],
    *,
    m: int,
    k: int,
    tokens: int,
    a: int,
    sf_a: int,
    b: int,
    sf_b: int,
    c: int,
    sf_c: int,
    route_map: int,
    per_token_sf_b: int,
    total_num_padded_tokens: int,
    cta_idx_xy_to_batch_idx: int,
    cta_idx_xy_to_mn_limit: int,
    num_non_exiting_ctas: int,
    encode: TensorMapEncoder,
) -> tuple[bytes, tuple[int, int, int]]:
    """KernelParamsSetup::setKernelParams and the grid of BatchedGemmInterface::run
    for this package's configs: transposed MMA output (batch N = tokens),
    dynamic batch with early exit, K-major A (weights, 3-D per expert),
    E4m3 A/B with DeepSeek FP8 scales, TMA store of C, no split-K/OOB trick.

    ``m``, ``k`` are the kernel's (transposed) problem: FC1 m = 4096, k = 7168;
    FC2 m = 7168, k = 2048. Returns (params, grid).
    """
    o = options
    expected = {
        "transpose_mma_output": 1,
        "is_static_batch": 0,
        "enables_early_exit": 1,
        "enables_delayed_early_exit": 0,
        "use_tma_store": 1,
        "use_tma_oob_opt": 0,
        "fused_act": 0,
        "layout_a": 0,
        "layout_b": 0,
        "num_slices_for_split_k": 1,
        "use_deepseek_fp8": 1,
        "dtype_a": DTYPE_E4M3,
        "dtype_b": DTYPE_E4M3,
        "cluster": [1, 1, 1],
    }
    for key, value in expected.items():
        if o[key] != value:
            raise ValueError(f"GEMM config {key}={o[key]} is not supported")
    if o["route_impl"] not in (ROUTE_IMPL_NO_ROUTE, ROUTE_IMPL_LDGSTS):
        raise ValueError("unsupported route implementation")
    tile_m, tile_n, tile_k = o["tile_m"], o["tile_n"], o["tile_k"]
    if m % tile_m:
        raise ValueError("mM must be a multiple of tileM")
    # Grid y = mMaxNumCtasInTokenDim: getMaxNumCtasInBatchDim over the local
    # experts (PermuteGemm1/Gemm2 pass local_num_experts as numExperts).
    cta_offset = max_num_ctas(tokens, LOCAL_EXPERTS)
    input_tokens = cta_offset * tile_n
    bits_a, bits_b, bits_c = o["dtype_a_bits"], o["dtype_b_bits"], o["dtype_c_bits"]
    # makeTmaShapeStrideAbc: A (weights) (B, M, K); B (activations) (padded, K);
    # C transposed (padded tokens, M) with the epilogue tile as box.
    tma_a = _nd_descriptor(
        encode,
        o["dtype_a"],
        bits_a,
        [k, m, LOCAL_EXPERTS],
        [1, k, k * m],
        [tile_k, tile_m],
        a,
    )
    tma_b = b""
    if o["route_impl"] == ROUTE_IMPL_NO_ROUTE:
        tma_b = _nd_descriptor(
            encode, o["dtype_b"], bits_b, [k, input_tokens], [1, k], [tile_k, tile_n], b
        )
    tma_c = _nd_descriptor(
        encode,
        o["dtype_c"],
        bits_c,
        [m, input_tokens],
        [1, m],
        [o["epilogue_tile_m"], o["epilogue_tile_n"]],
        c,
    )
    params = layout.pack(
        {
            "tmaA": tma_a,
            "tmaB": tma_b,  # unset (indeterminate upstream) for the ldgsts route
            "tmaC": tma_c,
            "tmaSfA": b"",  # unset: no block-scale descriptors for E4m3
            "tmaSfB": b"",
            "ptrA": a,
            "strideInBytesA": k * bits_a // 8,
            "ptrB": b,
            "strideInBytesB": k * bits_b // 8,
            "ptrC": 0,  # TMA store
            "ptrScaleC": 0,
            "ptrScaleGate": 0,
            "ptrClampLimit": 0,
            "ptrSwiGluAlpha": 0,
            "ptrSwiGluBeta": 0,
            "k": k,
            "nm": m,
            "tileStridePerBatch": m // tile_m,
            "ptrDqSfsC": sf_c if o["dtype_c"] == DTYPE_E4M3 else 0,
            "ptrSfA": sf_a,
            "ptrSfB": sf_b,
            "ptrPerTokenSfA": 0,
            # PermuteGemm1 passes the routing weights as perTokensSfA, which the
            # transposed runner stores here (unused: no per-token scaling).
            "ptrPerTokenSfB": per_token_sf_b,
            "ptrBias": 0,
            "ptrSfC": sf_c,
            "ptrRouteMap": route_map,
            "numTokens": tokens,
            "numBatches": LOCAL_EXPERTS,
            "ptrNumNonExitingCtas": num_non_exiting_ctas,
            "ptrTotalNumPaddedTokens": total_num_padded_tokens,
            "ptrCtaIdxXyToBatchIdx": cta_idx_xy_to_batch_idx,
            "ptrCtaIdxXyToMnLimit": cta_idx_xy_to_mn_limit,
            "totalNumPaddedTokens": 0,  # unset (indeterminate) for dynamic batch
            "ctaIdxXyToBatchIdx": b"",
            "ctaIdxXyToMnLimit": b"",
            "rank": 0,
            "tpGrpSize": 1,
            "ptrPartialRowMax": 0,
            "ptrRowMaxCompletionBars": 0,
        }
    )
    return params, (m // tile_m, cta_offset, 1)


# --- input generation ---------------------------------------------------------


def _generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _routing_choice(logits: torch.Tensor, bias: torch.Tensor):
    """(selected experts [T, 8] in kernel order, sigmoid scores, margin [T]).

    Upstream's top-k reductions order candidates by (value desc, index asc)
    (``TopKRedType`` packs ``65535 - idx`` below the value bits): of tied
    groups and tied experts the lower index wins. Both choices here use a
    stable descending sort to reproduce that.

    ``margin`` is the smallest gap the kernel's FP32 fast-math evaluation
    must resolve the same way: between every selected and unselected group
    and between consecutive experts of the top 9. Exactly equal values (in
    FP64, from identical logits and biases) are exact ties in the kernel too
    and are resolved by index; they do not count.
    """
    scores = (0.5 * torch.tanh(0.5 * logits.double()) + 0.5).float()
    biased = scores.double() + bias.double()
    tokens = logits.shape[0]
    top2 = biased.view(tokens, N_GROUP, EXPERTS_PER_GROUP).topk(2, dim=-1).values
    group = top2.sum(-1)
    gorder = group.sort(dim=-1, descending=True, stable=True).indices
    kept, dropped = gorder[:, :TOPK_GROUP], gorder[:, TOPK_GROUP:]
    gap = group.gather(1, kept)[:, :, None] - group.gather(1, dropped)[:, None, :]
    group_margin = gap.masked_fill(gap == 0, torch.inf).flatten(1).min(-1).values
    mask = torch.zeros_like(group, dtype=torch.bool).scatter_(1, kept, True)
    masked = biased.masked_fill(~mask.repeat_interleave(EXPERTS_PER_GROUP, -1), -1e30)
    values, order = masked.sort(dim=-1, descending=True, stable=True)
    steps = values[:, :TOP_K] - values[:, 1 : TOP_K + 1]
    expert_margin = steps.masked_fill(steps == 0, torch.inf).min(-1).values
    return order[:, :TOP_K], scores, torch.minimum(group_margin, expert_margin)


# Routing input distributions (``routing`` case parameter):
#
# * ``uniform``: logits ~ N(0, 1), bias ~ N(0, 0.1^2): near-balanced experts.
# * ``upstream``: logits ~ N(0, 1), bias ~ N(0, 1), as FlashInfer's
#   ``test_moe_quantization_classes`` (DSv3) samples them.
# * ``popular``: model-like skewed expert popularity: a fixed per-expert logit
#   shift ~ N(0, POPULARITY_STD^2) (one draw for all cases, like a trained
#   router) on top of per-token N(0, 1) logits.
# * ``hot``: ``uniform`` plus +0.5 bias on the local experts HOT_LOCAL: those
#   experts take (nearly) every token, so they span many 8-token tiles while
#   most other local experts stay empty.
# * ``remote``: ``uniform`` with bias -2 on every local expert: no token
#   routes to this rank (zero CTAs; early exit in the GEMMs).
# * ``ties``: logits on the lattice {0, 0.5, ..., 4} and zero bias: exact
#   ties between experts and between groups everywhere, resolved by index.
ROUTING_MODES = ("uniform", "upstream", "popular", "hot", "remote", "ties")
HOT_LOCAL = (0, 13, 31)
HOT_BIAS = 0.5
REMOTE_BIAS = -2.0
POPULARITY_STD = 0.7
POPULARITY_SEED = 0x9091


def _sample_logits(
    device: torch.device, rows: int, routing: str, g: torch.Generator
) -> torch.Tensor:
    logits = torch.randn((rows, NUM_EXPERTS), device=device, generator=g)
    if routing == "ties":
        logits = (2 * logits + 2).round().clamp(0, 8) / 2
    elif routing == "popular":
        shift = torch.randn(
            NUM_EXPERTS, generator=torch.Generator().manual_seed(POPULARITY_SEED)
        )
        logits += POPULARITY_STD * shift.to(device)
    return logits


def official_routing_inputs(
    device: torch.device,
    tokens: int,
    seed: int,
    offset: int = 0,
    routing: str = "uniform",
    require_local: bool = False,
):
    """(routing_logits [T, 256] f32, routing_bias [256] bf16) of ``routing``
    (see ROUTING_MODES), without near-ties (``_routing_choice``'s margin).

    With ``require_local``, logits are also resampled until at least one
    token routes to a local expert in ``[offset, offset + 32)``; otherwise the
    GEMM and activation stages would have no defined output to validate
    (not for ``remote``, whose point is that none does).
    """
    if routing not in ROUTING_MODES:
        raise ValueError(f"unknown routing distribution {routing!r}")
    # The bias is a model parameter: drawn on the CPU, the same on every device.
    bias = torch.randn(NUM_EXPERTS, generator=torch.Generator().manual_seed(seed))
    if routing == "ties":
        bias.zero_()
    elif routing != "upstream":
        bias *= 0.1
    local = slice(offset, offset + LOCAL_EXPERTS)
    if routing == "hot":
        bias[[offset + e for e in HOT_LOCAL]] += HOT_BIAS
    elif routing == "remote":
        bias[local] = REMOTE_BIAS
    bias = bias.to(device=device, dtype=torch.bfloat16)
    g = _generator(device, seed * 16 + 2)
    logits = _sample_logits(device, tokens, routing, g)
    require_local = require_local and routing != "remote"
    for _ in range(200):
        chosen, _, margin = _routing_choice(logits, bias)
        bad = (margin < TIE_MARGIN).nonzero().flatten()
        routed = (chosen >= offset) & (chosen < offset + LOCAL_EXPERTS)
        if not bad.numel() and require_local and not routed.any():
            bad = torch.arange(tokens, device=device)
        if not bad.numel():
            return logits, bias
        logits[bad] = _sample_logits(device, bad.numel(), routing, g)
    raise RuntimeError("could not sample tie-free routing logits")


def case_routing_inputs(device: torch.device, case: CaseSpec, require_local=False):
    """``official_routing_inputs`` for a case's tokens, offset and routing."""
    p = case.params
    return official_routing_inputs(
        device,
        p["tokens"],
        case.seed,
        p["offset"],
        p.get("routing", "uniform"),
        require_local,
    )


def official_activations(
    device: torch.device, tokens: int, seed: int, routing: str = "uniform"
):
    """(hidden_states [T, 7168] e4m3, hidden_states_scale [56, T] f32).

    Scales ~ U(0.5, 1); for ``upstream`` cases as FlashInfer's test makes
    them: E4M3 of 2 * N(0, 1) with every scale 2.
    """
    g = _generator(device, seed * 16 + 3)
    hidden = torch.randn((tokens, HIDDEN), device=device, generator=g)
    if routing == "upstream":
        hidden *= 2
    hidden = hidden.to(torch.float8_e4m3fn)
    g = _generator(device, seed * 16 + 4)
    scale = (
        torch.rand((HIDDEN // BLOCK, tokens), device=device, generator=g) * 0.5 + 0.5
    )
    if routing == "upstream":
        scale.fill_(2.0)
    return hidden, scale


def case_activations(device: torch.device, case: CaseSpec):
    """``official_activations`` of a case."""
    p = case.params
    return official_activations(
        device, p["tokens"], case.seed, p.get("routing", "uniform")
    )


@functools.lru_cache(maxsize=2)
def _weights(device_key: str, which: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device_key)
    rows, cols = (GEMM1_N, HIDDEN) if which == 1 else (HIDDEN, INTERMEDIATE)
    g = _generator(device, WEIGHT_SEED + which)
    weights = torch.empty(
        (LOCAL_EXPERTS, rows, cols), device=device, dtype=torch.float8_e4m3fn
    )
    for e in range(LOCAL_EXPERTS):  # expert by expert: no multi-GB FP32 temporary
        weights[e] = torch.randn((rows, cols), device=device, generator=g).to(
            torch.float8_e4m3fn
        )
    scale = (
        torch.rand(
            (LOCAL_EXPERTS, rows // BLOCK, cols // BLOCK), device=device, generator=g
        )
        * 0.05
        + 0.05
    )
    return weights, scale


def official_weights(device: torch.device, which: int):
    """gemm{1,2}_weights (e4m3) and gemm{1,2}_weights_scale (f32); shared by
    all cases (model parameters)."""
    key = str(
        device if device.type == "cpu" else torch.device("cuda", device.index or 0)
    )
    return _weights(key, which)


# --- PyTorch references of the stages ------------------------------------------
#
# Buffers are laid out as upstream's launcher allocates them. Rows and scale
# entries a kernel never writes (padding of each expert's last CTA, buffer
# tails) are zero in stage *inputs* and NaN in workload *references*, and are
# not compared.


class Routing(NamedTuple):
    """Reference routing outputs (upstream buffer layouts)."""

    expert_indexes: torch.Tensor  # int32 [T, 8] packed (bf16 score, int16 idx)
    expert_weights: torch.Tensor  # bf16 [T, 8]
    expanded_idx_to_permuted_idx: torch.Tensor  # int32 [T * 8], -1 if not local
    permuted_idx_to_token_idx: torch.Tensor  # int32 [max_padded], padding 0
    total_num_padded_tokens: torch.Tensor  # int32 [1]
    cta_idx_xy_to_batch_idx: torch.Tensor  # int32 [max_ctas], unused 0
    cta_idx_xy_to_mn_limit: torch.Tensor  # int32 [max_ctas], unused 0
    num_non_exiting_ctas: torch.Tensor  # int32 [1]


def routing_main_reference(logits, bias, scale: float):
    """(expert_indexes packed int32 [T, 8], expert_weights bf16 [T, 8])."""
    selected, scores, _ = _routing_choice(logits, bias)
    chosen = scores.gather(1, selected)
    # finalScore = scoreNorm * mRouteScale / redNorm, in FP32, then BF16.
    weights = (chosen * torch.tensor(scale, dtype=torch.float32)) / chosen.sum(
        -1, keepdim=True
    )
    weights = weights.to(torch.bfloat16)
    bits = weights.view(torch.int16).to(torch.int32) & 0xFFFF
    return bits | (selected.to(torch.int32) << 16), weights


def unpack_expert_indexes(packed: torch.Tensor):
    """(expert index int64 [T, 8], score bf16 [T, 8]) of PackedScoreIdx<bf16>."""
    idx = (packed >> 16).long()
    score = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    return idx, score


def routing_indices_reference(packed: torch.Tensor, offset: int) -> Routing:
    """routingPermutation / routingIndicesCoopKernel, with the tokens of one
    expert in expanded-index order (the kernels' order there is atomic)."""
    device = packed.device
    tokens = packed.shape[0]
    idx, score = unpack_expert_indexes(packed)
    local = idx.flatten() - offset
    is_local = (local >= 0) & (local < LOCAL_EXPERTS)
    counts = torch.bincount(local[is_local], minlength=LOCAL_EXPERTS)
    num_cta = (counts + TILE - 1) // TILE
    cta_offset = torch.cumsum(num_cta, 0) - num_cta  # CTAs in expert order
    total_ctas = int(num_cta.sum())
    i32 = torch.int32
    expanded = torch.full((tokens * TOP_K,), -1, dtype=i32, device=device)
    to_token = torch.zeros(max_permuted_padded_count(tokens), dtype=i32, device=device)
    positions = is_local.nonzero().flatten()
    experts = local[positions]
    order = torch.argsort(experts * (tokens * TOP_K) + positions)
    positions, experts = positions[order], experts[order]
    rank = (
        torch.arange(positions.numel(), device=device)
        - (torch.cumsum(counts, 0) - counts)[experts]
    )
    permuted = cta_offset[experts] * TILE + rank
    expanded[positions] = permuted.to(torch.int32)
    to_token[permuted] = (positions // TOP_K).to(torch.int32)
    batch = torch.zeros(max_num_ctas(tokens), dtype=i32, device=device)
    limit = torch.zeros(max_num_ctas(tokens), dtype=i32, device=device)
    for e in range(LOCAL_EXPERTS):
        n, start = int(num_cta[e]), int(cta_offset[e])
        if n:
            j = torch.arange(n, device=device)
            batch[start : start + n] = e
            limit[start : start + n] = torch.minimum(
                (start + j + 1) * TILE, start * TILE + counts[e]
            ).to(torch.int32)
    return Routing(
        packed,
        score.contiguous(),
        expanded,
        to_token,
        torch.tensor([total_ctas * TILE], dtype=i32, device=device),
        batch,
        limit,
        torch.tensor([total_ctas], dtype=i32, device=device),
    )


def routed_rows(expanded: torch.Tensor, batch: torch.Tensor):
    """(permuted rows, tokens, local experts) of every routed (token, k): a
    row's expert is its CTA's batch index, as the GEMMs read it."""
    flat = expanded.long()
    valid = (flat >= 0).nonzero().flatten()
    rows = flat[valid]
    return rows, valid // TOP_K, batch.long()[rows // TILE]


def quantize_blocks(values: torch.Tensor):
    """DeepSeek FP8 per row and 128 columns: scale = amax / 448."""
    rows, cols = values.shape
    blocks = values.view(rows, cols // BLOCK, BLOCK)
    scale = blocks.abs().amax(-1) / E4M3_MAX
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (blocks / safe[..., None]).to(torch.float8_e4m3fn).view(rows, cols)
    return q, scale  # scale [rows, cols / 128]


def _dequant_weights(weights, scale, e: int):
    return weights[e].float() * scale[e].repeat_interleave(BLOCK, 0).repeat_interleave(
        BLOCK, 1
    )


def _scale_view(flat: torch.Tensor, cols: int, total: int) -> torch.Tensor:
    """The [cols / 128, total_padded] DeepSeek scale layout inside a
    [cols / 128 * max_padded] buffer (row stride = runtime total_padded)."""
    return flat[: cols // BLOCK * total].view(cols // BLOCK, total)


def gemm1_reference(hidden, hidden_scale, w1, s1, expanded, total, batch):
    """(gemm1_output e4m3 [max_padded, 4096], gemm1_output_scale f32 flat)."""
    device = hidden.device
    max_padded = max_permuted_padded_count(hidden.shape[0])
    out = torch.zeros((max_padded, GEMM1_N), dtype=torch.float8_e4m3fn, device=device)
    out_scale = torch.zeros(GEMM1_N // BLOCK * max_padded, device=device)
    rows, tokens, experts = routed_rows(expanded, batch)
    view = _scale_view(out_scale, GEMM1_N, int(total[0]))
    x = hidden.float() * hidden_scale.T.repeat_interleave(BLOCK, -1)
    for e in experts.unique().tolist():
        sel = experts == e
        q, scale = quantize_blocks(x[tokens[sel]] @ _dequant_weights(w1, s1, e).T)
        out[rows[sel]] = q
        view[:, rows[sel]] = scale.T
    return out, out_scale


def activation_reference(gemm1_out, gemm1_scale, expanded, total):
    """activationDeepSeekKernel: up = first half, gate = second; out =
    silu(gate) * up, requantized per 128 columns."""
    device = gemm1_out.device
    max_padded, total_ = gemm1_out.shape[0], int(total[0])
    out = torch.zeros(
        (max_padded, INTERMEDIATE), dtype=torch.float8_e4m3fn, device=device
    )
    out_scale = torch.zeros(INTERMEDIATE // BLOCK * max_padded, device=device)
    rows = expanded.long()[expanded >= 0]
    if rows.numel():
        s = _scale_view(gemm1_scale, GEMM1_N, total_)[:, rows].T
        x = gemm1_out[rows].float() * s.repeat_interleave(BLOCK, -1)
        up, gate = x[:, :INTERMEDIATE], x[:, INTERMEDIATE:]
        q, scale = quantize_blocks(gate / (1.0 + torch.exp(-gate)) * up)
        out[rows] = q
        _scale_view(out_scale, INTERMEDIATE, total_)[:, rows] = scale.T
    return out, out_scale


def gemm2_reference(act, act_scale, w2, s2, expanded, total, batch):
    """FC2 into bf16 [max_padded, 7168]."""
    out = torch.zeros((act.shape[0], HIDDEN), dtype=torch.bfloat16, device=act.device)
    rows, _, experts = routed_rows(expanded, batch)
    if rows.numel():
        s = _scale_view(act_scale, INTERMEDIATE, int(total[0]))[:, rows].T
        x = act[rows].float() * s.repeat_interleave(BLOCK, -1)
        for e in experts.unique().tolist():
            sel = experts == e
            out[rows[sel]] = (x[sel] @ _dequant_weights(w2, s2, e).T).to(torch.bfloat16)
    return out


def finalize_reference(gemm2_out, expert_weights, expanded):
    """sum over k of weight[t, k] * gemm2_out[permuted(t, k)] in FP32 (k in
    order, unrouted k skipped), stored BF16."""
    tokens = expert_weights.shape[0]
    out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=gemm2_out.device)
    perm = expanded.view(tokens, TOP_K).long()
    for start in range(0, tokens, 2048):  # bounded FP32 temporaries
        rows = slice(start, start + 2048)
        acc = torch.zeros_like(out[rows], dtype=torch.float32)
        for k in range(TOP_K):
            valid = perm[rows, k] >= 0
            acc[valid] += (
                expert_weights[rows][valid, k].float()[:, None]
                * gemm2_out[perm[rows][valid, k]].float()
            )
        out[rows] = acc.to(torch.bfloat16)
    return out


def _undefined_rows(out: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``out`` with every row outside ``rows`` set to NaN (0x7F for E4M3)."""
    keep = torch.zeros(out.shape[0], dtype=torch.bool, device=out.device)
    keep[rows] = True
    out = out.clone()
    if out.dtype == torch.float8_e4m3fn:
        out.view(torch.uint8)[~keep] = 0x7F
    else:
        out[~keep] = torch.nan
    return out


def _undefined_scales(flat, cols, total, rows) -> torch.Tensor:
    """NaN except the [cols / 128, total] entries of ``rows``."""
    out = torch.full_like(flat, torch.nan)
    _scale_view(out, cols, total)[:, rows] = _scale_view(flat, cols, total)[:, rows]
    return out


def e4m3_step(values: torch.Tensor) -> torch.Tensor:
    """Spacing of E4M3 values at ``|values|``: 2^(exponent - 3), at least the
    subnormal step 2^-9."""
    exponent = torch.floor(torch.log2(values.abs().clamp(min=2.0**-6)))
    return torch.exp2(exponent - 3)


# Least fraction of defined E4M3 outputs that must equal the reference exactly.
EXACT_FP8_FRACTION = 0.99


def validate_fp8_blocks(ref: tuple, impl: tuple) -> None:
    """FP8 output + DeepSeek scales, where the reference is defined.

    Both sides quantize an FP32 result per (row, 128 columns) with scale =
    amax / 448; accumulation order and fast-math SiLU differ slightly (about
    1e-6 relative), so a value within that distance of a rounding midpoint
    may land on the neighbouring E4M3 value. Scales must agree within 1%.
    Every stored value must lie within one E4M3 step at the larger magnitude
    plus 1% of that magnitude (what a 1% scale difference moves it by); no
    absolute slack, so small values are checked as tightly as large ones.
    At most 1% of the values may differ at all: a midpoint flip needs the
    FP32 value within ~1e-6 relative of a midpoint (< 0.1% of Gaussian-like
    values), whereas a wrong scale, row or SwiGLU half, or zeroed small
    values (|q| < 4.5 is ~3% of a block), changes many more.
    """
    (q_ref, s_ref), (q, s) = ref, impl
    for a, b in ((q, q_ref), (s, s_ref)):
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            raise AssertionError("output shape/dtype/device mismatch")
    defined = ~torch.isnan(s_ref)
    torch.testing.assert_close(s[defined], s_ref[defined], rtol=1e-2, atol=1e-7)
    rows = (q_ref.view(torch.uint8) & 0x7F != 0x7F).all(-1)
    a, b = q[rows].float(), q_ref[rows].float()
    if torch.isnan(a).any():
        raise AssertionError("NaN in defined FP8 rows")
    larger = torch.maximum(a.abs(), b.abs())
    excess = (a - b).abs() - (e4m3_step(larger) + 0.01 * larger)
    if a.numel() and float(excess.max()) > 0:
        raise AssertionError(f"FP8 outputs differ (worst excess {float(excess.max())})")
    exact = float((q[rows].view(torch.uint8) == q_ref[rows].view(torch.uint8)).sum())
    if a.numel() and exact < EXACT_FP8_FRACTION * a.numel():
        raise AssertionError(
            f"only {exact / a.numel():.4f} of FP8 outputs match exactly"
        )


# --- workloads ------------------------------------------------------------------

# Smoke cases (tokens, local_expert_offset, routing); each workload runs the
# ones it serves. Token counts hit both sides of every dispatch boundary (16/17
# FC2, 32/33 FC1, 42/43 finalize, 1024/1025 routing); offsets both ends of
# [0, 224] and the group-unaligned 80; ``hot`` gives experts of 2 to ~340
# 8-token tiles with partial last tiles beside empty experts; the persistent
# GEMMs' grids reach thousands of CTAs (each of the 148 resident CTAs takes
# several tiles); ``remote`` leaves the rank without tokens; ``ties`` makes
# exact expert and group ties. ``tests/test_moe.py::SmokeHardPaths`` checks this.
SMOKE = (
    (4, 96, "uniform"),
    (8, 128, "hot"),
    (16, 224, "hot"),
    (17, 224, "popular"),
    (24, 0, "ties"),
    (32, 0, "hot"),
    (33, 80, "hot"),
    (42, 160, "popular"),
    (43, 96, "uniform"),
    (64, 64, "remote"),
    (300, 32, "hot"),
    (1025, 224, "uniform"),
    (3000, 128, "hot"),
)

# FlashInfer v0.2.10's test of this pipeline (tests/test_trtllm_gen_fused_moe.py)
# runs trtllm_fp8_block_scale_moe for the parametrization below; its routing
# config is this definition's (256 experts, top-8, 8 groups / 4 kept, scaling
# 2.5, BF16 bias ~ N(0, 1), FP32 logits ~ N(0, 1)) with local_expert_offset 0.
# Its hidden size 1024, intermediate sizes {1024, 768, 384}, 256 local experts
# and (for 1024 tokens) tile_tokens_dim 32 are fixed by the definition here
# to 7168, 2048, 32 and 8; its token counts, offset and routing distribution
# are the smoke cases below, served by every kernel's dispatch family.
UPSTREAM_TEST = (
    "tests/test_trtllm_gen_fused_moe.py::test_moe_quantization_classes"
    "[NoShuffle_MajorK-DSv3-FP8_Block-{1024,768,384}-1024-%d]"
)
UPSTREAM_REVISION = "7c79b41b8efd512eb40ec9bd56c9e6cda328d12e"  # v0.2.10
UPSTREAM_TOKENS = (1, 1024)
UPSTREAM_SEED = 7


def smoke_cases() -> list[CaseSpec]:
    cases = [
        upstream_case(
            f"tokens{t}_offset0_upstream",
            {"tokens": t, "offset": 0, "routing": "upstream"},
            UPSTREAM_TEST % t,
            # A seed whose N(0, 1) bias lets tokens reach experts 0..31 (the
            # upstream test makes all 256 experts local).
            seed=UPSTREAM_SEED,
            revision=UPSTREAM_REVISION,
        )
        for t in UPSTREAM_TOKENS
    ]
    cases += [
        CaseSpec(
            f"tokens{t}_offset{o}_{r}", {"tokens": t, "offset": o, "routing": r}, t
        )
        for t, o, r in SMOKE
    ]
    return sorted(cases, key=lambda c: c.params["tokens"])


def _scalar_int(value) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def _nothing() -> None:
    return None


class _MoEWorkload(Workload):
    """One kernel of the MoE pipeline; subclasses define ``serves``, inputs and
    ``_build`` (outputs, launch, and the work before each launch)."""

    package: ClassVar[str | None] = "moe"

    def __init__(self, cubin: bytes | None, *, device=None):
        super().__init__(cubin, device=device)
        self._sidecar: dict[str, Any] | None = None

    @classmethod
    def serves(cls, tokens: int) -> bool:
        raise NotImplementedError

    @property
    def sidecar(self) -> dict[str, Any]:
        """The compile stage's ``cubins/<arch>/<workload>.json``."""
        if self._sidecar is None:
            if self.arch is None:
                raise RuntimeError("reference-only workload has no kernel layout")
            path = self.cubin_path(self.arch).with_suffix(".json")
            sidecar = json.loads(path.read_text())
            if (
                sidecar["workload"] != self.name
                or sidecar["symbol"] != self.image_name()
            ):
                raise ValueError(f"{path} describes another kernel")
            if [(p.offset, p.size) for p in self.kernel_params] != [
                (0, sidecar["params"]["size"])
            ]:
                raise ValueError("kernel parameters do not match the recorded struct")
            self._sidecar = sidecar
        return self._sidecar

    @property
    def layout(self) -> Struct:
        return Struct(self.sidecar["params"])

    @property
    def block(self) -> tuple[int, int, int]:
        x, y, z = self.sidecar["block"]
        return x, y, z

    def get_cases(self):
        """The smoke and throughput cases whose token count upstream
        dispatches to this kernel."""
        cases = [
            c
            for c in smoke_cases() + throughput_cases()
            if self.serves(c.params["tokens"])
        ]
        if not any(c.suite == "smoke" for c in cases):
            raise AssertionError(f"{self.name}: no smoke case")
        return cases

    # -- stage inputs (PyTorch versions of the earlier kernels) ---------------

    def _routing(self, case: CaseSpec):
        logits, bias = case_routing_inputs(self.device, case, require_local=True)
        packed, _ = routing_main_reference(logits, bias, ROUTED_SCALING_FACTOR)
        return logits, bias, routing_indices_reference(packed, case.params["offset"])

    def _gemm1(self, case: CaseSpec, r: Routing):
        hidden, hidden_scale = case_activations(self.device, case)
        w1, s1 = official_weights(self.device, 1)
        out = gemm1_reference(
            hidden,
            hidden_scale,
            w1,
            s1,
            r.expanded_idx_to_permuted_idx,
            r.total_num_padded_tokens,
            r.cta_idx_xy_to_batch_idx,
        )
        return (hidden, hidden_scale, w1, s1), out

    def _activation(self, case: CaseSpec, r: Routing):
        _, (g1, g1_scale) = self._gemm1(case, r)
        return activation_reference(
            g1, g1_scale, r.expanded_idx_to_permuted_idx, r.total_num_padded_tokens
        )

    # -- launch ------------------------------------------------------------------

    def _check_supported(self) -> None:
        if not self.is_supported():
            raise RuntimeError(f"{self.arch} cubin cannot run on {self.device}")

    def _launch(self, spec: LaunchSpec) -> None:
        self.launch(
            spec.grid,
            spec.block,
            [spec.params],
            shared_mem=spec.shared_mem,
            cluster=spec.cluster,
            cooperative=spec.cooperative,
        )

    def _build(self, inputs) -> tuple[tuple, LaunchSpec, Callable[[], Any]]:
        raise NotImplementedError

    def run(self, inputs):
        self._check_supported()
        outputs, spec, before = self._build(inputs)
        before()
        self._launch(spec)
        return outputs

    def prepare(self, inputs):
        self._check_supported()
        outputs, spec, before = self._build(inputs)

        def call():
            before()
            self._launch(spec)
            return outputs

        return call, outputs


# --- routing ----------------------------------------------------------------------


def _routing_pointers(**tensors: torch.Tensor | None) -> Pointers:
    names = (
        "expert_count_histogram",
        "total_num_padded_tokens",
        "expanded_idx_to_permuted_idx",
        "permuted_idx_to_token_idx",
        "cta_idx_xy_to_batch_idx",
        "cta_idx_xy_to_mn_limit",
        "num_non_exiting_ctas",
        "expert_weights",
        "routing_logits",
        "expert_indexes",
        "routing_bias",
    )
    if set(tensors) - set(names):
        raise ValueError(f"unknown routing buffers {set(tensors) - set(names)}")
    out = {}
    for name in names:
        t = tensors.get(name)
        out[name] = t.data_ptr() if t is not None else 0
    return out


@register(name="moe_routing_main", supported_arches=("sm_86", "sm_100a"))
class MoERoutingMain(_MoEWorkload):
    """routingMainKernel <<<T, 256>>>: sigmoid + bias, top-2 per group, top-4
    groups, top-8 experts; writes packed (BF16 score, int16 index) pairs and
    the BF16 routing weights; for T > 1024 also zeroes the expert-count
    histogram the cooperative kernel accumulates into.

    Inputs: (routing_logits [T, 256] f32, routing_bias [256] bf16,
    local_expert_offset, routed_scaling_factor) (CPU scalars). Outputs:
    (expert_indexes int32 [T, 8], expert_weights bf16 [T, 8]) plus the
    histogram int32 [512] for T > 1024. Upstream passes every routing buffer
    in the shared parameter struct; the ones this kernel does not touch are
    null here (the byte-compared builder takes them all).
    """

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1

    def get_inputs(self, case):
        logits, bias = case_routing_inputs(self.device, case)
        return (
            logits,
            bias,
            self.scalar(case.params["offset"], torch.int32),
            self.scalar(ROUTED_SCALING_FACTOR),
        )

    def get_reference(self, inputs):
        logits, bias, _, scale = inputs
        out: tuple = routing_main_reference(logits, bias, float(scale))
        if logits.shape[0] > CLUSTER_MAX_TOKENS:
            out += (
                torch.zeros(2 * NUM_EXPERTS, dtype=torch.int32, device=logits.device),
            )
        return out

    def launch_spec(self, tokens: int, offset: int, scale: float, p: Pointers):
        return LaunchSpec(
            (tokens, 1, 1),
            self.block,
            routing_params(self.layout, tokens, offset, scale, p),
        )

    def _build(self, inputs):
        logits, bias, offset, scale = inputs
        tokens = logits.shape[0]
        packed = torch.empty((tokens, TOP_K), dtype=torch.int32, device=self.device)
        weights = torch.empty((tokens, TOP_K), dtype=torch.bfloat16, device=self.device)
        outputs: tuple = (packed, weights)
        histogram = None
        if tokens > CLUSTER_MAX_TOKENS:
            histogram = torch.empty(
                2 * NUM_EXPERTS, dtype=torch.int32, device=self.device
            )
            outputs += (histogram,)
        p = _routing_pointers(
            expert_count_histogram=histogram,
            expert_weights=weights,
            routing_logits=logits,
            expert_indexes=packed,
            routing_bias=bias,
        )
        spec = self.launch_spec(tokens, _scalar_int(offset), float(scale), p)
        return outputs, spec, _nothing

    def validate(self, ref, impl):
        if len(ref) != len(impl):
            raise AssertionError("output count mismatch")
        idx_ref, _ = unpack_expert_indexes(ref[0])
        idx, score = unpack_expert_indexes(impl[0])
        # Inputs are tie-free, so experts and their order are exact.
        torch.testing.assert_close(idx, idx_ref, rtol=0, atol=0)
        # The packed score is the stored weight; weights agree within BF16
        # rounding of an FP32 computation using a fast-math sigmoid.
        torch.testing.assert_close(score, impl[1], rtol=0, atol=0)
        self.assert_close(ref[1:2], impl[1:2], rtol=1e-2, atol=1e-3)
        if len(ref) == 3:
            torch.testing.assert_close(impl[2], ref[2], rtol=0, atol=0)


class _RoutingIndices(_MoEWorkload):
    """The permutation kernels (single cluster or cooperative grid).

    Inputs: (routing_logits, routing_bias, expert_indexes int32 [T, 8],
    local_expert_offset, routed_scaling_factor); the logits and bias are
    only addresses in the shared parameter struct, these kernels read the
    packed indices. Outputs: ([expert_weights bf16 [T, 8], cluster kernel
    only] expanded_idx_to_permuted_idx [T*8], permuted_idx_to_token_idx
    [max_padded], total_num_padded_tokens [1], cta_idx_xy_to_batch_idx
    [max_ctas], cta_idx_xy_to_mn_limit [max_ctas], num_non_exiting_ctas [1]).
    """

    writes_weights: ClassVar[bool] = False

    def get_inputs(self, case):
        logits, bias = case_routing_inputs(self.device, case)
        packed, _ = routing_main_reference(logits, bias, ROUTED_SCALING_FACTOR)
        return (
            logits,
            bias,
            packed,
            self.scalar(case.params["offset"], torch.int32),
            self.scalar(ROUTED_SCALING_FACTOR),
        )

    def get_reference(self, inputs):
        _, _, packed, offset, _ = inputs
        r = routing_indices_reference(packed, _scalar_int(offset))
        # Entries the kernels leave unwritten are -1 here (never compared).
        n = int(r.num_non_exiting_ctas[0])
        expanded = r.expanded_idx_to_permuted_idx
        to_token = torch.full_like(r.permuted_idx_to_token_idx, -1)
        rows = expanded.long()[expanded >= 0]
        to_token[rows] = r.permuted_idx_to_token_idx[rows]
        batch, limit = (
            r.cta_idx_xy_to_batch_idx.clone(),
            r.cta_idx_xy_to_mn_limit.clone(),
        )
        batch[n:] = limit[n:] = -1
        out = (
            r.expanded_idx_to_permuted_idx,
            to_token,
            r.total_num_padded_tokens,
            batch,
            limit,
            r.num_non_exiting_ctas,
        )
        return ((r.expert_weights,) if self.writes_weights else ()) + out

    def launch_spec(self, tokens: int, offset: int, scale: float, p: Pointers):
        raise NotImplementedError

    def _build(self, inputs):
        logits, bias, packed, offset, scale = inputs
        tokens = logits.shape[0]
        i32, dev = torch.int32, self.device
        weights = torch.empty((tokens, TOP_K), dtype=torch.bfloat16, device=self.device)
        expanded = torch.empty(tokens * TOP_K, dtype=i32, device=dev)
        to_token = torch.empty(max_permuted_padded_count(tokens), dtype=i32, device=dev)
        total = torch.empty(1, dtype=i32, device=dev)
        batch = torch.empty(max_num_ctas(tokens), dtype=i32, device=dev)
        limit = torch.empty(max_num_ctas(tokens), dtype=i32, device=dev)
        non_exiting = torch.empty(1, dtype=i32, device=dev)
        histogram = torch.zeros(2 * NUM_EXPERTS, dtype=i32, device=dev)
        p = _routing_pointers(
            expert_count_histogram=histogram,
            total_num_padded_tokens=total,
            expanded_idx_to_permuted_idx=expanded,
            permuted_idx_to_token_idx=to_token,
            cta_idx_xy_to_batch_idx=batch,
            cta_idx_xy_to_mn_limit=limit,
            num_non_exiting_ctas=non_exiting,
            expert_weights=weights,
            routing_logits=logits,
            expert_indexes=packed,
            routing_bias=bias,
        )
        outputs: tuple = (expanded, to_token, total, batch, limit, non_exiting)
        if self.writes_weights:
            outputs = (weights,) + outputs
        spec = self.launch_spec(tokens, _scalar_int(offset), float(scale), p)
        return outputs, spec, (histogram.zero_ if spec.cooperative else _nothing)

    def validate(self, ref, impl):
        if len(ref) != len(impl):
            raise AssertionError("output count mismatch")
        if self.writes_weights:
            torch.testing.assert_close(impl[0], ref[0], rtol=0, atol=0)
            ref, impl = ref[1:], impl[1:]
        expanded_ref, to_token_ref, total_ref, batch_ref, limit_ref, ctas_ref = ref
        expanded, to_token, total, batch, limit, ctas = impl
        torch.testing.assert_close(total, total_ref, rtol=0, atol=0)
        torch.testing.assert_close(ctas, ctas_ref, rtol=0, atol=0)
        n = int(ctas_ref[0])
        for a, b in (
            (expanded, expanded_ref),
            (to_token, to_token_ref),
            (batch, batch_ref),
        ):
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                raise AssertionError("output shape/dtype/device mismatch")
        torch.testing.assert_close(batch[:n], batch_ref[:n], rtol=0, atol=0)
        torch.testing.assert_close(limit[:n], limit_ref[:n], rtol=0, atol=0)
        local = expanded_ref >= 0
        torch.testing.assert_close(
            expanded[~local], expanded_ref[~local], rtol=0, atol=0
        )
        # Within an expert the kernels order tokens by atomics: require a
        # bijection onto the reference's rows of the same expert, each below
        # its CTA's mnLimit, mapping back to its token.
        got, want = expanded[local].long(), expanded_ref[local].long()
        if (
            got.unique().numel() != got.numel()
            or (got < 0).any()
            or (got >= n * TILE).any()
        ):
            raise AssertionError("permuted indices are not a valid permutation")
        experts = batch[:n].long()
        torch.testing.assert_close(
            experts[got // TILE], experts[want // TILE], rtol=0, atol=0
        )
        if (got >= limit[:n].long()[got // TILE]).any():
            raise AssertionError("permuted index beyond its CTA's mnLimit")
        tokens = local.nonzero().flatten() // TOP_K
        torch.testing.assert_close(to_token[got].long(), tokens, rtol=0, atol=0)


@register(name="moe_routing_cluster", supported_arches=("sm_100a",))
class MoERoutingCluster(_RoutingIndices):
    """routingIndicesClusterKernel <<<8, 256>>> (one 8-CTA cluster from
    ``__cluster_dims__``): per-expert permutation and the GEMMs' CTA tables;
    it also rewrites the BF16 routing weights from the packed scores. T <= 1024."""

    writes_weights = True

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return 1 <= tokens <= CLUSTER_MAX_TOKENS

    def launch_spec(self, tokens, offset, scale, p):
        p = dict(p, expert_count_histogram=0)  # nullptr on the single-cluster path
        return LaunchSpec(
            (8, 1, 1), self.block, routing_params(self.layout, tokens, offset, scale, p)
        )


@register(name="moe_routing_coop", supported_arches=("sm_100a",))
class MoERoutingCoop(_RoutingIndices):
    """routingIndicesCoopKernel <<<128, 256>>> with a cooperative launch (grid
    sync over the expert-count histogram); 1024 < T <= 262144. The histogram is
    scratch that must be zero at launch (upstream's routingMainKernel zeroes
    it). ``run`` zeroes it before the launch; ``prepare`` keeps that memset
    out of the timed callable (see there)."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return CLUSTER_MAX_TOKENS < tokens <= COOP_MAX_TOKENS

    def launch_spec(self, tokens, offset, scale, p):
        return LaunchSpec(
            (COOP_BLOCKS, 1, 1),
            self.block,
            routing_params(self.layout, tokens, offset, scale, p),
            cooperative=True,
        )

    def prepare(self, inputs):
        """Each timed call launches with the next of HISTOGRAM_RING pre-zeroed
        histograms (parameter structs prebuilt, only the histogram pointer
        differs), so it is the launch alone. After HISTOGRAM_RING calls the
        ring is re-zeroed by one memset inside that call (one call in
        HISTOGRAM_RING; the benchmark's default run makes 25)."""
        self._check_supported()
        outputs, spec, _ = self._build(inputs)
        ring = torch.zeros(
            (HISTOGRAM_RING, 2 * NUM_EXPERTS), dtype=torch.int32, device=self.device
        )
        offset = self.layout.fields["mPtrExpertCounts"]["offset"]
        specs = []
        for histogram in ring:
            params = bytearray(spec.params)
            struct.pack_into("<Q", params, offset, histogram.data_ptr())
            specs.append(spec._replace(params=bytes(params)))
        calls = 0

        def call():
            nonlocal calls
            if calls and calls % HISTOGRAM_RING == 0:
                ring.zero_()
            self._launch(specs[calls % HISTOGRAM_RING])
            calls += 1
            return outputs

        return call, outputs


# --- GEMMs --------------------------------------------------------------------------


class _GEMM(_MoEWorkload):
    """A trtllm-gen batched GEMM: one by-value KernelParams (0x4380 bytes),
    grid (M / 128, max CTAs over the 32 local experts, 1), the config's block
    and dynamic shared memory, cluster 1x1x1."""

    m: ClassVar[int]
    k: ClassVar[int]

    @classmethod
    def static_scheduler(cls, tokens: int) -> bool:
        # getValidConfigIndices' last comparator: persistent scheduler when the
        # estimated CTA count ceil(M / tileM) * ceil(T / tileN) exceeds 148.
        return ceil_div(cls.m, BLOCK) * ceil_div(tokens, TILE) <= 148

    @property
    def options(self) -> dict[str, Any]:
        return self.sidecar["gemm_options"]

    def configure(self, function: cuda_driver.Function) -> None:
        # trtllm::gen::launchKernel: opt in to more than 48 KiB of smem.
        smem = self.options["shared_mem"]
        if smem > 48 << 10:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )

    def gemm_launch(self, encode: TensorMapEncoder, **pointers: int) -> LaunchSpec:
        params, grid = gemm_params(
            self.layout, self.options, m=self.m, k=self.k, encode=encode, **pointers
        )
        x, y, z = self.options["cluster"]
        return LaunchSpec(
            grid,
            (self.options["threads"], 1, 1),
            params,
            shared_mem=self.options["shared_mem"],
            cluster=(x, y, z),
        )


class _GEMM1(_GEMM):
    """FC1 (PermuteGemm1): token rows gathered through permuted_idx_to_token_idx
    (ldgsts route); C = E4M3 [max_padded, 4096] with DeepSeek scales
    [32, total_padded].

    Inputs: (hidden_states e4m3 [T, 7168], hidden_states_scale [56, T],
    gemm1_weights e4m3 [32, 4096, 7168], gemm1_weights_scale [32, 32, 56],
    permuted_idx_to_token_idx, total_num_padded_tokens,
    cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit, num_non_exiting_ctas,
    expert_weights bf16 [T, 8] (only an address in the params),
    expanded_idx_to_permuted_idx (reference only)). Outputs: (gemm1_output,
    gemm1_output_scale flat [32 * max_padded]).
    """

    m, k = GEMM1_N, HIDDEN

    def get_inputs(self, case):
        _, _, r = self._routing(case)
        hidden, hidden_scale = case_activations(self.device, case)
        w1, s1 = official_weights(self.device, 1)
        return (
            hidden,
            hidden_scale,
            w1,
            s1,
            r.permuted_idx_to_token_idx,
            r.total_num_padded_tokens,
            r.cta_idx_xy_to_batch_idx,
            r.cta_idx_xy_to_mn_limit,
            r.num_non_exiting_ctas,
            r.expert_weights,
            r.expanded_idx_to_permuted_idx,
        )

    def get_reference(self, inputs):
        hidden, hidden_scale, w1, s1, _, total, batch, _, _, _, expanded = inputs
        out, scale = gemm1_reference(
            hidden, hidden_scale, w1, s1, expanded, total, batch
        )
        rows = expanded.long()[expanded >= 0]
        return (
            _undefined_rows(out, rows),
            _undefined_scales(scale, GEMM1_N, int(total[0]), rows),
        )

    def _build(self, inputs, encode: TensorMapEncoder | None = None):
        (
            hidden,
            hidden_scale,
            w1,
            s1,
            to_token,
            total,
            batch,
            limit,
            ctas,
            weights,
            _,
        ) = inputs
        max_padded = to_token.numel()
        out = torch.empty(
            (max_padded, GEMM1_N), dtype=torch.float8_e4m3fn, device=self.device
        )
        out_scale = torch.empty(GEMM1_N // BLOCK * max_padded, device=self.device)
        if encode is None:
            cuda_driver.ensure_context(self.device)
        spec = self.gemm_launch(
            encode or driver_encode,
            tokens=hidden.shape[0],
            a=w1.data_ptr(),
            sf_a=s1.data_ptr(),
            b=hidden.data_ptr(),
            sf_b=hidden_scale.data_ptr(),
            c=out.data_ptr(),
            sf_c=out_scale.data_ptr(),
            route_map=to_token.data_ptr(),
            per_token_sf_b=weights.data_ptr(),
            total_num_padded_tokens=total.data_ptr(),
            cta_idx_xy_to_batch_idx=batch.data_ptr(),
            cta_idx_xy_to_mn_limit=limit.data_ptr(),
            num_non_exiting_ctas=ctas.data_ptr(),
        )
        return (out, out_scale), spec, _nothing

    def validate(self, ref, impl):
        validate_fp8_blocks(ref, impl)


@register(name="moe_gemm1", supported_arches=("sm_100a",))
class MoEGEMM1(_GEMM1):
    """FC1 on the static-scheduler config upstream selects for T <= 32."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1 and cls.static_scheduler(tokens)


@register(name="moe_gemm1_persistent", supported_arches=("sm_100a",))
class MoEGEMM1Persistent(_GEMM1):
    """FC1 on the persistent-scheduler config upstream selects for T >= 33."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1 and not cls.static_scheduler(tokens)


class _GEMM2(_GEMM):
    """FC2 (Gemm2): activations through a TMA descriptor (no route); C = BF16
    [max_padded, 7168].

    Inputs: (activation_output e4m3 [max_padded, 2048], activation_output_scale
    flat [16 * max_padded], gemm2_weights e4m3 [32, 7168, 2048],
    gemm2_weights_scale [32, 56, 16], total_num_padded_tokens,
    cta_idx_xy_to_batch_idx, cta_idx_xy_to_mn_limit, num_non_exiting_ctas,
    expanded_idx_to_permuted_idx (reference only), num_tokens (CPU int32)).
    Output: (gemm2_output,).
    """

    m, k = HIDDEN, INTERMEDIATE

    def get_inputs(self, case):
        _, _, r = self._routing(case)
        act, act_scale = self._activation(case, r)
        w2, s2 = official_weights(self.device, 2)
        return (
            act,
            act_scale,
            w2,
            s2,
            r.total_num_padded_tokens,
            r.cta_idx_xy_to_batch_idx,
            r.cta_idx_xy_to_mn_limit,
            r.num_non_exiting_ctas,
            r.expanded_idx_to_permuted_idx,
            self.scalar(case.params["tokens"], torch.int32),
        )

    def get_reference(self, inputs):
        act, act_scale, w2, s2, total, batch, _, _, expanded, _ = inputs
        out = gemm2_reference(act, act_scale, w2, s2, expanded, total, batch)
        return (_undefined_rows(out, expanded.long()[expanded >= 0]),)

    def _build(self, inputs, encode: TensorMapEncoder | None = None):
        act, act_scale, w2, s2, total, batch, limit, ctas, _, tokens = inputs
        out = torch.empty(
            (act.shape[0], HIDDEN), dtype=torch.bfloat16, device=self.device
        )
        if encode is None:
            cuda_driver.ensure_context(self.device)
        spec = self.gemm_launch(
            encode or driver_encode,
            tokens=_scalar_int(tokens),
            a=w2.data_ptr(),
            sf_a=s2.data_ptr(),
            b=act.data_ptr(),
            sf_b=act_scale.data_ptr(),
            c=out.data_ptr(),
            sf_c=0,  # gemm2_output_scale = nullptr (BF16 output)
            route_map=0,
            per_token_sf_b=0,
            total_num_padded_tokens=total.data_ptr(),
            cta_idx_xy_to_batch_idx=batch.data_ptr(),
            cta_idx_xy_to_mn_limit=limit.data_ptr(),
            num_non_exiting_ctas=ctas.data_ptr(),
        )
        return (out,), spec, _nothing

    def validate(self, ref, impl):
        if len(ref) != 1 or len(impl) != 1:
            raise AssertionError("output count mismatch")
        rows = ~torch.isnan(ref[0]).all(-1)
        if impl[0].shape != ref[0].shape or impl[0].dtype != ref[0].dtype:
            raise AssertionError("output shape/dtype mismatch")
        # FP32 accumulation of identical FP8 operands (differing order), BF16 out.
        torch.testing.assert_close(impl[0][rows], ref[0][rows], rtol=2e-2, atol=2e-2)


@register(name="moe_gemm2", supported_arches=("sm_100a",))
class MoEGEMM2(_GEMM2):
    """FC2 on the static-scheduler config upstream selects for T <= 16."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1 and cls.static_scheduler(tokens)


@register(name="moe_gemm2_persistent", supported_arches=("sm_100a",))
class MoEGEMM2Persistent(_GEMM2):
    """FC2 on the persistent-scheduler config upstream selects for T >= 17."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1 and not cls.static_scheduler(tokens)


# --- activation and finalize ----------------------------------------------------


@register(name="moe_activation", supported_arches=("sm_86", "sm_100a"))
class MoEActivation(_MoEWorkload):
    """activationDeepSeekKernel <<<(32, 8, T), 128>>>: dequantize FC1's E4M3
    rows, SwiGLU (silu(second half) * first half), requantize per 128 columns.

    Inputs: (gemm1_output e4m3 [max_padded, 4096], gemm1_output_scale flat,
    expanded_idx_to_permuted_idx [T*8], total_num_padded_tokens [1]).
    Outputs: (activation_output e4m3 [max_padded, 2048],
    activation_output_scale flat [16 * max_padded]).
    """

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1

    def get_inputs(self, case):
        _, _, r = self._routing(case)
        _, (g1, g1_scale) = self._gemm1(case, r)
        return (g1, g1_scale, r.expanded_idx_to_permuted_idx, r.total_num_padded_tokens)

    def get_reference(self, inputs):
        g1, g1_scale, expanded, total = inputs
        out, scale = activation_reference(g1, g1_scale, expanded, total)
        rows = expanded.long()[expanded >= 0]
        return (
            _undefined_rows(out, rows),
            _undefined_scales(scale, INTERMEDIATE, int(total[0]), rows),
        )

    def launch_spec(self, tokens: int, p: Pointers) -> LaunchSpec:
        # activation::run: grid (innerDim / 128, topK, numTokens).
        return LaunchSpec(
            (GEMM1_N // BLOCK, TOP_K, tokens),
            self.block,
            activation_params(self.layout, tokens, p),
        )

    def _build(self, inputs):
        g1, g1_scale, expanded, total = inputs
        max_padded = g1.shape[0]
        out = torch.empty(
            (max_padded, INTERMEDIATE), dtype=torch.float8_e4m3fn, device=self.device
        )
        out_scale = torch.empty(INTERMEDIATE // BLOCK * max_padded, device=self.device)
        p = {
            "gemm1_output": g1.data_ptr(),
            "activation_output": out.data_ptr(),
            "gemm1_output_scale": g1_scale.data_ptr(),
            "activation_output_scale": out_scale.data_ptr(),
            "expanded_idx_to_permuted_idx": expanded.data_ptr(),
            "total_num_padded_tokens": total.data_ptr(),
        }
        return (
            (out, out_scale),
            self.launch_spec(expanded.numel() // TOP_K, p),
            _nothing,
        )

    def validate(self, ref, impl):
        validate_fp8_blocks(ref, impl)


class _Finalize(_MoEWorkload):
    """Weighted top-k reduction of FC2's rows into the BF16 output.

    Inputs: (gemm2_output bf16 [max_padded, 7168], expert_weights bf16 [T, 8],
    expanded_idx_to_permuted_idx [T*8], total_num_padded_tokens [1]).
    Output: (output bf16 [T, 7168],).
    """

    @staticmethod
    def blocks_x() -> int:
        return ceil_div(HIDDEN, FINALIZE_THREADS)

    @classmethod
    def small(cls, tokens: int) -> bool:
        # finalize::run: numBlocksX * min(8192, T) < 1184 (148 SMs x 8 CTAs).
        return cls.blocks_x() * min(8192, tokens) < 1184

    def get_inputs(self, case):
        _, _, r = self._routing(case)
        act, act_scale = self._activation(case, r)
        w2, s2 = official_weights(self.device, 2)
        g2 = gemm2_reference(
            act,
            act_scale,
            w2,
            s2,
            r.expanded_idx_to_permuted_idx,
            r.total_num_padded_tokens,
            r.cta_idx_xy_to_batch_idx,
        )
        return (
            g2,
            r.expert_weights,
            r.expanded_idx_to_permuted_idx,
            r.total_num_padded_tokens,
        )

    def get_reference(self, inputs):
        g2, weights, expanded, _ = inputs
        return (finalize_reference(g2, weights, expanded),)

    def launch_spec(self, tokens: int, p: Pointers) -> LaunchSpec:
        raise NotImplementedError

    def _build(self, inputs):
        g2, weights, expanded, total = inputs
        tokens = weights.shape[0]
        out = torch.empty((tokens, HIDDEN), dtype=torch.bfloat16, device=self.device)
        p = {
            "gemm2_output": g2.data_ptr(),
            "expert_weights": weights.data_ptr(),
            "output": out.data_ptr(),
            "expanded_idx_to_permuted_idx": expanded.data_ptr(),
            "total_num_padded_tokens": total.data_ptr(),
        }
        return (out,), self.launch_spec(tokens, p), _nothing

    def validate(self, ref, impl):
        # Same FP32 accumulation (k = 0..7, in order) of BF16 x BF16 products,
        # which are exact in FP32 (so FMA or not makes no difference); BF16
        # output. Bit-identical natively on sm_86 for every case; allowed: one
        # BF16 rounding step (2^-8 relative) in case of a different FP32
        # order, and 1e-3 absolute for near-zero sums (outputs are O(100)).
        self.assert_close(ref, impl, rtol=2.0**-8, atol=1e-3)


@register(name="moe_finalize", supported_arches=("sm_86", "sm_100a"))
class MoEFinalize(_Finalize):
    """finalizeKernel <<<(28, min(T, 8192)), 256>>>, upstream's choice while
    28 * min(T, 8192) < 1184 (T <= 42)."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1 and cls.small(tokens)

    def launch_spec(self, tokens, p):
        return LaunchSpec(
            (self.blocks_x(), min(8192, tokens), 1),
            self.block,
            finalize_params(self.layout, tokens, p),
        )


@register(name="moe_finalize_vec", supported_arches=("sm_86", "sm_100a"))
class MoEFinalizeVec(_Finalize):
    """finalizeKernelVecLoad <<<T, 256>>>, upstream's choice for T >= 43."""

    @classmethod
    def serves(cls, tokens: int) -> bool:
        return tokens >= 1 and not cls.small(tokens)

    def launch_spec(self, tokens, p):
        return LaunchSpec(
            (tokens, 1, 1), self.block, finalize_params(self.layout, tokens, p)
        )


THROUGHPUT = "moe"
# DeepSeek-V3 MoE layer at EP8 (this definition): decode batches and a prefill
# chunk per rank, with model-like expert popularity.
MODEL_TOKENS = ((128, 0), (512, 96), (4096, 224), (8192, 160))
STRESS_TOKENS = 16384


def throughput_cases():
    """Throughput (benchmark) cases of this package: every official inventory
    row (tokens and recorded local_expert_offset, model-like routing; the
    recorded logits are not replayed), DeepSeek-V3 token counts and
    16384-token stress (balanced, and three hot experts taking every token:
    ~2000 tiles each, ~200k FC1 CTAs)."""
    name = THROUGHPUT
    rows, _ = inventory(name)
    cases = [
        trace(
            name,
            f"tokens{r['axes']['seq_len']}",
            {
                "tokens": r["axes"]["seq_len"],
                "offset": r["inputs"]["local_expert_offset"]["value"],
                "routing": "popular",
            },
            r["axes"],
        )
        for r in sorted(rows, key=lambda r: r["axes"]["seq_len"])
    ]
    cases += [
        model_case(
            f"deepseek_v3_tokens{t}",
            {"tokens": t, "offset": o, "routing": "popular"},
            "deepseek_v3",
            "moe (EP8: 32 of 256 experts per rank)",
        )
        for t, o in MODEL_TOKENS
    ]
    cases += [
        synthetic(
            f"tokens{STRESS_TOKENS}_{routing}",
            {"tokens": STRESS_TOKENS, "offset": offset, "routing": routing},
            reason,
        )
        for offset, routing, reason in (
            (128, "uniform", "largest token count of the scope; near-balanced"),
            (224, "hot", "skew at scale: three local experts take every token"),
        )
    ]
    return cases

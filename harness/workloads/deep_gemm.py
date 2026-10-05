"""deep_gemm: m-grouped GEMMs, one registered workload per kernel.

The variants are generated from ``impls/deep_gemm/variants.json``, a compact
index written by ``impls/deep_gemm/compiler.py`` (registration parses no
cubin). Every variant wraps one kernel in one cubin and performs one launch.

sm_100a: DeepGEMM FP8 m-grouped GEMM (``deep_gemm_contig_*`` / ``_masked_*``)
    The 317 cubins of ``cubins/deep_gemm`` are FlashInfer v0.6.9's published
    DeepGEMM kernels (artifact ``a72d85b0.../deep-gemm``), each one
    ``deep_gemm::sm100_fp8_gemm_1d1d_impl`` instantiation (DeepGEMM 9da4a23 /
    aff9da0). They hold 236 distinct kernels: 81 contiguous cubins differ
    from another only in ``NUM_GROUPS``, which the contiguous scheduler never
    reads (identical code), so each such set is one workload
    ``deep_gemm_contig_n<N>_k<K>_bn<BLOCK_N>`` serving all its group counts;
    masked kernels depend on the group count and keep it in their names.
    FlashInfer's ``flashinfer/deep_gemm.py`` launches them from Python:
    ``m_grouped_fp8_gemm_nt_contiguous`` (``GemmType::MGroupedContiguous``)
    and ``m_grouped_fp8_gemm_nt_masked`` (``MGroupedMasked``), both
    ``D[bf16] = A[fp8, K-major] @ B[fp8, K-major]^T`` with per-128-K UE8M0
    scale factors, ``compiled_dims="nk"`` (N and K are template arguments, M
    is a runtime argument). The template arguments of each cubin are
    FlashInfer's config choice (``get_best_configs``) for some (M, N, K,
    num_groups, num_sms); the variant's ``dispatch`` entries record, per group
    count, the SM count and the ``ceil(M / 128)`` ranges that select it, and
    every case is one FlashInfer dispatches to this kernel (checked when the
    cases are built and against FlashInfer's own code in the tests). Launch
    preparation follows FlashInfer's ``*_kwargs_gen`` and
    ``SM100FP8GemmRuntime.launch``: six ``CUtensorMap`` (C aliases D), 256
    threads, a 1x1x1 cluster, ``num_min_sms`` persistent CTAs and the
    config's dynamic shared memory.

    Inputs carry the scale factors in the kernel-native layout FlashInfer's
    ``transform_sf_into_required_layout`` produces on sm_100: UE8M0 exponents
    (``2**(e - 127)``), one per row and 128-element K block, four K blocks
    packed little-endian per int32, MN-major. They are stored as int32
    ``[groups, ceil(K / 512), rows]`` (rows are multiples of 4, so
    FlashInfer's TMA-aligned MN-major view ``.mT`` is this storage).

    contiguous: ``(a[M, K], sfa[ceil(K/512), M], b[G, N, K],
    sfb[G, ceil(K/512), N], m_indices[M])`` -> ``d[M, N]`` bf16, case params
    ``groups``, ``rows`` (valid rows per group) and ``slack``. Each group's
    rows are padded with -1 to the 128-row alignment
    (``get_m_alignment_for_contiguous_layout``), empty groups take no rows
    and ``slack`` trailing all -1 blocks model an over-allocated buffer. The
    reference covers every valid row; padding rows are undefined upstream
    (DeepGEMM's tests mask them) and NaN, so ``validate`` skips them.

    masked: ``(a[G, C, K], sfa[G, ceil(K/512), C], b[G, N, K],
    sfb[G, ceil(K/512), N], masked_m[G], expected_m)`` -> ``d[G, C, N]``
    bf16, case params ``rows`` (masked_m), ``capacity`` C (% 128 == 0) and
    ``expected_m`` (a Python int that, as upstream, only selects the launch
    configuration). For group ``g`` the kernel computes and stores the
    128-row blocks covering ``masked_m[g]`` rows and leaves the remaining
    rows untouched; the reference marks those NaN and ``validate`` skips
    them.

    Cases (built per kernel by ``_CaseBuilder``):

    * upstream (``upstream_case``): FlashInfer's
      ``test_fp8_groupwise_group_deepgemm`` / ``..._batch_deepgemm_masked``
      parametrizations selecting the kernel at 148 SMs (smoke; the masked
      test's random expected_m may select any kernel up to its capacity, so
      each of them gets the parametrization), its
      ``bench_deepgemm_blackwell.py`` configurations (throughput) and the
      DeepGEMM test regimes replayed on the production (N, K) (smoke);
    * smoke: ragged long-tailed groups with empty groups (first, middle,
      last), single-row groups, partial blocks, slack blocks, more tiles than
      persistent CTAs where the dispatch range allows it, a single group or
      balanced groups, many groups with fewer blocks than groups; masked:
      full, empty, single-row and block-multiple groups and spare capacity;
    * throughput: for the production (N, K) -- the expert gate_up (4096,
      7168) and down (7168, 2048) GEMMs of DeepSeek-V3 / Kimi-K2 -- model
      cases routing tokens x top-8 rows to the rank's 256 / EP (384 / EP)
      experts with long-tailed loads: prefill up to 16k tokens per rank for
      the contiguous layout, decode (around 256, up to 4k tokens per rank)
      for the masked one, ``expected_m`` = rows per expert. FlashInfer's
      unit-test shapes (512, 128) / (128, 512) get the top of the kernel's
      dispatch range. Every case stays within 40 GB (``case_bytes``).

    Tolerance: FP8 x FP8 products and power-of-two scales are exact in FP32;
    kernel and reference both accumulate in FP32 and round once to bf16.
    ``rtol = 2**-6`` (two bf16 ulps) and ``atol = 1e-3`` (outputs are O(1)
    by construction of the scales) absorb summation order and the final
    rounding while a wrong group, block, scale or row is off by O(1); kept
    until the tcgen05 block-scaled accumulation is validated on a B200.

sm_86: FlashInfer CUTLASS segment GEMM, column-major weights
    (``deep_gemm_segment_{bf16,fp16}_{pipelined,multistage}``)
    ``SegmentGEMMWrapper.run(x, w, G, weight_column_major=True, seg_indptr,
    weight_indices)`` on the ``sm80`` backend: ``y[s] = x[s] @
    w[weight_indices[s]]^T`` per segment (``weight_indices[s] = s`` without
    indices), the BF16/FP16 analogue of DeepGEMM's m-grouped contiguous
    layout (weights ``[W, N, K]``, K-major). Kernels are stripped from the
    sm_80 JIT-cache module of FlashInfer 0.7.0 (``CutlassSegmentGEMMRun``);
    see the compiler's live/dead table for the pipelined (2-stage, dispatched
    when the SM has < 147968 B of shared memory, e.g. sm_86) and multistage
    (4-stage, dispatched on sm_80) mainloops. ``GemmGrouped::Params`` is
    rebuilt in Python from the compile-time probe's layout; the per-group
    device arrays FlashInfer fills with a Triton kernel
    (``compute_sm80_group_gemm_args``) are prepared with torch outside the
    timed launch. Upstream launches ``threadblock_count = 4`` persistent CTAs
    with ``alpha = beta = 1`` and ``C = D = y`` zero-initialised; ``run``
    does the same, so the timed callable of ``prepare`` accumulates into its
    output (as repeated upstream calls with a reused ``out`` would). Only the
    values change: the epilogue has no data-dependent control flow, so every
    launch does the same work.
    Inputs ``(x[M, K], w[W, N, K], seg_indptr[G + 1] int64,
    weight_indices[G] int64)`` -> ``y[M, N]``. Cases: FlashInfer's
    ``test_segment_gemm`` grid (column-major, sm80; recorded for the fp16
    kernels it tests), ragged segments with empty and single-row ones,
    partial 128 x 128 x 32 tiles, long K, permuted and shared weight banks,
    and MoE expert GEMMs of Mixtral, Qwen3-30B-A3B, gpt-oss-20b and Llama-4
    Scout as throughput cases (each within 6 GB).
    Tolerance: inputs are exact in FP32, both sides accumulate in FP32 and
    round once; ``rtol``/``atol`` of two output ulps near 1.
"""

from __future__ import annotations

import ctypes
import json
import math
import random
import struct
import zlib
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any, ClassVar, NamedTuple

import torch

from .. import cuda_driver
from ..cutlass_host import ceil_div, driver_encode
from ..models import MODELS
from ..registry import register_variant
from ..throughput import (
    model_case,
    skewed_lengths,
    split_lengths,
    synthetic,
    upstream_case,
)
from ..workload import IMPLS, CaseSpec, Workload

PACKAGE_DIR = IMPLS / "deep_gemm"
VARIANTS = PACKAGE_DIR / "variants.json"

# --- FlashInfer v0.6.9 flashinfer/deep_gemm.py (sm_100 path) ---------------

BLOCK_M = 128  # get_m_alignment_for_contiguous_layout(); BLOCK_M for both layouts
BLOCK_K = 128  # 128 // sizeof(fp8)
SM100_SMEM_CAPACITY = 232448
NUM_NON_EPILOGUE_THREADS = NUM_EPILOGUE_THREADS = 128
LAYOUT_AD_M = 128
GEMM_TYPE = {"contiguous": 1, "masked": 2}  # flashinfer GemmType values

# Raw CUtensorMap* enum values from cuda.h.
SWIZZLE = {0: 0, 16: 0, 32: 1, 64: 2, 128: 3}
L2_PROMOTION_256B = 3
INTERLEAVE_NONE = 0
OOB_FILL_NONE = 0


def round_up(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def swizzle_mode(block_size: int, elem_size: int) -> int:
    """get_swizzle_mode."""
    for mode in (128, 64, 32, 16):
        if (block_size * elem_size) % mode == 0:
            return mode
    return 0


def tmem_legal(block_m: int, block_n: int) -> bool:
    """is_tmem_size_legal for FP8 (UTCCP-aligned SF blocks)."""
    sf_m, sf_n = round_up(block_m, 128), round_up(block_n, 128)
    return 2 * block_n + sf_m // 32 + sf_n // 32 <= 512


def smem_size(block_m: int, block_n: int, num_stages: int) -> tuple[int, int]:
    """get_smem_config for K-major FP8 A/B, bf16 N-major D, no multicast:
    (shared memory bytes, CD swizzle mode)."""
    swizzle_cd = swizzle_mode(block_n, 2)
    smem = min(block_m, LAYOUT_AD_M) * swizzle_cd * 2
    smem += num_stages * block_m * BLOCK_K
    smem += num_stages * block_n * BLOCK_K
    smem += num_stages * round_up(block_m, 128) * 4
    smem += num_stages * round_up(block_n, 128) * 4
    smem += num_stages * 8 * 3 + 2 * 8 * 2 + 8
    smem += 4
    return smem, swizzle_cd


class Config(NamedTuple):
    num_sms: int  # num_min_sms: the persistent grid
    block_m: int
    block_n: int
    num_stages: int
    num_last_stages: int
    swizzle_cd: int
    smem: int


def best_config(layout: str, m: int, n: int, k: int, groups: int, sms: int) -> Config:
    """get_best_configs (+ NUM_LAST_STAGES) for K-major FP8 A/B and bf16 D.

    ``m`` is the total row count for the contiguous layout and
    ``expected_m`` for the masked one, as FlashInfer passes them. The
    multicast search is omitted: it can only pick multicast for
    ``GemmType::Normal``, which neither layout is.
    """
    if layout not in GEMM_TYPE:
        raise ValueError(f"unknown layout {layout!r}")
    block_ms = (BLOCK_M,)  # contiguous: alignment; masked: K-major B
    block_ns = tuple(range(16, 257, 16))  # K-major B

    def blocks(bm: int, bn: int) -> int:
        return ceil_div(m, bm) * ceil_div(n, bn) * groups

    def waves(bm: int, bn: int) -> int:
        return ceil_div(blocks(bm, bn), sms)

    def last_util(bm: int, bn: int) -> int:
        return blocks(bm, bn) % sms or sms

    best_m = best_n = 0
    for bm in block_ms:
        for bn in block_ns:
            if not best_m:
                success = True
            else:
                success = waves(bm, bn) < waves(best_m, best_n)
                if waves(bm, bn) == waves(best_m, best_n):
                    util, best_util = last_util(bm, bn), last_util(best_m, best_n)
                    success = util > best_util
                    if util == best_util:
                        success |= bm == best_m and bn < best_n
                        success |= bn == best_n and bm < best_m
                        success |= bm != best_m and bn > best_n
            if success and tmem_legal(bm, bn):
                best_m, best_n = bm, bn
    stages, smem, swizzle_cd = 0, 0, 0
    for stages in (8, 7, 6, 5, 4, 3, 2, 1):
        if stages > max(k // 128, 1):
            continue
        smem, swizzle_cd = smem_size(best_m, best_n, stages)
        if smem <= SM100_SMEM_CAPACITY:
            break
    num_min_sms = ceil_div(blocks(best_m, best_n), waves(best_m, best_n))
    return Config(
        num_min_sms,
        best_m,
        best_n,
        stages,
        ceil_div(k, BLOCK_K) % stages,
        swizzle_cd,
        smem,
    )


def unpack_scales(packed: torch.Tensor, k: int) -> torch.Tensor:
    """int32 ``[..., ceil(K/512), rows]`` UE8M0 packing -> float32 ``[...,
    rows, ceil(K/128)]`` scales ``2**(e - 127)``."""
    exponents = packed.transpose(-1, -2).contiguous().view(torch.uint8)
    exponents = exponents[..., : ceil_div(k, BLOCK_K)].to(torch.int32)
    return torch.exp2((exponents - 127).to(torch.float32))


def pack_scales(exponents: torch.Tensor) -> torch.Tensor:
    """uint8 UE8M0 ``[..., rows, ceil(K/128)]`` -> int32 ``[..., ceil(K/512),
    rows]`` (get_col_major_tma_aligned_packed_tensor's packing; rows % 4 == 0)."""
    *batch, rows, kb = exponents.shape
    padded = torch.zeros(
        (*batch, rows, round_up(kb, 4)), dtype=torch.uint8, device=exponents.device
    )
    padded[..., :kb] = exponents
    return padded.view(torch.int32).transpose(-1, -2).contiguous()


def dequantize(
    values: torch.Tensor, scales: torch.Tensor, k: int, dtype=torch.float32
) -> torch.Tensor:
    """FP8 ``[rows, K]`` times per-(row, 128-K block) scales ``[rows, K/128]``."""
    rows = values.shape[0]
    blocks = values.to(dtype).reshape(rows, k // BLOCK_K, BLOCK_K)
    return (blocks * scales.to(dtype)[..., None]).reshape(rows, k)


FP8_CHUNK = 1 << 26  # bytes transformed at a time by random_fp8
VALIDATE_CHUNK = 1 << 24  # output elements compared at a time


def random_fp8(shape: Sequence[int], generator, device) -> torch.Tensor:
    """FP8 E4M3 values of magnitude in [0.25, 3.75], random sign (no NaN).

    Random bytes are drawn in place into the output (the same stream as
    ``torch.randint(0, 256, dtype=uint8)``) and mapped to sign | (0x28 +
    low 5 bits) chunk by chunk, so a B tensor of several GiB (256 groups of
    4096 x 7168) needs no full-size temporary.
    """
    out = torch.empty(tuple(shape), dtype=torch.uint8, device=device)
    out.random_(0, 256, generator=generator)
    flat = out.view(-1)
    for start in range(0, flat.numel(), FP8_CHUNK):
        chunk = flat[start : start + FP8_CHUNK]
        sign = chunk & 0x80
        chunk.bitwise_and_(0x1F).add_(0x28).bitwise_or_(sign)
    return out.view(torch.float8_e4m3fn)


def random_exponents(
    shape: Sequence[int], center: int, generator, device
) -> torch.Tensor:
    """UE8M0 exponents ``127 + center + {-1, 0, 1}``."""
    jitter = torch.randint(
        -1, 2, tuple(shape), dtype=torch.int32, generator=generator, device=device
    )
    return (127 + center + jitter).to(torch.uint8)


def contiguous_indices(rows: Sequence[int], slack: int = 0) -> torch.Tensor:
    """int32 ``m_indices`` of FlashInfer's contiguous layout (CPU).

    Group ``g`` contributes ``rows[g]`` rows holding ``g`` followed by -1
    padding up to the 128-row alignment (``get_m_alignment_for_contiguous_
    layout``); empty groups contribute nothing. ``slack`` trailing 128-row
    blocks of -1 model a buffer allocated larger than the routed rows.
    """
    parts = []
    for group, count in enumerate(rows):
        if count:
            parts.append(torch.full((count,), group, dtype=torch.int32))
            parts.append(torch.full((round_up(count, BLOCK_M) - count,), -1))
    parts.append(torch.full((slack * BLOCK_M,), -1))
    return torch.cat([p.to(torch.int32) for p in parts])


def contiguous_blocks(params: dict[str, Any]) -> int:
    """ceil(M / 128) of a contiguous case (M is always a multiple of 128)."""
    rows = params["rows"]
    return sum(ceil_div(r, BLOCK_M) for r in rows) + params.get("slack", 0)


class LaunchSpec(NamedTuple):
    grid: int
    block: int
    args: list[Any]
    shared_mem: int
    cluster: tuple[int, int, int] | None


Encoder = Callable[..., bytes]


class Dispatch(NamedTuple):
    """One (group count, SM count) for which FlashInfer selects a kernel:
    the ``ceil(M / 128)`` ranges (searched up to MAX_M_BLOCKS; a range
    ending there is open-ended) and the cubin FlashInfer loads for it."""

    num_groups: int
    sms: int
    m_blocks: tuple[tuple[int, int], ...]
    cubin: str


MAX_M_BLOCKS = 1024  # the compiler's dispatch search bound


def range_values(entry: Dispatch, limit: int) -> list[int]:
    """``ceil(M / 128)`` values of ``entry``'s ranges up to ``limit``
    (open-ended ranges continue past MAX_M_BLOCKS; ``config`` checks)."""
    values: list[int] = []
    for lo, hi in entry.m_blocks:
        top = limit if hi >= MAX_M_BLOCKS else min(hi, limit)
        values.extend(range(lo, top + 1))
    return values


@cache
def variant_index() -> dict[str, Any]:
    if not VARIANTS.is_file():
        return {}
    return json.loads(VARIANTS.read_text())


# --- upstream parametrizations (FlashInfer v0.6.9, DeepGEMM 78b6900) --------

FLASHINFER_V069 = "a1aa676196f798435248d9ea205c67674476f473"
DEEPGEMM_TESTS = "78b69000794d0937b47ae3387eff7663410264d1"
FI_TEST = "tests/gemm/test_groupwise_scaled_gemm_fp8.py"
FI_BENCH = "benchmarks/bench_deepgemm_blackwell.py"
DG_TEST = "tests/test_fp8_fp4.py"
FI_NK = ((128, 512), (512, 128), (4096, 7168), (7168, 2048))
FI_TEST_M = (128, 256, 512, 1024)
FI_GROUPS = (1, 4, 8, 64, 128, 256)
FI_BENCH_M = (128, 256, 1024, 8192, 16384)
# tests/generators.py enumerate_m_grouped_{contiguous,masked}: (num_groups,
# expected m per group); masked max_m. Their (N, K) are not compiled into
# any of these kernels, so the regimes are replayed on the kernels' (N, K);
# the masked num_groups=6 rows have no 6-group kernel.
DG_CONTIGUOUS = ((4, 8192), (8, 4096))
DG_MASKED = ((32, 192), (32, 20), (6, 1024), (6, 20))
DG_MASKED_MAX_M = 4096
UPSTREAM_SMS = 148  # FlashInfer's B200 CI


def fi_contiguous_tests() -> list[tuple[int, int]]:
    """test_fp8_groupwise_group_deepgemm (m, group_size) that run: m rows,
    m // group_size (>= 128) rows per group, every row valid."""
    return [(m, g) for m in FI_TEST_M for g in FI_GROUPS if m // g >= 128]


def fi_contiguous_bench() -> list[tuple[int, int]]:
    """bench_deepgemm_grouped_fp8_blackwell (batch_size, m): m rows per group."""
    return [
        (g, m)
        for g in FI_GROUPS
        for m in FI_BENCH_M
        if m // g >= 128 and m * g <= 16384
    ]


def fi_masked_bench() -> list[tuple[int, int]]:
    """bench_deepgemm_batch_fp8_blackwell (batch_size, m): capacity m."""
    return [(g, m) for g in FI_GROUPS for m in FI_BENCH_M if m * g <= 16384]


def fi_contiguous_test_id(m: int, n: int, k: int, groups: int) -> str:
    return (
        f"{FI_TEST}::test_fp8_groupwise_group_deepgemm"
        f"[m={m}-nk=({n}, {k})-group_size={groups}]"
    )


def fi_masked_test_id(m: int, n: int, k: int, groups: int) -> str:
    return (
        f"{FI_TEST}::test_fp8_groupwise_batch_deepgemm_masked"
        f"[m={m}-nk=({n}, {k})-group_size={groups}]"
    )


def fi_contiguous_bench_id(groups: int, m: int, n: int, k: int) -> str:
    return (
        f"{FI_BENCH}::bench_deepgemm_grouped_fp8_blackwell"
        f"(batch_size={groups}, m={m}, n={n}, k={k})"
    )


def fi_masked_bench_id(groups: int, m: int, n: int, k: int) -> str:
    return (
        f"{FI_BENCH}::bench_deepgemm_batch_fp8_blackwell"
        f"(batch_size={groups}, m={m}, n={n}, k={k})"
    )


def dg_contiguous_id(groups: int, expected: int, n: int, k: int) -> str:
    return (
        f"{DG_TEST}::test_m_grouped_gemm_contiguous[num_groups={groups}, "
        f"expected_m_per_group={expected}; (N, K) replayed as ({n}, {k})]"
    )


def dg_masked_id(groups: int, expected: int, n: int, k: int) -> str:
    return (
        f"{DG_TEST}::test_m_grouped_gemm_masked[num_groups={groups}, "
        f"max_m={DG_MASKED_MAX_M}, expected_m_per_group={expected}; "
        f"(N, K) replayed as ({n}, {k})]"
    )


def _stable_seed(*parts: Any) -> int:
    return zlib.crc32(":".join(map(str, parts)).encode()) & 0x7FFFFFFF


def dg_contiguous_rows(groups: int, expected: int, n: int, k: int) -> list[int]:
    """generate_m_grouped_contiguous: int(expected * U(0.7, 1.3)) rows per
    group (aligned with -1 padding), seeded."""
    rng = random.Random(_stable_seed("dg_contiguous", groups, expected, n, k))
    return [int(expected * rng.uniform(0.7, 1.3)) for _ in range(groups)]


def dg_masked_rows(groups: int, expected: int, n: int, k: int) -> tuple[list[int], int]:
    """generate_m_grouped_masked: masked_m = int(expected * U(0.7, 1.3)),
    and test_m_grouped_gemm_masked's expected_m = int(expected * 1.2)."""
    rng = random.Random(_stable_seed("dg_masked", groups, expected, n, k))
    rows = [int(expected * rng.uniform(0.7, 1.3)) for _ in range(groups)]
    return rows, int(expected * 1.2)


def fi_natural_expected(m: int) -> int:
    """expected_m = min(int(mean(masked_m)) + 1, m) of FlashInfer's masked
    test/benchmark at the mean of masked_m ~ U{0, m - 1}."""
    return min(m // 2 + 1, m)


def uniform_rows_with_mean(groups: int, m: int, expected: int, seed: int) -> list[int]:
    """masked_m ~ randint(0, m) as FlashInfer draws it, adjusted so that
    min(int(mean) + 1, m) == expected (sum == (expected - 1) * groups)."""
    rng = random.Random(seed)
    target = (expected - 1) * groups
    values = [rng.randrange(m) for _ in range(groups)]
    total = sum(values)
    if total:
        values = [min(m - 1, round(v * target / total)) for v in values]
    else:
        values = [0] * groups
    diff, i = target - sum(values), 0
    while diff:
        step = 1 if diff > 0 else -1
        if 0 <= values[i % groups] + step <= m - 1:
            values[i % groups] += step
            diff -= step
        i += 1
    return values


# --- case construction --------------------------------------------------------

TOP_K = 8  # routed experts per token of DeepSeek-V3 and Kimi-K2
MODEL_TOKENS = 16384  # largest tokens per rank of the model cases
SMOKE_B_BYTES = 512 << 20  # weight bytes of ordinary smoke cases
MANY_GROUPS_B_BYTES = 1 << 30  # ... of the many-groups smoke case
CASE_BUDGET = 40 * 10**9  # inputs + outputs + reference temporaries (B200)
# (N, K) of the production kernels: expert GEMMs of DeepSeek-V3 / Kimi-K2.
PRODUCTION_LAYERS = {(4096, 7168): "expert_gate_up", (7168, 2048): "expert_down"}
STRESS_TOKENS = (16384, 12288, 8192, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512)
STRESS_TOKENS += (384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1)
TYPICAL_TOKENS = (2048, 1024, 4096, 512, 256, 8192, 128, 64, 32, 16, 8, 4, 2, 1)


def ragged_rows(
    blocks: int, groups: int, rng: random.Random, *, empty: int = 0
) -> list[int]:
    """Valid rows per group of a contiguous layout with ``blocks`` 128-row
    blocks: long-tailed block counts, ``empty`` (or more, when there are
    fewer blocks than groups) empty groups including the last one, partial
    last blocks (1-127 padding rows), one fully valid group and, where a
    group has a single block, one group with a single row."""
    if blocks <= 0:
        return [0] * groups
    candidates = list(range(groups))
    if empty and groups > 1:
        drop = {groups - 1}
        if empty > 1 and groups > 2:
            drop |= set(rng.sample(range(1, groups - 1), min(empty - 1, groups - 2)))
        candidates = [g for g in candidates if g not in drop]
    if len(candidates) > blocks:
        candidates = sorted(rng.sample(candidates, blocks))
    counts = skewed_lengths(blocks, len(candidates), rng.randrange(1 << 30), 1.0, 1)
    # The largest group is fully valid when another group has padding.
    top = max(range(len(counts)), key=counts.__getitem__) if len(counts) > 1 else -1
    rows, single = [0] * groups, False
    for i, (group, count) in enumerate(zip(candidates, counts)):
        if count == 1 and not single and i != top:
            rows[group], single = 1, True
        elif i == top:
            rows[group] = count * BLOCK_M
        else:
            rows[group] = count * BLOCK_M - rng.randrange(1, BLOCK_M)
    return rows


def nice_tokens(lo: int, hi: int, target: int) -> int:
    """A round token count in [lo, hi] near ``target``."""
    value = min(max(target, lo), hi)
    for shift in range(14, -1, -1):
        step = 1 << shift
        candidate = round(value / step) * step
        if lo <= candidate <= hi:
            return candidate
    return value


def contiguous_params(groups: int, rows: Sequence[int], slack: int = 0, **extra):
    return {"groups": groups, "rows": list(rows), "slack": slack, **extra}


def masked_params(rows: Sequence[int], capacity: int, expected_m: int, **extra):
    return {"rows": list(rows), "capacity": capacity, "expected_m": expected_m, **extra}


class _CaseBuilder:
    """Cases of one FP8 variant; every case is checked to be dispatched to
    the variant by FlashInfer's get_best_configs (``config``)."""

    def __init__(self, cls: type[_DeepGemmFP8]):
        self.cls = cls
        self.n, self.k = cls.shape_n, cls.shape_k
        self.nb = ceil_div(self.n, cls.block_n)
        self.layer = PRODUCTION_LAYERS.get((self.n, self.k))
        self.cases: dict[str, CaseSpec] = {}

    def seed(self, label: str) -> int:
        return _stable_seed(self.cls.name, label)

    def add(self, case: CaseSpec) -> None:
        cls = self.cls
        if cls.layout == "contiguous":
            cls.config(case.params["groups"], contiguous_blocks(case.params) * BLOCK_M)
        else:
            rows = case.params["rows"]
            if len(rows) != cls.num_groups or max(rows) > case.params["capacity"]:
                raise ValueError(f"{cls.name}/{case.name}: bad masked rows")
            cls.config(cls.num_groups, case.params["expected_m"])
        self.cases.setdefault(case.name, case)

    def model(self, groups: int, alternate: bool) -> tuple[str, int]:
        """(model, routed experts): Kimi-K2 for alternate cases when its 384
        experts split into ``groups`` per rank, else DeepSeek-V3."""
        name = (
            "kimi_k2"
            if alternate and MODELS["kimi_k2"].experts % groups == 0
            else "deepseek_v3"
        )
        spec = MODELS[name]
        if spec.linear_shapes()[self.layer] != (self.n, self.k) or spec.top_k != TOP_K:
            raise AssertionError(f"{name} {self.layer} is not ({self.n}, {self.k})")
        return name, spec.experts

    def build(self) -> list[CaseSpec]:
        if self.cls.layout == "contiguous":
            self.contiguous_upstream()
            self.contiguous_smoke()
            self.contiguous_throughput()
        else:
            self.masked_upstream()
            self.masked_smoke()
            self.masked_throughput()
        return list(self.cases.values())

    # -- contiguous ---------------------------------------------------------------

    def contiguous_upstream(self) -> None:
        cls, n, k = self.cls, self.n, self.k
        for m, groups in fi_contiguous_tests():
            if cls.selects(groups, m // BLOCK_M, UPSTREAM_SMS):
                params = contiguous_params(groups, [m // groups] * groups)
                test = fi_contiguous_test_id(m, n, k, groups)
                label = f"fi_test_m{m}_g{groups}"
                self.add(
                    upstream_case(
                        label,
                        params,
                        test,
                        seed=self.seed(label),
                        revision=FLASHINFER_V069,
                    )
                )
        for groups, m in fi_contiguous_bench():
            if cls.selects(groups, groups * m // BLOCK_M, UPSTREAM_SMS):
                params = contiguous_params(groups, [m] * groups)
                test = fi_contiguous_bench_id(groups, m, n, k)
                label = f"fi_bench_g{groups}_m{m}"
                self.add(
                    upstream_case(
                        label,
                        params,
                        test,
                        suite="throughput",
                        seed=self.seed(label),
                        revision=FLASHINFER_V069,
                    )
                )
        if self.layer is None:
            return
        for groups, expected in DG_CONTIGUOUS:
            rows = dg_contiguous_rows(groups, expected, n, k)
            params = contiguous_params(groups, rows)
            if cls.selects(groups, contiguous_blocks(params), UPSTREAM_SMS):
                label = f"deepgemm_g{groups}_m{expected}"
                test = dg_contiguous_id(groups, expected, n, k)
                self.add(
                    upstream_case(
                        label,
                        params,
                        test,
                        seed=self.seed(label),
                        revision=DEEPGEMM_TESTS,
                    )
                )

    def contiguous_smoke(self) -> None:
        cls, n, k, nb = self.cls, self.n, self.k, self.nb
        options = []
        for entry in cls.dispatch:
            groups = entry.num_groups
            if groups * n * k > SMOKE_B_BYTES:
                continue
            for blocks in range_values(entry, 96):
                grid = cls.config(groups, blocks * BLOCK_M).num_sms
                used = min(groups, blocks - (blocks >= 3))
                score = (blocks * nb > grid, used >= 2, used, -blocks)
                options.append((score, groups, blocks))
        if not options:
            entry = min(cls.dispatch, key=lambda e: e.num_groups)
            options.append(((), entry.num_groups, entry.m_blocks[0][0]))
        _, groups, blocks = max(options)
        rng = random.Random(self.seed("smoke_ragged"))
        slack = int(blocks >= 3)
        empty = max(1, groups // 4) if groups >= 3 else 0
        rows = ragged_rows(blocks - slack, groups, rng, empty=empty)
        label = f"smoke_ragged_g{groups}_b{blocks}"
        self.add(
            CaseSpec(label, contiguous_params(groups, rows, slack), self.seed(label))
        )
        first = (groups, blocks)
        if 0 not in rows:
            # No empty group yet: leave the first group empty, with S1's
            # group count or the smallest multi-group count of this code.
            multi = [
                (e.num_groups, b)
                for e in sorted(cls.dispatch, key=lambda e: e.num_groups)
                for b in range_values(e, 96)
                if e.num_groups >= 2
                and b >= 2
                and e.num_groups * n * k <= SMOKE_B_BYTES
            ]
            if groups >= 2:
                multi.insert(0, (groups, blocks))
            if multi:
                groups, blocks = multi[0]
                slack = int(blocks >= 3)
                rows = [0] + ragged_rows(blocks - slack, groups - 1, rng)
                label = f"smoke_empty_first_g{groups}_b{blocks}"
                params = contiguous_params(groups, rows, slack)
                self.add(CaseSpec(label, params, self.seed(label)))
        # A single group with a partial last block, or balanced fully valid
        # groups when no single-group cubin has this code.
        single = [e for e in cls.dispatch if e.num_groups == 1]
        entry = single[0] if single else min(cls.dispatch, key=lambda e: e.num_groups)
        values = [b for b in range_values(entry, 96) if (entry.num_groups, b) != first]
        if values:
            blocks = max([b for b in values if b <= 64] or [min(values)])
            if single:
                label = f"smoke_single_b{blocks}"
                params = contiguous_params(1, [blocks * BLOCK_M - 37])
            else:
                groups = entry.num_groups
                used = min(groups, blocks)
                counts = split_lengths(blocks, used) + [0] * (groups - used)
                label = f"smoke_balanced_g{groups}_b{blocks}"
                params = contiguous_params(groups, [c * BLOCK_M for c in counts])
            self.add(CaseSpec(label, params, self.seed(label)))
        # Many groups, fewer blocks than groups: most experts empty.
        many = [
            e
            for e in cls.dispatch
            if e.num_groups > first[0] and e.num_groups * n * k <= MANY_GROUPS_B_BYTES
        ]
        if many:
            entry = max(many, key=lambda e: e.num_groups)
            groups = entry.num_groups
            values = range_values(entry, 256)
            blocks = min(values, key=lambda b: (abs(b - (groups // 2 + 1)), b))
            slack = int(blocks >= 2)
            rng = random.Random(self.seed("smoke_many_groups"))
            rows = ragged_rows(blocks - slack, groups, rng)
            label = f"smoke_many_groups_g{groups}_b{blocks}"
            self.add(
                CaseSpec(
                    label, contiguous_params(groups, rows, slack), self.seed(label)
                )
            )

    def contiguous_tokens(self, entry: Dispatch, order: Sequence[int]) -> int | None:
        groups, seed = entry.num_groups, self.seed(f"model_g{entry.num_groups}")
        for tokens in order:
            rows = skewed_lengths(TOP_K * tokens, groups, seed, 0.6, 0)
            if self.cls.selects(groups, sum(ceil_div(r, BLOCK_M) for r in rows)):
                return tokens
        return None

    def contiguous_exact_tokens(self, entry: Dispatch) -> int | None:
        """Largest token count whose routed rows select the kernel, for
        kernels whose narrow ranges the coarse token lists miss."""
        groups, seed = entry.num_groups, self.seed(f"model_g{entry.num_groups}")

        def blocks(tokens: int) -> int:
            rows = skewed_lengths(TOP_K * tokens, groups, seed, 0.6, 0)
            return sum(ceil_div(r, BLOCK_M) for r in rows)

        top = blocks(MODEL_TOKENS)
        for target in sorted(set(range_values(entry, top)), reverse=True):
            lo, hi = 1, MODEL_TOKENS
            while lo < hi:  # smallest token count reaching ``target`` blocks
                mid = (lo + hi) // 2
                lo, hi = (lo, mid) if blocks(mid) >= target else (mid + 1, hi)
            found = None
            for tokens in range(lo, min(lo + 256, MODEL_TOKENS) + 1):
                value = blocks(tokens)
                if value == target and self.cls.selects(groups, value):
                    found = tokens
                elif value > target:
                    break
            if found is not None:
                return found
        return None

    def contiguous_model_case(
        self, entry: Dispatch, tokens: int, alternate: bool
    ) -> CaseSpec:
        groups = entry.num_groups
        model, experts = self.model(groups, alternate)
        rows = skewed_lengths(
            TOP_K * tokens, groups, self.seed(f"model_g{groups}"), 0.6, 0
        )
        ep = experts // groups
        params = contiguous_params(groups, rows, tokens=tokens, ep=ep, experts=experts)
        label = f"{model}_ep{ep}_t{tokens}"
        return model_case(label, params, model, self.layer, seed=self.seed(label))

    def contiguous_throughput(self) -> None:
        cls = self.cls
        found = False
        if self.layer is not None:
            # Prefill: tokens per rank x top-8 rows routed to the rank's
            # experts (256 / EP of DeepSeek-V3), long-tailed per expert.
            best = None
            for entry in cls.dispatch:
                tokens = self.contiguous_tokens(entry, STRESS_TOKENS)
                if tokens is None:
                    tokens = self.contiguous_exact_tokens(entry)
                if tokens is not None:
                    key = (tokens, -abs(math.log2(entry.num_groups / 8)))
                    if best is None or key > best[0]:
                        best = (key, entry, tokens)
            if best is not None:
                _, entry, stress = best
                self.add(self.contiguous_model_case(entry, stress, False))
                found = True
                order = sorted(
                    cls.dispatch, key=lambda e: abs(math.log2(e.num_groups / 32))
                )
                for entry in order:
                    typical = [t for t in TYPICAL_TOKENS if 2 * t <= stress]
                    tokens = self.contiguous_tokens(entry, typical)
                    if tokens is not None:
                        self.add(self.contiguous_model_case(entry, tokens, True))
                        break
        if found:
            return
        # Toy (FlashInfer unit-test) shapes: the top of the dispatch range.
        best = None
        for entry in cls.dispatch:
            values = [
                b
                for b in range_values(entry, MAX_M_BLOCKS)
                if cls.selects(entry.num_groups, b)
            ]
            if values and (best is None or (max(values), entry.num_groups) > best[:2]):
                best = (max(values), entry.num_groups)
        assert best is not None, cls.name
        blocks, groups = best
        rng = random.Random(self.seed("stress"))
        rows = ragged_rows(blocks, groups, rng, empty=groups // 8)
        reason = (
            f"FlashInfer unit-test shape (N, K) = ({self.n}, {self.k}) (no model "
            "layer); M at the top of this kernel's dispatch range, ragged groups"
            if self.layer is None
            else "no model token count selects this kernel; top of its dispatch range"
        )
        self.add(
            synthetic(
                f"stress_g{groups}_b{blocks}", contiguous_params(groups, rows), reason
            )
        )

    # -- masked -------------------------------------------------------------------

    def masked_upstream(self) -> None:
        cls, n, k, groups = self.cls, self.n, self.k, self.cls.num_groups
        for m in FI_TEST_M:
            choices = [
                b
                for b in range(1, m // BLOCK_M + 1)
                if cls.selects(groups, b, UPSTREAM_SMS)
            ]
            if not choices:
                continue
            natural = fi_natural_expected(m)
            first = ceil_div(natural, BLOCK_M)
            blocks = min(choices, key=lambda b: (abs(b - first), b))
            expected = natural if blocks == first else (blocks - 1) * BLOCK_M + 64
            label = f"fi_test_m{m}_e{expected}"
            rows = uniform_rows_with_mean(groups, m, expected, self.seed(label))
            test = fi_masked_test_id(m, n, k, groups)
            self.add(
                upstream_case(
                    label,
                    masked_params(rows, m, expected),
                    test,
                    seed=self.seed(label),
                    revision=FLASHINFER_V069,
                )
            )
        for bench_groups, m in fi_masked_bench():
            expected = fi_natural_expected(m)
            if bench_groups == groups and cls.selects(
                groups, ceil_div(expected, BLOCK_M), UPSTREAM_SMS
            ):
                label = f"fi_bench_m{m}"
                rows = uniform_rows_with_mean(groups, m, expected, self.seed(label))
                test = fi_masked_bench_id(groups, m, n, k)
                self.add(
                    upstream_case(
                        label,
                        masked_params(rows, m, expected),
                        test,
                        suite="throughput",
                        seed=self.seed(label),
                        revision=FLASHINFER_V069,
                    )
                )
        if self.layer is None:
            return
        for dg_groups, per_group in DG_MASKED:
            if dg_groups != groups:
                continue
            rows, expected = dg_masked_rows(groups, per_group, n, k)
            if cls.selects(groups, ceil_div(expected, BLOCK_M), UPSTREAM_SMS):
                label = f"deepgemm_m{per_group}"
                test = dg_masked_id(groups, per_group, n, k)
                params = masked_params(rows, DG_MASKED_MAX_M, expected)
                self.add(
                    upstream_case(
                        label,
                        params,
                        test,
                        seed=self.seed(label),
                        revision=DEEPGEMM_TESTS,
                    )
                )

    def masked_smoke(self) -> None:
        cls, groups, nb = self.cls, self.cls.num_groups, self.nb
        entry = cls.dispatch[0]
        values = range_values(entry, 64) or [entry.m_blocks[0][0]]
        persistent = [
            b
            for b in values
            if b * nb * groups > cls.config(groups, b * BLOCK_M).num_sms
        ]
        blocks = persistent[0] if persistent else values[-1]
        rng = random.Random(self.seed("smoke_ragged"))
        expected = (blocks - 1) * BLOCK_M + rng.randrange(1, BLOCK_M + 1)
        capacity = (blocks + 1) * BLOCK_M
        rows = [
            min(capacity, int(expected * rng.lognormvariate(0.0, 0.6)))
            for _ in range(groups)
        ]
        rows[0] = capacity  # full group
        if groups >= 2:
            rows[-1] = 0  # empty expert
        if groups >= 3:
            rows[1] = 1  # single row
        if groups >= 4:
            rows[2] = blocks * BLOCK_M  # exact multiple of the block
        label = f"smoke_ragged_e{expected}"
        self.add(
            CaseSpec(label, masked_params(rows, capacity, expected), self.seed(label))
        )
        # Spare capacity, uniform masked_m as FlashInfer's test draws it.
        others = [b for b in values if b != blocks]
        second = min(others) if others else blocks
        rng = random.Random(self.seed("smoke_spare"))
        expected = second * BLOCK_M - rng.randrange(0, BLOCK_M - 1)
        capacity = (2 * second + 1) * BLOCK_M
        rows = [rng.randrange(capacity) for _ in range(groups)]
        if groups >= 2:
            rows[-1] = capacity
        label = f"smoke_spare_e{expected}"
        self.add(
            CaseSpec(label, masked_params(rows, capacity, expected), self.seed(label))
        )

    def masked_model_case(self, tokens: int, alternate: bool) -> CaseSpec | None:
        groups = self.cls.num_groups
        model, experts = self.model(groups, alternate)
        label_seed = self.seed(f"model_t{tokens}")
        rows = skewed_lengths(TOP_K * tokens, groups, label_seed, 0.6, 0)
        expected = ceil_div(TOP_K * tokens, groups)
        capacity = round_up(max(max(rows), 1), BLOCK_M)
        ep = experts // groups
        params = masked_params(
            rows, capacity, expected, tokens=tokens, ep=ep, experts=experts
        )
        label = f"{model}_ep{ep}_t{tokens}"
        case = model_case(label, params, model, self.layer, seed=self.seed(label))
        if self.cls.case_bytes(case) > CASE_BUDGET:
            return None
        return case

    def masked_throughput(self) -> None:
        cls, groups = self.cls, self.cls.num_groups
        entry = cls.dispatch[0]
        if self.layer is not None:
            # Decode: tokens per rank x top-8 rows routed to the rank's
            # experts; expected_m = rows per expert (ceil(8T / G)), so
            # ceil(expected_m / 128) == b for 16 (b - 1) G < T <= 16 b G.
            top = ceil_div(ceil_div(TOP_K * MODEL_TOKENS, groups), BLOCK_M)
            spans = []
            for b in range_values(entry, top):
                lo, hi = 16 * (b - 1) * groups + 1, min(16 * b * groups, MODEL_TOKENS)
                if lo <= hi and cls.selects(groups, b):
                    spans.append((lo, hi))
            if spans:
                typical = min(
                    (nice_tokens(lo, hi, 256) for lo, hi in spans),
                    key=lambda t: (abs(math.log2(t / 256)), t),
                )
                case = self.masked_model_case(typical, False)
                if case is not None:
                    self.add(case)
                capped = [(lo, min(hi, 4096)) for lo, hi in spans if lo <= 4096]
                if capped:
                    lo, hi = max(capped, key=lambda s: s[1])
                    stress = nice_tokens(lo, hi, hi)
                    if stress >= 2 * typical:
                        case = self.masked_model_case(stress, True)
                        if case is not None:
                            self.add(case)
                if any(
                    c.suite == "throughput" and c.source.get("kind") == "model_shape"
                    for c in self.cases.values()
                ):
                    return
        cap = max(1, min(MAX_M_BLOCKS, (1 << 20) // (groups * BLOCK_M)))
        values = [b for b in range_values(entry, cap) if cls.selects(groups, b)]
        blocks = max(values) if values else entry.m_blocks[0][0]
        expected = blocks * BLOCK_M - 64
        rng = random.Random(self.seed("stress"))
        rows = [
            max(0, int(expected * rng.lognormvariate(0.0, 0.3))) for _ in range(groups)
        ]
        capacity = round_up(max(max(rows), 1), BLOCK_M)
        reason = (
            f"FlashInfer unit-test shape (N, K) = ({self.n}, {self.k}) (no model "
            "layer); expected_m at the top of this kernel's dispatch range"
            if self.layer is None
            else "no decode token count selects this kernel; top of its dispatch range"
        )
        self.add(
            synthetic(
                f"stress_e{expected}", masked_params(rows, capacity, expected), reason
            )
        )


@cache
def _variant_cases(cls: type[_DeepGemmFP8]) -> tuple[CaseSpec, ...]:
    return tuple(_CaseBuilder(cls).build())


class _DeepGemmFP8(Workload):
    """DeepGEMM ``sm100_fp8_gemm_1d1d_impl`` m-grouped variant (see module doc).

    Class attributes come from the variant index: ``layout``, ``shape_n``,
    ``shape_k``, ``block_n``, ``num_stages``, ``num_last_stages``,
    ``swizzle_cd``, ``smem`` and ``dispatch`` (one ``Dispatch`` per group
    count the kernel's code serves: FlashInfer's cubin for it, the SM count
    that selects it -- 148, the B200's, when any M does -- and the
    ``ceil(M / 128)`` ranges). ``num_groups`` is the compiled group count of
    a masked kernel; a contiguous kernel takes the group count from ``b``.
    """

    package = "deep_gemm"
    layout: ClassVar[str]
    shape_n: ClassVar[int]
    shape_k: ClassVar[int]
    block_n: ClassVar[int]
    num_stages: ClassVar[int]
    num_last_stages: ClassVar[int]
    swizzle_cd: ClassVar[int]
    smem: ClassVar[int]
    dispatch: ClassVar[tuple[Dispatch, ...]]
    num_groups: ClassVar[int]  # masked only
    rtol, atol = 2.0**-6, 1e-3

    def __init__(self, cubin, *, device=None):
        super().__init__(cubin, device=device)
        # cuTensorMapEncodeTiled; tests substitute a recording encoder on GPUs
        # without TMA (the driver refuses to encode there).
        self.encode: Encoder = driver_encode

    # -- dispatch ----------------------------------------------------------------

    @classmethod
    def entry(cls, groups: int) -> Dispatch:
        for entry in cls.dispatch:
            if entry.num_groups == groups:
                return entry
        served = [e.num_groups for e in cls.dispatch]
        raise ValueError(f"{cls.name}: no cubin for {groups} groups (serves {served})")

    @classmethod
    def config(cls, groups: int, m: int) -> Config:
        """FlashInfer's config for ``m`` (total rows / expected_m) and
        ``groups`` at the group count's SM count; raises unless it selects
        this kernel."""
        entry = cls.entry(groups)
        config = best_config(cls.layout, m, cls.shape_n, cls.shape_k, groups, entry.sms)
        expected = (BLOCK_M, cls.block_n, cls.num_stages, cls.num_last_stages)
        if config[1:5] != expected or config.swizzle_cd != cls.swizzle_cd:
            raise ValueError(
                f"{cls.name}: FlashInfer dispatches m={m}, groups={groups} to a "
                f"different kernel (block_n {config.block_n}, stages "
                f"{config.num_stages})"
            )
        return config

    @classmethod
    def selects(cls, groups: int, m_blocks: int, sms: int | None = None) -> bool:
        """Whether FlashInfer dispatches ``ceil(M / 128) = m_blocks`` with
        ``groups`` groups to this kernel (at ``sms`` SMs, if given)."""
        try:
            if sms is not None and cls.entry(groups).sms != sms:
                return False
            cls.config(groups, m_blocks * BLOCK_M)
        except ValueError:
            return False
        return True

    def get_cases(self) -> list[CaseSpec]:
        return list(_variant_cases(type(self)))

    @classmethod
    def case_bytes(cls, case: CaseSpec) -> int:
        """Peak bytes of a case's inputs, output and reference (with its
        per-group float temporaries), without allocating."""
        n, k, p = cls.shape_n, cls.shape_k, case.params
        k4 = ceil_div(k, 4 * BLOCK_K)
        if cls.layout == "contiguous":
            groups, m = p["groups"], contiguous_blocks(p) * BLOCK_M
            rows, largest = m, max(round_up(r, BLOCK_M) for r in p["rows"])
            inputs = m * (k + 4 * k4 + 4)
        else:
            groups, capacity = cls.num_groups, p["capacity"]
            rows, largest = groups * capacity, capacity
            inputs = rows * (k + 4 * k4) + 4 * groups
        inputs += groups * n * (k + 4 * k4)
        outputs = 2 * 2 * rows * n  # kernel output and reference
        temporaries = 4 * (largest * k + n * k + largest * n) + 12 * rows
        return inputs + outputs + temporaries

    # -- inputs ------------------------------------------------------------------

    def _operands(self, rows: int, groups_a: int, groups: int, g) -> tuple:
        """A, SFA, B, SFB with O(1) outputs: A/B magnitudes ~1.4, SFB
        exponents centred on -log2(1.6 sqrt(K))."""
        k, n = self.shape_k, self.shape_n
        kb = ceil_div(k, BLOCK_K)
        shape_a = (rows, k) if groups_a == 0 else (groups_a, rows, k)
        a = random_fp8(shape_a, g, self.device)
        sfa = random_exponents((*shape_a[:-1], kb), 0, g, self.device)
        b = random_fp8((groups, n, k), g, self.device)
        center = -round(math.log2(1.6 * math.sqrt(k)))
        # Per-128x128-block B scales (FlashInfer's recipe (1, 128, 128)),
        # expanded to rows as transform_sf_into_required_layout does.
        sfb_blocks = random_exponents(
            (groups, ceil_div(n, 128), kb), center, g, self.device
        )
        sfb = sfb_blocks.repeat_interleave(128, dim=1)[:, :n]
        return a, pack_scales(sfa), b, pack_scales(sfb)

    def get_inputs(self, case: CaseSpec) -> tuple:
        g = self.generator(case)
        p = case.params
        if self.layout == "contiguous":
            indices = contiguous_indices(p["rows"], p.get("slack", 0))
            a, sfa, b, sfb = self._operands(indices.numel(), 0, p["groups"], g)
            return a, sfa, b, sfb, indices.to(self.device)
        groups, capacity = self.num_groups, p["capacity"]
        a, sfa, b, sfb = self._operands(capacity, groups, groups, g)
        masked = torch.tensor(p["rows"], dtype=torch.int32, device=self.device)
        return a, sfa, b, sfb, masked, int(p["expected_m"])

    # -- reference ----------------------------------------------------------------

    def get_reference(self, inputs: tuple) -> tuple:
        """Exactly what the kernel defines: every valid row of the
        contiguous layout (padding rows, -1, are NaN: their values are
        undefined upstream) / the 128-row blocks covering ``masked_m[g]``
        rows of each masked group (other rows untouched, NaN)."""
        k, n = self.shape_k, self.shape_n
        if self.layout == "contiguous":
            a, sfa, b, sfb, indices = inputs
            m = a.shape[0]
            out = torch.full((m, n), torch.nan, dtype=torch.bfloat16, device=a.device)
            scale_a = unpack_scales(sfa, k)
            row_group = indices.long()
            for group in torch.unique(row_group[row_group >= 0]).tolist():
                rows = (row_group == group).nonzero().flatten()
                lhs = dequantize(a[rows], scale_a[rows], k)
                rhs = dequantize(b[group], unpack_scales(sfb[group], k), k)
                out[rows] = (lhs @ rhs.T).to(torch.bfloat16)
            return (out,)
        a, sfa, b, sfb, masked, _ = inputs
        groups, m, _ = a.shape
        out = torch.full(
            (groups, m, n), torch.nan, dtype=torch.bfloat16, device=a.device
        )
        for group, count in enumerate(masked.tolist()):
            rows = min(m, round_up(count, BLOCK_M))
            if not rows:
                continue
            lhs = dequantize(a[group, :rows], unpack_scales(sfa[group], k)[:rows], k)
            rhs = dequantize(b[group], unpack_scales(sfb[group], k), k)
            out[group, :rows] = (lhs @ rhs.T).to(torch.bfloat16)
        return (out,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != 1 or len(impl) != 1:
            raise AssertionError("expected one output")
        expected, actual = ref[0], impl[0]
        if not isinstance(actual, torch.Tensor) or (
            actual.shape,
            actual.dtype,
            actual.device,
        ) != (expected.shape, expected.dtype, expected.device):
            raise AssertionError("output shape, dtype or device mismatch")
        # Compared in chunks: assert_close's float temporaries over a whole
        # 256-group masked output would take several GiB.
        expected, actual = expected.reshape(-1), actual.reshape(-1)
        for start in range(0, expected.numel(), VALIDATE_CHUNK):
            want = expected[start : start + VALIDATE_CHUNK]
            got = actual[start : start + VALIDATE_CHUNK]
            written = ~torch.isnan(want)
            try:
                self.assert_close(
                    (want[written],), (got[written],), rtol=self.rtol, atol=self.atol
                )
            except AssertionError as error:
                raise AssertionError(
                    f"output elements [{start}, {start + want.numel()}): {error}"
                ) from None

    # -- launch ------------------------------------------------------------------

    def configure(self, function: cuda_driver.Function) -> None:
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, self.smem
        )

    def _tma_2d(
        self,
        t: torch.Tensor,
        inner: int,
        outer: int,
        smem_inner: int,
        smem_outer: int,
        outer_stride: int,
        swizzle: int,
        address: int | None = None,
    ) -> bytes:
        """make_tma_2d_desc (+ make_tma_xd_desc)."""
        if swizzle:
            smem_inner = swizzle // t.element_size()
        return self.encode(
            cuda_driver.CU_TENSOR_MAP_DATA_TYPE[t.dtype],
            t.data_ptr() if address is None else address,
            [inner, outer],
            [outer_stride * t.element_size()],
            [smem_inner, smem_outer],
            [1, 1],
            INTERLEAVE_NONE,
            SWIZZLE[swizzle],
            L2_PROMOTION_256B,
            OOB_FILL_NONE,
        )

    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        """Outputs and the complete launch (FlashInfer's *_kwargs_gen +
        SM100FP8GemmRuntime.launch); runs nothing."""
        k, n = self.shape_k, self.shape_n
        a, sfa, b, sfb, layout_tensor = inputs[:5]
        if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
            raise ValueError("a and b must be float8_e4m3fn")
        groups = b.shape[0] if self.layout == "contiguous" else self.num_groups
        if b.shape != (groups, n, k) or not b.is_contiguous():
            raise ValueError(f"b must be contiguous [{groups}, {n}, {k}]")
        if layout_tensor.dtype != torch.int32 or not layout_tensor.is_contiguous():
            raise ValueError("m_indices / masked_m must be contiguous int32")
        k4 = ceil_div(k, 4 * BLOCK_K)
        if self.layout == "contiguous":
            m = a.shape[0]
            if a.shape != (m, k) or layout_tensor.shape != (m,):
                raise ValueError("a must be [M, K] and m_indices [M]")
            config = self.config(groups, m)
            d = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
            rows_a, groups_a = m, 1
        else:
            m, expected_m = a.shape[1], int(inputs[5])
            if a.shape != (groups, m, k) or layout_tensor.shape != (groups,):
                raise ValueError("a must be [G, M, K] and masked_m [G]")
            if groups > 1 and m % BLOCK_M:
                raise ValueError("masked M must be a multiple of 128")
            config = self.config(groups, expected_m)
            d = torch.empty((groups, m, n), dtype=torch.bfloat16, device=a.device)
            rows_a, groups_a = m, groups
        if not a.is_contiguous() or m % 4:
            raise ValueError("a must be contiguous with M % 4 == 0")
        if sfa.dtype != torch.int32 or sfa.shape != (
            *((groups,) if self.layout == "masked" else ()),
            k4,
            m,
        ):
            raise ValueError("sfa must be packed int32 [(G,) ceil(K/512), M]")
        if sfb.dtype != torch.int32 or sfb.shape != (groups, k4, n):
            raise ValueError("sfb must be packed int32 [G, ceil(K/512), N]")
        if not (sfa.is_contiguous() and sfb.is_contiguous()):
            raise ValueError("scale factors must be contiguous")
        # NOTES (upstream): A, SFA and D of the contiguous layout carry no
        # group dimension (num_groups=1 descriptors).
        tensor_map_a = self._tma_2d(a, k, rows_a * groups_a, BLOCK_K, BLOCK_M, k, 128)
        tensor_map_b = self._tma_2d(b, k, n * groups, BLOCK_K, self.block_n, k, 128)
        tensor_map_d = self._tma_2d(
            d,
            n,
            rows_a * groups_a,
            self.block_n,
            min(BLOCK_M, LAYOUT_AD_M),
            n,
            self.swizzle_cd,
        )
        # make_tma_sf_desc: MN-major, TMA-aligned rows (M and N are % 4).
        tensor_map_sfa = self._tma_2d(sfa, m, k4 * groups_a, BLOCK_M, 1, m, 0)
        tensor_map_sfb = self._tma_2d(sfb, n, k4 * groups, self.block_n, 1, n, 0)
        args = [
            layout_tensor,  # grouped_layout
            ctypes.c_uint32(m),
            ctypes.c_uint32(n),
            ctypes.c_uint32(round_up(k, 128)),  # aligned_k
            tensor_map_a,
            tensor_map_b,
            tensor_map_sfa,
            tensor_map_sfb,
            tensor_map_d,  # TENSOR_MAP_C (unused: no accumulation)
            tensor_map_d,
        ]
        block = NUM_NON_EPILOGUE_THREADS + NUM_EPILOGUE_THREADS
        spec = LaunchSpec(config.num_sms, block, args, config.smem, (1, 1, 1))
        return spec, (d,)

    def _launch(self, spec: LaunchSpec) -> None:
        self.launch(
            spec.grid,
            spec.block,
            spec.args,
            shared_mem=spec.shared_mem,
            cluster=spec.cluster,
        )

    def run(self, inputs: tuple) -> tuple:
        spec, outputs = self.configure_launch(inputs)
        self._launch(spec)
        return outputs

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple | None]:
        spec, outputs = self.configure_launch(inputs)

        def launch() -> tuple:
            self._launch(spec)
            return outputs

        return launch, outputs


# --- sm_86: FlashInfer CUTLASS segment GEMM (column-major weights) ----------


@cache
def segment_layout(sidecar: str) -> dict[str, Any]:
    return json.loads((PACKAGE_DIR / sidecar).read_text())


def pack_segment_params(
    layout: dict[str, Any],
    *,
    problem_sizes: int,
    problem_count: int,
    ptr_a: int,
    ptr_b: int,
    ptr_c: int,
    ptr_d: int,
    lda: int,
    ldb: int,
    ldc: int,
    ldd: int,
    threadblock_count: int = 4,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> bytes:
    """GemmGrouped<...>::Params(args, nullptr) as CutlassSegmentGEMMRun builds
    it, at the probe's field offsets (padding zero)."""
    values: dict[str, tuple[str, Any]] = {
        "problem_visitor.problem_sizes": ("Q", problem_sizes),
        "problem_visitor.problem_count": ("i", problem_count),
        "problem_visitor.workspace": ("Q", 0),
        "problem_visitor.tile_count": ("i", 0),
        "threadblock_count": ("i", threadblock_count),
        "output_op.alpha": ("f", alpha),
        "output_op.beta": ("f", beta),
        "output_op.alpha_ptr": ("Q", 0),
        "output_op.beta_ptr": ("Q", 0),
        "output_op.alpha_ptr_array": ("Q", 0),
        "output_op.beta_ptr_array": ("Q", 0),
        "ptr_A": ("Q", ptr_a),
        "ptr_B": ("Q", ptr_b),
        "ptr_C": ("Q", ptr_c),
        "ptr_D": ("Q", ptr_d),
        "lda": ("Q", lda),
        "ldb": ("Q", ldb),
        "ldc": ("Q", ldc),
        "ldd": ("Q", ldd),
    }
    fields = layout["fields"]
    if set(fields) != set(values):
        raise ValueError(f"probe fields {sorted(fields)} != builder fields")
    buffer = bytearray(layout["params_size"])
    for name, (fmt, value) in values.items():
        offset, size = fields[name]
        if struct.calcsize(fmt) != size:
            raise ValueError(f"{name}: probe size {size} != {fmt}")
        struct.pack_into("<" + fmt, buffer, offset, value)
    return bytes(buffer)


def padding_mask(layout: dict[str, Any]) -> list[bool]:
    covered = [False] * layout["params_size"]
    for offset, size in layout["fields"].values():
        covered[offset : offset + size] = [True] * size
    return covered


class _SegmentGemm(Workload):
    """FlashInfer ``CutlassSegmentGEMMRun<DType>`` with column-major weights.

    Class attributes from the variant index: ``dtype`` (``torch.bfloat16`` /
    ``torch.float16``), ``sidecar`` (probe layout, relative to
    ``impls/deep_gemm``), ``mainloop`` and ``upstream_dispatch``.
    """

    package = "deep_gemm"
    dtype: ClassVar[torch.dtype]
    sidecar: ClassVar[str]
    mainloop: ClassVar[str]
    upstream_dispatch: ClassVar[str]

    def get_cases(self) -> list[CaseSpec]:
        return list(_segment_cases(self.dtype))

    @property
    def layout(self) -> dict[str, Any]:
        return segment_layout(self.sidecar)

    @staticmethod
    def case_bytes(case: CaseSpec) -> int:
        """Peak bytes of inputs, output and reference (with its per-segment
        float temporaries and the per-weight generation temporary)."""
        p = case.params
        lengths, n, k = p["lengths"], p["n"], p["k"]
        rows, weights = sum(lengths), p.get("weights", len(lengths))
        longest = max(lengths, default=0)
        inputs = 2 * (rows * k + weights * n * k) + 16 * len(lengths)
        outputs = 2 * 2 * rows * n
        temporaries = 4 * (longest * k + n * k + longest * n) + 64 * len(lengths)
        return inputs + outputs + temporaries

    def get_inputs(self, case: CaseSpec) -> tuple:
        g = self.generator(case)
        p = case.params
        lengths, n, k = p["lengths"], p["n"], p["k"]
        groups, weights = len(lengths), p.get("weights", len(lengths))
        x = self.randn((sum(lengths), k), g, self.dtype)
        # One weight at a time: no float32 temporary of the whole bank.
        w = torch.empty((weights, n, k), dtype=self.dtype, device=self.device)
        for i in range(weights):
            w[i] = self.randn((n, k), g, self.dtype, scale=k**-0.5)
        kind = p.get("indices", "identity")
        if kind == "permuted":  # distinct weights drawn from a larger bank
            cpu = torch.Generator().manual_seed(case.seed)
            indices = torch.randperm(weights, generator=cpu)[:groups]
        else:  # identity, or segments sharing weights (modulo the bank)
            indices = torch.arange(groups) % weights
        indptr = torch.tensor([0] + list(lengths), dtype=torch.int64).cumsum(0)
        return x, w, indptr.to(self.device), indices.to(self.device)

    def get_reference(self, inputs: tuple) -> tuple:
        x, w, indptr, indices = inputs
        y = torch.empty((x.shape[0], w.shape[1]), dtype=self.dtype, device=x.device)
        bounds, weight = indptr.tolist(), indices.tolist()
        for group in range(len(bounds) - 1):
            lo, hi = bounds[group], bounds[group + 1]
            rhs = w[weight[group]].float()
            y[lo:hi] = (x[lo:hi].float() @ rhs.T).to(self.dtype)
        return (y,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(ref) != 1 or len(impl) != 1:
            raise AssertionError("expected one output")
        tol = 1.6e-2 if self.dtype == torch.bfloat16 else 2e-3
        self.assert_close(ref, impl, rtol=tol, atol=tol)

    def configure(self, function: cuda_driver.Function) -> None:
        smem = self.layout["shared_storage"]
        if smem >= 48 << 10:  # BaseGrouped::initialize
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )

    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple, tuple]:
        """(launch, outputs, device argument arrays kept alive)."""
        x, w, indptr, indices = inputs
        groups, (_, n, k) = indptr.shape[0] - 1, w.shape
        if x.dtype != self.dtype or w.dtype != self.dtype:
            raise ValueError(f"x and w must be {self.dtype}")
        if (
            x.dim() != 2
            or x.shape[1] != k
            or not (x.is_contiguous() and w.is_contiguous())
        ):
            raise ValueError("x must be contiguous [M, K] and w [W, N, K]")
        if indptr.dtype != torch.int64 or indptr.dim() != 1 or groups < 0:
            raise ValueError("seg_indptr must be int64 [G + 1]")
        if indices.dtype != torch.int64 or indices.shape != (groups,):
            raise ValueError("weight_indices must be int64 [G]")
        if k % 8 or n % 8:
            raise ValueError("K and N must be multiples of 8 (128-bit accesses)")
        if any(t.device.type != "cuda" for t in inputs):
            raise ValueError("native launch needs CUDA tensors")
        # The upstream epilogue computes y = x w^T + 1.0 * y: zero-initialise.
        y = torch.zeros((x.shape[0], n), dtype=self.dtype, device=x.device)
        esize = x.element_size()
        # compute_sm80_group_gemm_args (FlashInfer's Triton kernel), in torch:
        # weight i of segment i, or weight_indices[i] when given.
        starts = indptr[:-1]
        problems = torch.stack(
            (
                indptr[1:] - starts,
                torch.full_like(starts, n),
                torch.full_like(starts, k),
            ),
            dim=1,
        ).to(torch.int32)
        x_data = x.data_ptr() + starts * k * esize
        w_data = w.data_ptr() + indices * k * n * esize
        y_data = y.data_ptr() + starts * n * esize
        x_ld = torch.full_like(starts, k)
        w_ld = torch.full_like(starts, k)  # column-major weights
        y_ld = torch.full_like(starts, n)
        arrays = (problems, x_data, w_data, y_data, x_ld, w_ld, y_ld)
        params = pack_segment_params(
            self.layout,
            problem_sizes=problems.data_ptr(),
            problem_count=groups,
            ptr_a=x_data.data_ptr(),
            ptr_b=w_data.data_ptr(),
            ptr_c=y_data.data_ptr(),
            ptr_d=y_data.data_ptr(),
            lda=x_ld.data_ptr(),
            ldb=w_ld.data_ptr(),
            ldc=y_ld.data_ptr(),
            ldd=y_ld.data_ptr(),
        )
        spec = LaunchSpec(
            4, self.layout["threads"], [params], self.layout["shared_storage"], None
        )
        return spec, (y,), arrays

    def _launch(self, spec: LaunchSpec) -> None:
        self.launch(spec.grid, spec.block, spec.args, shared_mem=spec.shared_mem)

    def run(self, inputs: tuple) -> tuple:
        spec, outputs, arrays = self.configure_launch(inputs)
        if inputs[2].shape[0] > 1:  # GemmGrouped::run returns early without problems
            self._launch(spec)
        self._keepalive = arrays  # until the next run (asynchronous launch)
        return outputs

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple | None]:
        spec, outputs, arrays = self.configure_launch(inputs)

        def launch() -> tuple:
            if inputs[2].shape[0] > 1:  # as run: no launch without problems
                self._launch(spec)
            assert arrays  # the device argument arrays live with the closure
            return outputs

        return launch, outputs


# --- sm_86 segment GEMM cases -----------------------------------------------

FLASHINFER_V070 = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
SEGMENT_TEST = "tests/gemm/test_group_gemm.py::test_segment_gemm"
SEGMENT_TEST_BATCH = (1, 77, 199)
SEGMENT_TEST_ROWS = (3, 10, 99)
SEGMENT_TEST_DIMS = (128, 1024, 4096)
SEGMENT_WEIGHT_BYTES = 3 << 30  # larger upstream banks are indexed modulo
SEGMENT_BUDGET = 6 * 10**9  # sm_86: inputs + outputs + reference
# (model, layer, tokens): MoE expert GEMMs with column-major [E, N, K] weights.
SEGMENT_MODELS = (
    ("mixtral_8x7b", "expert_gate_up", 4096),
    ("qwen3_30b_a3b", "expert_gate_up", 4096),
    ("qwen3_30b_a3b", "expert_down", 8192),
    ("gpt_oss_20b", "expert_gate_up", 2048),
    ("llama4_scout", "expert_down", 8192),
)


def segment_tests() -> list[tuple[int, int, int, int]]:
    """test_segment_gemm (batch_size, num_rows_per_batch, d_in, d_out) that
    run (batch_size * num_rows_per_batch <= 8192)."""
    return [
        (b, r, d_in, d_out)
        for b in SEGMENT_TEST_BATCH
        for r in SEGMENT_TEST_ROWS
        for d_in in SEGMENT_TEST_DIMS
        for d_out in SEGMENT_TEST_DIMS
        if b * r <= 8192
    ]


def segment_test_id(batch: int, rows: int, d_in: int, d_out: int) -> str:
    return (
        f"{SEGMENT_TEST}[batch_size={batch}-num_rows_per_batch={rows}-d_in={d_in}"
        f"-d_out={d_out}-use_weight_indices=False-column_major=True"
        "-dtype=torch.float16-device=cuda:0-backend=sm80]"
    )


def segment_test_params(batch: int, rows: int, d_in: int, d_out: int) -> dict:
    params: dict[str, Any] = {"lengths": [rows] * batch, "n": d_out, "k": d_in}
    weight = d_out * d_in * 2
    if batch * weight > SEGMENT_WEIGHT_BYTES:
        params.update(weights=SEGMENT_WEIGHT_BYTES // weight, indices="modulo")
    return params


@cache
def _segment_cases(dtype: torch.dtype) -> tuple[CaseSpec, ...]:
    """Designed smoke cases (ragged, empty and single-row segments, partial
    128 x 128 x 32 tiles, long K loops, weight banks), the upstream test grid
    (FlashInfer tests fp16; the bf16 kernels get the same shapes) and MoE
    expert GEMMs of models as throughput cases. Every launch has 4
    persistent CTAs (upstream's threadblock_count), so each one loops over
    many tiles."""
    rng = random.Random(86)
    cases = [
        CaseSpec("four_segments", {"lengths": [1, 2, 3, 4], "n": 256, "k": 128}, 1),
        CaseSpec(
            "empty_and_ragged",
            {"lengths": [0, 37, 300, 0, 129], "n": 520, "k": 1032},
            2,
        ),
        CaseSpec(
            "eight_experts",
            {"lengths": [256, 1, 640, 128, 0, 77, 513, 431], "n": 2048, "k": 1024},
            3,
        ),
        CaseSpec(
            "weight_bank_permuted",
            {
                "lengths": skewed_lengths(2000, 64, 5, 1.2, 0),
                "n": 264,
                "k": 392,
                "weights": 96,
                "indices": "permuted",
            },
            5,
        ),
        CaseSpec(
            "long_k_shared_weights",
            {
                "lengths": [700, 3, 0, 1500, 1],
                "n": 384,
                "k": 4104,
                "weights": 2,
                "indices": "modulo",
            },
            6,
        ),
        CaseSpec(
            "many_skewed_segments",
            {"lengths": skewed_lengths(6000, 199, 7, 1.5, 0), "n": 136, "k": 1544},
            7,
        ),
    ]
    for batch, rows, d_in, d_out in segment_tests():
        label = f"fi_test_b{batch}_r{rows}_k{d_in}_n{d_out}"
        params = segment_test_params(batch, rows, d_in, d_out)
        seed = rng.randrange(1 << 30)
        if dtype == torch.float16:
            test = segment_test_id(batch, rows, d_in, d_out)
            cases.append(
                upstream_case(label, params, test, seed=seed, revision=FLASHINFER_V070)
            )
        else:
            cases.append(CaseSpec(label, params, seed))
    for model, layer, tokens in SEGMENT_MODELS:
        spec = MODELS[model]
        n, k = spec.linear_shapes()[layer]
        lengths = skewed_lengths(tokens * spec.top_k, spec.experts, 11, 0.6, 0)
        params = {"lengths": lengths, "n": n, "k": k, "tokens": tokens}
        cases.append(
            model_case(f"{model}_{layer}_t{tokens}", params, model, layer, seed=12)
        )
    for case in cases:
        if _SegmentGemm.case_bytes(case) > SEGMENT_BUDGET:
            raise AssertionError(f"segment case {case.name} exceeds the sm_86 budget")
    return tuple(cases)


def check_param_layouts() -> bool:
    """Byte-compare Params rebuilt in Python with the probe's reference
    structs (padding excluded) for every sm_86 variant."""
    for name, entry in variant_index().get("sm_86", {}).items():
        layout = segment_layout(entry["sidecar"])
        fixtures = json.loads(
            (
                PACKAGE_DIR / entry["sidecar"].replace(".json", ".fixtures.json")
            ).read_text()
        )
        mask = padding_mask(layout)
        for example in fixtures["examples"]:
            base = example["base"]
            built = pack_segment_params(
                layout,
                problem_sizes=base,
                problem_count=example["problem_count"],
                ptr_a=base + 0x1000,
                ptr_b=base + 0x2000,
                ptr_c=base + 0x3000,
                ptr_d=base + 0x3000,
                lda=base + 0x4000,
                ldb=base + 0x5000,
                ldc=base + 0x6000,
                ldd=base + 0x6000,
            )
            expected = bytes.fromhex(example["bytes"])
            if [b for b, m in zip(built, mask) if m] != [
                b for b, m in zip(expected, mask) if m
            ]:
                raise AssertionError(f"{name}: Params differ from the probe")
    return True


def _register() -> None:
    index = variant_index()
    for name, entry in index.get("sm_100a", {}).items():
        dispatch = tuple(
            Dispatch(
                d["num_groups"],
                d["sms"],
                tuple(tuple(r) for r in d["m_blocks"]),
                d["cubin"],
            )
            for d in entry["dispatch"]
        )
        register_variant(
            _DeepGemmFP8,
            name=name,
            supported_arches=("sm_100a",),
            layout=entry["layout"],
            shape_n=entry["n"],
            shape_k=entry["k"],
            block_n=entry["block_n"],
            num_stages=entry["num_stages"],
            num_last_stages=entry["num_last_stages"],
            swizzle_cd=entry["swizzle_cd"],
            smem=entry["smem"],
            dispatch=dispatch,
            num_groups=dispatch[0].num_groups if entry["layout"] == "masked" else 0,
        )
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    for name, entry in index.get("sm_86", {}).items():
        register_variant(
            _SegmentGemm,
            name=name,
            supported_arches=("sm_86",),
            dtype=dtypes[entry["dtype"]],
            sidecar=entry["sidecar"],
            mainloop=entry["mainloop"],
            upstream_dispatch=entry["upstream_dispatch"],
        )


_register()

__all__ = [
    "Config",
    "best_config",
    "check_param_layouts",
    "pack_scales",
    "pack_segment_params",
    "unpack_scales",
    "variant_index",
]

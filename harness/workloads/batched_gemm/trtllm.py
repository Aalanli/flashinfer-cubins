"""trtllm-gen batched (MoE) and dense GEMM kernels of FlashInfer v0.6.9 (sm_100a).

One workload per live kernel of ``cubins/batched_gemm`` and ``cubins/gemm``
(see ``impls/batched_gemm/compiler.py`` for the live/dead table), generated
from the variant index ``impls/batched_gemm/variants/sm_100a.json``. Two base
classes implement the contract from each variant's trtllm-gen options:

:class:`MoEGemm` -- ``TrtllmGenBatchedGemmRunner`` as FlashInfer's fused MoE
    runs it (``csrc/trtllm_fused_moe_runner.cu``): dynamic batching over the
    local experts with early exit, transposed MMA output (weights are the
    kernel's A, token rows its B), the CTA tables of the routing kernels.

    * FC1 (``PermuteGemm1``, routed): token rows are gathered from the
      unpermuted hidden states through ``permuted_idx_to_token_idx``
      (ldgsts or TMA gather). C rows are permuted token slots. With
      ``acc' = s_token * acc + bias`` (routing scales on the input and the
      FP32 bias in accumulator units, logical rows; KernelParamsDecl.h),
      fused gated configs write ``scale_c * (x0 + beta) * g * act(alpha *
      g)`` with ``g = scale_gate * x1`` (``x0``: the first half of the
      logical rows, ``x1``: the second; SwiGlu ``act = sigmoid`` with the
      clamp ``g <= limit``, ``|x0| <= limit`` first; GeGlu ``act = phi`` in
      the tanh form every GeGlu cubin evaluates, see :func:`gelu_tanh_phi`),
      eltwise ReLU2 configs ``scale_c * relu(scale_gate * acc') ** 2``,
      DeepSeek FP8 configs the raw ``acc`` (their activation kernel runs
      separately).
    * FC2 (``Gemm2``, no route): the permuted activations are read in place
      through TMA; C = BF16 ``scale_c * (acc + bias)``.

    Optional buffers are null where FlashInfer's launchers pass null
    (``host.moe_buffers``; scales then count as 1, alpha as 1, beta as 0,
    the clamp as infinite). A case's ``bias`` / ``gated_act`` /
    ``routing_scales`` / ``zero`` params switch on what a launcher can pass
    for the config (``host.MOE_FEATURES``); cases with ``gated_act`` use unit
    output scales so that the clamp acts on the values the formula names.

:class:`DenseGemm` -- the dense runners: ``mm_fp4`` / ``mm_mxfp8`` (trtllm
    backend, autotuned ``TrtllmGenGemmRunner``), ``gemm_fp8_nt_groupwise``
    (DeepSeek FP8, ``select_kernel_fp8``) and ``trtllm_low_latency_gemm``.
    ``out[m, n] = alpha * sum_k a[m, k] * w[n, k]`` in BF16.

Inputs are in the exact kernel-native formats FlashInfer prepares (see
``formats``): weights with the gated-activation row interleave (fused gated
FC1), the epilogue row shuffle (``useShuffledMatrix``), BlockMajorK blocks,
scale factors in R128c4 (weights), linear (routed MoE activations), R128c4/
R8c4 (``mSfLayoutB``) or DeepSeek's float ``[K / 128, rows]``. Values are
generated as exactly representable codes; references decode the inputs
(undoing every layout) and compute in FP32 from exact operands.

Cases (``impls/batched_gemm/compiler.py``: case policy) are problems the
upstream runners accept for this kernel (probe-verified at compile time:
the MoE runners' ``isValidConfigIndex``, ``getValidTactics`` or the tactic
``-1`` heuristic); each case's ``source`` records whether FlashInfer's tile
selection offers the kernel's tile for it (``dispatch``), its regime
(``host.moe_regime``) and where its shape comes from (an exact upstream test
problem, a model shape, or a stress shape). ``KernelParams``, grid,
cluster, launch attributes and every ``cuTensorMapEncodeTiled`` call are
rebuilt by ``host`` and compared byte for byte with the probes' recordings
of every case by :func:`check_param_layouts`.

Validation (``validate``) compares the dequantized kernel output with the
reference's pre-quantization FP32 values on every defined element (rows of
routed token slots; padding slots of an expert's last CTA are undefined).
The MoE reference also carries those values ideally quantized into the
kernel's output buffers, so ``validate(ref, ref)`` exercises the same
dequantization and checks as a kernel output:

* BF16, E4m3 (per-tensor) and DeepSeek E4m3 outputs: per element within
  one quantization step (BF16: ``2**-6`` relative; E4m3: ``2**-3`` relative
  plus a subnormal step; DeepSeek E4m3 with its float block scale ``amax /
  448``), and in aggregate a relative RMS error at most 1.5x that of an
  ideal quantizer of the format (plus ``1e-3``), so a systematic error
  (wrong expert, row, scale or activation half) fails;
* block-scaled outputs (NVFP4: E2m1 codes, E4m3 scale per 16; MxFP8: E4m3
  codes, UE8m0 scale per 32) are judged against the kernel's own scales
  (:func:`check_block_scaled`): every block's scale must be a legitimate
  choice for the block's reference amax -- within one E4m3 ulp of
  ``e4m3(amax / 6)`` for NVFP4 (one ulp: the kernel's FP32 amax may cross a
  rounding boundary), and within one binade of ``2**ceil(log2(amax / 448))``
  for MxFP8 (the split-K kernels use the OCP MX ``2**(floor(log2 amax) -
  8)``, which saturates values above ``448 * s``; others the ceiling) --
  every element within one code of the correctly rounded, saturating
  quantization of the reference value at that scale (one code: a value
  within FP32 summation-order noise of a rounding midpoint may round either
  way), and in aggregate a relative RMS error at most 1.5x (plus ``1e-3``)
  that of correct rounding at the kernel's scales. On the B200 the kernels'
  codes equal that correct rounding exactly; the per-element bound is
  needed where E4m3 scales are subnormal (``amax / 6 < 2**-6``: a block
  whose scale rounds to 0 or to ``2**-9`` legitimately flushes or clips).
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Any, ClassVar, NamedTuple

import torch

from ... import cuda_driver
from ...cutlass_host import driver_encode
from ...registry import register_variant
from ...workload import IMPLS, CaseSpec, Workload
from . import formats as fmt
from . import host

PACKAGE_DIR = IMPLS / "batched_gemm"
VARIANT_FILE = PACKAGE_DIR / "variants" / "sm_100a.json"
LAYOUT_FILE = PACKAGE_DIR / "variants" / "sm_100a_layouts.json"
FIXTURE_FILE = PACKAGE_DIR / "fixtures" / "sm_100a.jsonl.gz"

D = host.DTYPES
# Fake device addresses of the probes (bmm_probe.cu / gemm_probe.cu kBuffers).
BMM_PROBE_BUFFERS = (
    "a", "sf_a", "b", "sf_b", "c", "sf_c", "scale_c", "scale_gate", "bias",
    "alpha", "beta", "clamp_limit", "route_map", "total_num_padded_tokens",
    "cta_idx_xy_to_batch_idx", "cta_idx_xy_to_mn_limit", "num_non_exiting_ctas",
    "per_token_sf_a", "per_token_sf_b", "workspace",
)  # fmt: skip
GEMM_PROBE_BUFFERS = ("a", "sf_a", "b", "sf_b", "c", "sf_c", "scale_c", "workspace")
DYNAMIC_TABLES = (
    "total_num_padded_tokens",
    "cta_idx_xy_to_batch_idx",
    "cta_idx_xy_to_mn_limit",
    "num_non_exiting_ctas",
)


def fake_pointers(names: Sequence[str]) -> dict[str, int]:
    return {n: 0x7F0000000000 + ((i + 1) << 30) for i, n in enumerate(names)}


@cache
def variant_index() -> dict[str, Any]:
    if not VARIANT_FILE.is_file():
        return {}
    return json.loads(VARIANT_FILE.read_text())


@cache
def layouts() -> dict[str, Any]:
    return json.loads(LAYOUT_FILE.read_text())


# --- element formats ----------------------------------------------------------------


class Format(NamedTuple):
    """How a trtllm-gen dtype is stored and scaled."""

    name: str  # bf16 | fp16 | e4m3 | e2m1 | mxe2m1 | mxe4m3 | mxint4
    block: int  # scale-factor block (0: none)
    packed: bool  # two elements per byte


FORMATS = {
    D["Bfloat16"]: Format("bf16", 0, False),
    D["Fp16"]: Format("fp16", 0, False),
    D["E4m3"]: Format("e4m3", 0, False),
    D["E2m1"]: Format("e2m1", 16, True),
    D["MxE2m1"]: Format("mxe2m1", 32, True),
    D["MxE4m3"]: Format("mxe4m3", 32, False),
    D["MxInt4"]: Format("mxint4", 32, True),
}
CODE_RMS = {"e2m1": 2.926, "mxe2m1": 2.926, "mxint4": 4.61, "mxe4m3": 16.0}


def _pow2(x: float) -> float:
    return 2.0 ** round(math.log2(max(x, 2.0**-60)))


def _sf_values(shape, g, device, kind: str, target: float) -> torch.Tensor:
    """Exactly representable scale factors near ``target`` (x2 jitter):
    ``2**e * (1 + m / 8)`` for E4m3/BF16 scales, ``2**e`` for UE8m0."""
    e0 = round(math.log2(target))
    e = torch.randint(e0 - 1, e0 + 1, shape, generator=g, device=device).float()
    if kind == "ue8m0":
        return torch.exp2(e)
    m = torch.randint(0, 8, shape, generator=g, device=device).float()
    return torch.exp2(e) * (1 + m / 8)


def _encode_sf(values: torch.Tensor, name: str) -> torch.Tensor:
    if name == "e2m1":
        return values.to(torch.float8_e4m3fn)
    if name == "mxint4":
        return values.to(torch.bfloat16)
    return (torch.log2(values).round() + 127).to(torch.uint8)


def _decode_sf(storage: torch.Tensor, name: str) -> torch.Tensor:
    if name == "e2m1":
        return fmt.e4m3_decode(storage)
    if name == "mxint4":
        return storage.view(torch.bfloat16).float()
    return fmt.ue8m0_decode(storage)


class Operand(NamedTuple):
    data: torch.Tensor  # storage (packed bytes, fp8, bf16), logical row order
    sf: torch.Tensor | None  # scale factors [..., K / block] (values' dtype)
    values: torch.Tensor  # exact float32 values [..., K]


def make_operand(
    dtype: int, shape: Sequence[int], g: torch.Generator, device, target_rms: float
) -> Operand:
    """Seeded exact operand of ``dtype`` with RMS about ``target_rms``."""
    f = FORMATS[dtype]
    *rows, k = shape
    if f.name in ("bf16", "fp16"):
        t = torch.bfloat16 if f.name == "bf16" else torch.float16
        data = (torch.randn(shape, generator=g, device=device) * target_rms).to(t)
        return Operand(data, None, data.float())
    if f.name == "e4m3":
        data = fmt.e4m3_encode(
            torch.randn(shape, generator=g, device=device) * target_rms
        )
        return Operand(data, None, data.float())
    if f.name in ("e2m1", "mxe2m1", "mxint4"):
        codes = torch.randint(
            0, 16, shape, generator=g, device=device, dtype=torch.uint8
        )
        elems = fmt.int4_decode(codes) if f.name == "mxint4" else fmt.e2m1_decode(codes)
        data = fmt.pack_nibbles(codes)
    else:  # mxe4m3
        data = fmt.e4m3_encode(torch.randn(shape, generator=g, device=device) * 16)
        elems = data.float()
    kind = "ue8m0" if f.name.startswith("mxe") else "e4m3"
    sf_vals = _sf_values(
        (*rows, k // f.block), g, device, kind, target_rms / CODE_RMS[f.name]
    )
    sf = _encode_sf(sf_vals, f.name)
    values = elems * _decode_sf(sf, f.name).repeat_interleave(f.block, -1)
    return Operand(data, sf, values)


def decode_operand(dtype: int, data: torch.Tensor, sf: torch.Tensor | None, k: int):
    """Float32 values of a logical-order operand ``[..., K]``."""
    f = FORMATS[dtype]
    if f.name in ("bf16", "fp16", "e4m3"):
        return data.float()
    if f.packed:
        codes = fmt.unpack_nibbles(data)
        elems = fmt.int4_decode(codes) if f.name == "mxint4" else fmt.e2m1_decode(codes)
    else:
        elems = data.float()
    assert sf is not None
    return elems * _decode_sf(sf, f.name).repeat_interleave(f.block, -1)[..., :k]


# --- weights: kernel-native preparation -----------------------------------------------


def weight_rows(o: Mapping[str, Any], rows: int) -> torch.Tensor:
    """Physical row i holds logical row idx[i]: FlashInfer's gated
    interleave (fused gated FC1) composed with the epilogue shuffle."""
    idx = torch.arange(rows)
    if o.get("mFusedAct") and o.get("mRouteImpl", 0) != host.ROUTE_NONE:
        idx = fmt.gated_rows(rows)
    if o["mUseShuffledMatrix"]:
        idx = idx[fmt.shuffle_rows(rows, o["mEpilogueTileM"])]
    return idx


def storage_block(dtype: int, block_k: int) -> int:
    return block_k // 2 if FORMATS[dtype].packed else block_k


def prepare_weights(
    o: Mapping[str, Any], dtype: int, data: torch.Tensor, sf: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Logical [B, M, K] weights -> kernel-native (data, flat SF in
    ``mSfLayoutA``). Batches are independent: with M a multiple of 128 every
    R128c4 block lies inside one batch, so the flat SF buffer is the
    concatenation of the per-batch buffers."""
    batches, rows = data.shape[0], data.shape[1]
    idx = weight_rows(o, rows).to(data.device)
    data = data[:, idx].contiguous()
    if o["mLayoutA"] == host.LAYOUT_BLOCK_MAJOR_K:
        data = fmt.to_block_major_k(data, storage_block(dtype, o["mBlockK"]))
    if sf is not None:
        sf = sf[:, idx].reshape(batches * rows, -1)
        sf = fmt.sf_to_layout(sf, o["mSfLayoutA"])
    return data, sf


def restore_weights(
    o: Mapping[str, Any],
    dtype: int,
    data: torch.Tensor,
    sf: torch.Tensor | None,
    batches: int,
    rows: int,
    k: int,
) -> torch.Tensor:
    """Kernel-native weights -> logical float32 [B, M, K]."""
    if o["mLayoutA"] == host.LAYOUT_BLOCK_MAJOR_K:
        data = fmt.from_block_major_k(data)
    data = data.reshape(batches, rows, -1)
    sf_rows = None
    f = FORMATS[dtype]
    if sf is not None and f.block:
        sf_rows = fmt.sf_from_layout(sf, batches * rows, k // f.block, o["mSfLayoutA"])
        sf_rows = sf_rows.reshape(batches, rows, -1)
    phys = decode_operand(dtype, data, sf_rows, k)
    idx = weight_rows(o, rows).to(phys.device)
    logical = torch.empty_like(phys)
    logical[:, idx] = phys
    return logical


def expert_slice(
    flat: torch.Tensor | None, expert: int, rows: int, cols: int, layout: int
) -> torch.Tensor | None:
    """One expert's part of a flat per-expert-concatenated SF buffer."""
    if flat is None:
        return None
    size = fmt.sf_storage_size(rows, cols, layout)
    return flat[expert * size : (expert + 1) * size]


# --- DeepSeek FP8 block scales -----------------------------------------------------------

DS_BLOCK = 128


def ds_operand(
    shape, g, device, target_rms: float, scale_shape
) -> tuple[torch.Tensor, torch.Tensor]:
    """E4m3 values with power-of-two float scales (exact products)."""
    data = fmt.e4m3_encode(torch.randn(shape, generator=g, device=device) * 16)
    e0 = round(math.log2(target_rms / 16))
    e = torch.randint(e0 - 1, e0 + 1, scale_shape, generator=g, device=device)
    return data, torch.exp2(e.float())


def ds_scale_view(flat: torch.Tensor, cols: int, total: int) -> torch.Tensor:
    """The [cols / 128, total] DeepSeek scale layout inside a flat buffer
    (row stride = the runtime total padded tokens)."""
    return flat[: cols // DS_BLOCK * total].view(cols // DS_BLOCK, total)


# --- MoE routing -------------------------------------------------------------------------


class Routing(NamedTuple):
    expanded_idx_to_permuted_idx: torch.Tensor  # int32 [T * top_k]
    permuted_idx_to_token_idx: torch.Tensor  # int32 [rows]; padding 0
    total_num_padded_tokens: torch.Tensor  # int32 [1]
    cta_idx_xy_to_batch_idx: torch.Tensor  # int32 [max CTAs]; unused 0
    cta_idx_xy_to_mn_limit: torch.Tensor  # int32 [max CTAs]; unused 0
    num_non_exiting_ctas: torch.Tensor  # int32 [1]


def expert_weights(experts: int, skew: float = 0.0, empty: int = 0, seed: int = 0):
    """Routing preference per expert: the 4-expert smoke cases use a fixed
    skew (expert 0 gets several CTAs, the last expert none); otherwise seeded
    log-normal weights of spread ``skew`` (0: uniform, the balanced routing
    FlashInfer's tile heuristic assumes), the last ``empty`` experts unused."""
    if experts == 4 and not skew and not empty:
        return torch.tensor([8.0, 4.0, 2.0, 0.0])
    g = torch.Generator().manual_seed(seed + 7919)
    w = (
        torch.exp(torch.randn(experts, generator=g) * skew)
        if skew
        else torch.ones(experts)
    )
    if empty:
        w[experts - empty :] = 0
    return w


def make_routing(
    tokens: int,
    top_k: int,
    experts: int,
    tile: int,
    seed: int,
    skew: float = 0.0,
    empty: int = 0,
) -> Routing:
    """Top-k routing and the tables FlashInfer's routing kernels write
    (``RoutingKernel.cuh``): per expert, its tokens in expanded-index order
    from ``cta_offset * tile``; CTA tables with mnLimit = min((cta + 1) *
    tile, start * tile + count)."""
    g = torch.Generator().manual_seed(seed)
    weights = expert_weights(experts, skew, empty, seed)
    if int((weights > 0).sum()) < top_k:
        raise ValueError("fewer usable experts than top_k")
    choice = torch.multinomial(
        weights.expand(tokens, experts), top_k, replacement=False, generator=g
    )
    flat = choice.flatten()
    counts = torch.bincount(flat, minlength=experts)
    num_cta = (counts + tile - 1) // tile
    cta_offset = torch.cumsum(num_cta, 0) - num_cta
    max_ctas = host.max_num_ctas_in_batch_dim(tokens, top_k, experts, tile)
    rows = max_ctas * tile
    order = torch.argsort(flat * (tokens * top_k) + torch.arange(flat.numel()))
    expanded = torch.empty(tokens * top_k, dtype=torch.int32)
    to_token = torch.zeros(rows, dtype=torch.int32)
    start = torch.cumsum(counts, 0) - counts
    sorted_experts = flat[order]
    rank = torch.arange(flat.numel()) - start[sorted_experts]
    permuted = cta_offset[sorted_experts] * tile + rank
    expanded[order] = permuted.to(torch.int32)
    to_token[permuted] = (order // top_k).to(torch.int32)
    used = num_cta > 0
    batch = torch.repeat_interleave(torch.arange(experts)[used], num_cta[used])
    cta = torch.arange(batch.numel())
    limit = torch.minimum((cta + 1) * tile, cta_offset[batch] * tile + counts[batch])
    total = int(num_cta.sum())
    if total > max_ctas:
        raise AssertionError("routing exceeds getMaxNumCtasInBatchDim")
    batch_t = torch.zeros(max_ctas, dtype=torch.int32)
    limit_t = torch.zeros(max_ctas, dtype=torch.int32)
    batch_t[:total] = batch.to(torch.int32)
    limit_t[:total] = limit.to(torch.int32)
    return Routing(
        expanded,
        to_token,
        torch.tensor([total * tile], dtype=torch.int32),
        batch_t,
        limit_t,
        torch.tensor([total], dtype=torch.int32),
    )


# --- output formats, references and validation -------------------------------------------


def output_kind(o: Mapping[str, Any]) -> str:
    c = o["mDtypeC"]
    if c == D["Bfloat16"]:
        return "bf16"
    if c == D["E4m3"]:
        return "ds_e4m3" if o["mUseDeepSeekFp8"] else "e4m3"
    if c == D["E2m1"]:
        return "nvfp4"
    if c == D["MxE4m3"]:
        return "mxfp8"
    raise ValueError(f"unsupported output dtype {c:#x}")


def _block_amax(x: torch.Tensor, block: int) -> torch.Tensor:
    rows, cols = x.shape
    return (
        x.abs().view(rows, cols // block, block).amax(-1).repeat_interleave(block, -1)
    )


def quantize_like(x: torch.Tensor, kind: str) -> torch.Tensor:
    """An ideal quantizer of each output format (for the aggregate bound)."""
    if kind == "bf16":
        return x.to(torch.bfloat16).float()
    if kind == "e4m3":
        return fmt.e4m3_encode(x).float()
    if kind == "ds_e4m3":
        s = _block_amax(x, DS_BLOCK) / fmt.E4M3_MAX
        s = torch.where(s > 0, s, torch.ones_like(s))
        return fmt.e4m3_encode(x / s).float() * s
    if kind == "nvfp4":
        s = fmt.e4m3_encode(_block_amax(x, 16) / fmt.E2M1_MAX).float()
        safe = torch.where(s > 0, s, torch.ones_like(s))
        return fmt.e2m1_decode(fmt.e2m1_encode(x / safe)) * s
    if kind == "mxfp8":
        amax = _block_amax(x, 32)
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=2.0**-100) / fmt.E4M3_MAX)))
        return fmt.e4m3_encode(x / s).float() * s
    raise ValueError(kind)


def element_tolerance(expected: torch.Tensor, kind: str) -> torch.Tensor:
    mag = expected.abs()
    if kind == "bf16":
        scale = expected.pow(2).mean().sqrt() if expected.numel() else torch.tensor(0.0)
        return 2.0**-6 * mag + 2.0**-10 * scale + 1e-6
    if kind == "e4m3":
        return 2.0**-3 * mag + 2.0**-8
    if kind == "ds_e4m3":
        return 2.0**-3 * mag + 2.0**-8 * _block_amax(expected, DS_BLOCK) / fmt.E4M3_MAX
    if kind == "nvfp4":
        return 1.25 * _block_amax(expected, 16) / fmt.E2M1_MAX + 1e-6
    if kind == "mxfp8":
        smax = 2 * _block_amax(expected, 32) / fmt.E4M3_MAX
        return 2.0**-3 * mag + 2.0**-8 * smax + 1e-6
    raise ValueError(kind)


def check_output(
    expected: torch.Tensor, actual: torch.Tensor, kind: str, what: str
) -> None:
    """``expected``: reference FP32 values, NaN rows undefined; ``actual``:
    dequantized kernel output of the same shape."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    rows = ~torch.isnan(expected).any(-1)
    exp, act = expected[rows].float(), actual[rows].float()
    if not torch.isfinite(act).all():
        raise AssertionError(f"{what}: non-finite values in defined rows")
    err = (act - exp).abs()
    tol = element_tolerance(exp, kind)
    bad = err > tol
    if bad.any():
        i = int(bad.flatten().nonzero()[0])
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {err.numel()} elements outside the "
            f"{kind} tolerance; first at {divmod(i, exp.shape[-1])}: "
            f"{act.flatten()[i].item()} vs {exp.flatten()[i].item()}"
        )
    norm = exp.pow(2).mean().sqrt()
    if norm > 0:
        ideal = (quantize_like(exp, kind) - exp).pow(2).mean().sqrt() / norm
        rel = err.pow(2).mean().sqrt() / norm
        if rel > 1.5 * ideal + 1e-3:
            raise AssertionError(
                f"{what}: relative RMS error {rel.item():.4g} exceeds 1.5x the "
                f"ideal {kind} quantizer's {ideal.item():.4g}"
            )


BLOCK_SCALED = {"nvfp4": (16, "e2m1"), "mxfp8": (32, "e4m3")}


def ideal_block_scale(amax: torch.Tensor, kind: str) -> torch.Tensor:
    """NVFP4: ``e4m3(amax / 6)``; MxFP8: the non-saturating UE8m0
    ``2**ceil(log2(amax / 448))``."""
    if kind == "nvfp4":
        return fmt.e4m3_encode(amax / fmt.E2M1_MAX).float()
    return torch.exp2(torch.ceil(torch.log2(amax.clamp(min=2.0**-100) / fmt.E4M3_MAX)))


def round_at_scale(x: torch.Tensor, scale: torch.Tensor, elem: str) -> torch.Tensor:
    """Correct (round-to-nearest, saturating) quantization of ``x`` with
    per-element ``scale``; a zero scale flushes to zero."""
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    if elem == "e2m1":
        q = fmt.e2m1_decode(fmt.e2m1_encode(x / safe))
    else:
        q = fmt.e4m3_encode(x / safe).float()
    return torch.where(scale > 0, q * scale, torch.zeros_like(x))


def check_block_scaled(
    expected: torch.Tensor,
    actual: torch.Tensor,
    scales: torch.Tensor,
    kind: str,
    what: str,
) -> None:
    """Block-scaled output check (module docstring). ``expected``: reference
    FP32 values, NaN rows undefined; ``actual``: the kernel's dequantized
    values; ``scales``: its decoded block scales ``[rows, cols / block]``."""
    block, elem = BLOCK_SCALED[kind]
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    rows = ~torch.isnan(expected).any(-1)
    e, a, s = expected[rows].float(), actual[rows].float(), scales[rows].float()
    if not (torch.isfinite(a).all() and torch.isfinite(s).all()):
        raise AssertionError(f"{what}: non-finite values in defined rows")
    n, cols = e.shape
    amax = e.abs().view(n, cols // block, block).amax(-1)
    ideal = ideal_block_scale(amax, kind)
    if kind == "nvfp4":
        scale_ok = (s - ideal).abs() <= fmt.spacing(ideal, "e4m3")
    else:
        ratio = s / ideal
        scale_ok = (amax == 0) | ((ratio >= 0.5) & (ratio <= 2.0))
    if not scale_ok.all():
        i = int((~scale_ok).flatten().nonzero()[0])
        r, b = divmod(i, cols // block)
        raise AssertionError(
            f"{what}: {int((~scale_ok).sum())} of {scale_ok.numel()} {kind} block "
            f"scales are no legitimate choice; first (row {r}, block {b}): "
            f"{s[r, b].item()} for amax {amax[r, b].item()} (ideal {ideal[r, b].item()})"
        )
    sv = s.repeat_interleave(block, -1)
    q = round_at_scale(e, sv, elem)
    step = fmt.spacing(q / torch.where(sv > 0, sv, torch.ones_like(sv)), elem) * sv
    bad = (a - q).abs() > step
    if bad.any():
        i = int(bad.flatten().nonzero()[0])
        r, c = divmod(i, cols)
        raise AssertionError(
            f"{what}: {int(bad.sum())} of {a.numel()} {kind} elements differ by more "
            f"than one code from correct rounding at the kernel's scale; first at "
            f"({r}, {c}): {a[r, c].item()} vs {q[r, c].item()} (reference "
            f"{e[r, c].item()}, scale {sv[r, c].item()})"
        )
    norm = e.pow(2).mean().sqrt()
    if norm > 0:
        ideal_rms = (q - e).pow(2).mean().sqrt() / norm
        rel = (a - e).pow(2).mean().sqrt() / norm
        if rel > 1.5 * ideal_rms + 1e-3:
            raise AssertionError(
                f"{what}: relative RMS error {rel.item():.4g} exceeds 1.5x that of "
                f"correct rounding at the kernel's {kind} scales ({ideal_rms.item():.4g})"
            )


def gelu_tanh_phi(x: torch.Tensor) -> torch.Tensor:
    """The kernels' GeGlu phi: the tanh form ``0.5 * (1 + tanh(sqrt(2 / pi) *
    (x + 0.044715 x^3)))`` (every GeGlu cubin computes MUFU.TANH of
    0.7978845608 * (x + 0.044715 x^3); none evaluates erf)."""
    return 0.5 * (
        1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3)))
    )


def gated(
    act_type: int,
    x0: torch.Tensor,
    gate: torch.Tensor,
    alpha: torch.Tensor | float = 1.0,
    beta: torch.Tensor | float = 0.0,
    limit: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(x0 + beta) * gate * act(alpha * gate)`` (KernelParamsDecl.h:
    ``out_glu = x_glu * sigmoid(alpha * x_glu) * (x_linear + beta)``, GeGlu
    with phi), the SwiGlu clamp first: ``x_glu.clamp(max=limit)``,
    ``x_linear.clamp(-limit, limit)``."""
    if limit is not None:
        gate = torch.minimum(gate, limit)
        x0 = torch.maximum(torch.minimum(x0, limit), -limit)
    if act_type == host.ACT_SWIGLU:
        act = torch.sigmoid(alpha * gate)
    elif act_type == host.ACT_GEGLU:
        act = gelu_tanh_phi(alpha * gate)
    else:
        raise ValueError(f"unsupported gated activation {act_type}")
    return (x0 + beta) * gate * act


# --- workloads ---------------------------------------------------------------------------


MOE_DIMS = ("tokens", "top_k", "experts", "hidden", "intermediate")
MOE_INPUTS = (
    "b",
    "sf_b",
    "a",
    "sf_a",
    "scale_c",
    "scale_gate",
    "bias",
    "alpha",
    "beta",
    "clamp_limit",
    "per_token_sf_b",
    "route_map",
    *DYNAMIC_TABLES,
    "expanded_idx_to_permuted_idx",
    "dims",
)


class _TrtllmGen(Workload):
    """Common launch plumbing of the trtllm-gen variants (class attributes
    from the variant index: ``entry``)."""

    package = "batched_gemm"
    entry: ClassVar[dict[str, Any]]
    pdl_fields: ClassVar[tuple[str, ...]]
    layout_key: ClassVar[str]

    @property
    def options(self) -> dict[str, Any]:
        return self.entry["options"]

    @property
    def layout(self) -> dict[str, Any]:
        return layouts()[self.layout_key]

    def get_cases(self) -> list[CaseSpec]:
        return [
            CaseSpec(
                c["name"], dict(c["params"]), c["seed"], c["suite"], c.get("source", {})
            )
            for c in self.entry["cases"]
        ]

    def configure(self, function: cuda_driver.Function) -> None:
        # trtllm::gen::launchKernel: opt in to more than 48 KiB of smem.
        smem = self.entry["shared_mem"]
        if smem > 48 << 10:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )
        o = self.options
        if o["mClusterDimX"] * o["mClusterDimY"] * o["mClusterDimZ"] > 8:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1
            )

    def spec(self, grid: tuple[int, ...], params: bytes) -> host.LaunchSpec:
        return host.launch_spec(
            {**self.entry, "options": self.options},
            grid,
            params[: self.entry["param_size"]],
            pdl_safe_fields=self.pdl_fields,
        )

    def _launch(self, spec: host.LaunchSpec) -> None:
        self.launch(
            spec.grid,
            spec.block,
            [spec.params],
            shared_mem=spec.shared_mem,
            cluster=spec.cluster,
            programmatic_serialization=spec.pdl,
            cluster_scheduling_policy=spec.scheduling_policy,
        )

    def _sm_count(self) -> int:
        return torch.cuda.get_device_properties(self.device).multi_processor_count

    def _encoder(self) -> host.TensorMapEncoder:
        cuda_driver.ensure_context(self.device)
        return driver_encode

    def _build(self, inputs: tuple) -> tuple[host.LaunchSpec, tuple, tuple]:
        raise NotImplementedError

    def run(self, inputs: tuple) -> tuple:
        spec, outputs, _ = self._build(inputs)
        self._launch(spec)
        return outputs

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple | None]:
        spec, outputs, keep = self._build(inputs)

        def launch() -> tuple:
            self._launch(spec)
            assert keep is not None  # buffers referenced by the params live here
            return outputs

        return launch, outputs

    def _check_cuda(self, tensors: Sequence[Any]) -> None:
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.device.type != "cuda":
                raise ValueError("native launch needs CUDA tensors")


class MoEGemm(_TrtllmGen):
    """One MoE FC1/FC2 trtllm-gen batched GEMM (see module docstring).

    Inputs (``MOE_INPUTS`` order): FC1 ``b`` = hidden states ``[T, K]``
    (linear SFs ``[T, K / block]``, DeepSeek ``[K / 128, T]``), FC2 ``b`` =
    permuted activations ``[rows, K]`` (SFs in ``mSfLayoutB``, DeepSeek
    ``[K / 128, total]`` in a ``[K / 128 * rows]`` buffer); ``a`` = weights
    (kernel-native) and their SFs (``mSfLayoutA``, per expert; DeepSeek float
    ``[E, M / 128, K / 128]``); per-expert float32 ``scale_c``,
    ``scale_gate``, ``alpha``, ``beta``, ``clamp_limit``, float32 ``bias [E,
    M]`` (logical rows), BF16 routing scales ``per_token_sf_b [T]``; the
    routing tables; ``expanded_idx_to_permuted_idx`` for the reference only;
    ``dims`` a CPU int64 tensor (tokens, top_k, experts, hidden,
    intermediate). Buffers FlashInfer would not pass are None. Outputs:
    ``(c,)`` or ``(c, sf_c)`` with ``rows = max CTAs * tileN`` C rows.

    The reference is ``(*encoded, expected, total_num_padded_tokens)``:
    ``expected`` the FP32 values (NaN rows undefined) and ``encoded`` them
    quantized by an ideal quantizer into the kernel's output buffers, so that
    ``validate(ref, ref)`` runs the same dequantization and checks as a
    kernel output.
    """

    pdl_fields = host.BMM_PDL_FIELDS
    layout_key = "bmm"

    # -- shapes ----------------------------------------------------------------

    def problem(self, params: Mapping[str, Any]) -> host.BmmProblem:
        return host.moe_problem(self.options, params)

    @property
    def fc1(self) -> bool:
        return host.moe_role(self.options) == "fc1"

    def _out_cols(self, p: host.BmmProblem) -> int:
        return p.m // 2 if self.options["mFusedAct"] else p.m

    @property
    def outputs_count(self) -> int:
        return 2 if output_kind(self.options) in ("nvfp4", "mxfp8", "ds_e4m3") else 1

    # -- inputs ----------------------------------------------------------------

    def _weights(
        self, p: host.BmmProblem, g: torch.Generator, w_rms: float
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        """Kernel-native weights generated one expert at a time into
        preallocated buffers (bounded temporaries), their SFs and the mean
        square of the values."""
        o, dev = self.options, self.device
        a_dtype = o["mDtypeA"]
        ds = bool(o["mUseDeepSeekFp8"])
        data: torch.Tensor | None = None
        sfs: torch.Tensor | None = None
        ms = 0.0
        for e in range(p.num_batches):
            if ds:
                d, s = ds_operand(
                    (1, p.m, p.k), g, dev, w_rms, (1, p.m // DS_BLOCK, p.k // DS_BLOCK)
                )
                vals = d.float() * s.repeat_interleave(DS_BLOCK, 1).repeat_interleave(
                    DS_BLOCK, 2
                )
                store, _ = prepare_weights(o, a_dtype, d, None)
                sf: torch.Tensor | None = s[0]
            else:
                op = make_operand(a_dtype, (1, p.m, p.k), g, dev, w_rms)
                vals = op.values
                store, sf = prepare_weights(o, a_dtype, op.data, op.sf)
            ms += vals.pow(2).mean().item() / p.num_batches
            del vals
            if data is None:
                data = torch.empty((p.num_batches, *store.shape[1:]), dtype=store.dtype,
                                   device=dev)  # fmt: skip
            data[e] = store[0]
            if sf is not None:
                if sfs is None:
                    sfs = torch.empty(
                        (p.num_batches, *sf.shape), dtype=sf.dtype, device=dev
                    )
                sfs[e] = sf
        assert data is not None
        if sfs is not None and not ds:
            sfs = sfs.reshape(-1)  # per-expert flat buffers, concatenated
        return data, sfs, ms

    def get_inputs(self, case: CaseSpec) -> tuple:
        o, params = self.options, case.params
        p = self.problem(params)
        g = self.generator(case)
        dev = self.device
        tokens, experts, top_k = params["tokens"], params["experts"], params["top_k"]
        r = make_routing(
            tokens,
            top_k,
            experts,
            o["mTileN"],
            case.seed,
            params.get("skew", 0.0),
            params.get("empty", 0),
        )
        rows = p.max_num_ctas * o["mTileN"]
        buffers = set(host.moe_buffers(o, params))
        ds = bool(o["mUseDeepSeekFp8"])
        b_dtype = o["mDtypeB"]
        b_rows = tokens if self.fc1 else rows
        zero = bool(params.get("zero"))
        # Operands: activations ~1, weights ~1/sqrt(K) where nothing rescales
        # them (E4m3 per-tensor weights stay O(1); the output scales adapt).
        w_rms = 1.0 if (o["mDtypeA"] == D["E4m3"] and not ds) else p.k**-0.5
        sf_b: torch.Tensor | None = None
        if ds:
            b_data, b_scale = ds_operand(
                (b_rows, p.k), g, dev, 1.0, (p.k // DS_BLOCK, b_rows)
            )
            b_vals = b_data.float() * b_scale.T.repeat_interleave(DS_BLOCK, -1)
        else:
            b_op = make_operand(b_dtype, (b_rows, p.k), g, dev, 1.0)
            b_data, b_vals = b_op.data, b_op.values
        keep = torch.ones(b_rows, dtype=torch.bool, device=dev)
        if not self.fc1:
            # Permuted activations: routed slots hold data, padding is zero.
            keep = torch.zeros(rows, dtype=torch.bool, device=dev)
            keep[r.expanded_idx_to_permuted_idx.long().to(dev)] = True
        if zero:
            keep = torch.zeros_like(keep)
        if not bool(keep.all()):
            b_vals = b_vals * keep[:, None]
            if FORMATS.get(b_dtype, FORMATS[D["E4m3"]]).packed and not ds:
                b_data = b_data * keep[:, None].to(b_data.dtype)
            elif b_data.dtype == torch.float8_e4m3fn:
                b_data = fmt.e4m3_encode(b_data.float() * keep[:, None])
            else:
                b_data = (b_data.float() * keep[:, None]).to(b_data.dtype)
        x_ms = b_vals.pow(2).mean().item() if not zero else 1.0
        del b_vals
        if ds:
            if self.fc1:
                sf_b = b_scale.contiguous()  # [K / 128, T]
            else:
                total = int(r.total_num_padded_tokens[0])
                flat = torch.zeros(p.k // DS_BLOCK * rows, device=dev)
                ds_scale_view(flat, p.k, total).copy_(b_scale[:, :total])
                sf_b = flat
        elif b_op.sf is not None:
            sf_b = (
                b_op.sf.contiguous()  # linear [T, K / block]
                if self.fc1
                else fmt.sf_to_layout(b_op.sf, o["mSfLayoutB"])
            )
        a_store, sf_a, w_ms = self._weights(p, g, w_rms)
        acc_rms = math.sqrt(p.k * x_ms * w_ms)
        # Output scales (per local expert), powers of two chosen from the
        # operands so that outputs are O(1) (the 2x per-expert step makes a
        # wrong expert index visible). With gated_act, scale 1: the clamp then
        # acts on the values KernelParamsDecl.h's formula names.
        step = torch.tensor([2.0 ** (e % 2) for e in range(experts)], device=dev)
        unit = params.get("gated_act") and host.moe_supports(o, "gated_act")
        inputs: dict[str, Any] = {name: None for name in MOE_INPUTS}
        if "scale_gate" in buffers:
            inputs["scale_gate"] = (
                torch.ones(experts, device=dev) if unit else _pow2(1.0 / acc_rms) * step
            )
        if "scale_c" in buffers:
            if unit:
                inputs["scale_c"] = torch.ones(experts, device=dev)
            elif self.fc1 and o["mFusedAct"]:
                inputs["scale_c"] = _pow2(2.0 / acc_rms) / step
            elif self.fc1:
                inputs["scale_c"] = torch.full((experts,), 2.0, device=dev) / step
            else:
                inputs["scale_c"] = _pow2(1.0 / acc_rms) * step
        if "bias" in buffers:  # raw accumulator units, logical rows
            inputs["bias"] = (
                torch.randn((experts, p.m), generator=g, device=dev) * 0.5 * acc_rms
            )
        if "alpha" in buffers:  # gpt-oss's 1.702, and 1
            inputs["alpha"] = torch.tensor(
                [(1.702, 1.0)[e % 2] for e in range(experts)], device=dev
            )
            inputs["beta"] = torch.tensor(
                [(1.0, -0.5)[e % 2] for e in range(experts)], device=dev
            )
        if "clamp_limit" in buffers:  # binds for a large share of the values
            inputs["clamp_limit"] = torch.tensor(
                [(0.75, 1.25)[e % 2] * acc_rms for e in range(experts)], device=dev
            )
        if "per_token_sf_b" in buffers:  # top-1 routing weights in (0, 1]
            u = torch.rand(tokens, generator=g, device=dev)
            inputs["per_token_sf_b"] = (0.25 + 0.75 * u).to(torch.bfloat16)
        inputs.update(
            b=b_data,
            sf_b=sf_b,
            a=a_store,
            sf_a=sf_a,
            route_map=r.permuted_idx_to_token_idx.to(dev) if self.fc1 else None,
            total_num_padded_tokens=r.total_num_padded_tokens.to(dev),
            cta_idx_xy_to_batch_idx=r.cta_idx_xy_to_batch_idx.to(dev),
            cta_idx_xy_to_mn_limit=r.cta_idx_xy_to_mn_limit.to(dev),
            num_non_exiting_ctas=r.num_non_exiting_ctas.to(dev),
            expanded_idx_to_permuted_idx=r.expanded_idx_to_permuted_idx.to(dev),
            dims=torch.tensor([params[k] for k in MOE_DIMS], dtype=torch.int64),
        )
        return tuple(inputs[name] for name in MOE_INPUTS)

    def _problem_of(self, inputs: tuple) -> host.BmmProblem:
        return self.problem(dict(zip(MOE_DIMS, inputs[-1].tolist())))

    # -- reference ---------------------------------------------------------------

    def _activations(self, i: Mapping[str, Any], p: host.BmmProblem, total: int):
        o = self.options
        b, sf_b = i["b"], i["sf_b"]
        if o["mUseDeepSeekFp8"]:
            if self.fc1:
                return b.float() * sf_b.T.repeat_interleave(DS_BLOCK, -1)
            x = torch.zeros(b.shape[0], p.k, device=b.device)
            s = ds_scale_view(sf_b, p.k, total).T
            x[:total] = b[:total].float() * s.repeat_interleave(DS_BLOCK, -1)
            return x
        f = FORMATS[o["mDtypeB"]]
        sf_rows = None
        if sf_b is not None and f.block:
            sf_rows = (
                sf_b
                if self.fc1
                else fmt.sf_from_layout(
                    sf_b, b.shape[0], p.k // f.block, o["mSfLayoutB"]
                )
            )
        return decode_operand(o["mDtypeB"], b, sf_rows, p.k)

    def expert_weights_f32(self, i: Mapping[str, Any], p: host.BmmProblem, e: int):
        """Logical float32 [M, K] weights of local expert ``e``."""
        o = self.options
        a = i["a"][e : e + 1]
        if o["mUseDeepSeekFp8"]:
            w = restore_weights(o, D["E4m3"], a, None, 1, p.m, p.k)[0]
            s = i["sf_a"][e]
            return w * s.repeat_interleave(DS_BLOCK, 0).repeat_interleave(DS_BLOCK, 1)
        f = FORMATS[o["mDtypeA"]]
        sf = None
        if f.block:
            sf = expert_slice(i["sf_a"], e, p.m, p.k // f.block, o["mSfLayoutA"])
        return restore_weights(o, o["mDtypeA"], a, sf, 1, p.m, p.k)[0]

    def get_reference(self, inputs: tuple) -> tuple:
        o = self.options
        i = dict(zip(MOE_INPUTS, inputs))
        p = self._problem_of(inputs)
        tile = o["mTileN"]
        rows = p.max_num_ctas * tile
        total = int(i["total_num_padded_tokens"][0])
        top_k = int(i["dims"][1])
        dev = i["b"].device
        x = self._activations(i, p, total)
        expanded = i["expanded_idx_to_permuted_idx"].long()
        slots = expanded[expanded >= 0]
        token_of = (expanded >= 0).nonzero().flatten() // top_k
        row_expert = i["cta_idx_xy_to_batch_idx"].long()[slots // tile]
        expected = torch.full((rows, self._out_cols(p)), torch.nan, device=dev)

        def per_expert(name: str, e: int) -> Any:
            t = i[name]
            return t[e] if t is not None else None

        for e in row_expert.unique().tolist():
            sel = row_expert == e
            s_slots, s_tokens = slots[sel], token_of[sel]
            w = self.expert_weights_f32(i, p, e)
            acc = (x[s_tokens] if self.fc1 else x[s_slots]) @ w.T
            del w
            if i["per_token_sf_b"] is not None:  # routing scales on the input
                acc = acc * i["per_token_sf_b"][s_tokens].float()[:, None]
            if i["bias"] is not None:
                acc = acc + i["bias"][e]
            sc = per_expert("scale_c", e)
            sg = per_expert("scale_gate", e)
            sc = 1.0 if sc is None else sc
            sg = 1.0 if sg is None else sg
            if self.fc1 and o["mFusedAct"]:
                half = p.m // 2
                alpha, beta = per_expert("alpha", e), per_expert("beta", e)
                out = sc * gated(
                    o["mActType"],
                    acc[:, :half],
                    sg * acc[:, half:],
                    1.0 if alpha is None else alpha,
                    0.0 if beta is None else beta,
                    per_expert("clamp_limit", e),
                )
            elif self.fc1 and o["mEltwiseActType"] == host.ELTWISE_RELU2:
                out = sc * torch.relu(sg * acc).pow(2)
            elif self.fc1 and o["mEltwiseActType"] != host.ELTWISE_NONE:
                raise ValueError("unsupported eltwise activation")
            else:
                out = sc * acc
            expected[s_slots] = out
        total_t = i["total_num_padded_tokens"].clone()
        return (*self.encode_output(expected, total), expected, total_t)

    # -- launch ---------------------------------------------------------------

    def _outputs(self, p: host.BmmProblem) -> tuple[torch.Tensor, torch.Tensor | None]:
        o, dev = self.options, self.device
        rows = p.max_num_ctas * o["mTileN"]
        cols = self._out_cols(p)
        kind = output_kind(o)
        sf = None
        if kind == "bf16":
            c = torch.empty(rows, cols, dtype=torch.bfloat16, device=dev)
        elif kind == "nvfp4":
            c = torch.empty(rows, cols // 2, dtype=torch.uint8, device=dev)
        else:
            c = torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device=dev)
        if "sf_c" in host.moe_buffers(o):
            if kind == "nvfp4":
                sf = torch.empty(fmt.sf_storage_size(rows, cols // 16, 3), dtype=torch.uint8,
                                 device=dev)  # fmt: skip
            elif kind == "mxfp8":
                sf = torch.empty(fmt.sf_storage_size(rows, cols // 32, 3), dtype=torch.uint8,
                                 device=dev)  # fmt: skip
            else:  # DeepSeek scales / the FP8 per-tensor launcher's spare buffer
                sf = torch.empty(
                    -(-p.m // DS_BLOCK) * rows, dtype=torch.float32, device=dev
                )
        return c, sf

    def pointers(
        self, inputs: tuple, c: torch.Tensor, sf_c: torch.Tensor | None
    ) -> dict[str, int]:
        """Device addresses of every KernelParams pointer: the inputs that
        are present (get_inputs creates exactly ``host.moe_buffers``), the
        outputs; 0 (nullptr) otherwise."""
        tensors = dict(zip(MOE_INPUTS, inputs))
        tensors.update(c=c, sf_c=sf_c)
        return {
            name: (tensors[name].data_ptr() if tensors.get(name) is not None else 0)
            for name in host.BMM_POINTER_FIELDS
        }

    def _build(self, inputs: tuple) -> tuple[host.LaunchSpec, tuple, tuple]:
        self._check_cuda(inputs[:-1])
        p = self._problem_of(inputs)
        c, sf_c = self._outputs(p)
        o = self.options
        params = host.bmm_params(
            self.layout, o, p, self.pointers(inputs, c, sf_c), self._encoder()
        )
        spec = self.spec(host.bmm_grid(o, p, self._sm_count()), params)
        outputs: tuple = (c, sf_c) if self.outputs_count == 2 else (c,)
        return spec, outputs, (inputs, c, sf_c)

    # -- validation ---------------------------------------------------------------

    def dequantize(self, impl: tuple, cols: int, total: int) -> torch.Tensor:
        o = self.options
        kind = output_kind(o)
        c = impl[0]
        if kind in ("bf16", "e4m3"):
            return c.float()
        rows = c.shape[0]
        sf = impl[1]
        if kind == "ds_e4m3":
            out = torch.full((rows, cols), torch.nan, device=c.device)
            s = ds_scale_view(sf, cols, total).T  # [total, cols / 128]
            out[:total] = c[:total].float() * s.repeat_interleave(DS_BLOCK, -1)
            return out
        layout = o["mSfLayoutC"]
        if kind == "nvfp4":
            codes = fmt.unpack_nibbles(c)
            scales = fmt.e4m3_decode(fmt.sf_from_layout(sf, rows, cols // 16, layout))
            return fmt.e2m1_decode(codes) * scales.repeat_interleave(16, -1)
        scales = fmt.ue8m0_decode(fmt.sf_from_layout(sf, rows, cols // 32, layout))
        return c.float() * scales.repeat_interleave(32, -1)

    def block_scales(self, impl: tuple, cols: int) -> torch.Tensor:
        """Decoded per-block scales ``[rows, cols / block]`` of a
        block-scaled output (NVFP4 / MxFP8)."""
        kind = output_kind(self.options)
        block = BLOCK_SCALED[kind][0]
        flat = fmt.sf_from_layout(
            impl[1], impl[0].shape[0], cols // block, self.options["mSfLayoutC"]
        )
        return fmt.e4m3_decode(flat) if kind == "nvfp4" else fmt.ue8m0_decode(flat)

    def encode_output(
        self, expected: torch.Tensor, total: int, mx_floor: bool = False
    ) -> tuple:
        """``expected`` quantized into the kernel's output buffers by an
        ideal quantizer (undefined rows as zeros). MxFP8 scales are
        ``2**ceil(log2(amax / 448))``, or with ``mx_floor`` the OCP MX
        ``2**(floor(log2 amax) - 8)`` (saturating)."""
        o = self.options
        kind = output_kind(o)
        v = expected.nan_to_num(0.0)
        rows, cols = v.shape
        if kind == "bf16":
            return (v.to(torch.bfloat16),)
        if kind == "e4m3":
            return (fmt.e4m3_encode(v),)
        if kind == "ds_e4m3":
            s = v.abs().view(rows, cols // DS_BLOCK, DS_BLOCK).amax(-1) / fmt.E4M3_MAX
            s = torch.where(s > 0, s, torch.ones_like(s))
            c = fmt.e4m3_encode(v / s.repeat_interleave(DS_BLOCK, -1))
            flat = torch.zeros(cols // DS_BLOCK * rows, device=v.device)
            ds_scale_view(flat, cols, total).copy_(s[:total].T)
            return c, flat
        layout = o["mSfLayoutC"]
        if kind == "nvfp4":
            s = fmt.e4m3_encode(
                v.abs().view(rows, cols // 16, 16).amax(-1) / fmt.E2M1_MAX
            )
            sv = s.float().repeat_interleave(16, -1)
            codes = fmt.e2m1_encode(v / torch.where(sv > 0, sv, torch.ones_like(sv)))
            return fmt.pack_nibbles(codes), fmt.sf_to_layout(
                s.view(torch.uint8), layout
            )
        amax = v.abs().view(rows, cols // 32, 32).amax(-1)
        if mx_floor:
            e = torch.floor(torch.log2(amax.clamp(min=2.0**-100))) - 8
        else:
            e = torch.ceil(torch.log2(amax.clamp(min=2.0**-100) / fmt.E4M3_MAX))
        e = e.clamp(-127, 127)
        c = fmt.e4m3_encode(v / torch.exp2(e).repeat_interleave(32, -1))
        return c, fmt.sf_to_layout((e + 127).to(torch.uint8), layout)

    def validate(self, ref: tuple, impl: tuple) -> None:
        expected, total = ref[-2], int(ref[-1][0])
        kind = output_kind(self.options)
        if len(impl) < self.outputs_count:
            raise AssertionError(f"{self.name}: expected {self.outputs_count} outputs")
        impl = impl[: self.outputs_count]
        cols = expected.shape[1]
        actual = self.dequantize(impl, cols, total).to(expected.device)
        if kind in BLOCK_SCALED:
            scales = self.block_scales(impl, cols).to(expected.device)
            check_block_scaled(expected, actual, scales, kind, self.name)
        else:
            check_output(expected, actual, kind, self.name)


class DenseGemm(_TrtllmGen):
    """One dense trtllm-gen GEMM (see module docstring).

    Inputs: ``(act [m, K], act_sf, weights [n, K] kernel-native, weights_sf,
    alpha)``; FP4: E2m1 packed, E4m3 SFs (``mSfLayoutB`` for the
    activations, R128c4 shuffled for the weights), ``alpha`` float32 [1];
    MxFP8: E4m3 + UE8m0 SFs; DeepSeek FP8: float scales ``[K / 128, m]`` and
    ``[n / 128, K / 128]``; low latency: E4m3 weights in BlockMajorK, no
    SFs, ``alpha``. Output: ``(c [m, n] bf16,)``.
    """

    pdl_fields = host.GEMM_PDL_FIELDS
    layout_key = "gemm"

    def get_inputs(self, case: CaseSpec) -> tuple:
        o, role = self.options, self.entry["role"]
        m, n, k = case.params["m"], case.params["n"], case.params["k"]
        g, dev = self.generator(case), self.device
        alpha: torch.Tensor | None = None
        if role == "gemm_fp8_blockscale":
            act, act_scale = ds_operand((m, k), g, dev, 1.0, (k // DS_BLOCK, m))
            w, w_scale = ds_operand(
                (1, n, k), g, dev, k**-0.5, (n // DS_BLOCK, k // DS_BLOCK)
            )
            return act, act_scale, w[0].contiguous(), w_scale, alpha
        a_op = make_operand(o["mDtypeB"], (m, k), g, dev, 1.0)
        per_tensor = role == "gemm_low_latency"
        w_op = make_operand(
            o["mDtypeA"], (1, n, k), g, dev, 1.0 if per_tensor else k**-0.5
        )
        w_store, w_sf = prepare_weights(o, o["mDtypeA"], w_op.data, w_op.sf)
        act_sf: torch.Tensor | None = None
        if a_op.sf is not None:
            act_sf = fmt.sf_to_layout(a_op.sf, o["mSfLayoutB"])
        if "scale_c" in self.entry["buffers"]:
            acc_rms = math.sqrt(
                k * a_op.values.pow(2).mean().item() * w_op.values.pow(2).mean().item()
            )
            alpha = torch.tensor([_pow2(1.0 / acc_rms)], device=dev)
        return a_op.data, act_sf, w_store[0].contiguous(), w_sf, alpha

    def _dims(self, inputs: tuple) -> host.GemmProblem:
        act, _, w, *_ = inputs
        m, k = act.shape[0], act.shape[1] * (
            2 if FORMATS[self.options["mDtypeB"]].packed else 1
        )
        n = w.numel() * (2 if FORMATS[self.options["mDtypeA"]].packed else 1) // k
        return host.GemmProblem(n, m, k)

    def get_reference(self, inputs: tuple) -> tuple:
        o, role = self.options, self.entry["role"]
        act, act_sf, w, w_sf, alpha = inputs
        p = self._dims(inputs)
        if role == "gemm_fp8_blockscale":
            x = act.float() * act_sf.T.repeat_interleave(DS_BLOCK, -1)
            wv = w.float() * w_sf.repeat_interleave(DS_BLOCK, 0).repeat_interleave(
                DS_BLOCK, 1
            )
        else:
            f = FORMATS[o["mDtypeB"]]
            sf_rows = None
            if act_sf is not None:
                sf_rows = fmt.sf_from_layout(
                    act_sf, p.n, p.k // f.block, o["mSfLayoutB"]
                )
            x = decode_operand(o["mDtypeB"], act, sf_rows, p.k)
            wv = restore_weights(o, o["mDtypeA"], w[None], w_sf, 1, p.m, p.k)[0]
        out = x @ wv.T
        if alpha is not None:
            out = out * alpha
        return (out,)

    def _build(self, inputs: tuple) -> tuple[host.LaunchSpec, tuple, tuple]:
        self._check_cuda(inputs)
        act, act_sf, w, w_sf, alpha = inputs
        p = self._dims(inputs)
        c = torch.empty(p.n, p.m, dtype=torch.bfloat16, device=self.device)
        tensors = {
            "a": w,
            "b": act,
            "c": c,
            "sf_a": w_sf,
            "sf_b": act_sf,
            "scale_c": alpha,
        }
        allowed = {"a", "b", "c", *self.entry["buffers"]}
        ptr = {}
        for name in host.GEMM_POINTER_FIELDS:
            t = tensors.get(name)
            if name in allowed:
                if t is None:
                    raise ValueError(f"{self.name}: {name} is required")
                ptr[name] = t.data_ptr()
            else:
                ptr[name] = 0
        o = self.options
        params = host.gemm_params(self.layout, o, p, ptr, self._encoder())
        spec = self.spec(host.gemm_grid(o, p, self._sm_count()), params)
        return spec, (c,), (inputs, c)

    def validate(self, ref: tuple, impl: tuple) -> None:
        if len(impl) != 1:
            raise AssertionError("expected one output")
        check_output(ref[0], impl[0].float().to(ref[0].device), "bf16", self.name)


# --- probe fixtures -----------------------------------------------------------------------


def _probe_bytes(launch: Mapping[str, Any]) -> bytes:
    out = bytearray(launch["params_size"])
    for offset, data in launch["params"]:
        raw = bytes.fromhex(data)
        out[offset : offset + len(raw)] = raw
    return bytes(out)


def _zero(data: bytes, ranges: Sequence[Sequence[int]]) -> bytes:
    out = bytearray(data)
    for lo, hi in ranges:
        out[lo:hi] = bytes(hi - lo)
    return bytes(out)


def _recording_encoder(calls: list[dict[str, Any]]) -> host.TensorMapEncoder:
    def encode(data_type, address, dims, strides, box, element_strides, interleave,
               swizzle, l2_promotion, oob_fill) -> bytes:  # fmt: skip
        calls.append(
            {
                "data_type": data_type, "address": address, "dims": list(dims),
                "strides": list(strides), "box": list(box),
                "element_strides": list(element_strides), "interleave": interleave,
                "swizzle": swizzle, "l2_promotion": l2_promotion, "oob_fill": oob_fill,
            }
        )  # fmt: skip
        return host.fake_encode(
            data_type, address, dims, strides, box, element_strides, interleave,
            swizzle, l2_promotion, oob_fill,
        )  # fmt: skip

    return encode


def rebuild_fixture(name: str, case: Mapping[str, Any]) -> tuple[host.LaunchSpec, list]:
    """The launch the Python host mirror builds for a probe fixture case
    (fake addresses, recording encoder): (spec, encode calls)."""
    entry = variant_index()[name]
    o = entry["options"]
    calls: list[dict[str, Any]] = []
    encode = _recording_encoder(calls)
    if entry["role"] in ("fc1", "fc2"):
        p = host.moe_problem(o, case["params"])
        fake = fake_pointers(BMM_PROBE_BUFFERS)
        buffers = host.moe_buffers(o, case["params"])
        allowed = {"a", "b", "c", *DYNAMIC_TABLES, *buffers}
        ptr = {n: (fake[n] if n in allowed else 0) for n in host.BMM_POINTER_FIELDS}
        params = host.bmm_params(layouts()["bmm"], o, p, ptr, encode)
        grid = host.bmm_grid(o, p, 148)
        fields: tuple[str, ...] = host.BMM_PDL_FIELDS
    else:
        q = case["params"]
        gp = host.GemmProblem(q["n"], q["m"], q["k"])
        fake = fake_pointers(GEMM_PROBE_BUFFERS)
        allowed = {"a", "b", "c", *entry["buffers"]}
        ptr = {n: (fake[n] if n in allowed else 0) for n in host.GEMM_POINTER_FIELDS}
        params = host.gemm_params(layouts()["gemm"], o, gp, ptr, encode)
        grid = host.gemm_grid(o, gp, 148)
        fields = host.GEMM_PDL_FIELDS
    spec = host.launch_spec(entry, grid, params, pdl_safe_fields=fields)
    return spec, calls


def check_param_layouts() -> bool:
    """Byte-compare the KernelParams, grid/block/smem, launch attributes and
    every cuTensorMapEncodeTiled call rebuilt in Python with the probes'
    recordings of FlashInfer's host code (indeterminate bytes excluded), for
    every case of every registered sm_100a variant."""
    if not FIXTURE_FILE.is_file():
        return True
    index = variant_index()
    seen = set()
    with gzip.open(FIXTURE_FILE, "rt") as f:
        for line in f:
            row = json.loads(line)
            name, case_name = row["key"].split("/", 1)
            if name not in index:
                continue
            case = next(c for c in index[name]["cases"] if c["name"] == case_name)
            spec, calls = rebuild_fixture(name, case)
            (launch,) = row["launches"]
            what = f"{name}/{case_name}"
            expected = _probe_bytes(launch)
            built = _zero(spec.params, launch["indeterminate"])
            if len(spec.params) != launch["params_size"] or built != expected:
                diff = next(
                    (i for i, (x, y) in enumerate(zip(built, expected)) if x != y), None
                )
                raise AssertionError(f"{what}: KernelParams differ at byte {diff}")
            if calls != launch["tensor_maps"]:
                raise AssertionError(f"{what}: cuTensorMapEncodeTiled calls differ")
            if list(spec.grid) != launch["grid"] or list(spec.block) != launch["block"]:
                raise AssertionError(f"{what}: grid/block differ")
            if spec.shared_mem != launch["shared_mem"]:
                raise AssertionError(f"{what}: dynamic shared memory differs")
            if host.recorded_attrs(spec) != launch["attrs"]:
                raise AssertionError(f"{what}: launch attributes differ")
            seen.add(what)
    missing = {f"{n}/{c['name']}" for n, e in index.items() for c in e["cases"]} - seen
    if missing:
        raise AssertionError(f"no probe fixture for {sorted(missing)[:5]}")
    return True


def _register() -> None:
    for name, entry in variant_index().items():
        base = MoEGemm if entry["role"] in ("fc1", "fc2") else DenseGemm
        register_variant(base, name=name, supported_arches=("sm_100a",), entry=entry)


_register()

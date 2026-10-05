"""GDN (Gated DeltaNet, Qwen3-Next linear attention) as two single-kernel workloads.

One package serves both official definitions, the TP = 4 shard of
Qwen3-Next-80B-A3B's linear-attention layers (16 key / 32 value heads become
Hq = Hk = 4, Hv = 8; head size 128; FP32 k-last state ``[N, 8, V, K]``):

* ``gdn_decode_qk4_v8_d128_k_last`` (``resources/gdn_decode.json``): one token
  per sequence (``[B, 1, H, 128]`` tensors, here viewed as ``[B, H, 128]``);
* ``gdn_prefill_qk4_v8_d128_k_last`` (``resources/gdn_prefill.json``): ragged
  ``[T, H, 128]`` tensors with ``cu_seqlens``.

Gates ``g = exp(-exp(A_log) * softplus(a + dt_bias))``, ``beta = sigmoid(b)``;
per token and value head (query/key head ``h // 2``), state ``S`` [V, K]:
``S = g S; S += beta (v - S k) k^T; o = scale S q``.

``gdn_gates`` (sm_86, sm_100a)
    Project glue ``prepare_gates``: FP32 ``gate``/``beta`` ``[T, 8]`` and int32
    ``cu_seqlens`` from the raw inputs (int64 ``cu_seqlens``; ``None`` for
    decode, where the kernel writes ``0..B``).
``gdn_chunk`` (sm_100a)
    FlashInfer's generated Blackwell (tcgen05) DV-split chunked prefill
    ``kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial``: output ``[T,
    8, 128]`` (BF16) and the final state (FP32, a separate buffer) from ``q, k,
    v``, the initial state, ``g``, ``beta`` and int32 ``cu_seqlens``. Decode
    is B one-token sequences.

The former ``gdn_decode`` and ``gdn_prefill`` packages shipped byte-identical
copies of both kernels; they are merged here (``impls/gdn/compiler.py``).
Inventory names stay ``gdn_decode`` / ``gdn_prefill``.

All launch preparation happens in Python, mirroring the upstream TVM-FFI host
shim and ``flashinfer/gdn_prefill.py`` (``_run_cake_gdn_prefill``): four BF16
CUtensorMaps (box 64x64x1, 128 B swizzle, 256 B L2 promotion) over ``[T, H,
128]`` views, ``total_tiles = N * 8 * 2`` (DV-split), the persistent grid
``min(sm_count, total_tiles)``, a per-CTA tensor-map workspace of ``grid *
512`` bytes, one-element dummies for the unused state-index and checkpoint
buffers, 384 threads and 226048 bytes of dynamic shared memory. The probe
sidecar records what the upstream host code builds; ``tests/test_gdn.py``
compares byte for byte.

Cases (``CaseSpec.params``): ``batch`` (decode: B one-token sequences) or
``lengths`` (ragged prefill, empty sequences allowed); ``gates``: ``model``
(Qwen3-Next initialisation: ``A = exp(A_log)`` uniform in [1, 16], ``dt_bias``
the inverse softplus of ``dt`` log-uniform in [1e-3, 1e-1], so most gates are
close to 1: long memory), ``wide`` (standard normal ``A_log``/``dt_bias``),
``small`` (upstream decode tests: ``0.1 *`` normal ``A_log``/``dt_bias``/``a``),
or, for the chunk kernel only (no raw equivalent), ``uniform`` (upstream
prefill tests: ``g, beta ~ U(0, 1)``), ``unit_alpha`` / ``unit_beta`` (upstream
``alpha=False`` / ``beta=False``: ones), ``near_one`` (``U(0.99, 1)``);
``zero_state``; ``scale`` (0 selects 1/sqrt(128)); ``q``: ``unit``
(L2-normalised as Qwen3-Next does) or ``randn``; ``v_std``; ``state_std``.
Keys are always L2-normalised.

No sm_100a result here has been validated numerically on Blackwell hardware.
"""

from __future__ import annotations

import ctypes
import json
import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import cached_property
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from .. import cuda_driver
from ..cutlass_host import driver_encode
from ..registry import register
from ..throughput import model_case, skewed_lengths, synthetic, trace, upstream_case
from ..workload import CaseSpec, Workload

NUM_Q_HEADS = 4
NUM_V_HEADS = 8
GROUP = NUM_V_HEADS // NUM_Q_HEADS
HEAD_SIZE = 128
DEFAULT_SCALE = 1 / math.sqrt(HEAD_SIZE)
VALUE_SPLITS = 2  # DV-split: each (sequence, head) is two 64-column tiles
STATE_HEAD_STRIDE = HEAD_SIZE * HEAD_SIZE
GATES_BLOCK = 256
WORKSPACE_PER_CTA = 512  # four 128-byte CUtensorMaps per persistent CTA
DEFAULT_SM_COUNT = 148  # B200; used only when building arguments off-GPU
REFERENCE_BLOCK = 256  # sequences per reference step (128 MiB FP32 states)

# Raw CUtensorMap* enum values from cuda.h (as the upstream host shim uses).
INTERLEAVE_NONE, SWIZZLE_128B, L2_PROMOTION_256B, OOB_FILL_NONE = 0, 3, 3, 0
TMA_BOX = (64, 64, 1)

RAW_GATES = ("model", "wide", "small")
DIRECT_GATES = ("uniform", "unit_alpha", "unit_beta", "near_one")
DEFAULTS: dict[str, Any] = {
    "gates": "model",
    "zero_state": False,
    "scale": 0.0,
    "q": "unit",
    "v_std": 1.0,
    "state_std": 0.1,
}

# Encodes one CUtensorMap: (data_type, address, dims, strides_bytes, box,
# element_strides, interleave, swizzle, l2_promotion, oob_fill).
TensorMapEncoder = Callable[..., Any]


class LaunchSpec(NamedTuple):
    grid: int
    block: int
    args: list[Any]
    shared_mem: int


def case_lengths(params: dict[str, Any]) -> list[int]:
    """Per-sequence token counts: ``lengths``, or ``batch`` one-token sequences."""
    if "lengths" in params:
        return list(params["lengths"])
    return [1] * params["batch"]


def param(case: CaseSpec, key: str) -> Any:
    return case.params.get(key, DEFAULTS[key])


def gdn_gates(
    alog: torch.Tensor, a: torch.Tensor, bias: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``(gate, beta)`` of shape ``a.shape``, computed in FP64.

    ``softplus`` uses PyTorch's (and the kernel's) threshold 20.
    """
    x = a.double() + bias.double()
    gate = torch.exp(-alog.double().exp() * F.softplus(x))
    return gate.float(), torch.sigmoid(b.double()).float()


def delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    lengths: Sequence[int],
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-serial gated delta rule (the definitions' reference), FP32.

    ``q/k [T, 4, 128]``, ``v [T, 8, 128]``, ``state [N, 8, V, K]``, ``gate/beta
    [T, 8]``. Sequences advance in lockstep (one step per token position), so
    a decode batch is a single step. An empty sequence passes its initial state
    through, as the kernel does (``num_chunks == 0`` copies it).
    """
    tokens, device = q.shape[0], q.device
    qf = q.float().repeat_interleave(GROUP, dim=1)
    kf = k.float().repeat_interleave(GROUP, dim=1)
    vf, gate, beta = v.float(), gate.float(), beta.float()
    output = torch.empty(
        (tokens, NUM_V_HEADS, HEAD_SIZE), dtype=torch.bfloat16, device=device
    )
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    # Longest first: the sequences still active at step i are a prefix.
    order = sorted(range(len(lengths)), key=lambda s: -lengths[s])
    perm = torch.tensor(order, dtype=torch.long, device=device)
    first = torch.tensor([starts[s] for s in order], dtype=torch.long, device=device)
    sorted_lengths = [lengths[s] for s in order]
    h = state.float()[perm].clone()
    active = len(order)
    for i in range(max(lengths, default=0)):
        while sorted_lengths[active - 1] <= i:
            active -= 1
        for lo in range(0, active, REFERENCE_BLOCK):  # bounds temporaries
            hi = min(active, lo + REFERENCE_BLOCK)
            t = first[lo:hi] + i
            s = h[lo:hi] * gate[t][:, :, None, None]
            old = (s @ kf[t][..., None])[..., 0]  # [n, H, V]
            s += (beta[t][..., None] * (vf[t] - old))[..., None] * kf[t][:, :, None, :]
            output[t] = ((s @ qf[t][..., None])[..., 0] * scale).to(torch.bfloat16)
            h[lo:hi] = s
    final = torch.empty_like(h)
    final[perm] = h
    return output, final


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scaled_close(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float,
    atol_rms: float,
    rel_l2: float,
) -> None:
    """Raise unless ``|a - e| <= atol_rms * rms(e) + rtol * |e|`` element-wise
    and ``||a - e||_2 <= rel_l2 * ||e||_2``: tolerances relative to the
    tensor's own scale, so a uniform relative error cannot hide below an
    absolute tolerance larger than the values."""
    e, a = expected.double(), actual.double()
    if not bool(torch.isfinite(a).all()):
        raise AssertionError(f"{name}: non-finite values")
    norm = e.norm()
    rms = float(norm) / math.sqrt(max(e.numel(), 1))
    diff = (a - e).abs()
    excess = diff - (atol_rms * rms + rtol * e.abs())
    if bool((excess > 0).any()):
        worst = int(excess.argmax())
        raise AssertionError(
            f"{name}: {int((excess > 0).sum())}/{e.numel()} elements exceed "
            f"{atol_rms} * rms ({rms:.3e}) + {rtol} * |ref|; worst at flat index "
            f"{worst}: {a.flatten()[worst].item():.6e} vs {e.flatten()[worst].item():.6e}"
        )
    error = float(diff.norm())
    if error > rel_l2 * float(norm):
        raise AssertionError(
            f"{name}: relative L2 error {error / max(float(norm), 1e-300):.3e} > {rel_l2}"
        )


class _GDN(Workload):
    """Shared cases and generation of the definitions' inputs."""

    package = "gdn"

    def get_cases(self) -> list[CaseSpec]:
        return smoke_cases() + throughput_cases()

    def definition_inputs(self, case: CaseSpec) -> tuple:
        """``(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)`` in the
        prefill definition's layout (decode tensors are their ``[B, 1, ...]``
        views); int64 ``cu_seqlens``, CPU scalar ``scale``."""
        g, lengths = self.generator(case), case_lengths(case.params)
        t, n = sum(lengths), len(lengths)
        if param(case, "q") == "unit":
            q = F.normalize(
                self.randn((t, NUM_Q_HEADS, HEAD_SIZE), g, torch.float32), dim=-1
            )
        else:
            q = self.randn((t, NUM_Q_HEADS, HEAD_SIZE), g, torch.float32)
        k = F.normalize(
            self.randn((t, NUM_Q_HEADS, HEAD_SIZE), g, torch.float32), dim=-1
        )
        v = self.randn((t, NUM_V_HEADS, HEAD_SIZE), g, scale=param(case, "v_std"))
        state = self.randn(
            (n, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE),
            g,
            torch.float32,
            scale=param(case, "state_std"),
        )
        if param(case, "zero_state"):
            state.zero_()
        mode = param(case, "gates")
        if mode == "model":
            uniform = torch.rand((2, NUM_V_HEADS), device=self.device, generator=g)
            alog = torch.log(1 + 15 * uniform[0])
            dt = torch.exp(math.log(1e-3) + uniform[1] * math.log(100.0))
            bias = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
            a, b = self.randn((t, NUM_V_HEADS), g), self.randn((t, NUM_V_HEADS), g)
        else:  # "wide"; "small" (upstream decode tests); direct modes use wide
            s = 0.1 if mode == "small" else 1.0
            alog = self.randn((NUM_V_HEADS,), g, torch.float32, scale=s)
            bias = self.randn((NUM_V_HEADS,), g, torch.float32, scale=s)
            a = self.randn((t, NUM_V_HEADS), g, scale=s)
            b = self.randn((t, NUM_V_HEADS), g)
        cu = torch.tensor([0, *lengths], device=self.device, dtype=torch.int64)
        scale = self.scalar(param(case, "scale"))
        return (
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v,
            state,
            alog,
            a,
            bias,
            b,
            cu.cumsum(0),
            scale,
        )

    def chunk_gates(
        self, case: CaseSpec, raw: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP32 ``(gate, beta)`` the chunk kernel reads: ``gdn_gates`` of the raw
        inputs, or the directly specified upstream-test values."""
        _, _, _, _, alog, a, bias, b, _, _ = raw
        mode = param(case, "gates")
        if mode not in DIRECT_GATES:
            return gdn_gates(alog, a, bias, b)
        g = torch.Generator(device=self.device).manual_seed(case.seed + 7919)
        u = torch.rand((2, *a.shape), device=self.device, generator=g)
        if mode == "near_one":
            u = 0.99 + 0.01 * u
        gate, beta = u[0], u[1]
        if mode == "unit_alpha":
            gate = torch.ones_like(gate)
        if mode == "unit_beta":
            beta = torch.ones_like(beta)
        return gate.contiguous(), beta.contiguous()

    @abstractmethod
    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        """Allocate outputs/scratch and build the complete launch; no launch."""

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
    def check_outputs(ref: tuple, impl: tuple) -> None:
        if not (isinstance(ref, tuple) and isinstance(impl, tuple)):
            raise AssertionError("outputs must be tuples")
        if len(ref) != len(impl):
            raise AssertionError(f"expected {len(ref)} outputs, got {len(impl)}")
        for expected, actual in zip(ref, impl):
            if not isinstance(actual, torch.Tensor):
                raise AssertionError("every output must be a torch tensor")
            if (actual.shape, actual.dtype, actual.device) != (
                expected.shape,
                expected.dtype,
                expected.device,
            ):
                raise AssertionError(
                    f"output {tuple(actual.shape)} {actual.dtype} {actual.device} "
                    f"!= {tuple(expected.shape)} {expected.dtype} {expected.device}"
                )


@register(name="gdn_gates", supported_arches=("sm_86", "sm_100a"))
class GDNGates(_GDN):
    """``prepare_gates`` (project glue launched before the chunk kernel).

    Inputs: FP32 ``A_log [8]``, BF16 ``a [T, 8]``, FP32 ``dt_bias [8]``, BF16
    ``b [T, 8]`` (contiguous) and int64 ``cu_seqlens [N + 1]`` (prefill cases)
    or ``None`` (decode cases: the kernel's ``cu == nullptr`` branch writes
    ``0..B``). Outputs: FP32 ``gate [T, 8]``, FP32 ``beta [T, 8]``, int32
    ``cu_seqlens [N + 1]``; every element is written. Cases with directly
    specified gates (chunk-only regimes) are not served here.

    Tolerance against the FP64 reference: the kernel uses IEEE
    ``expf``/``log1pf`` (no fast math), a few ULP each, amplified in ``gate``
    by at most ``|exp(A_log) * softplus(x)|`` (< ~200 for the generated
    inputs), i.e. well below rtol 1e-4; atol 1e-30 only admits denormal
    rounding of gates that underflow. ``beta`` is checked at rtol 1e-5;
    ``cu_seqlens`` exactly.
    """

    def get_cases(self) -> list[CaseSpec]:
        return [c for c in super().get_cases() if param(c, "gates") in RAW_GATES]

    def get_inputs(self, case: CaseSpec) -> tuple:
        _, _, _, _, alog, a, bias, b, cu, _ = self.definition_inputs(case)
        return alog, a, bias, b, (None if "batch" in case.params else cu)

    def get_reference(self, inputs: tuple) -> tuple:
        alog, a, bias, b, cu = inputs
        gate, beta = gdn_gates(alog, a, bias, b)
        if cu is None:
            offsets = torch.arange(a.shape[0] + 1, dtype=torch.int32, device=a.device)
        else:
            offsets = cu.to(torch.int32)
        return gate, beta, offsets

    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        alog, a, bias, b, cu = inputs
        tokens = a.shape[0]
        seqs = tokens if cu is None else cu.numel() - 1
        device = a.device
        checks = [
            ("A_log", alog, torch.float32, (NUM_V_HEADS,)),
            ("a", a, torch.bfloat16, (tokens, NUM_V_HEADS)),
            ("dt_bias", bias, torch.float32, (NUM_V_HEADS,)),
            ("b", b, torch.bfloat16, (tokens, NUM_V_HEADS)),
        ]
        if cu is not None:
            checks.append(("cu_seqlens", cu, torch.int64, (seqs + 1,)))
        for name, tensor, dtype, shape in checks:
            _require(
                tensor.dtype == dtype
                and tuple(tensor.shape) == shape
                and tensor.is_contiguous()
                and tensor.device == device,
                f"{name} must be contiguous {dtype} {list(shape)} on {device}",
            )
        _require(
            tokens >= 1 and seqs >= 1 and tokens * NUM_V_HEADS < 2**31,
            f"unsupported tokens={tokens}, seqs={seqs}",
        )
        gate = torch.empty((tokens, NUM_V_HEADS), dtype=torch.float32, device=device)
        beta = torch.empty_like(gate)
        offsets = torch.empty(seqs + 1, dtype=torch.int32, device=device)
        args = [
            alog,
            a,
            bias,
            b,
            ctypes.c_void_p(None) if cu is None else cu,
            gate,
            beta,
            offsets,
            ctypes.c_int32(tokens),
            ctypes.c_int32(seqs),
        ]
        elements = max(tokens * NUM_V_HEADS, seqs + 1)
        grid = (elements + GATES_BLOCK - 1) // GATES_BLOCK
        return LaunchSpec(grid, GATES_BLOCK, args, 0), (gate, beta, offsets)

    def validate(self, ref: tuple, impl: tuple) -> None:
        self.check_outputs(ref, impl)
        self.assert_close(ref[:1], impl[:1], rtol=1e-4, atol=1e-30)
        self.assert_close(ref[1:2], impl[1:2], rtol=1e-5, atol=1e-7)
        self.assert_close(ref[2:], impl[2:], rtol=0, atol=0)


@register(name="gdn_chunk", supported_arches=("sm_100a",))
class GDNChunk(_GDN):
    """``kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial`` (Hv = 8).

    Inputs: BF16 ``q, k [T, 4, 128]``, ``v [T, 8, 128]``, FP32 initial state
    ``[N, 8, 128, 128]`` (k-last), FP32 ``gate, beta [T, 8]`` (what
    ``gdn_gates`` writes, generated with its FP64 reference, or the upstream
    tests' direct values), int32 ``cu_seqlens [N + 1]`` and the CPU scalar
    ``scale`` (0 selects 1/sqrt(128)). Outputs, every element written: BF16
    ``output [T, 8, 128]`` and FP32 ``new_state [N, 8, 128, 128]`` (a separate
    buffer; empty sequences copy their initial state).

    Tolerance (per output, relative to its own RMS, see ``scaled_close``):
    ``|err| <= 0.1 rms + 2e-2 |ref|`` element-wise and relative L2 error
    ``<= 1.5e-2``. The kernel runs the chunked (WY) form with BF16 Tensor
    Core operands (q, k, v, the state tile, the 64x64 inverse and QK^T tiles
    and the pseudo-values are BF16 in shared/tensor memory; FP32 accumulation;
    log2-space gate cumsum with ``g + 1e-10``) and ``--use_fast_math``. An
    emulation of those roundings (chunk 64, BF16 operands, FP32 accumulation)
    run on every smoke and throughput case (up to 32k tokens) gave relative
    L2 errors <= 4.4e-3 and worst ``|err| - 2e-2 |ref|`` of 3.8e-2 rms, for
    output and state alike; the bounds are 2.6-3.4x that. The former bound, ``rtol = atol = 1e-2`` absolute, exceeded the
    model-scale outputs themselves (rms ~1e-2): a uniform x1.3 error passed.
    These bounds are from emulation and must be re-confirmed on a B200 (only
    the former tolerance has passed there).
    """

    rtol, atol_rms, rel_l2 = 2e-2, 0.1, 1.5e-2
    encode: TensorMapEncoder = staticmethod(driver_encode)
    sm_count: int | None = None  # default: the device's SM count

    def get_inputs(self, case: CaseSpec) -> tuple:
        raw = self.definition_inputs(case)
        q, k, v, state, _, _, _, _, cu, scale = raw
        gate, beta = self.chunk_gates(case, raw)
        return q, k, v, state, gate, beta, cu.to(torch.int32), scale

    def get_reference(self, inputs: tuple) -> tuple:
        q, k, v, state, gate, beta, cu, scale = inputs
        lengths = cu.diff().tolist()
        return delta_rule(
            q, k, v, state, gate, beta, lengths, float(scale) or DEFAULT_SCALE
        )

    @cached_property
    def constants(self) -> dict[str, int]:
        """Launch constants of the generated kernel (compile-time sidecar)."""
        if self.arch is None:
            raise RuntimeError("reference-only workload has no launch constants")
        sidecar = self.cubin_path(self.arch).with_suffix(".json")
        return json.loads(sidecar.read_text())["constants"]

    def configure(self, function: cuda_driver.Function) -> None:
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            self.constants["dynamic_smem"],
        )

    def active_sm_count(self, device: torch.device) -> int:
        if self.sm_count is not None:
            return self.sm_count
        if device.type == "cuda":
            return torch.cuda.get_device_properties(device).multi_processor_count
        return DEFAULT_SM_COUNT

    def tensor_map(self, tensor: torch.Tensor) -> Any:
        """``EncodeTma_{Q,K,V,O}`` of the upstream shim for a ``[T, H, 128]``
        BF16 tensor: dims ``{128, T, H}``, byte strides ``{token, head}``."""
        tokens, heads, dim = tensor.shape
        _require(
            tensor.dtype == torch.bfloat16 and dim == HEAD_SIZE,
            "TMA source must be BF16 [T, H, 128]",
        )
        _require(tensor.stride(2) == 1, "TMA source needs a unit innermost stride")
        token_stride, head_stride = tensor.stride(0) * 2, tensor.stride(1) * 2
        _require(
            tensor.data_ptr() % 16 == 0
            and token_stride % 16 == 0
            and head_stride % 16 == 0,
            "TMA source address and strides must be 16-byte aligned",
        )
        return self.encode(
            cuda_driver.CU_TENSOR_MAP_DATA_TYPE[torch.bfloat16],
            tensor.data_ptr(),
            [dim, tokens, heads],
            [token_stride, head_stride],
            list(TMA_BOX),
            [1, 1, 1],
            INTERLEAVE_NONE,
            SWIZZLE_128B,
            L2_PROMOTION_256B,
            OOB_FILL_NONE,
        )

    def configure_launch(self, inputs: tuple) -> tuple[LaunchSpec, tuple]:
        q, k, v, state, gate, beta, cu, scale = inputs
        tokens, seqs = q.shape[0], cu.numel() - 1
        device = q.device
        _require(tokens >= 1 and seqs >= 1, "needs at least one token and sequence")
        for name, tensor, dtype, shape in (
            ("q", q, torch.bfloat16, (tokens, NUM_Q_HEADS, HEAD_SIZE)),
            ("k", k, torch.bfloat16, (tokens, NUM_Q_HEADS, HEAD_SIZE)),
            ("v", v, torch.bfloat16, (tokens, NUM_V_HEADS, HEAD_SIZE)),
            ("state", state, torch.float32, (seqs, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE)),
            ("gate", gate, torch.float32, (tokens, NUM_V_HEADS)),
            ("beta", beta, torch.float32, (tokens, NUM_V_HEADS)),
            ("cu_seqlens", cu, torch.int32, (seqs + 1,)),
        ):
            _require(
                tensor.dtype == dtype
                and tuple(tensor.shape) == shape
                and tensor.is_contiguous()
                and tensor.device == device,
                f"{name} must be contiguous {dtype} {list(shape)} on {device}",
            )
        _require(
            isinstance(scale, torch.Tensor) and scale.numel() == 1,
            "scale must be a scalar tensor",
        )
        scale_value = float(scale) or DEFAULT_SCALE

        output = torch.empty_like(v)
        new_state = torch.empty_like(state)
        total_tiles = seqs * NUM_V_HEADS * VALUE_SPLITS  # (sequence, head, V half)
        grid = min(self.active_sm_count(device), total_tiles)
        workspace = torch.empty(
            grid * WORKSPACE_PER_CTA, dtype=torch.uint8, device=device
        )
        _require(
            not workspace.is_cuda or workspace.data_ptr() % 128 == 0,
            "tensor-map workspace must be 128-byte aligned",
        )
        # FlashInfer passes one-element dummies for the unused optional buffers.
        dummy_i32 = torch.empty(1, dtype=torch.int32, device=device)
        dummy_f32 = torch.empty(1, dtype=torch.float32, device=device)
        args = [
            self.tensor_map(q),
            self.tensor_map(k),
            self.tensor_map(v),
            self.tensor_map(output),
            gate,
            beta,
            cu,
            dummy_i32,  # state_indices (USE_STATE_INDICES = 0)
            state,  # initial_state
            new_state,  # output_state
            dummy_f32,  # checkpoint_state (ENABLE_CHECKPOINTS = 0)
            dummy_i32,  # cu_checkpoints
            workspace,  # tensormap_workspace
            ctypes.c_int64(state.stride(0)),  # initial_state_stride_slot
            ctypes.c_int64(new_state.stride(0)),  # output_state_stride_slot
            ctypes.c_int32(0),  # checkpoint_every_n_tokens
            ctypes.c_float(scale_value),
            ctypes.c_int32(seqs),  # num_seqs
            ctypes.c_int32(NUM_Q_HEADS),
            ctypes.c_int32(NUM_V_HEADS),
            ctypes.c_int32(total_tiles),
        ]
        # The scratch tensors stay alive with the argument list (prepare).
        spec = LaunchSpec(
            grid, self.constants["threads"], args, self.constants["dynamic_smem"]
        )
        return spec, (output, new_state)

    def validate(self, ref: tuple, impl: tuple) -> None:
        self.check_outputs(ref, impl)
        for name, expected, actual in zip(("output", "new_state"), ref, impl):
            scaled_close(
                name,
                expected,
                actual,
                rtol=self.rtol,
                atol_rms=self.atol_rms,
                rel_l2=self.rel_l2,
            )


# -- cases ---------------------------------------------------------------------

PREFILL_TESTS = "tests/gdn/test_prefill_delta_rule.py"
CAKE_PREFILL_TESTS = "tests/gdn/test_cake_gdn_prefill_gpu.py"
DECODE_TESTS = "tests/gdn/test_decode_delta_rule.py"
CAKE_DECODE_TESTS = "tests/gdn/test_cake_gdn_decode_gpu.py"
UPSTREAM_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"

# Upstream FlashInfer GDN test parametrizations projected onto the definitions
# (BF16 I/O, FP32 state, Hq/Hk/Hv = 4/4/8 for the upstream 2/2/4 and the TP4
# shard of 16/16/32): (test id, the case params it requires). Gate regimes:
# upstream alpha/beta=True draw torch.rand ("uniform"), alpha=False / beta=False
# pass ones ("unit_alpha" / "unit_beta"; both False is skipped upstream);
# initial_state=None runs with a zero state. tests/test_gdn.py checks that
# every entry matches a gdn_chunk case.
_CHUNKED = (  # test_chunked_prefill second call (initial state = first's state)
    ([128], "uniform", 1.0),
    ([61], "uniform", 0.0),
    ([256, 256], "unit_alpha", 1.0),
    ([511, 501], "unit_alpha", 0.0),
    ([64, 128, 512], "unit_beta", 1.0),
    ([123, 150, 500], "unit_beta", 0.0),
)
_BASIC = ([64], [128], [256], [256, 256], [64, 128, 512])
_NONFULL = ([31], [61], [91], [121], [251], [511, 501], [31, 63, 93, 123, 150, 500])
_UPSTREAM_GATES = ("uniform", "unit_alpha", "unit_beta")


def _upstream_tests() -> list[tuple[str, dict[str, Any]]]:
    tests: list[tuple[str, dict[str, Any]]] = []
    for lengths, gates, scale in _CHUNKED:
        tests.append(
            (
                f"{PREFILL_TESTS}::test_chunked_prefill[bfloat16-64-seq_lens2={lengths}"
                f"-(2,2,4)-128-scale={scale or 'auto'}-{gates}]",
                {
                    "lengths": lengths,
                    "gates": gates,
                    "scale": scale,
                    "zero_state": False,
                },
            )
        )
    for name, sets in (("basic", _BASIC), ("nonfull", _NONFULL)):
        for i, lengths in enumerate(sets):
            gates, scale = _UPSTREAM_GATES[i % 3], (1.0, 0.0)[i % 2]
            tests.append(
                (
                    f"{PREFILL_TESTS}::test_prefill_kernel_{name}[bfloat16-{lengths}"
                    f"-(2,2,4)-128-False-scale={scale or 'auto'}-{gates}]",
                    {
                        "lengths": lengths,
                        "gates": gates,
                        "scale": scale,
                        "zero_state": True,
                    },
                )
            )
    for length in (256, 255):
        tests.append(
            (
                f"{PREFILL_TESTS}::test_prefill_kernel_zero_length_sequence"
                f"[bfloat16-{length}-False]",
                {
                    "lengths": [length, 0],
                    "gates": "uniform",
                    "scale": 0.1,
                    "zero_state": True,
                },
            )
        )
    tests.append(
        (
            f"{PREFILL_TESTS}::test_prefill_zero_length_sequence_state_untouched"
            "[bfloat16-False]",
            {"lengths": [256, 0], "gates": "uniform", "scale": 0.1, "zero_state": True},
        )
    )
    tests.append(
        (
            f"{PREFILL_TESTS}::test_prefill_block_end_decay",
            {
                "lengths": [64, 111, 192],
                "gates": "near_one",
                "scale": 1.0,
                "zero_state": True,
            },
        )
    )
    cake = {"gates": "uniform", "v_std": 0.1, "state_std": 0.05, "zero_state": False}
    for name, lengths in (
        ("raw_ffi_bf16_indexed_checkpoint_b7_t421", [52, 93, 15, 107, 72, 61, 21]),
        ("api_bf16_indexed_checkpoint_b5_t4296", [849, 835, 862, 897, 853]),
        ("matches_independent_recurrence[seq_lens=(128,)]", [128]),
        ("checkpoint_is_cuda_graph_safe", [64]),
        ("matches_independent_recurrence[seq_lens=(1,)]", [1]),
        ("raw_abi_invalid_slot_is_noop[seq_lens=(128, 192, 64)]", [128, 192, 64]),
    ):
        tests.append(
            (
                f"{CAKE_PREFILL_TESTS}::test_public_cake_gdn_prefill_{name}",
                {"lengths": lengths, **cake},
            )
        )
    decode = {"gates": "small", "q": "randn", "state_std": 1.0, "zero_state": False}
    for batch in (1, 4, 16, 32, 512):
        tests.append(
            (
                f"{DECODE_TESTS}::test_decode_kernel_basic_{{pre,non}}transpose / "
                f"pretranspose_pool / output_state_indices [float32 state, batch={batch}]; "
                f"{CAKE_DECODE_TESTS}::test_public_cake_gdn_fp32_*_t1_* (batch 1, 32)",
                {"batch": batch, "scale": 1.0, **decode},
            )
        )
    for tokens in (2, 4, 8):
        for batch in (1, 4, 8, 16, 64):
            tests.append(
                (
                    f"{DECODE_TESTS}::test_mtp_fp32_state_with_cache_and_state_update"
                    f"[bfloat16-{batch}-{tokens}]; {CAKE_DECODE_TESTS}::"
                    "test_public_cake_gdn_fp32_mtp_rows_* (subset)",
                    {"lengths": [tokens] * batch, "scale": 0.0, **decode},
                )
            )
    return tests


UPSTREAM_TESTS = _upstream_tests()


def _upstream_label(params: dict[str, Any]) -> str:
    lengths = case_lengths(params)
    if "batch" in params:
        shape = f"decode_b{params['batch']}"
    elif len(set(lengths)) == 1 and len(lengths) > 1:
        shape = f"b{len(lengths)}x{lengths[0]}"
    elif len(lengths) <= 3:
        shape = "l" + "_".join(map(str, lengths))
    else:
        shape = f"b{len(lengths)}_t{sum(lengths)}"
    scale = params.get("scale", 0.0)
    parts = [
        "upstream",
        shape,
        params.get("gates", "model"),
        "zero" if params.get("zero_state") else "init",
        f"s{scale:g}" if scale else "sauto",
    ]
    if params.get("q", "unit") != "unit":
        parts.append("q" + params["q"])
    if "v_std" in params or "state_std" in params:
        parts.append(f"v{params.get('v_std', 1.0):g}_h{params.get('state_std', 0.1):g}")
    return "_".join(parts)


def smoke_cases() -> list[CaseSpec]:
    """Smoke cases of both kernels (the gates kernel skips direct-gate cases).

    Own cases first (pre-existing ones kept), then one case per distinct
    upstream parametrization (``UPSTREAM_TESTS``)."""
    pair_edges = [63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257]
    own = [
        # Former gdn_decode cases.
        CaseSpec(
            "decode_b1_zero_state_wide",
            {"batch": 1, "zero_state": True, "gates": "wide"},
            1,
        ),
        CaseSpec("decode_b3_wide", {"batch": 3, "gates": "wide"}, 3),
        CaseSpec("decode_b16_model", {"batch": 16}, 16),
        # More decode CTAs than SMs: every CTA runs several tiles.
        CaseSpec("decode_b200_model_scale1", {"batch": 200, "scale": 1.0}, 200),
        # Former gdn_prefill cases.
        CaseSpec("prefill_single", {"lengths": [1], "gates": "wide"}, 1),
        CaseSpec("prefill_ragged", {"lengths": [7, 1, 17], "gates": "wide"}, 2),
        CaseSpec("prefill_chunk_boundary", {"lengths": [65, 129], "gates": "wide"}, 3),
        # 64-token chunks are processed in pairs: every partial-pair edge, 192
        # tiles (> 148 CTAs), model gates.
        CaseSpec("prefill_pair_edges_model", {"lengths": pair_edges}, 4),
        # Empty sequences first, in the middle and last (state passes through).
        CaseSpec(
            "prefill_empty_seqs", {"lengths": [0, 37, 0, 200, 0], "gates": "wide"}, 5
        ),
        # Long-tailed batch: 48 sequences (768 tiles), several length-1.
        CaseSpec(
            "prefill_skewed_b48_model",
            {"lengths": skewed_lengths(3000, 48, seed=6, sigma=1.5)},
            6,
        ),
        # One long sequence with long-memory gates: 24 chunks of accumulation.
        CaseSpec("prefill_long_1536_model", {"lengths": [1536]}, 7),
        CaseSpec(
            "prefill_zero_state_model", {"lengths": [300, 77, 1], "zero_state": True}, 8
        ),
        CaseSpec(
            "prefill_q_randn_scale1_wide",
            {"lengths": [200, 31], "gates": "wide", "q": "randn", "scale": 1.0},
            9,
        ),
    ]
    # One case per distinct parametrization; tests sharing one are joined.
    groups: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for test, params in UPSTREAM_TESTS:
        key = json.dumps(params, sort_keys=True)
        groups.setdefault(key, (params, []))[1].append(test)
    cases, labels = [], set()
    for index, (params, tests) in enumerate(groups.values()):
        label = _upstream_label(params)
        if label in labels:
            label += f"_{index}"
        labels.add(label)
        cases.append(
            upstream_case(
                label,
                params,
                " | ".join(tests),
                seed=100 + index,
                revision=UPSTREAM_REVISION,
            )
        )
    return own + cases


QWEN3_NEXT = "qwen3_next_80b_a3b"
LAYER = "linear_attention (Gated DeltaNet), TP=4 shard: 4/4/8 heads, d=128"
# (total tokens, sequences) of official gdn_prefill rows: the smallest, the
# median (139 tokens), single long sequences and the 8192-token batches.
PREFILL_ROWS = (
    (6, 1),
    (35, 2),
    (82, 3),
    (139, 3),
    (401, 4),
    (525, 1),
    (983, 2),
    (2107, 1),
    (3028, 5),
    (3999, 13),
    (4124, 15),
    (5709, 2),
    (8192, 20),
    (8192, 32),
    (8192, 57),
)


def throughput_cases() -> list[CaseSpec]:
    """Throughput (benchmark) cases of both kernels.

    Official rows of both inventories (decode: every batch size; prefill:
    small latency-bound to 8192-token rows, skewed per-sequence lengths since
    the inventory does not record them), and Qwen3-Next stress shapes. Heads
    are independent, so batch B here equals TP1 batch B / 4 (32 value heads).
    """
    cases = [
        trace("gdn_decode", f"decode_b{b}", {"batch": b}, {"batch_size": b})
        for b in (1, 4, 8, 16, 32, 48, 64)
    ]
    for total, n in PREFILL_ROWS:
        lengths = skewed_lengths(total, n, seed=total + n) if n > 1 else [total]
        cases.append(
            trace(
                "gdn_prefill",
                f"prefill_t{total}_n{n}",
                {"lengths": lengths},
                dict(num_seqs=n, len_cu_seqlens=n + 1, total_seq_len=total),
            )
        )
    cases += [
        synthetic(
            "decode_b256",
            {"batch": 256},
            "128 MiB recurrent state read and written; 4096 tiles over 148 SMs",
        ),
        model_case(
            "decode_b512", {"batch": 512}, QWEN3_NEXT, LAYER + "; TP1 batch 128"
        ),
        model_case(
            "decode_b1024", {"batch": 1024}, QWEN3_NEXT, LAYER + "; TP1 batch 256"
        ),
        model_case(
            "prefill_t16384_n64",
            {"lengths": skewed_lengths(16384, 64, seed=64, sigma=1.5)},
            QWEN3_NEXT,
            LAYER + "; 16k-token chunked prefill, long-tailed lengths",
        ),
        model_case(
            "prefill_t32768_n128",
            {"lengths": skewed_lengths(32768, 128, seed=128, sigma=2.0)},
            QWEN3_NEXT,
            LAYER + "; 32k tokens: a few long and many short sequences",
        ),
        model_case(
            "prefill_t32768_n1",
            {"lengths": [32768]},
            QWEN3_NEXT,
            LAYER + "; one 32k-token long-context sequence (16 tiles, 512 chunks)",
        ),
    ]
    return cases

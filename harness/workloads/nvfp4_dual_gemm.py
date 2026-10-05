"""NVFP4 dual GEMM ``half(silu(A @ B1^T) * (A @ B2^T))``, split into its kernels.

The previous package ran, per call: ``pack_scales`` x 4 (SFA, SFB1, SFA again,
SFB2) -> FlashInfer's CUTLASS SM100 block-scaled GEMM x 2 (the same kernel for
B1 and B2, FP32 outputs) -> ``silu_product`` (see the live/dead table in
``impls/nvfp4_dual_gemm/compiler.py``). Each live kernel is one workload:

* ``nvfp4_dual_gemm_pack_scales`` (sm_86, sm_100a): logical E4M3 scales
  ``[rows, k/16, l]`` -> CUTLASS's blocked ``Sm1xxBlkScaledConfig`` layout,
  flat ``[l, padded_rows * padded_cols]`` (``pack_scales_reference``).
* ``nvfp4_dual_gemm_gemm`` (sm_100a): ``h = alpha * (A * SFA) @ (B * SFB)^T``
  in FP32, ``[m, n, l]``, taking already packed scales.
* ``nvfp4_dual_gemm_silu_product`` (sm_86, sm_100a): ``half(silu(h1) * h2)``.

Mid-pipeline inputs are produced with torch versions of the earlier stages
(``pack_scales_reference``, ``quantization.fp4_matmul``). Tensor conventions
follow ``harness/workloads/quantization.py``: packed E2M1 as uint8 ``[rows,
k/2, l]`` views of contiguous ``[l, rows, k/2]`` storage, and batch-major
``[m, n, l]`` outputs. The GEMM is the FP32-output instantiation of
nvfp4_gemm's template, so its by-value CUTLASS ``Params`` comes from the same
builder (``nvfp4_gemm.CutlassFp4Gemm``) with this package's probe sidecar
(``cubins/sm_100a/nvfp4_dual_gemm_gemm.json``); ``tests/test_nvfp4_dual_gemm.py``
checks it byte for byte against the probe's fixtures.

Cases are shared by the three stages (same seeded tensors). The default
distribution is the task generator's (``TASK_DIST``: 0xBB-masked E2M1, U[0, 1)
scales with zeros and subnormals); ``alpha`` (FlashInfer's ``global_sf``,
1 in the task) is a GEMM input, applied to both GEMMs of a case.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from ..cutlass_host import ceil_div
from ..registry import register
from ..throughput import gemm_case, model_case
from ..workload import ROOT, CaseSpec, Workload
from .nvfp4_gemm import (
    CutlassFp4Gemm,
    LaunchConfig,
    fake_encode,
    fp4_case,
    task_cases,
    task_shapes,
)
from .quantization import (
    SF_COLS,
    SF_ROWS,
    fp4_matmul,
    from_blocked_scales,
    make_fp4,
    to_blocked_scales,
)

__all__ = ["LaunchConfig", "fake_encode"]

INT32_MAX = (1 << 31) - 1
BLOCK = 256  # threads per block of both glue kernels (as previously launched)
TASK = ROOT / "resources/nvfp4_dual_gemm/task.yml"
# The task generator's distributions (reference.py): 0xBB-masked E2M1 bytes,
# U[0, 1) scales rounded to E4M3 (quantization.make_fp4).
TASK_DIST = {"values": "restricted", "scales": "unit"}
# Per-case GPU memory the sm_86 silu_product stage may use (inputs, reference,
# output and the test's input snapshots: about 36 bytes per output element).
SILU_BUDGET = 5 << 30

# Smoke cases of the pipeline: one tile; batched; AlongN; partial M/N/K tiles
# with 6 scale columns (a partial scale atom); partial tiles at l = 3 with
# alpha != 1; a 9-tile K loop (beyond the 5-stage pipeline) ending in a partial
# tile; 195 tiles (> 148 SMs: CLC hands CTAs further tiles); m = 1; FP32 rows
# with n % 8 == 4; unrestricted E2M1 with integer scales (exact FP32 sums;
# alpha = 2**-6, exact, keeps silu(h1) * h2 within FP16); three-binade scales.
SMOKE = (
    fp4_case(128, 128, 128, seed=128),
    fp4_case(128, 256, 256, 2, seed=256),
    fp4_case(256, 128, 512, seed=512),
    fp4_case(200, 136, 96, seed=1),
    fp4_case(300, 264, 384, 3, seed=2, alpha=0.37),
    fp4_case(129, 1032, 2080, 2, seed=3),
    fp4_case(640, 1544, 768, 3, seed=4),
    fp4_case(1, 520, 512, seed=5),
    fp4_case(17, 2052, 4096, seed=6),
    fp4_case(200, 136, 96, seed=7, values="full", scales="int0_3", alpha=0.015625),
    fp4_case(192, 520, 1024, 2, seed=8, scales="uniform"),
)


def dual_gemm_cases() -> list[CaseSpec]:
    """The pipeline's cases: smoke, every task test line, throughput."""
    return [*SMOKE, *task_cases(TASK, "nvfp4_dual_gemm"), *throughput_cases()]


def dual_gemm_inputs(workload: Workload, case: CaseSpec) -> tuple:
    """The dual-GEMM inputs ``(a, b1, b2, sa, s1, s2, alpha)`` of ``case``."""
    p, g = case.params, workload.generator(case)
    dist = {key: p.get(key, TASK_DIST[key]) for key in TASK_DIST}
    a, sa = make_fp4(workload, g, p["m"], p["k"], p["l"], **dist)
    b1, s1 = make_fp4(workload, g, p["n"], p["k"], p["l"], **dist)
    b2, s2 = make_fp4(workload, g, p["n"], p["k"], p["l"], **dist)
    # alpha as the FP32 value the GEMM reads (and the references multiply by).
    alpha = torch.tensor(p.get("alpha", 1.0), dtype=torch.float32).item()
    return a, b1, b2, sa, s1, s2, alpha


def packed_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    """(padded rows, padded columns) of one batch of packed scales."""
    return ceil_div(rows, SF_ROWS) * SF_ROWS, ceil_div(cols, SF_COLS) * SF_COLS


def pack_scales_reference(scales: torch.Tensor) -> torch.Tensor:
    """Torch version of the ``pack_scales`` kernel: ``scales [rows, cols, l]``
    (any strides) -> contiguous ``[l, padded_rows * padded_cols]``, the flat
    view of ``quantization.to_blocked_scales``."""
    return to_blocked_scales(scales).reshape(scales.shape[2], -1)


def unpack_scales_reference(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``pack_scales_reference``: ``[rows, cols, l]`` logical scales."""
    padded_rows, padded_cols = packed_scale_shape(rows, cols)
    blocked = packed.view(
        packed.shape[0], padded_rows // SF_ROWS, padded_cols // SF_COLS, 32, 4, 4
    )
    return from_blocked_scales(blocked, rows, cols)


def check_tensor(
    tensor: torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    strides: Sequence[int] | None = None,
) -> None:
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(
            f"expected {dtype} {tuple(shape)}, got {tensor.dtype} {tuple(tensor.shape)}"
        )
    if strides is None:
        if not tensor.is_contiguous():
            raise ValueError("expected a contiguous tensor")
        return
    for size, actual, expected in zip(tensor.shape, tensor.stride(), strides):
        if size > 1 and actual != expected:
            raise ValueError(
                f"expected strides {tuple(strides)}, got {tuple(tensor.stride())}"
            )


def batched_strides(rows: int, cols: int) -> tuple[int, int, int]:
    """Strides of a ``[rows, cols, l]`` view of contiguous ``[l, rows, cols]``."""
    return cols, 1, rows * cols


def empty_batched(
    rows: int, cols: int, batches: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """``[rows, cols, l]`` view of contiguous ``[l, rows, cols]`` storage."""
    return torch.empty((batches, rows, cols), dtype=dtype, device=device).permute(
        1, 2, 0
    )


def check_int32(**values: int) -> None:
    for name, value in values.items():
        if value > INT32_MAX:
            raise ValueError(f"{name} = {value} exceeds the kernels' int32 range")


@register(name="nvfp4_dual_gemm_gemm", supported_arches=("sm_100a",))
class NVFP4DualGEMMGemm(CutlassFp4Gemm):
    """One FP32-output GEMM of the dual GEMM (the B1 GEMM's data).

    Wraps ``genericFp4GemmKernelLauncher<float, 128, 128, 256, 1, 1, 1, _1SM>``
    (128x128x256 block-scaled tcgen05 tile, dynamic cluster (1,1,1), CLC
    persistent scheduler). The B2 GEMM is the same launch with B2/SFB2 and
    another output. Inputs ``(a, b, sfa, sfb, alpha)``: uint8 ``a [m, k/2, l]``
    and ``b [n, k/2, l]`` (two E2M1 values per byte, K contiguous), packed
    E4M3 scales ``[l, padded_rows * padded_cols]`` (``pack_scales_reference``)
    and the FP32 device scalar alpha. Output FP32 ``[m, n, l]`` (batch-major
    storage). Requires k % 32 == 0 and n % 4 == 0 (16-byte TMA strides).
    """

    package = "nvfp4_dual_gemm"
    OUT_DTYPE = torch.float32

    def operands(self, inputs, out):
        a, b, sfa, sfb, _ = inputs
        m, half_k, batches = a.shape
        n, k = b.shape[0], 2 * half_k
        rows_a, cols = packed_scale_shape(m, k // 16)
        rows_b, _ = packed_scale_shape(n, k // 16)
        check_tensor(a, (m, half_k, batches), torch.uint8, batched_strides(m, half_k))
        check_tensor(b, (n, half_k, batches), torch.uint8, batched_strides(n, half_k))
        check_tensor(sfa, (batches, rows_a * cols), torch.float8_e4m3fn)
        check_tensor(sfb, (batches, rows_b * cols), torch.float8_e4m3fn)
        check_tensor(out, (m, n, batches), torch.float32, batched_strides(m, n))
        return m, n, k, batches

    # -- workload contract ---------------------------------------------------------

    def get_cases(self):
        return dual_gemm_cases()

    def get_inputs(self, case):
        a, b1, _, sa, s1, _, alpha = dual_gemm_inputs(self, case)
        return (
            a,
            b1,
            pack_scales_reference(sa),
            pack_scales_reference(s1),
            torch.full((1,), alpha, device=self.device),
        )

    def get_reference(self, inputs):
        a, b, sfa, sfb, alpha = inputs
        k = 2 * a.shape[1]
        sa = unpack_scales_reference(sfa, a.shape[0], k // 16)
        sb = unpack_scales_reference(sfb, b.shape[0], k // 16)
        return (fp4_matmul(a, b, sa, sb, alpha=float(alpha.item())),)

    def validate(self, ref, impl):
        # Both accumulate exactly representable E2M1 x E4M3 products in FP32
        # (with integer scales exactly, in any order). With U[0, 1) or
        # three-binade scales only the summation order differs: FP32 sums in
        # forward/reversed K order stay within 0.13 of this tolerance of the
        # FP64 sum (all distributions, k up to 53248). The task's tolerance.
        self.assert_close(ref, impl, rtol=1e-3, atol=1e-3)


@register(name="nvfp4_dual_gemm_pack_scales", supported_arches=("sm_86", "sm_100a"))
class NVFP4DualGEMMPackScales(Workload):
    """``pack_scales``: logical E4M3 scales -> CUTLASS's blocked layout.

    Input ``scales [rows, k/16, l]`` (a view of contiguous ``[l, rows, k/16]``
    storage, as produced by ``make_fp4``); output contiguous
    ``[l, padded_rows * padded_cols]`` (``pack_scales_reference``). Each dual
    GEMM case yields an SFA (rows = m) and an SFB (rows = n) case; the previous
    package launched this kernel for SFA, SFB1, SFA and SFB2 on every call.
    """

    package = "nvfp4_dual_gemm"

    def get_cases(self):
        cases = []
        for case in dual_gemm_cases():
            p = case.params
            for operand, rows in (("sfa", p["m"]), ("sfb", p["n"])):
                cases.append(
                    CaseSpec(
                        f"{case.name}_{operand}",
                        dict(operand=operand, rows=rows, cols=p["k"] // 16, **p),
                        case.seed,
                        case.suite,
                        case.source,
                    )
                )
        return cases

    def get_inputs(self, case):
        _, _, _, sa, s1, _, _ = dual_gemm_inputs(self, case)
        return (sa if case.params["operand"] == "sfa" else s1,)

    def get_reference(self, inputs):
        return (pack_scales_reference(inputs[0]),)

    def setup_launch(self, inputs):
        """(outputs, grid, args): checks and allocation, outside the launch."""
        (scales,) = inputs
        rows, cols, batches = scales.shape
        check_tensor(
            scales, (rows, cols, batches), torch.float8_e4m3fn, (cols, 1, rows * cols)
        )
        padded_rows, padded_cols = packed_scale_shape(rows, cols)
        total = batches * padded_rows * padded_cols
        check_int32(total=total + BLOCK, input=batches * rows * cols)
        out = torch.empty(
            (batches, padded_rows * padded_cols),
            dtype=scales.dtype,
            device=scales.device,
        )
        args = [
            scales,
            out,
            ctypes.c_int(rows),
            ctypes.c_int(cols),
            ctypes.c_int(batches),
        ]
        return (out,), ceil_div(total, BLOCK), args

    def run(self, inputs):
        outputs, grid, args = self.setup_launch(inputs)
        self.launch(grid, BLOCK, args)
        return outputs

    def prepare(self, inputs):
        outputs, grid, args = self.setup_launch(inputs)

        def launch() -> tuple:
            self.launch(grid, BLOCK, args)
            return outputs

        return launch, outputs

    def validate(self, ref, impl):
        # A byte permutation: exact.
        self.assert_close(
            tuple(t.view(torch.uint8) for t in ref),
            tuple(t.view(torch.uint8) for t in impl),
            rtol=0,
            atol=0,
        )


def silu_bytes(params: dict) -> int:
    """GPU memory of one silu_product case (see ``SILU_BUDGET``)."""
    m, n, k, batches = (params[key] for key in ("m", "n", "k", "l"))
    return batches * (36 * m * n + 4 * m * k)


@register(name="nvfp4_dual_gemm_silu_product", supported_arches=("sm_86", "sm_100a"))
class NVFP4DualGEMMSiluProduct(Workload):
    """``silu_product``: ``half(silu(h1) * h2)`` over FP32 ``[m, n, l]``.

    Inputs are the two GEMM outputs, generated with the torch reference GEMM
    (``fp4_matmul``) from the dual GEMM case; all three tensors use the
    batch-major ``[m, n, l]`` storage of ``empty_fp4_output``, which the kernel
    treats as ``m * n * l`` contiguous elements. Cases: the pipeline's cases
    within the sm_86 budget (``SILU_BUDGET``; the largest model throughput
    shapes are GEMM-only).
    """

    package = "nvfp4_dual_gemm"

    def get_cases(self):
        return [c for c in dual_gemm_cases() if silu_bytes(c.params) <= SILU_BUDGET]

    def get_inputs(self, case):
        a, b1, b2, sa, s1, s2, alpha = dual_gemm_inputs(self, case)
        h1 = fp4_matmul(a, b1, sa, s1, alpha=alpha)
        h2 = fp4_matmul(a, b2, sa, s2, alpha=alpha)
        return h1, h2

    def get_reference(self, inputs):
        h1, h2 = inputs
        out = empty_batched(*h1.shape, torch.float16, h1.device)
        out.copy_(F.silu(h1) * h2)
        return (out,)

    def setup_launch(self, inputs):
        """(outputs, grid, args): checks and allocation, outside the launch."""
        h1, h2 = inputs
        m, n, batches = h1.shape
        strides = batched_strides(m, n)
        check_tensor(h1, (m, n, batches), torch.float32, strides)
        check_tensor(h2, (m, n, batches), torch.float32, strides)
        size = m * n * batches
        check_int32(size=size + BLOCK)
        out = empty_batched(*h1.shape, torch.float16, h1.device)
        return (out,), ceil_div(size, BLOCK), [h1, h2, out, ctypes.c_int(size)]

    def run(self, inputs):
        outputs, grid, args = self.setup_launch(inputs)
        self.launch(grid, BLOCK, args)
        return outputs

    def prepare(self, inputs):
        outputs, grid, args = self.setup_launch(inputs)

        def launch() -> tuple:
            self.launch(grid, BLOCK, args)
            return outputs

        return launch, outputs

    def validate(self, ref, impl):
        # FP32 silu with expf versus torch's FP32 silu, then FP16 rounding: at
        # most one FP16 ulp (2^-10 relative) apart.
        self.assert_close(ref, impl, rtol=1e-3, atol=1e-3)


THROUGHPUT = "nvfp4_dual_gemm"

# (model, MLP, m values): the gate/up projection of each MLP as the dual GEMM
# (n = intermediate size, k = hidden size); dense MLPs from decode to 16k-token
# prefill, MoE experts at per-expert token counts.
MODEL_PLAN = (
    ("deepseek_v3", "dense", (1, 16, 4096, 16384)),
    ("llama3_70b", "dense", (1, 16, 4096, 16384)),
    ("llama3_405b", "dense", (1, 16, 4096, 16384)),
    ("qwen3_32b", "dense", (1, 16, 4096, 16384)),
    ("llama4_maverick", "dense", (1, 16, 4096, 16384)),  # shared expert
    ("deepseek_v3", "expert", (16, 512, 4096)),
    ("qwen3_235b_a22b", "expert", (16, 512, 4096)),
    ("gpt_oss_120b", "expert", (16, 512, 4096)),
    ("llama4_maverick", "expert", (16, 512, 4096)),
    ("mixtral_8x7b", "expert", (16, 512, 4096)),
)


def throughput_cases():
    """Throughput (benchmark) cases: every task benchmark line and the MLP
    gate/up projections of DeepSeek-V3, Llama-3.1 70B/405B, Qwen3-32B/235B,
    Llama-4 Maverick, gpt-oss-120b and Mixtral."""
    from ..models import MODELS

    name = THROUGHPUT
    source = {
        "kind": "official_shape",
        "snapshot": f"resources/{name}/task.yml",
        "url": f"https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/{name}/task.yml",
        "selection": "All benchmark (not just test) shapes; generated using the harness quantization contract.",
    }
    cases = [
        gemm_case(name, (s["m"], s["n"], s["k"], s["l"]), source)
        for s in task_shapes(TASK, "benchmarks")
    ]
    for model, mlp, ms in MODEL_PLAN:
        spec = MODELS[model]
        n = spec.intermediate if mlp == "dense" else spec.moe_intermediate
        layer = "gate_up" if mlp == "dense" else "expert_gate_up"
        for m in ms:
            cases.append(
                model_case(
                    f"{model}_{layer}_m{m}",
                    dict(m=m, n=n, k=spec.hidden, l=1),
                    model,
                    layer,
                )
            )
    return cases

"""RMSNorm (hidden 7168, BF16) as a single-kernel workload.

Official definition ``rmsnorm_h7168`` (``resources/rmsnorm.json``)::

    y = (x.float() * rsqrt(mean(x.float() ** 2, -1) + 1e-6)) * weight.float()
    output = y.to(bfloat16)          # x: [batch, 7168] bf16, weight: [7168] bf16

One kernel, one cubin, one launch: FlashInfer ``norm::RMSNormKernel<8,
__nv_bfloat16>`` (``impls/rmsnorm/compiler.py`` documents the live/dead
kernel decisions). The launch mirrors FlashInfer's host ``norm::RMSNorm``:

* ``vec_size = gcd(16 / sizeof(bf16), d) = 8`` (the compiled instantiation),
* grid ``(batch, 1, 1)``, block ``(32, ceil(min(1024, d / vec_size) / 32), 1)
  = (32, 28, 1)``, dynamic shared memory ``28 * sizeof(float) = 112`` bytes,
  and ``cudaFuncAttributeMaxDynamicSharedMemorySize`` set to that size,
* arguments ``(input, weight, output, d, input.stride(0), output.stride(0),
  weight_bias = 0, eps = 1e-6)`` as ``(ptr, ptr, ptr, u32, u32, u32, f32, f32)``,
* programmatic stream serialization (PDL) on sm_90+ only, as
  ``flashinfer.norm.rmsnorm(enable_pdl=None)`` resolves it.

The compile-stage probe ran upstream ``norm::RMSNorm`` with intercepted launch
calls and recorded these constants and example parameter bytes in
``cubins/<arch>/rmsnorm.json``; the workload checks its formula against the
sidecar constants and ``tests/test_rmsnorm.py`` byte-compares the arguments.
"""

from __future__ import annotations

import ctypes
import json
import math
from functools import cached_property
from typing import Any, NamedTuple

import torch

from .. import cuda_driver
from ..registry import register
from ..throughput import model_case, trace, upstream_case
from ..workload import CaseSpec, Workload, arch_capability

HIDDEN = 7168
EPS = 1e-6
WEIGHT_BIAS = 0.0
# cudaFuncAttributeMaxDynamicSharedMemorySize, as recorded by the probe.
CUDA_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE = 8
U32_MAX = 2**32 - 1
GRID_X_MAX = 2**31 - 1


class LaunchSpec(NamedTuple):
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_mem: int
    args: tuple[Any, ...]
    pdl: bool


def flashinfer_launch_shape(
    d: int, element_size: int = 2
) -> tuple[int, tuple[int, int, int], int]:
    """(vec_size, block, dynamic smem) exactly as ``norm::RMSNorm`` computes them."""
    vec_size = math.gcd(16 // element_size, d)
    block_size = min(1024, d // vec_size)
    num_warps = -(-block_size // 32)
    return vec_size, (32, num_warps, 1), num_warps * 4  # sizeof(float)


def flashinfer_enable_pdl(arch: str) -> bool:
    """``flashinfer.utils.device_support_pdl``: compute capability major >= 9."""
    return arch_capability(arch)[0] >= 9


def build_launch(
    arch: str,
    batch: int,
    input_ptr: int,
    weight_ptr: int,
    output_ptr: int,
    stride_input: int,
    stride_output: int,
) -> LaunchSpec:
    """The full launch FlashInfer's ``norm::RMSNorm`` performs; no GPU needed."""
    if not 0 < batch <= GRID_X_MAX:
        raise ValueError(f"rmsnorm: batch {batch} outside [1, {GRID_X_MAX}]")
    for name, stride in (("input", stride_input), ("output", stride_output)):
        if not HIDDEN <= stride <= U32_MAX:
            raise ValueError(f"rmsnorm: {name} row stride {stride} is not a u32 >= d")
        # The kernel indexes rows with u32 arithmetic (blockIdx.x * stride).
        if (batch - 1) * stride + HIDDEN > U32_MAX:
            raise ValueError(f"rmsnorm: {name} offsets exceed the kernel's u32 range")
    _, block, shared_mem = flashinfer_launch_shape(HIDDEN)
    args = (
        ctypes.c_void_p(input_ptr),
        ctypes.c_void_p(weight_ptr),
        ctypes.c_void_p(output_ptr),
        ctypes.c_uint32(HIDDEN),
        ctypes.c_uint32(stride_input),
        ctypes.c_uint32(stride_output),
        ctypes.c_float(WEIGHT_BIAS),
        ctypes.c_float(EPS),
    )
    return LaunchSpec(
        (batch, 1, 1), block, shared_mem, args, flashinfer_enable_pdl(arch)
    )


@register(name="rmsnorm", supported_arches=("sm_86", "sm_100a"))
class RMSNorm(Workload):
    """``out = rmsnorm(x) * weight`` with FlashInfer ``RMSNormKernel<8, bf16>``.

    Cases (``CaseSpec.params``): ``batch`` rows; ``stride`` (input row stride
    in elements, default 7168; larger values make ``hidden_states`` a
    non-contiguous view, as upstream's ``contiguous=False``); ``weight``:
    ``model`` (``1 + 0.25 N(0, 1)``, trained norm weights scatter around 1)
    or ``randn`` (upstream tests); ``x``: ``randn``, ``outliers`` (LLM
    hidden states: 8 channels scaled by 64, plus all-zero rows, whose output
    is exactly 0, and rows scaled by 1000) or ``zeros``.
    """

    package = "rmsnorm"

    def get_cases(self) -> list[CaseSpec]:
        return smoke_cases() + throughput_cases()

    def get_inputs(self, case: CaseSpec) -> tuple:
        g = self.generator(case)
        batch = case.params["batch"]
        stride = case.params.get("stride", HIDDEN)
        mode = case.params.get("x", "randn")
        x = torch.randn(
            (batch, stride), device=self.device, generator=g, dtype=torch.bfloat16
        )
        if mode == "outliers":
            channels = torch.randperm(HIDDEN, device=self.device, generator=g)[:8]
            x[:, channels] *= 64
            x[::7] = 0
            x[3::11] *= 1000
        elif mode == "zeros":
            x.zero_()
        x = x[:, :HIDDEN]
        if case.params.get("weight", "model") == "model":
            weight = self.randn((HIDDEN,), g, scale=0.25) + 1
        else:
            weight = self.randn((HIDDEN,), g)
        return x, weight.to(torch.bfloat16)

    def get_reference(self, inputs: tuple) -> tuple:
        """The definition's formula, in row blocks so that FP32 temporaries
        stay small for the 64k-row stress cases."""
        x, weight = inputs
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        w = weight.float()
        for start in range(0, x.shape[0], REFERENCE_ROWS):
            xf = x[start : start + REFERENCE_ROWS].float()
            inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
            out[start : start + REFERENCE_ROWS] = ((xf * inv_rms) * w).to(x.dtype)
        return (out,)

    def validate(self, ref: tuple, impl: tuple) -> None:
        """At most one BF16 rounding step per element, and only rarely.

        Kernel and reference compute the same FP32 formula and differ only in
        the summation order and the ``rsqrt`` rounding, far below one BF16
        step; rounding to BF16 then agrees except for values within that
        distance of a rounding boundary, where they differ by one step,
        ``2^-8 .. 2^-7`` relative: rtol 8e-3 (atol 1e-3 for values near 0).
        Measured on sm_86: 4e-6 of the elements differ, by one step. A count
        bound (0.1% of the elements, plus 8) rejects a systematic one-step
        error, which the element-wise bound alone would admit.
        """
        if not (isinstance(ref, tuple) and isinstance(impl, tuple)):
            raise AssertionError("outputs must be tuples")
        if len(ref) != 1 or len(impl) != 1:
            raise AssertionError("expected one output")
        (expected,), (actual,) = ref, impl
        if not isinstance(actual, torch.Tensor) or (
            actual.shape,
            actual.dtype,
            actual.device,
        ) != (expected.shape, expected.dtype, expected.device):
            raise AssertionError("output shape, dtype or device differs")
        mismatches = 0
        for start in range(0, expected.shape[0], REFERENCE_ROWS):
            e = expected[start : start + REFERENCE_ROWS]
            a = actual[start : start + REFERENCE_ROWS]
            torch.testing.assert_close(a, e, rtol=8e-3, atol=1e-3)
            mismatches += int((a != e).sum())
        allowed = expected.numel() // 1000 + 8
        if mismatches > allowed:
            raise AssertionError(
                f"{mismatches} of {expected.numel()} elements differ (> {allowed})"
            )

    # -- launch ----------------------------------------------------------------

    @cached_property
    def sidecar(self) -> dict[str, Any]:
        if self.arch is None:
            raise RuntimeError("reference-only workload has no launch record")
        record = json.loads(self.cubin_path(self.arch).with_suffix(".json").read_text())
        vec_size, block, shared_mem = flashinfer_launch_shape(HIDDEN)
        launch = record["launch"]
        expected = {
            "hidden": HIDDEN,
            "eps": EPS,
            "weight_bias": WEIGHT_BIAS,
            "vec_size": vec_size,
            "block": list(block),
            "shared_mem": shared_mem,
            "func_attributes": [
                {
                    "attr": CUDA_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_MEMORY_SIZE,
                    "value": shared_mem,
                }
            ],
            "enable_pdl": flashinfer_enable_pdl(self.arch),
        }
        if launch != expected:
            raise ValueError(f"{self.name}: recorded launch {launch} != {expected}")
        return record

    def configure(self, function: cuda_driver.Function) -> None:
        # norm::RMSNorm sets the max dynamic smem attribute before each launch.
        for attribute in self.sidecar["launch"]["func_attributes"]:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                attribute["value"],
            )

    def _check(self, name: str, tensor: Any, shape: tuple[int, ...]) -> None:
        ok = (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype == torch.bfloat16
            and tensor.device.type == "cuda"
            and self.device.index in (None, tensor.device.index)
            and tensor.dim() == len(shape)
            and tuple(tensor.shape) == shape
            and tensor.stride(-1) == 1
            and tensor.data_ptr() % 16 == 0
            # 16-byte vector loads of every row.
            and (tensor.dim() == 1 or tensor.stride(0) % 8 == 0)
        )
        if not ok:
            raise ValueError(
                f"{self.name}: {name} must be a bf16 {shape} CUDA tensor on {self.device} "
                "with unit inner stride, 16-byte aligned rows"
            )

    def setup_launch(self, inputs: tuple) -> tuple[tuple, LaunchSpec]:
        if len(inputs) != 2:
            raise ValueError(f"{self.name}: expected (hidden_states, weight)")
        x, weight = inputs
        if not isinstance(x, torch.Tensor) or x.dim() != 2:
            raise ValueError(f"{self.name}: hidden_states must be [batch, {HIDDEN}]")
        batch = x.shape[0]
        self._check("hidden_states", x, (batch, HIDDEN))
        self._check("weight", weight, (HIDDEN,))
        out = torch.empty((batch, HIDDEN), dtype=x.dtype, device=x.device)
        assert self.arch is not None
        spec = build_launch(
            self.arch,
            batch,
            x.data_ptr(),
            weight.data_ptr(),
            out.data_ptr(),
            x.stride(0),
            out.stride(0),
        )
        return (out,), spec

    def _launch(self, spec: LaunchSpec) -> None:
        self.launch(
            spec.grid,
            spec.block,
            spec.args,
            shared_mem=spec.shared_mem,
            programmatic_serialization=spec.pdl,
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


REFERENCE_ROWS = 4096  # rows per reference/validation block
THROUGHPUT = "rmsnorm"
UPSTREAM_TEST = "tests/utils/test_norm.py::test_norm"
UPSTREAM_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
# test_norm's batch x contiguous grid (contiguous=False: rows of a [batch,
# 2 * hidden] tensor). Upstream runs FP16 at hidden sizes of its own; the
# definition fixes BF16 and 7168, which selects this kernel (vec_size 8).
# specify_out only chooses who allocates the (contiguous) output and
# enable_pdl is a launch attribute (this package follows FlashInfer's
# default: PDL on sm_90+), so neither changes the kernel's work.
UPSTREAM_GRID = tuple(
    (batch, contiguous) for batch in (1, 19, 99, 989) for contiguous in (True, False)
)


def smoke_cases() -> list[CaseSpec]:
    """Small latency rows, upstream test_norm's grid, strided and outlier rows."""
    cases = [
        CaseSpec("batch1", {"batch": 1}, 1),
        CaseSpec("batch7", {"batch": 7}, 7),
        CaseSpec("batch128", {"batch": 128}, 128),
        CaseSpec("batch64_outliers", {"batch": 64, "x": "outliers"}, 64),
        CaseSpec("batch3_zeros", {"batch": 3, "x": "zeros"}, 3),
        # Padded rows (row stride 7168 + 8: 16-byte aligned, not 32).
        CaseSpec("batch33_stride7176", {"batch": 33, "stride": 7176}, 33),
    ]
    for batch, contiguous in UPSTREAM_GRID:
        params: dict[str, Any] = {"batch": batch, "weight": "randn"}
        if not contiguous:
            params["stride"] = 2 * HIDDEN
        label = f"upstream_batch{batch}" + ("" if contiguous else "_noncontiguous")
        test = (
            f"{UPSTREAM_TEST}[batch_size={batch}, contiguous={contiguous}; every "
            "hidden_size with vec_size 8, specify_out, enable_pdl]"
        )
        cases.append(
            upstream_case(label, params, test, seed=batch, revision=UPSTREAM_REVISION)
        )
    return cases


def throughput_cases():
    """Every official inventory row (batch 1 to 14,521; median 48) and
    stress rows of 16k-64k tokens at the definition's hidden size, which is
    DeepSeek-V3's / Kimi-K2's (7168). 64k rows move 0.9 GiB each way."""
    cases = [
        trace(THROUGHPUT, f"batch{b}", {"batch": b}, {"batch_size": b})
        for b in (1, 7, 18, 32, 64, 539, 11949, 14521)
    ]
    cases += [
        model_case(
            f"batch{b}",
            {"batch": b},
            "deepseek_v3",
            f"input_layernorm; {b // 1024}k-token prefill batch",
        )
        for b in (16384, 32768)
    ]
    cases.append(
        model_case(
            "batch65536_outliers",
            {"batch": 65536, "x": "outliers"},
            "kimi_k2",
            "input_layernorm; 64k tokens with outlier channels",
        )
    )
    return cases

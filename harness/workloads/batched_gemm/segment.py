"""FlashInfer CUTLASS segment GEMM with row-major weights (sm_86).

``SegmentGEMMWrapper.run(x, w, batch_size, weight_column_major=False,
seg_indptr)`` on the ``sm80`` backend (FlashInfer 0.7.0): ``y[s] = x[s] @
w[g]`` for every segment ``s`` of ``x`` (rows ``seg_indptr[g]`` to
``seg_indptr[g + 1]``), weights ``[G, K, N]`` (row-major B). The kernels are
the four ``cutlass::Kernel<GemmGrouped<...>>`` instantiations of
``CutlassSegmentGEMMRun`` with ``WEIGHT_LAYOUT = RowMajor``, stripped from the
sm_80 JIT-cache module (``impls/batched_gemm/compiler.py``: live/dead table;
the 2-stage ``MmaPipelined`` mainloop is upstream's sm_86 dispatch, the
4-stage ``MmaMultistage`` one its sm_80 dispatch, whose SASS runs unchanged
on sm_86).

``GemmGrouped::Params`` is rebuilt in Python from the compile-time probe's
layout (``kernels/segment_gemm_rowmajor_probe.cu``); the per-group device
arrays FlashInfer fills with a Triton kernel (``compute_sm80_group_gemm_args``:
problem sizes, pointers, leading dimensions ``x_ld = K``, ``w_ld = N``,
``y_ld = N``) are prepared with torch outside the timed launch. Upstream
launches ``threadblock_count = 4`` persistent CTAs with ``alpha = beta = 1``
and ``C = D = y`` zero-initialised; ``run`` does the same, so the timed
callable of ``prepare`` accumulates into its output (as repeated upstream
calls with a reused ``out`` would). Only the values change: the epilogue
has no data-dependent control flow, so every launch does the same work.

Cases come from the variant index (``impls/batched_gemm/compiler.py``
``segment_cases``): smoke shapes, every row-major sm80 problem of
FlashInfer's ``test_segment_gemm`` and model-shape throughput cases.

Inputs ``(x [M, K], w [G, K, N], seg_indptr [G + 1] int64)`` -> ``y [M, N]``.
Tolerance: inputs are exact in FP32, both sides accumulate in FP32 and round
once; ``rtol``/``atol`` of two output ulps near 1 (BF16 ``1.6e-2``, FP16
``2e-3``).
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable
from functools import cache
from typing import Any, ClassVar, NamedTuple

import torch

from ... import cuda_driver
from ...registry import register_variant
from ...workload import IMPLS, CaseSpec, Workload

PACKAGE_DIR = IMPLS / "batched_gemm"
VARIANT_FILE = PACKAGE_DIR / "variants" / "sm_86.json"
THREADBLOCK_COUNT = 4  # CutlassSegmentGEMMRun


@cache
def variant_index() -> dict[str, Any]:
    if not VARIANT_FILE.is_file():
        return {}
    return json.loads(VARIANT_FILE.read_text())


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
    threadblock_count: int = THREADBLOCK_COUNT,
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
    for name, (code, value) in values.items():
        offset, size = fields[name]
        if struct.calcsize(code) != size:
            raise ValueError(f"{name}: probe size {size} != {code}")
        struct.pack_into("<" + code, buffer, offset, value)
    return bytes(buffer)


def padding_mask(layout: dict[str, Any]) -> list[bool]:
    covered = [False] * layout["params_size"]
    for offset, size in layout["fields"].values():
        covered[offset : offset + size] = [True] * size
    return covered


class LaunchSpec(NamedTuple):
    grid: int
    block: int
    params: bytes
    shared_mem: int


class SegmentGemmRowMajor(Workload):
    """``CutlassSegmentGEMMRun<DType>`` with row-major weights. Class
    attributes from the variant index: ``dtype``, ``sidecar``, ``mainloop``
    and ``upstream_dispatch``."""

    package = "batched_gemm"
    dtype: ClassVar[torch.dtype]
    sidecar: ClassVar[str]
    mainloop: ClassVar[str]
    upstream_dispatch: ClassVar[str]

    cases: ClassVar[list[dict[str, Any]]]

    def get_cases(self) -> list[CaseSpec]:
        return [
            CaseSpec(
                c["name"], dict(c["params"]), c["seed"], c["suite"], c.get("source", {})
            )
            for c in self.cases
        ]

    @property
    def layout(self) -> dict[str, Any]:
        return segment_layout(self.sidecar)

    def get_inputs(self, case: CaseSpec) -> tuple:
        g = self.generator(case)
        lengths, n, k = case.params["lengths"], case.params["n"], case.params["k"]
        x = self.randn((sum(lengths), k), g, self.dtype)
        # One group at a time: no FP32 temporary of all the weights.
        w = torch.empty((len(lengths), k, n), dtype=self.dtype, device=self.device)
        for group in range(len(lengths)):
            w[group] = self.randn((k, n), g, self.dtype, scale=k**-0.5)
        indptr = torch.tensor([0] + lengths, dtype=torch.int64).cumsum(0)
        return x, w, indptr.to(self.device)

    def get_reference(self, inputs: tuple) -> tuple:
        x, w, indptr = inputs
        y = torch.empty((x.shape[0], w.shape[2]), dtype=self.dtype, device=x.device)
        bounds = indptr.tolist()
        for group in range(w.shape[0]):
            lo, hi = bounds[group], bounds[group + 1]
            y[lo:hi] = (x[lo:hi].float() @ w[group].float()).to(self.dtype)
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
        x, w, indptr = inputs
        groups, k, n = w.shape
        if x.dtype != self.dtype or w.dtype != self.dtype:
            raise ValueError(f"x and w must be {self.dtype}")
        if (
            x.dim() != 2
            or x.shape[1] != k
            or not (x.is_contiguous() and w.is_contiguous())
        ):
            raise ValueError("x must be contiguous [M, K] and w [G, K, N]")
        if indptr.dtype != torch.int64 or indptr.shape != (groups + 1,):
            raise ValueError("seg_indptr must be int64 [G + 1]")
        if k % 8 or n % 8:
            raise ValueError("K and N must be multiples of 8 (128-bit accesses)")
        if any(t.device.type != "cuda" for t in inputs):
            raise ValueError("native launch needs CUDA tensors")
        # The upstream epilogue computes y = x w + 1.0 * y: zero-initialise.
        y = torch.zeros((x.shape[0], n), dtype=self.dtype, device=x.device)
        esize = x.element_size()
        # compute_sm80_group_gemm_args (FlashInfer's Triton kernel), in torch.
        starts = indptr[:-1]
        group_ids = torch.arange(groups, device=x.device, dtype=torch.int64)
        problems = torch.stack(
            (
                indptr[1:] - starts,
                torch.full_like(starts, n),
                torch.full_like(starts, k),
            ),
            dim=1,
        ).to(torch.int32)
        x_data = x.data_ptr() + starts * k * esize
        w_data = w.data_ptr() + group_ids * k * n * esize
        y_data = y.data_ptr() + starts * n * esize
        x_ld = torch.full_like(starts, k)
        w_ld = torch.full_like(starts, n)  # row-major weights
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
            THREADBLOCK_COUNT,
            self.layout["threads"],
            params,
            self.layout["shared_storage"],
        )
        return spec, (y,), arrays

    def _launch(self, spec: LaunchSpec) -> None:
        self.launch(spec.grid, spec.block, [spec.params], shared_mem=spec.shared_mem)

    def run(self, inputs: tuple) -> tuple:
        spec, outputs, arrays = self.configure_launch(inputs)
        if inputs[1].shape[0]:  # GemmGrouped::run returns early without problems
            self._launch(spec)
        self._keepalive = arrays  # until the next run (asynchronous launch)
        return outputs

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple | None]:
        spec, outputs, arrays = self.configure_launch(inputs)

        def launch() -> tuple:
            if inputs[1].shape[0]:  # as run: no launch without problems
                self._launch(spec)
            assert arrays  # the device argument arrays live with the closure
            return outputs

        return launch, outputs


def check_param_layouts() -> bool:
    """Byte-compare Params rebuilt in Python with the probe's reference
    structs (padding excluded) for every sm_86 variant."""
    for name, entry in variant_index().items():
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
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    for name, entry in variant_index().items():
        register_variant(
            SegmentGemmRowMajor,
            name=name,
            supported_arches=("sm_86",),
            dtype=dtypes[entry["dtype"]],
            sidecar=entry["sidecar"],
            mainloop=entry["mainloop"],
            upstream_dispatch=entry["upstream_dispatch"],
            cases=entry.get("cases", []),
        )


_register()

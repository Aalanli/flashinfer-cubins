"""deep_gemm: dispatch, launch preparation and cases versus upstream code.

sm_100a: FlashInfer v0.6.9's ``flashinfer/deep_gemm.py`` (from
``resources/flashinfer-<v0.6.9>``) is loaded with stub modules for its package
imports and a fake ``cuda.bindings.driver`` whose ``cuTensorMapEncodeTiled``
records its arguments (as raw ``cuda.h`` enum values). Against it the tests
check that

* ``harness.workloads.deep_gemm.best_config`` equals ``get_best_configs`` for
  every served (layout, N, K, groups) at 148 SMs and at the SM count of each
  dispatch entry, for every ceil(M / 128) in 1..1024;
* FlashInfer's cubin name (``load("fp8_m_grouped_gemm", generate(static
  kwargs))``) for the ends of every dispatch range is that entry's cubin, the
  ranges are maximal, and every published cubin is one entry; cubins merged
  into one workload have identical code;
* ``configure_launch`` builds the same six TMA descriptors (encode
  arguments), scalars, grid, block and shared memory as FlashInfer's
  ``*_kwargs_gen`` for every case (so FlashInfer dispatches every case to its
  workload's code), using a recording encoder (the RTX 3090 has no TMA);
* every parametrization of FlashInfer's deep_gemm tests and benchmark, and
  the replayed DeepGEMM test regimes, is a case of the kernel FlashInfer
  selects for it; FlashInfer's column-major segment GEMM test grid is a case
  of both fp16 sm_86 kernels;
* cases fit their memory budgets and smoke cases reach the hard paths.

sm_86: the stripped cubins match their sidecars (Params layouts are
byte-compared by ``tests/test_workloads.py``'s ``check_param_layouts``).
Nothing here launches a kernel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
import unittest
from functools import cache
from pathlib import Path
from typing import Any
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import get_workload  # noqa: E402
from harness.cutlass_host import ceil_div  # noqa: E402
from harness.models import MODELS  # noqa: E402
from harness.workload import cubin_info  # noqa: E402
from harness.workloads import deep_gemm  # noqa: E402

PACKAGE = ROOT / "impls" / "deep_gemm"
FLASHINFER = ROOT / "resources" / "flashinfer-a1aa676196f798435248d9ea205c67674476f473"
UPSTREAM_SOURCE = FLASHINFER / "flashinfer" / "deep_gemm.py"
NUM_SMS = 148
MAX_M_BLOCKS = 1024

# cuda.h values of the CUtensorMap enum members FlashInfer names.
CUDA_ENUMS = {
    "CU_TENSOR_MAP_DATA_TYPE_UINT8": 0,
    "CU_TENSOR_MAP_DATA_TYPE_UINT16": 1,
    "CU_TENSOR_MAP_DATA_TYPE_UINT32": 2,
    "CU_TENSOR_MAP_DATA_TYPE_INT32": 3,
    "CU_TENSOR_MAP_DATA_TYPE_UINT64": 4,
    "CU_TENSOR_MAP_DATA_TYPE_INT64": 5,
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT16": 6,
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT32": 7,
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT64": 8,
    "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16": 9,
    "CU_TENSOR_MAP_SWIZZLE_NONE": 0,
    "CU_TENSOR_MAP_SWIZZLE_32B": 1,
    "CU_TENSOR_MAP_SWIZZLE_64B": 2,
    "CU_TENSOR_MAP_SWIZZLE_128B": 3,
    "CU_TENSOR_MAP_INTERLEAVE_NONE": 0,
    "CU_TENSOR_MAP_L2_PROMOTION_NONE": 0,
    "CU_TENSOR_MAP_L2_PROMOTION_L2_64B": 1,
    "CU_TENSOR_MAP_L2_PROMOTION_L2_128B": 2,
    "CU_TENSOR_MAP_L2_PROMOTION_L2_256B": 3,
    "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE": 0,
}


class _Enum:
    """``cbd.<EnumType>``: members resolve to their raw cuda.h value."""

    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, member: str) -> Any:
        if member in CUDA_ENUMS:
            return CUDA_ENUMS[member]
        return f"{self.name}.{member}"


class Recorder:
    """Records encode calls as tuples of the cuTensorMapEncodeTiled arguments
    (data type, address, dims, byte strides, box, element strides,
    interleave, swizzle, L2 promotion, OOB fill)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def record(self, *args: Any) -> tuple:
        dtype, address, dims, strides, box, elem, il, sw, l2, oob = args
        entry = (
            int(dtype),
            int(address),
            tuple(int(x) for x in dims),
            tuple(int(x) for x in strides),
            tuple(int(x) for x in box),
            tuple(int(x) for x in elem),
            int(il),
            int(sw),
            int(l2),
            int(oob),
        )
        self.calls.append(entry)
        return entry

    def harness_encode(self, *args: Any) -> bytes:
        """``harness.cutlass_host.driver_encode``'s signature."""
        self.record(*args)
        return bytes(128)

    def flashinfer_encode(self, dtype, rank, address, dims, strides, box, elem, *rest):
        """``cbd.cuTensorMapEncodeTiled``: (CUresult, CUtensorMap)."""
        if rank != len(dims):
            raise AssertionError("rank differs from len(dims)")
        return 0, self.record(dtype, address, dims, strides, box, elem, *rest)


RECORDER = Recorder()


def _fake_driver() -> types.ModuleType:
    driver = types.ModuleType("cuda.bindings.driver")
    vars(driver).update(
        cuuint64_t=int,
        cuuint32_t=int,
        cuTensorMapEncodeTiled=lambda *args: RECORDER.flashinfer_encode(*args),
        __getattr__=_Enum,  # PEP 562: every other name is an enum type
    )
    return driver


def _identity_decorator(*_args: Any, **_kwargs: Any) -> Any:
    return lambda function: function


@cache
def upstream() -> types.ModuleType:
    """FlashInfer v0.6.9 ``flashinfer/deep_gemm.py`` on stubbed imports."""
    package = "_flashinfer_v069"
    stubs: dict[str, types.ModuleType] = {}

    def stub(name: str, **attributes: Any) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        stubs[name] = module
        return module

    stub(package, __path__=[])
    stub(f"{package}.jit", __path__=[])
    stub(
        f"{package}.artifacts",
        ArtifactPath=types.SimpleNamespace(DEEPGEMM="deep-gemm"),
    )
    stub(f"{package}.cuda_utils", checkCudaErrors=lambda result: result[1])
    stub(f"{package}.jit.cubin_loader", get_artifact=lambda *_: True)
    stub(f"{package}.jit.env", FLASHINFER_CUBIN_DIR=Path("/nonexistent"))
    stub(
        f"{package}.utils",
        ceil_div=ceil_div,
        round_up=deep_gemm.round_up,
        supported_compute_capability=_identity_decorator,
        backend_requirement=_identity_decorator,
    )
    driver = _fake_driver()
    bindings = stub("cuda.bindings", driver=driver, __path__=[])
    stub("cuda", bindings=bindings, __path__=[])
    stubs["cuda.bindings.driver"] = driver
    spec = importlib.util.spec_from_file_location(
        f"{package}.deep_gemm", UPSTREAM_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    vars(module)["get_device_arch"] = lambda: "100a"
    return module


def upstream_device(num_sms: int):
    """The CUDA queries of FlashInfer's kwargs generators, faked."""
    properties = types.SimpleNamespace(multi_processor_count=num_sms)
    stream = types.SimpleNamespace(cuda_stream=0)
    return mock.patch.multiple(
        torch.cuda,
        get_device_properties=mock.Mock(return_value=properties),
        current_stream=mock.Mock(return_value=stream),
    )


def fp8_workload(name: str) -> type[deep_gemm._DeepGemmFP8]:
    cls = get_workload(name)
    assert issubclass(cls, deep_gemm._DeepGemmFP8), name
    return cls


def variants() -> dict[str, dict[str, Any]]:
    return deep_gemm.variant_index().get("sm_100a", {})


def manifest() -> dict[str, dict[str, str]]:
    return json.loads((PACKAGE / "kernels.json").read_text())


def gemm_type(fi, layout: str) -> Any:
    return {
        "contiguous": fi.GemmType.GroupedContiguous,
        "masked": fi.GemmType.GroupedMasked,
    }[layout]


def static_kwargs(
    fi, layout: str, n: int, k: int, groups: int, m_blocks: int, num_sms: int
) -> dict:
    """FlashInfer's static kwargs (compiled_dims "nk") for ceil(M/128)."""
    m = m_blocks * 128
    major_k, major_d = fi.MajorTypeAB.KMajor, fi.MajorTypeCD.NMajor
    with upstream_device(num_sms):
        if layout == "contiguous":
            _, kwargs = fi.m_grouped_fp8_gemm_nt_contiguous_static_kwargs_gen(
                m, n, k, k, groups, major_k, major_k, major_d, "nk", torch.bfloat16
            )
        else:
            _, kwargs = fi.m_grouped_fp8_gemm_nt_masked_static_kwargs_gen(
                m, n, k, m, k, groups, major_k, major_k, major_d,
                "nk", torch.bfloat16,
            )  # fmt: skip
    return kwargs


@cache
def flashinfer_cubin(
    layout: str, n: int, k: int, groups: int, m_blocks: int, num_sms: int
) -> str:
    """The cubin FlashInfer loads for the shape (its own config and hash)."""
    fi = upstream()
    kwargs = static_kwargs(fi, layout, n, k, groups, m_blocks, num_sms)
    code = fi.SM100FP8GemmRuntime.generate(kwargs)
    return f"kernel.fp8_m_grouped_gemm.{fi.hash_to_hex('fp8_m_grouped_gemm$$' + code)}"


@cache
def served_by() -> dict[str, tuple[str, int]]:
    """FlashInfer cubin stem -> (workload, group count) serving it."""
    return {
        entry["cubin"]: (name, entry["num_groups"])
        for name, variant in variants().items()
        for entry in variant["dispatch"]
    }


def compiler() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "impls_deep_gemm_compiler", PACKAGE / "compiler.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def meta_inputs(cls, params: dict[str, Any]) -> tuple:
    """Operands of a case's shapes without storage (data_ptr() == 0)."""
    n, k, fp8 = cls.shape_n, cls.shape_k, torch.float8_e4m3fn
    k4 = ceil_div(k, 512)

    def empty(shape, dtype):
        return torch.empty(shape, dtype=dtype, device="meta")

    if cls.layout == "contiguous":
        groups, m = params["groups"], deep_gemm.contiguous_blocks(params) * 128
        return (
            empty((m, k), fp8),
            empty((k4, m), torch.int32),
            empty((groups, n, k), fp8),
            empty((groups, k4, n), torch.int32),
            empty((m,), torch.int32),
        )
    groups, m = cls.num_groups, params["capacity"]
    return (
        empty((groups, m, k), fp8),
        empty((groups, k4, m), torch.int32),
        empty((groups, n, k), fp8),
        empty((groups, k4, n), torch.int32),
        empty((groups,), torch.int32),
        params["expected_m"],
    )


def cases_by_test(name: str) -> dict[str, list[Any]]:
    """Upstream test id -> cases of workload ``name`` recording it."""
    found: dict[str, list[Any]] = {}
    for case in get_workload(name)(None, device="cpu").get_cases():
        test = case.source.get("test")
        if test:
            found.setdefault(test, []).append(case)
    return found


@unittest.skipUnless(UPSTREAM_SOURCE.is_file(), "FlashInfer v0.6.9 source missing")
class FlashInferDispatch(unittest.TestCase):
    def test_best_config_matches_get_best_configs(self):
        fi = upstream()
        major_k, major_d = fi.MajorTypeAB.KMajor, fi.MajorTypeCD.NMajor
        shapes: dict[tuple, set[int]] = {}
        for variant in variants().values():
            for entry in variant["dispatch"]:
                key = (variant["layout"], variant["n"], variant["k"])
                key += (entry["num_groups"],)
                shapes.setdefault(key, {NUM_SMS}).add(entry["sms"])
        self.assertTrue(shapes)
        for (layout, n, k, groups), sms_values in sorted(shapes.items()):
            for sms in sorted(sms_values):
                for m_blocks in range(1, MAX_M_BLOCKS + 1):
                    m = m_blocks * 128
                    ours = deep_gemm.best_config(layout, m, n, k, groups, sms)
                    num_sms, bm, bn, bk, stages, multicast, smem = (
                        fi.get_best_configs(
                            gemm_type(fi, layout), m, n, k, groups,
                            major_k, major_k, major_d,
                            torch.float8_e4m3fn, torch.bfloat16, sms,
                        )
                    )  # fmt: skip
                    theirs = deep_gemm.Config(
                        num_sms,
                        bm,
                        bn,
                        stages,
                        ceil_div(k, bk) % stages,
                        smem.swizzle_cd_mode,
                        smem.smem_size,
                    )
                    if ours != theirs or multicast.num_multicast != 1:
                        self.fail(
                            f"{layout} n={n} k={k} g={groups} sms={sms} m={m}: "
                            f"{ours} != {theirs} (multicast "
                            f"{multicast.num_multicast})"
                        )

    def test_dispatch_selects_registered_cubin(self):
        """Upstream maps the ends of every dispatch range, and nothing next
        to them, to the entry's cubin; the workload's own cubin is its first
        entry's, and its code is the compiler's."""
        fi, build = upstream(), compiler()
        for name, variant in variants().items():
            with self.subTest(workload=name):
                layout, n, k = variant["layout"], variant["n"], variant["k"]
                for entry in variant["dispatch"]:
                    groups, sms, stem = (
                        entry["num_groups"],
                        entry["sms"],
                        entry["cubin"],
                    )
                    for lo, hi in entry["m_blocks"]:
                        for m_blocks in {lo, hi}:
                            self.assertEqual(
                                flashinfer_cubin(layout, n, k, groups, m_blocks, sms),
                                stem,
                            )
                        for outside in (lo - 1, hi + 1):
                            if 1 <= outside <= MAX_M_BLOCKS:
                                self.assertNotEqual(
                                    flashinfer_cubin(
                                        layout, n, k, groups, outside, sms
                                    ),
                                    stem,
                                )
                first = variant["dispatch"][0]
                path = PACKAGE / manifest()["sm_100a"][name]
                self.assertEqual(path.name, first["cubin"] + ".cubin")
                kwargs = static_kwargs(
                    fi, layout, n, k, first["num_groups"], first["m_blocks"][0][0],
                    first["sms"],
                )  # fmt: skip
                symbol = build.decode_symbol(cubin_info(path.read_bytes())[1][0])
                self.assertEqual(
                    fi.SM100FP8GemmRuntime.generate(kwargs),
                    build.flashinfer_code(symbol),
                )

    def test_launch_matches_kwargs_gen(self):
        """TMA descriptors, scalars, grid, block and shared memory of every
        case equal FlashInfer's for the same tensors (so FlashInfer
        dispatches every case to this kernel's code)."""
        fi = upstream()
        major_k = fi.MajorTypeAB.KMajor
        for name in variants():
            cls = fp8_workload(name)
            workload = cls(None, device="cpu")
            recorder = Recorder()
            workload.encode = recorder.harness_encode
            for case in workload.get_cases():
                with self.subTest(workload=name, case=case.name):
                    inputs = meta_inputs(cls, case.params)
                    groups = inputs[2].shape[0]
                    recorder.calls.clear()
                    spec, (d,) = workload.configure_launch(inputs)
                    ours = list(recorder.calls)
                    RECORDER.calls.clear()
                    with upstream_device(cls.entry(groups).sms):
                        if cls.layout == "contiguous":
                            _, kwargs = fi.m_grouped_fp8_gemm_nt_contiguous_kwargs_gen(
                                *inputs[:4], d, inputs[4], major_k, major_k, "nk"
                            )
                        else:
                            _, kwargs = fi.m_grouped_fp8_gemm_nt_masked_kwargs_gen(
                                *inputs[:4], d, *inputs[4:6], major_k, major_k, "nk"
                            )
                    # Upstream encodes A, B, D, SFA, SFB; the harness A, B,
                    # D, SFA, SFB too, then passes D again as C.
                    self.assertEqual(ours, RECORDER.calls)
                    descriptor = {
                        key: kwargs[f"TENSOR_MAP_{key}"]
                        for key in ("A", "B", "SFA", "SFB", "C", "D")
                    }
                    a, b, d_, sfa, sfb = ours
                    self.assertEqual(
                        [a, b, sfa, sfb, d_, d_],
                        [descriptor[k] for k in ("A", "B", "SFA", "SFB", "C", "D")],
                    )
                    scalars = [int(x.value) for x in spec.args[1:4]]
                    self.assertEqual(scalars, [kwargs["M"], kwargs["N"], kwargs["K"]])
                    self.assertIs(spec.args[0], inputs[4])
                    self.assertEqual(spec.grid, kwargs["NUM_SMS"])
                    self.assertEqual(
                        spec.block,
                        kwargs["NUM_NON_EPILOGUE_THREADS"]
                        + kwargs["NUM_EPILOGUE_THREADS"],
                    )
                    self.assertEqual(spec.shared_mem, kwargs["SMEM_SIZE"])
                    self.assertEqual(spec.cluster, (kwargs["NUM_MULTICAST"], 1, 1))

    def test_launch_addresses(self):
        """With real (CPU) tensors every descriptor points at its tensor."""
        for layout in ("contiguous", "masked"):
            name = min(
                (n for n, v in variants().items() if v["layout"] == layout),
                key=lambda n: variants()[n]["n"] * variants()[n]["k"],
            )
            cls = fp8_workload(name)
            workload = cls(None, device="cpu")
            recorder = Recorder()
            workload.encode = recorder.harness_encode
            case = min(workload.get_cases(), key=cls.case_bytes)
            inputs = workload.get_inputs(case)
            _, (d,) = workload.configure_launch(inputs)
            addresses = [call[1] for call in recorder.calls]
            a, sfa, b, sfb = inputs[:4]
            expected = [t.data_ptr() for t in (a, b, d, sfa, sfb)]
            self.assertEqual(addresses, expected, name)


@unittest.skipUnless(UPSTREAM_SOURCE.is_file(), "FlashInfer v0.6.9 source missing")
class UpstreamCoverage(unittest.TestCase):
    """Upstream parametrizations are cases of the kernel FlashInfer's own
    get_best_configs / cubin naming selects for them at 148 SMs."""

    def kernel(self, layout: str, n: int, k: int, groups: int, m_blocks: int):
        stem = flashinfer_cubin(layout, n, k, groups, m_blocks, deep_gemm.UPSTREAM_SMS)
        self.assertIn(stem, served_by(), f"{layout} {n} {k} {groups} {m_blocks}")
        name, served_groups = served_by()[stem]
        self.assertEqual(served_groups, groups)
        return name

    def test_flashinfer_contiguous_test(self):
        count = 0
        for m, groups in deep_gemm.fi_contiguous_tests():
            for n, k in deep_gemm.FI_NK:
                name = self.kernel("contiguous", n, k, groups, m // 128)
                test = deep_gemm.fi_contiguous_test_id(m, n, k, groups)
                cases = cases_by_test(name).get(test, [])
                self.assertTrue(cases, f"{name} lacks {test}")
                for case in cases:
                    self.assertEqual(case.suite, "smoke")
                    self.assertEqual(case.params["groups"], groups)
                    self.assertEqual(case.params["rows"], [m // groups] * groups)
                    self.assertEqual(case.params["slack"], 0)
                count += 1
        self.assertEqual(count, 28)

    def test_flashinfer_masked_test(self):
        """masked_m ~ randint(0, m) on the GPU: expected_m can fall into any
        128-row block up to m, so every kernel it can select is checked."""
        count = 0
        for m in deep_gemm.FI_TEST_M:
            for n, k in deep_gemm.FI_NK:
                for groups in deep_gemm.FI_GROUPS:
                    test = deep_gemm.fi_masked_test_id(m, n, k, groups)
                    for m_blocks in range(1, m // 128 + 1):
                        name = self.kernel("masked", n, k, groups, m_blocks)
                        cases = cases_by_test(name).get(test, [])
                        self.assertTrue(cases, f"{name} lacks {test}")
                        for case in cases:
                            self.assertEqual(case.params["capacity"], m)
                            self.assertLess(max(case.params["rows"]), m)
                            mean = sum(case.params["rows"]) / groups
                            self.assertEqual(
                                case.params["expected_m"], min(int(mean) + 1, m)
                            )
                        count += 1
        self.assertEqual(count, 4 * 6 * (1 + 2 + 4 + 8))

    def test_flashinfer_benchmark(self):
        for groups, m in deep_gemm.fi_contiguous_bench():
            for n, k in deep_gemm.FI_NK:
                name = self.kernel("contiguous", n, k, groups, groups * m // 128)
                test = deep_gemm.fi_contiguous_bench_id(groups, m, n, k)
                (case,) = cases_by_test(name)[test]
                self.assertEqual(case.suite, "throughput")
                self.assertEqual(case.params["rows"], [m] * groups)
        for groups, m in deep_gemm.fi_masked_bench():
            expected = deep_gemm.fi_natural_expected(m)
            for n, k in deep_gemm.FI_NK:
                name = self.kernel("masked", n, k, groups, ceil_div(expected, 128))
                test = deep_gemm.fi_masked_bench_id(groups, m, n, k)
                (case,) = cases_by_test(name)[test]
                self.assertEqual(case.suite, "throughput")
                self.assertEqual(case.params["capacity"], m)
                self.assertEqual(case.params["expected_m"], expected)

    def test_deepgemm_regimes(self):
        for n, k in deep_gemm.PRODUCTION_LAYERS:
            for groups, expected in deep_gemm.DG_CONTIGUOUS:
                rows = deep_gemm.dg_contiguous_rows(groups, expected, n, k)
                blocks = sum(ceil_div(r, 128) for r in rows)
                name = self.kernel("contiguous", n, k, groups, blocks)
                test = deep_gemm.dg_contiguous_id(groups, expected, n, k)
                (case,) = cases_by_test(name)[test]
                self.assertEqual(case.params["rows"], rows)
            for groups, expected in deep_gemm.DG_MASKED:
                rows, expected_m = deep_gemm.dg_masked_rows(groups, expected, n, k)
                if not any(
                    v["layout"] == "masked" and v["dispatch"][0]["num_groups"] == groups
                    for v in variants().values()
                ):
                    self.assertEqual(groups, 6)  # documented: no 6-group kernel
                    continue
                name = self.kernel("masked", n, k, groups, ceil_div(expected_m, 128))
                test = deep_gemm.dg_masked_id(groups, expected, n, k)
                (case,) = cases_by_test(name)[test]
                self.assertEqual(case.params["rows"], rows)
                self.assertEqual(case.params["capacity"], deep_gemm.DG_MASKED_MAX_M)
                self.assertEqual(case.params["expected_m"], expected_m)

    def test_segment_gemm_test(self):
        tests = deep_gemm.segment_tests()
        self.assertEqual(len(tests), 72)
        for dtype in ("fp16",):
            for mainloop in ("pipelined", "multistage"):
                name = f"deep_gemm_segment_{dtype}_{mainloop}"
                found = cases_by_test(name)
                for batch, rows, d_in, d_out in tests:
                    test = deep_gemm.segment_test_id(batch, rows, d_in, d_out)
                    (case,) = found[test]
                    self.assertEqual(case.params["lengths"], [rows] * batch)
                    self.assertEqual(
                        (case.params["n"], case.params["k"]), (d_out, d_in)
                    )


class Cases(unittest.TestCase):
    def test_budgets_and_throughput_cases(self):
        """Every kernel has a throughput case; every case fits its arch's
        budget (inputs + outputs + reference temporaries)."""
        for name in variants():
            cls = fp8_workload(name)
            cases = cls(None, device="cpu").get_cases()
            with self.subTest(workload=name):
                self.assertTrue(any(c.suite == "throughput" for c in cases))
                for case in cases:
                    self.assertLessEqual(cls.case_bytes(case), deep_gemm.CASE_BUDGET)
        for name in deep_gemm.variant_index().get("sm_86", {}):
            cls = get_workload(name)
            for case in cls(None, device="cpu").get_cases():
                self.assertLessEqual(cls.case_bytes(case), deep_gemm.SEGMENT_BUDGET)

    def test_model_cases(self):
        """Model cases route tokens x top-8 rows to the rank's experts of a
        model whose expert GEMM is the kernel's (N, K); every
        production-shape kernel has one."""
        for name in variants():
            cls = fp8_workload(name)
            cases = cls(None, device="cpu").get_cases()
            models = [c for c in cases if c.source.get("kind") == "model_shape"]
            layer = deep_gemm.PRODUCTION_LAYERS.get((cls.shape_n, cls.shape_k))
            with self.subTest(workload=name):
                self.assertEqual(bool(models), layer is not None)
                for case in models:
                    spec = MODELS[case.source["model"]]
                    p = case.params
                    self.assertEqual(case.source["layer"], layer)
                    self.assertEqual(
                        spec.linear_shapes()[layer], (cls.shape_n, cls.shape_k)
                    )
                    groups = (
                        p["groups"] if cls.layout == "contiguous" else cls.num_groups
                    )
                    self.assertEqual(p["ep"] * groups, spec.experts)
                    self.assertLessEqual(p["tokens"], deep_gemm.MODEL_TOKENS)
                    self.assertEqual(sum(p["rows"]), spec.top_k * p["tokens"])
                    if cls.layout == "masked":
                        self.assertEqual(
                            p["expected_m"], ceil_div(spec.top_k * p["tokens"], groups)
                        )

    def test_smoke_paths(self):
        """Smoke cases reach multi-block M, partial blocks, empty groups,
        padding and persistent second tiles wherever the kernel's dispatch
        range allows them."""
        for name in variants():
            cls = fp8_workload(name)
            nb = ceil_div(cls.shape_n, cls.block_n)
            smoke = [
                c for c in cls(None, device="cpu").get_cases() if c.suite == "smoke"
            ]
            with self.subTest(workload=name):
                self.assertGreaterEqual(len(smoke), 2 if cls.layout == "masked" else 1)
                persistent, partial, empty, multi = False, False, False, False
                for case in smoke:
                    p = case.params
                    rows = p["rows"]
                    partial |= any(r % 128 for r in rows)
                    if cls.layout == "contiguous":
                        groups, blocks = p["groups"], deep_gemm.contiguous_blocks(p)
                        empty |= 0 in rows
                        tiles = blocks * nb
                    else:
                        groups, blocks = cls.num_groups, ceil_div(p["expected_m"], 128)
                        empty |= 0 in rows
                        tiles = sum(ceil_div(r, 128) for r in rows) * nb
                    multi |= blocks >= 2 or sum(ceil_div(r, 128) for r in rows) >= 2
                    persistent |= tiles > cls.config(groups, blocks * 128).num_sms
                values = [
                    (e.num_groups, b)
                    for e in cls.dispatch
                    for b in deep_gemm.range_values(e, 96)
                ]
                if cls.layout == "masked":
                    self.assertTrue(partial and multi)
                    if cls.num_groups >= 2:
                        self.assertTrue(empty)
                    reachable = any(
                        b * nb * g > cls.config(g, b * 128).num_sms for g, b in values
                    )
                else:
                    self.assertTrue(partial or all(b == 1 for _, b in values))
                    if any(g >= 2 and b >= 2 for g, b in values):
                        self.assertTrue(empty)
                    reachable = any(
                        b * nb > cls.config(g, b * 128).num_sms
                        and g * cls.shape_n * cls.shape_k <= deep_gemm.SMOKE_B_BYTES
                        for g, b in values
                    )
                if reachable:
                    self.assertTrue(persistent)


class Artifacts(unittest.TestCase):
    def test_cubins_match_artifact(self):
        """Each workload's cubin and every merged duplicate match the pinned
        artifact; duplicates have the workload's code; every published cubin
        serves exactly one (workload, group count)."""
        build = compiler()
        base = ROOT / "resources" / "trtllm-gen-artifacts" / build.ARTIFACT
        if not (base / "kernel_map.json").is_file():
            self.skipTest("run scripts/fetch_resources.py --only trtllm-gen")
        sums, kernel_map = build.ImplCompiler("sm_100a")._artifact_files()
        names = manifest()["sm_100a"]
        self.assertEqual(set(variants()), set(names))
        self.assertEqual(set(served_by()), set(kernel_map))
        workload_module = sys.modules[cubin_info.__module__]
        for name, relative in names.items():
            with self.subTest(workload=name):
                variant = variants()[name]
                image = (PACKAGE / relative).read_bytes()
                code = build.ImplCompiler._code_key(workload_module, image)
                for entry in variant["dispatch"]:
                    path = build.CUBIN_DIR / (entry["cubin"] + ".cubin")
                    data = path.read_bytes()
                    symbol, digest = kernel_map[entry["cubin"]]
                    self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
                    self.assertEqual(sums[path.name], digest)
                    arch, kernels, _ = cubin_info(data)
                    self.assertEqual((arch, kernels), ("sm_100f", [symbol]))
                    args = build.decode_symbol(symbol)
                    self.assertEqual(build.cubin_name(args), entry["cubin"])
                    self.assertEqual(
                        build.ImplCompiler._code_key(workload_module, data), code
                    )
                    self.assertEqual(
                        (
                            {1: "contiguous", 2: "masked"}[args["GEMM_TYPE"]],
                            args["N"],
                            args["K"],
                            args["NUM_GROUPS"],
                            args["BLOCK_N"],
                            args["NUM_STAGES"],
                            args["NUM_LAST_STAGES"],
                            args["SWIZZLE_CD_MODE"],
                        ),
                        (
                            variant["layout"],
                            variant["n"],
                            variant["k"],
                            entry["num_groups"],
                            variant["block_n"],
                            variant["num_stages"],
                            variant["num_last_stages"],
                            variant["swizzle_cd"],
                        ),
                    )

    def test_segment_sidecars(self):
        for name, entry in deep_gemm.variant_index().get("sm_86", {}).items():
            with self.subTest(workload=name):
                sidecar = json.loads((PACKAGE / entry["sidecar"]).read_text())
                path = PACKAGE / manifest()["sm_86"][name]
                if not path.is_file():
                    self.skipTest("run compile_kernels.py --package deep_gemm")
                image = path.read_bytes()
                self.assertEqual(
                    hashlib.sha256(image).hexdigest(), sidecar["cubin_sha256"]
                )
                arch, kernels, params = cubin_info(image)
                self.assertEqual((arch, kernels), ("sm_80", [sidecar["symbol"]]))
                self.assertEqual([p.size for p in params], [sidecar["params_size"]])


class Inputs(unittest.TestCase):
    def test_random_fp8_chunks(self):
        """Chunked generation equals the elementwise map of randint bytes
        across a chunk boundary, with finite magnitudes in [0.25, 3.75]."""
        shape = (2, deep_gemm.FP8_CHUNK // 2 + 3)
        values = deep_gemm.random_fp8(shape, torch.Generator().manual_seed(3), "cpu")
        raw = torch.randint(
            0, 256, shape, dtype=torch.uint8, generator=torch.Generator().manual_seed(3)
        )
        expected = (raw & 0x80) | (0x28 + (raw & 0x1F))
        self.assertTrue(torch.equal(values.view(torch.uint8), expected))
        magnitude = values.float().abs()
        self.assertGreaterEqual(magnitude.min().item(), 0.25)
        self.assertLessEqual(magnitude.max().item(), 3.75)

    def test_contiguous_indices(self):
        """Groups are 128-row aligned with -1 padding; empty groups take no
        rows; slack blocks are all -1."""
        indices = deep_gemm.contiguous_indices([3, 0, 130, 128], slack=1).tolist()
        expected = [0] * 3 + [-1] * 125 + [2] * 130 + [-1] * 126 + [3] * 128
        self.assertEqual(indices, expected + [-1] * 128)

    def test_validate_chunks(self):
        """Every chunk is compared; NaN (unwritten) elements are skipped."""
        name = next(n for n, v in variants().items() if v["layout"] == "masked")
        workload = get_workload(name)(None, device="cpu")
        size = deep_gemm.VALIDATE_CHUNK + 1000
        ref = torch.ones(size, dtype=torch.bfloat16)
        ref[5] = torch.nan
        impl = ref.clone()
        impl[5] = 123.0  # unwritten: ignored
        workload.validate((ref,), (impl,))
        impl[size - 1] = 2.0
        with self.assertRaises(AssertionError):
            workload.validate((ref,), (impl,))


if __name__ == "__main__":
    unittest.main()

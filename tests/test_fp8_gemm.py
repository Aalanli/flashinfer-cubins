"""fp8_gemm / fp8_gemm_small_m: Python-built CUTLASS Params versus CUTLASS's own host code.

``impls/fp8_gemm/compiler.py`` records, per supported architecture and
workload, the Params bytes CUTLASS's ``GemmKernel::to_underlying_arguments``
produced for several problems from the Arguments the upstream FlashInfer host
function builds (for ``fp8_gemm_small_m`` including its swap-AB and
KernelHardwareInfo query) in ``cubins/<arch>/<workload>.fixtures.json``. The
probe replaces the opaque CUtensorMap encoding with a deterministic packing of
the encode arguments, mirrored by ``harness.workloads.fp8_gemm.fake_encode``,
so byte equality covers every Params field and every descriptor's encode
arguments. None of this launches a kernel: it runs on any machine, with or
without GPU.
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.workload import CaseSpec  # noqa: E402
from harness.workloads.fp8_gemm import (  # noqa: E402
    FP8GEMM,
    SMALL_M_MAX_M,
    UNSERVED_UPSTREAM,
    FP8GEMMSmallM,
    fake_encode,
    fast_divmod,
)

WORKLOADS = (FP8GEMM, FP8GEMMSmallM)
FLASHINFER = ROOT / "resources/flashinfer-aa7c67f2b876b89be34c7a70ac022369a56e60d5"
DEEPGEMM = ROOT / "resources/DeepGEMM"


def parametrize(path: Path, function: str) -> dict[str, list]:
    """``@pytest.mark.parametrize(name, [literals])`` lists of a test function
    (None for non-literal lists, e.g. dtypes)."""
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == function
    )
    values: dict[str, list] = {}
    for decorator in node.decorator_list:
        if getattr(getattr(decorator, "func", None), "attr", "") == "parametrize":
            name = ast.literal_eval(decorator.args[0])
            try:
                values[name] = ast.literal_eval(decorator.args[1])
            except ValueError:
                values[name] = None
    return values


def assignments(path: Path, function: str) -> dict[str, object]:
    """Literal (tuple) assignments inside ``function``."""
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == function
    )
    values: dict[str, object] = {}
    for stmt in ast.walk(node):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        pairs = (
            zip(target.elts, stmt.value.elts)
            if isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Tuple)
            else [(target, stmt.value)]
        )
        for name, value in pairs:
            if isinstance(name, ast.Name):
                try:
                    values[name.id] = ast.literal_eval(value)
                except ValueError:
                    pass
    return values


def fixtures(cls, arch: str) -> dict:
    path = cls.cubin_path(arch).with_name(f"{cls.name}.fixtures.json")
    return json.loads(path.read_text())


def fixture_hardware(fixture: dict) -> tuple[int, int]:
    """(sm_count, max_active_clusters) CUTLASS stored: a failed query (< 0) is 0."""
    return fixture["sm_count"], max(fixture["max_active_clusters"], 0)


def shapes(problems: list[dict]) -> set[tuple[int, int, int]]:
    return {(f["problem"]["m"], f["problem"]["n"], f["problem"]["k"]) for f in problems}


class FP8GEMMParams(unittest.TestCase):
    def test_params_match_cutlass(self):
        for cls in WORKLOADS:
            for arch in cls.supported_arches:
                workload = cls.from_arch(arch, device="cpu")
                data = fixtures(cls, arch)
                self.assertEqual(data["workload"], cls.name)
                problems = data["problems"]
                self.assertGreaterEqual(len(problems), 10)
                for fixture in problems:
                    q, p = fixture["problem"], fixture["pointers"]
                    with self.subTest(
                        workload=cls.name,
                        arch=arch,
                        **q,
                        driver=fixture["driver_version"],
                    ):
                        config = workload.build_params(
                            q["m"],
                            q["n"],
                            q["k"],
                            p["a"],
                            p["b"],
                            p["sfa"],
                            p["sfb"],
                            p["d"],
                            encode=fake_encode,
                            driver=fixture["driver_version"],
                            hardware=fixture_hardware(fixture),
                        )
                        expected = bytes.fromhex(fixture["params_hex"])
                        self.assertEqual(len(config.params), len(expected))
                        padding = {
                            i for s, e in fixture["padding"] for i in range(s, e)
                        }
                        mismatched = [
                            i
                            for i, (x, y) in enumerate(
                                zip(config.params, expected, strict=True)
                            )
                            if i not in padding and x != y
                        ]
                        self.assertEqual(mismatched, [])
                        self.assertFalse(any(config.params[i] for i in padding))
                        self.assertEqual(list(config.grid), fixture["grid"])
                        self.assertEqual(list(config.block), fixture["block"])
                        self.assertEqual(config.shared_mem, fixture["shared_mem"])
                        cluster = fixture["cluster"]
                        self.assertEqual(
                            config.cluster, None if cluster is None else tuple(cluster)
                        )

    def test_fixtures_cover_regimes(self):
        """Partial tiles, both raster orders, the driver descriptor fix-up, and
        every case shape of the workload."""
        for cls in WORKLOADS:
            for arch in cls.supported_arches:
                problems = fixtures(cls, arch)["problems"]
                workload = cls.from_arch(arch, device="cpu")
                tile_m, tile_n, _ = workload.layout["constants"]["cta_shape_mnk"]
                swap = workload.layout["constants"]["swap_ab"]
                with self.subTest(workload=cls.name, arch=arch):
                    self.assertEqual(
                        {f["driver_version"] for f in problems}, {13010, 13020}
                    )
                    self.assertTrue(all(k % 128 == 0 for _, _, k in shapes(problems)))
                    # The kernel's own (km, kn): partial tiles in both dimensions.
                    kernel = [
                        (n, m) if swap else (m, n) for m, n, _ in shapes(problems)
                    ]
                    self.assertTrue(any(m % tile_m and n % tile_n for m, n in kernel))
                    # Both rasterizations (grid transposed for AlongN).
                    orders = {
                        tuple(f["grid"][:2])
                        == (
                            -(-f["problem"]["n" if swap else "m"] // tile_m),
                            -(-f["problem"]["m" if swap else "n"] // tile_n),
                        )
                        for f in problems
                        if f["grid"][0] != f["grid"][1]
                    }
                    self.assertEqual(orders, {True, False})
                    for case in workload.get_cases():
                        c = case.params
                        self.assertIn((c["m"], c["n"], c["k"]), shapes(problems))

    def test_small_m_problems_follow_upstream_dispatch(self):
        for arch in FP8GEMMSmallM.supported_arches:
            problems = fixtures(FP8GEMMSmallM, arch)["problems"]
            self.assertTrue(all(m <= SMALL_M_MAX_M for m, _, _ in shapes(problems)))
            self.assertIn(1, {m for m, _, _ in shapes(problems)})
            self.assertTrue(any(n % 128 for _, n, _ in shapes(problems)))
            layout = FP8GEMMSmallM.from_arch(arch, device="cpu").layout
            self.assertTrue(layout["constants"]["swap_ab"])
            self.assertEqual(layout["constants"]["cta_shape_mnk"], [128, 16, 128])
            self.assertNotIn("c", layout["tensor_maps"])
            self.assertEqual(layout["cluster"], [1, 1, 1])
            # make_kernel_hardware_info's queries, as the Python builder repeats.
            for fixture in problems:
                calls = [q["call"] for q in fixture["hardware_queries"]]
                self.assertEqual(
                    calls,
                    [
                        "cudaGetDevice",
                        "cudaDeviceGetAttribute",
                        "cudaOccupancyMaxActiveClusters",
                    ],
                )
            self.assertEqual({fixture_hardware(f)[1] for f in problems}, {0, 37})

    def test_case_assignment_follows_upstream_dispatch(self):
        """Each case runs on the kernel upstream dispatches its m to, at least
        three smoke cases per workload, including partial tiles."""
        for cls in WORKLOADS:
            workload = cls(None, device="cpu")
            cases = workload.get_cases()
            smoke = [c for c in cases if c.suite == "smoke"]
            with self.subTest(workload=cls.name):
                self.assertGreaterEqual(len(smoke), 3)
                self.assertTrue(
                    any(c.params["m"] % 16 or c.params["n"] % 128 for c in smoke)
                )
                for case in cases:
                    small = case.params["m"] <= SMALL_M_MAX_M
                    self.assertEqual(small, cls is FP8GEMMSmallM, case.name)
        names = [
            {c.name for c in cls(None, device="cpu").get_cases()} for cls in WORKLOADS
        ]
        self.assertFalse(names[0] & names[1])

    def test_small_m_restricted_to_validated_shapes(self):
        """Shapes that race on a B200 are neither cases nor runnable."""
        workload = FP8GEMMSmallM(None, device="cpu")
        for case in workload.get_cases():
            self.assertLessEqual(case.params["m"], 16)
        self.assertFalse(FP8GEMMSmallM.validated(32, 384))
        self.assertFalse(FP8GEMMSmallM.validated(17, 256))
        self.assertTrue(FP8GEMMSmallM.validated(16, 4096))
        inputs = workload.get_inputs(CaseSpec("x", {"m": 32, "n": 128, "k": 384}, 1))
        with self.assertRaisesRegex(ValueError, "validated range"):
            workload._check_supported(inputs)

    def test_fast_divmod(self):
        for divisor in (1, 2, 3, 7, 128, 1000):
            d, multiplier, shift = fast_divmod(divisor)
            for dividend in (0, 1, 5, 127, 128, 1 << 20, (1 << 31) - 1):
                quotient = (
                    dividend if d == 1 else (dividend * multiplier >> 32) >> shift
                )
                self.assertEqual(quotient, dividend // divisor)

    def test_upstream_parametrizations_are_cases(self):
        """Every parametrization of the pinned upstream tests a kernel here
        serves is one of its cases (read from the upstream sources); the
        unserved m = 32 small-batch cases are exactly UNSERVED_UPSTREAM."""
        served = {
            cls: {
                (c.params["m"], c.params["n"], c.params["k"])
                for c in cls(None, device="cpu").get_cases()
            }
            for cls in WORKLOADS
        }

        def kernel(m):
            return FP8GEMMSmallM if m <= SMALL_M_MAX_M else FP8GEMM

        tests = FLASHINFER / "tests/gemm/test_groupwise_scaled_gemm_fp8.py"
        wanted = []
        p = parametrize(tests, "test_fp8_groupwise_gemm")
        self.assertIn("cutlass", p["backend"])
        self.assertIn("K", p["scale_major_mode"])
        wanted += [(m, n, k) for m in p["m"] for n in p["n"] for k in p["k"]]
        p = parametrize(tests, "test_fp8_groupwise_gemm_small_batch_size")
        self.assertIn("K", p["scale_major_mode"])
        small = [(m, n, k) for m in p["m"] for n in p["n"] for k in p["k"]]
        self.assertEqual(
            [s for s in small if not FP8GEMMSmallM.validated(s[0], s[2])],
            UNSERVED_UPSTREAM,
        )
        wanted += [s for s in small if s not in UNSERVED_UPSTREAM]
        # DeepGEMM enumerate_normal: FP8 legacy forward, BF16 output.
        names = assignments(DEEPGEMM / "tests/generators.py", "enumerate_normal")
        wanted += [
            (m, n, k) for m in names["m_fwd_list"] for n, k in names["bf16_output_nk"]
        ]
        self.assertGreaterEqual(len(wanted), 125 + 6 + 21)
        for shape in wanted:
            self.assertIn(shape, served[kernel(shape[0])], shape)

    def test_small_m_throughput_covers_decode(self):
        """fp8_gemm_small_m is benchmarked at m = 1, 4, 8, 16 on DeepSeek-V3's
        linear layers, including n = 24576 (192 tiles: CTAs take several)."""
        cases = [
            c
            for c in FP8GEMMSmallM(None, device="cpu").get_cases()
            if c.suite == "throughput"
        ]
        shapes = {(c.params["m"], c.params["n"], c.params["k"]) for c in cases}
        for m in (1, 4, 8, 16):
            self.assertIn((m, 24576, 1536), shapes)

    def test_rejects_bad_operands(self):
        for cls in WORKLOADS:
            workload = cls.from_arch(cls.supported_arches[0], device="cpu")
            case = workload.get_cases()[0]
            a, b, sfa, sfb = workload.get_inputs(case)
            (out,) = workload.allocate_outputs((a, b, sfa, sfb))
            with self.assertRaises(ValueError):
                workload.launch_config((a, b, sfa.t(), sfb), out, encode=fake_encode)
            with self.assertRaises(ValueError):
                workload.launch_config(
                    (a, b, sfa, sfb), out.float(), encode=fake_encode
                )


if __name__ == "__main__":
    unittest.main()

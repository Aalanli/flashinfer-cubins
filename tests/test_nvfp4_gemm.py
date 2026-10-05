"""nvfp4_gemm: Python-built CUTLASS Params versus CUTLASS's own host code.

``impls/nvfp4_gemm/compiler.py`` records, per supported architecture, the
Params bytes CUTLASS's ``GemmKernel::to_underlying_arguments`` produced for
regime-edge problems and every case shape from Arguments built like
FlashInfer's ``prepareGemmArgs`` (``cubins/<arch>/nvfp4_gemm.fixtures.json``).
The probe replaces the opaque CUtensorMap encoding with a deterministic
packing of the encode arguments, mirrored by
``harness.workloads.nvfp4_gemm.fake_encode``, so byte equality covers every
Params field and every descriptor's encode arguments. None of this launches a
kernel: it runs on any machine, with or without GPU.
"""

from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.workload import CaseSpec  # noqa: E402
from harness.workloads.nvfp4_gemm import (  # noqa: E402
    NVFP4GEMM,
    TASK,
    TASK_DIST,
    fake_encode,
    from_task_scales,
    task_shapes,
)

FLASHINFER = ROOT / "resources/flashinfer-aa7c67f2b876b89be34c7a70ac022369a56e60d5"


def load(arch: str, device: str) -> NVFP4GEMM:
    workload = NVFP4GEMM.from_arch(arch, device=device)
    assert isinstance(workload, NVFP4GEMM)
    return workload


def fixtures(arch: str) -> dict:
    path = NVFP4GEMM.cubin_path(arch).with_name("nvfp4_gemm.fixtures.json")
    return json.loads(path.read_text())


def mm_fp4_cases() -> list[tuple]:
    """``_SMOKE_CASES`` + ``_CUTEDSL_LOW_LATENCY_MODEL_CASES`` of FlashInfer's
    tests/gemm/test_mm_fp4.py (dtypes as their attribute names)."""
    tree = ast.parse((FLASHINFER / "tests/gemm/test_mm_fp4.py").read_text())
    rows = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id in (
            "_SMOKE_CASES",
            "_CUTEDSL_LOW_LATENCY_MODEL_CASES",
        ):
            for element in node.value.elts:
                rows.append(
                    tuple(
                        v.attr if isinstance(v, ast.Attribute) else ast.literal_eval(v)
                        for v in element.elts
                    )
                )
    return rows


class NVFP4GEMMParams(unittest.TestCase):
    def test_params_match_cutlass(self):
        for arch in NVFP4GEMM.supported_arches:
            workload = load(arch, "cpu")
            problems = fixtures(arch)["problems"]
            self.assertGreaterEqual(len(problems), 20)
            for fixture in problems:
                q, p = fixture["problem"], fixture["pointers"]
                with self.subTest(arch=arch, **q, driver=fixture["driver_version"]):
                    config = workload.build_params(
                        q["m"],
                        q["n"],
                        q["k"],
                        q["l"],
                        p["a"],
                        p["b"],
                        p["sfa"],
                        p["sfb"],
                        p["alpha"],
                        p["d"],
                        encode=fake_encode,
                        driver=fixture["driver_version"],
                    )
                    expected = bytes.fromhex(fixture["params_hex"])
                    self.assertEqual(len(config.params), len(expected))
                    padding = {i for s, e in fixture["padding"] for i in range(s, e)}
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
                    self.assertEqual(list(config.cluster), fixture["cluster"])
                    self.assertEqual(fixture["cluster_fallback"], [1, 1, 1])
                    self.assertTrue(config.programmatic_serialization)

    def test_fixtures_cover_regimes(self):
        """Partial tiles, batches, both raster orders, the descriptor fix-up and
        every case the workload serves."""
        for arch in NVFP4GEMM.supported_arches:
            problems = fixtures(arch)["problems"]
            shapes = {
                (
                    f["problem"]["m"],
                    f["problem"]["n"],
                    f["problem"]["k"],
                    f["problem"]["l"],
                )
                for f in problems
            }
            self.assertTrue(any(m % 128 and n % 128 for m, n, _, _ in shapes))
            self.assertTrue(any(k % 256 for _, _, k, _ in shapes))
            self.assertTrue(any((k // 16) % 4 for _, _, k, _ in shapes))
            self.assertTrue(any(batch > 1 for *_, batch in shapes))
            self.assertEqual({f["driver_version"] for f in problems}, {13010, 13020})
            grids = {tuple(f["grid"]) for f in problems}
            self.assertTrue(any(x != y for x, y, _ in grids))
            workload = load(arch, "cpu")
            for case in workload.get_cases():
                c = case.params
                self.assertIn((c["m"], c["n"], c["k"], c["l"]), shapes)

    def test_upstream_parametrizations_are_cases(self):
        """Every task test/benchmark line (task distribution and seed for the
        tests) and FlashInfer's mm_fp4 cutlass/nvfp4 cases are cases."""
        cases = NVFP4GEMM(None, device="cpu").get_cases()
        params = [c.params for c in cases]
        by_shape = {tuple(c.params[k] for k in "mnkl"): c for c in cases}
        for spec in task_shapes(TASK, "tests"):
            case = by_shape[tuple(spec[k] for k in "mnkl")]
            self.assertEqual(case.seed, spec["seed"])
            self.assertEqual(case.source["kind"], "upstream_test")
            self.assertFalse(set(TASK_DIST) & set(case.params))
        for spec in task_shapes(TASK, "benchmarks"):
            self.assertIn({k: spec[k] for k in "mnkl"}, params)
        served = [r for r in mm_fp4_cases() if r[4] == "cutlass" and r[7] == "nvfp4"]
        self.assertGreaterEqual(len(served), 2)
        for m, n, k, *_ in served:
            case = by_shape[(m, n, k, 1)]
            self.assertNotEqual(case.params.get("alpha", 1.0), 1.0)
            self.assertEqual(case.source["kind"], "upstream_test")

    def test_reference_matches_task_definition(self):
        """``get_reference`` equals the NVIDIA reference's math (FP64 here) on
        logical scales, times alpha, for unrestricted values and l = 2."""
        workload = NVFP4GEMM(None, device="cpu")
        case = CaseSpec("tiny", {"m": 130, "n": 136, "k": 96, "l": 2, "alpha": 0.37}, 5)
        a, b, sfa, sfb, alpha = workload.get_inputs(case)
        (ref,) = workload.get_reference((a, b, sfa, sfb, alpha))
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float64,
        )
        sa = from_task_scales(sfa, 130, 6).double()
        sb = from_task_scales(sfb, 136, 6).double()
        self.assertTrue((sa == 0).any())  # the task's integer scales 0..3

        def dequant(packed, scales):
            values = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
            return lut[values.long()] * scales.repeat_interleave(16, dim=1)

        for batch in range(2):
            expected = (
                dequant(a[:, :, batch], sa[:, :, batch])
                @ dequant(b[:, :, batch], sb[:, :, batch]).T
            ) * alpha.double()
            torch.testing.assert_close(
                ref[:, :, batch], expected.half(), rtol=1e-3, atol=1e-3
            )
        self.assertEqual(tuple(ref.shape), (130, 136, 2))
        self.assertTrue(ref.permute(2, 0, 1).is_contiguous())

    def test_rejects_bad_operands(self):
        workload = load(NVFP4GEMM.supported_arches[0], "cpu")
        case = workload.get_cases()[1]  # batched
        inputs = workload.get_inputs(case)
        a, b, sfa, sfb, alpha = inputs
        (out,) = workload.allocate_outputs(inputs)
        kwargs = dict(encode=fake_encode, driver=13020)
        workload.launch_config(inputs, out, **kwargs)
        logical = from_task_scales(sfa, a.shape[0], 2 * a.shape[1] // 16)
        bad = (
            ((a, b, logical, sfb, alpha), out),  # logical instead of blocked scales
            ((a.contiguous(), b, sfa, sfb, alpha), out),  # batch innermost
            (inputs, out.float()),
            ((a, b, sfa, sfb, alpha.double()), out),
            ((a, b, sfa, sfb, torch.ones(2)), out),
        )
        for operands, output in bad:
            with self.assertRaises(ValueError):
                workload.launch_config(operands, output, **kwargs)
        with self.assertRaises(ValueError):
            workload.build_params(128, 128, 48, 1, 0, 0, 0, 0, 0, 0, **kwargs)
        if not workload.is_supported():
            # Never execute sm_100a code on another GPU.
            with self.assertRaises(RuntimeError):
                workload.run(inputs)


if __name__ == "__main__":
    unittest.main()

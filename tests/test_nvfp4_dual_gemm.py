"""nvfp4_dual_gemm: Python-built CUTLASS Params, pipeline cases, upstream coverage.

``impls/nvfp4_dual_gemm/compiler.py`` records the Params bytes CUTLASS's
``GemmKernel::to_underlying_arguments`` produced for Arguments built exactly as
FlashInfer's ``prepareGemmArgs_*`` builds them, for regime-edge problems and
every case shape (``cubins/sm_100a/nvfp4_dual_gemm_gemm.fixtures.json``). The
scale relayout is tested once, in ``tests/test_quantization.py``. The probe replaces
the opaque CUtensorMap encoding with a deterministic packing of the encode
arguments, mirrored by ``fake_encode``, so byte equality covers every Params
field and every descriptor's encode arguments. Nothing here launches the
sm_100a kernel; it runs on any machine, with or without GPU.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.workloads.nvfp4_dual_gemm import (  # noqa: E402
    TASK,
    TASK_DIST,
    NVFP4DualGEMMGemm,
    NVFP4DualGEMMPackScales,
    NVFP4DualGEMMSiluProduct,
    fake_encode,
    silu_bytes,
)
from harness.workloads.nvfp4_gemm import task_shapes  # noqa: E402

GEMM = NVFP4DualGEMMGemm


def fixtures(arch: str) -> dict:
    path = GEMM.cubin_path(arch).with_name("nvfp4_dual_gemm_gemm.fixtures.json")
    return json.loads(path.read_text())


class NVFP4DualGEMMParams(unittest.TestCase):
    def test_params_match_cutlass(self):
        for arch in GEMM.supported_arches:
            workload = GEMM.from_arch(arch, device="cpu")
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
                    self.assertEqual(config.cluster, (1, 1, 1))
                    self.assertTrue(config.programmatic_serialization)

    def test_fixtures_cover_regimes(self):
        """Partial tiles, batches, both raster orders, the descriptor fix-up and
        every case shape."""
        for arch in GEMM.supported_arches:
            problems = [f["problem"] for f in fixtures(arch)["problems"]]
            shapes = {(q["m"], q["n"], q["k"], q["l"]) for q in problems}
            self.assertTrue(any(m % 128 and n % 128 for m, n, _, _ in shapes))
            self.assertTrue(any(k % 256 for _, _, k, _ in shapes))
            self.assertTrue(any(b > 1 for *_, b in shapes))
            self.assertTrue(any(m > n for m, n, _, _ in shapes))
            self.assertTrue(any(m < n for m, n, _, _ in shapes))
            drivers = {f["driver_version"] for f in fixtures(arch)["problems"]}
            self.assertEqual(drivers, {13010, 13020})
            workload = GEMM.from_arch(arch, device="cpu")
            for case in workload.get_cases():
                c = case.params
                self.assertIn((c["m"], c["n"], c["k"], c["l"]), shapes)

    def test_rejects_bad_operands(self):
        workload = GEMM.from_arch(GEMM.supported_arches[0], device="cpu")
        case = workload.get_cases()[1]  # l = 2
        inputs = workload.get_inputs(case)
        a, b, sfa, sfb, alpha = inputs
        (out,) = workload.allocate_outputs(inputs)
        workload.launch_config(inputs, out, encode=fake_encode, driver=1)
        bad = (
            ((a.contiguous(), b, sfa, sfb, alpha), out),  # wrong batch stride
            ((a, b, sfa[:, 1:], sfb, alpha), out),  # wrong packed size
            (inputs, out.contiguous()),  # wrong output layout
            (inputs, out.half()),
            ((a, b, sfa, sfb, alpha.half()), out),
        )
        for operands, output in bad:
            with self.assertRaises(ValueError):
                workload.launch_config(operands, output, encode=fake_encode, driver=1)
        with self.assertRaises(ValueError):
            workload.build_params(128, 128, 48, 1, *([1 << 20] * 6), encode=fake_encode)
        if not workload.is_supported():
            # Never execute sm_100a code on another GPU.
            with self.assertRaises(RuntimeError):
                workload.run(inputs)


class Pipeline(unittest.TestCase):
    def test_pipeline_cases_share_inputs(self):
        """Stage workloads see the dual GEMM's tensors of the same case (with
        alpha != 1 applied to both GEMMs)."""
        pack = NVFP4DualGEMMPackScales(None, device="cpu")
        gemm = NVFP4DualGEMMGemm(None, device="cpu")
        silu = NVFP4DualGEMMSiluProduct(None, device="cpu")
        case = next(c for c in gemm.get_cases() if c.params.get("alpha", 1) != 1)
        a, b, sfa, sfb, alpha = gemm.get_inputs(case)
        self.assertEqual(alpha.item(), torch.tensor(case.params["alpha"]).item())
        packed = {
            c.name: pack.get_reference(pack.get_inputs(c))[0]
            for c in pack.get_cases()
            if c.name.startswith(case.name + "_")
        }
        self.assertTrue(torch.equal(packed[case.name + "_sfa"], sfa))
        self.assertTrue(torch.equal(packed[case.name + "_sfb"], sfb))
        h1, _ = silu.get_inputs(case)
        (ref,) = gemm.get_reference((a, b, sfa, sfb, alpha))
        self.assertTrue(torch.equal(ref, h1))
        self.assertEqual(ref.stride(), h1.stride())

    def test_upstream_parametrizations_are_cases(self):
        """Every task test line (task distribution and seed) is a case of all
        three stages (pack_scales: both operands); every benchmark line is a
        GEMM throughput case."""
        stages = {
            cls: {c.name: c for c in cls(None, device="cpu").get_cases()}
            for cls in (GEMM, NVFP4DualGEMMPackScales, NVFP4DualGEMMSiluProduct)
        }
        gemm_shapes = {
            tuple(c.params[k] for k in "mnkl"): c for c in stages[GEMM].values()
        }
        for spec in task_shapes(TASK, "tests"):
            case = gemm_shapes[tuple(spec[k] for k in "mnkl")]
            self.assertEqual(case.seed, spec["seed"])
            self.assertEqual(case.source["kind"], "upstream_test")
            self.assertFalse(set(TASK_DIST) & set(case.params))
            self.assertIn(case.name, stages[NVFP4DualGEMMSiluProduct])
            for operand in ("sfa", "sfb"):
                self.assertIn(f"{case.name}_{operand}", stages[NVFP4DualGEMMPackScales])
        for spec in task_shapes(TASK, "benchmarks"):
            case = gemm_shapes[tuple(spec[k] for k in "mnkl")]
            self.assertEqual(case.suite, "throughput")

    def test_silu_stage_budget(self):
        """silu_product (an sm_86 workload) takes every pipeline case within
        its memory budget, including all smoke and task cases."""
        silu = {
            c.name for c in NVFP4DualGEMMSiluProduct(None, device="cpu").get_cases()
        }
        for case in GEMM(None, device="cpu").get_cases():
            if case.suite == "smoke":
                self.assertIn(case.name, silu)
            if case.name in silu:
                self.assertLessEqual(silu_bytes(case.params), 6 << 30)


if __name__ == "__main__":
    unittest.main()

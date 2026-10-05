"""nvfp4_group_gemm: Python-built CUTLASS Params and device arrays versus CUTLASS.

``impls/nvfp4_group_gemm/compiler.py`` records, per supported architecture,
the Params bytes, launch shape, workspace size and per-group device-array
bytes CUTLASS's own host code produced for many group configurations
(``cubins/<arch>/nvfp4_group_gemm.fixtures.json``). The probe replaces the
opaque CUtensorMap encoding with a deterministic packing of the encode
arguments, mirrored by ``harness.workloads.nvfp4_group_gemm.fake_encode``, so
byte equality covers every Params field and every descriptor's encode
arguments. None of this launches a kernel: it runs on any machine.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.workloads.nvfp4_gemm import task_shapes  # noqa: E402
from harness.workloads.nvfp4_group_gemm import (  # noqa: E402
    ARRAYS,
    TASK,
    TASK_DIST,
    NVFP4GroupGEMM,
    example75_groups,
    fake_encode,
    fast_divmod_u64,
    from_blocked,
)

TASK_SHAPES = TASK


def fixtures(arch: str) -> dict:
    path = NVFP4GroupGEMM.cubin_path(arch).with_name("nvfp4_group_gemm.fixtures.json")
    return json.loads(path.read_text())


class NVFP4GroupGEMMParams(unittest.TestCase):
    def test_params_and_arrays_match_cutlass(self):
        for arch in NVFP4GroupGEMM.supported_arches:
            workload = NVFP4GroupGEMM.from_arch(arch, device="cpu")
            problems = fixtures(arch)["problems"]
            self.assertGreaterEqual(len(problems), 30)
            for fixture in problems:
                groups = [tuple(p) for p in fixture["problems"]]
                pointers = fixture["pointers"]
                with self.subTest(
                    arch=arch,
                    groups=groups,
                    sm_count=fixture["sm_count"],
                    driver=fixture["driver_version"],
                ):
                    config = workload.build_params(
                        groups,
                        {name: pointers[name] for name in ARRAYS},
                        pointers["workspace"],
                        fixture["host_problem_shapes"],
                        fixture["sm_count"],
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
                    self.assertEqual(config.workspace_size, fixture["workspace_size"])
                    self.assertIsNone(config.cluster)
                    arrays = workload.array_bytes(
                        groups,
                        {k: pointers[k] for k in ("a", "b", "sfa", "sfb", "d")},
                    )
                    self.assertEqual(set(arrays), set(fixture["arrays_hex"]))
                    for name, value in fixture["arrays_hex"].items():
                        self.assertEqual(arrays[name].hex(), value, name)

    def test_fixtures_cover_regimes(self):
        """Partial M/N tiles, persistent-grid truncation, the driver fix-up and
        every harness and NVIDIA task shape."""
        task = TASK_SHAPES.read_text()
        for arch in NVFP4GroupGEMM.supported_arches:
            problems = fixtures(arch)["problems"]
            configurations = {tuple(tuple(p) for p in f["problems"]) for f in problems}
            self.assertEqual({f["driver_version"] for f in problems}, {13010, 13020})
            self.assertTrue(
                any(m % 128 or n % 128 for c in configurations for m, n, _ in c)
            )
            self.assertTrue(any(f["grid"][0] == f["sm_count"] for f in problems))
            self.assertTrue(any(f["grid"][0] < f["sm_count"] for f in problems))
            workload = NVFP4GroupGEMM.from_arch(arch, device="cpu")
            for case in workload.get_cases():
                self.assertIn(tuple(map(tuple, case.params["groups"])), configurations)
            # Every test/benchmark line of the task is a fixture configuration.
            lines = [line for line in task.splitlines() if '"m": [' in line]
            self.assertEqual(len(lines), 14)
            for line in lines:
                spec = json.loads(line.strip().removeprefix("- "))
                groups = tuple(zip(spec["m"], spec["n"], spec["k"], strict=True))
                self.assertIn(groups, configurations)

    def test_fast_divmod_u64(self):
        for divisor in (1, 2, 3, 7, 128, 192, 1000):
            d, multiplier, shift, round_up = fast_divmod_u64(divisor)
            for dividend in (0, 1, 5, 127, 128, 1 << 20, (1 << 40) + 3):
                x = (
                    ((dividend + round_up) * multiplier) >> 64
                    if multiplier
                    else dividend
                )
                self.assertEqual(x >> shift, dividend // divisor)

    def test_launch_config_wires_tensors(self):
        """launch_config's device arrays hold every group's tensor pointers in
        group order, next to a workspace of CUTLASS's size (never launched)."""
        workload = NVFP4GroupGEMM.from_arch(
            NVFP4GroupGEMM.supported_arches[0], device="cpu"
        )
        case = next(c for c in workload.get_cases() if c.name == "distinct")
        inputs = workload.get_inputs(case)
        outputs = workload.allocate_outputs(inputs)
        config = workload.launch_config(
            inputs, outputs, encode=fake_encode, driver=13020, sm_count=148
        )
        args, workspace = config.keepalive[:2]
        self.assertEqual(workspace.numel(), config.workspace_size)
        memory = args.numpy().tobytes()
        groups = workload.groups_of(inputs)
        for slot, tensors in (
            *((i, [g[i] for g in groups]) for i in range(4)),
            (4, list(outputs)),
        ):
            pointers = b"".join(t.data_ptr().to_bytes(8, "little") for t in tensors)
            self.assertIn(pointers, memory, ARRAYS[1 + slot])
        if not workload.is_supported():
            # Never execute sm_100a code on another GPU.
            with self.assertRaises(RuntimeError):
                workload.run(inputs)

    def test_upstream_parametrizations_are_cases(self):
        """Every task test (task distribution and seed) and benchmark line and
        CUTLASS example 75's random problems are cases; MoE throughput cases
        have skewed groups including 1-row groups."""
        cases = NVFP4GroupGEMM(None, device="cpu").get_cases()
        by_groups = {tuple(map(tuple, c.params["groups"])): c for c in cases}
        for section in ("tests", "benchmarks"):
            for spec in task_shapes(TASK, section):
                groups = tuple(zip(spec["m"], spec["n"], spec["k"], strict=True))
                case = by_groups[groups]
                if section == "tests":
                    self.assertEqual(case.seed, spec["seed"])
                    self.assertEqual(case.source["kind"], "upstream_test")
                    self.assertFalse(set(TASK_DIST) & set(case.params))
                else:
                    self.assertEqual(case.suite, "throughput")
        example = by_groups[tuple(example75_groups())]
        self.assertEqual(example.source["kind"], "upstream_test")
        self.assertEqual(len(example.params["groups"]), 10)
        moe = [c for c in cases if c.source.get("kind") == "model_shape"]
        self.assertTrue(moe)
        self.assertTrue(any(min(m for m, _, _ in c.params["groups"]) == 1 for c in moe))
        sizes = [m for c in moe for m, _, _ in c.params["groups"]]
        self.assertGreaterEqual(max(sizes), 1000)

    def test_rejects_bad_operands(self):
        workload = NVFP4GroupGEMM.from_arch(
            NVFP4GroupGEMM.supported_arches[0], device="cpu"
        )
        case = workload.get_cases()[1]
        inputs = workload.get_inputs(case)
        outputs = workload.allocate_outputs(inputs)
        kwargs = dict(encode=fake_encode, driver=13020, sm_count=148)
        logical = from_blocked(inputs[2], 17, 128 // 16)
        bad = (inputs[0], inputs[1], logical) + inputs[3:]
        with self.assertRaises(ValueError):
            workload.launch_config(bad, outputs, **kwargs)
        with self.assertRaises(ValueError):
            workload.launch_config(
                inputs, (outputs[0].float(),) + outputs[1:], **kwargs
            )
        with self.assertRaises(ValueError):
            workload.launch_config(inputs[:-1], outputs, **kwargs)
        with self.assertRaises(ValueError):
            workload.launch_config(inputs, outputs[:-1], **kwargs)


if __name__ == "__main__":
    unittest.main()

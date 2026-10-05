"""Shape/provenance checks for every registered workload, without large allocations."""

import hashlib
import json
import re
import unittest
from collections import defaultdict
from importlib import import_module

from harness import enumerate_workloads, get_workload
from harness.throughput import DATA, inventory, throughput_cases
from harness.workload import ROOT


def workload_classes():
    """(impls package, class) for every registered single-kernel workload."""
    for cls in enumerate_workloads():
        yield cls.manifest_path().parent.name, cls


class ThroughputCases(unittest.TestCase):
    def test_cases_are_well_formed(self):
        for name, cls in workload_classes():
            with self.subTest(workload=cls.name):
                cases = cls(None, device="cpu").get_cases()
                self.assertEqual(len(cases), len({c.name for c in cases}))
                self.assertTrue(any(c.suite == "smoke" for c in cases))
                for c in cases:
                    self.assertIn(c.suite, ("smoke", "throughput"))
                    if c.suite != "throughput":
                        continue
                    self.assertTrue(c.source)
                    if "axes" in c.source:
                        rows, _ = inventory(c.source["inventory"])
                        self.assertTrue(
                            any(
                                r["axes"] == c.source["axes"]
                                and r["uuid"] == c.source["uuid"]
                                for r in rows
                            )
                        )

    def test_every_throughput_shape_is_served(self):
        """Pipelines are split per kernel and dispatch by shape; together each
        package's kernels must cover every throughput case of the package."""
        served = defaultdict(set)
        for name, cls in workload_classes():
            for c in cls(None, device="cpu").get_cases():
                if c.suite == "throughput":
                    served[name].add(c.name)
        packages = {name for name, _ in workload_classes()}
        self.assertEqual(set(served), packages)
        for name, names in served.items():
            module = import_module(f"harness.workloads.{name}")
            if not hasattr(module, "throughput_cases"):
                continue  # variant packages: each kernel has its own cases
            with self.subTest(package=name):
                wanted = {c.name for c in throughput_cases(name)}
                self.assertGreaterEqual(len(wanted), 3)
                self.assertLessEqual(wanted, names)

    def test_inventory_integrity(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertEqual(len(manifest), 7)
        for name, entry in manifest.items():
            blob = (DATA / f"{name}.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), entry["sha256"])
            self.assertEqual(len(blob.splitlines()), entry["rows"])
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")

    def test_nvidia_benchmark_shapes(self):
        for task_name, name in (
            ("nvfp4_gemm", "nvfp4_gemm"),
            ("nvfp4_dual_gemm", "nvfp4_dual_gemm_gemm"),
            ("nvfp4_group_gemm", "nvfp4_group_gemm"),
        ):
            task = (
                (ROOT / "resources" / task_name / "task.yml")
                .read_text()
                .split("\nbenchmarks:", 1)[1]
            )
            shapes = [json.loads(s) for s in re.findall(r"^\s*- (\{.*\})", task, re.M)]
            cases = [
                c
                for c in get_workload(name)(None, device="cpu").get_cases()
                if c.suite == "throughput"
            ]
            # The task's benchmark shapes are a subset of the throughput cases.
            if task_name == "nvfp4_group_gemm":
                served = [[tuple(g) for g in c.params["groups"]] for c in cases]
                wanted = [list(zip(s["m"], s["n"], s["k"])) for s in shapes]
            else:
                served = [c.params for c in cases]
                wanted = [{k: s[k] for k in ("m", "n", "k", "l")} for s in shapes]
            for shape in wanted:
                self.assertIn(shape, served)

    def test_trace_shape_geometry(self):
        traced = (
            "gqa_decode",
            "gdn_prefill",
            "dsa_indexer",
            "dsa_attention",
            "moe",
            "rmsnorm",
            "gdn_decode",
        )
        for name, cls in workload_classes():
            for c in cls(None, device="cpu").get_cases():
                axes = c.source.get("axes")
                if not axes:
                    continue
                name = c.source["inventory"]
                self.assertIn(name, traced)
                p = c.params
                if name in ("gqa_decode", "dsa_attention"):
                    # The softmax scale is replayed from the recorded row.
                    (row,) = [
                        r for r in inventory(name)[0] if r["uuid"] == c.source["uuid"]
                    ]
                    scale = row["inputs"]["sm_scale"]["value"]
                    self.assertEqual(p["sm_scale"], scale)
                if name == "gqa_decode":
                    self.assertEqual(sum(p["lengths"]), axes["num_kv_indices"])
                    self.assertEqual(len(p["lengths"]), axes["batch_size"])
                    self.assertEqual(p["pages"], axes["num_pages"])
                    # Long-tailed, not balanced, lengths where batch > 1.
                    if len(p["lengths"]) > 2:
                        self.assertGreater(max(p["lengths"]), 2 * min(p["lengths"]))
                elif name == "gdn_prefill":
                    self.assertEqual(sum(p["lengths"]), axes["total_seq_len"])
                    self.assertEqual(len(p["lengths"]), axes["num_seqs"])
                    self.assertGreater(min(p["lengths"]), 0)
                elif name == "gdn_decode":
                    # B one-token sequences of the merged gdn package.
                    self.assertEqual(p["batch"], axes["batch_size"])
                    self.assertNotIn("lengths", p)
                elif name == "dsa_indexer":
                    self.assertEqual(len(p["lengths"]), axes["batch_size"])
                    self.assertEqual(max(p["lengths"]), axes["max_num_pages"] * 64)
                    self.assertEqual(p["width"], axes["max_num_pages"])
                    self.assertEqual(p["pages"], axes["num_pages"])
                    if len(p["lengths"]) > 2:
                        self.assertGreater(max(p["lengths"]), 2 * min(p["lengths"]))
                elif name == "dsa_attention":
                    self.assertEqual(p["tokens"], axes["num_tokens"])
                    self.assertEqual(p["pages"], axes["num_pages"])
                elif name == "moe":
                    self.assertEqual(p["tokens"], axes["seq_len"])
                    self.assertLessEqual(p["tokens"], 16384)
                else:
                    self.assertEqual(p["batch"], axes["batch_size"])


if __name__ == "__main__":
    unittest.main()

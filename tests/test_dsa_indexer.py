"""dsa_indexer: launch arguments, logits layout and the upstream subset.

``LaunchArguments`` builds both kernels' complete launches for real tensors
(nothing is launched; the logits kernel's TMA descriptors come from a
recording encoder because the sm_86 driver refuses
``cuTensorMapEncodeTiled``): the logits row stride follows DeepGEMM's
``align(max_context_len, 256)`` independently of the block-table stride, so
the official 43- and 91-page tables are served as they are.

``UpstreamCoverage`` runs DeepGEMM's own ``enumerate_paged_mqa_logits`` (from
the pinned ``tests/test_attention.py``, as on SM100) and requires every
parametrization these kernels serve among the cases.
"""

from __future__ import annotations

import ast
import ctypes
import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import cuda_driver  # noqa: E402
from harness.workload import ctypes_value  # noqa: E402
from harness.workloads import dsa_indexer as di  # noqa: E402

CUDA = torch.cuda.is_available()
DEEPGEMM_TEST = ROOT / "resources" / "DeepGEMM" / "tests" / "test_attention.py"


class RecordingEncoder:
    """Stands in for cuTensorMapEncodeTiled; records its arguments."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, dtype, address, dims, strides, box, element_strides, **kw):
        self.calls.append((dtype, address, list(dims), list(strides), list(box)))
        return cuda_driver.TensorMap()


def own_smoke(workload):
    own = {c.name for c in di.smoke_cases()}
    return [c for c in workload.get_cases() if c.name in own]


@unittest.skipUnless(CUDA, "launch arguments are built from CUDA tensors")
class LaunchArguments(unittest.TestCase):
    def sizes(self, spec, workload):
        sizes = [ctypes.sizeof(ctypes_value(a)) for a in spec.args]
        self.assertEqual(sizes, [p.size for p in workload.kernel_params])

    def test_metadata(self):
        workload = di.DSAIndexerMetadata.from_arch("sm_100a", device="cuda")
        for case in own_smoke(workload):
            with self.subTest(case=case.name):
                spec, (schedule,) = workload.configure_launch(workload.get_inputs(case))
                self.sizes(spec, workload)
                batch = len(case.params["lengths"])
                self.assertEqual((spec.grid, spec.block), (1, 1024))
                self.assertEqual(spec.shared_mem, (batch + 33) * 4)
                self.assertEqual(spec.args[0].value, batch)
                self.assertEqual(schedule.shape, (149, 2))

    def test_logits(self):
        workload = di.DSAIndexerLogits.from_arch("sm_100a", device="cuda")
        try:
            for case in own_smoke(workload):
                with self.subTest(case=case.name):
                    encoder = RecordingEncoder()
                    workload.tensor_map_encoder = encoder
                    inputs = workload.get_inputs(case)
                    q, cache, weights, lengths, table, schedule = inputs
                    spec, (logits,) = workload.configure_launch(inputs)
                    self.sizes(spec, workload)
                    batch, width = table.shape
                    columns = di.logits_width(width)
                    self.assertEqual(columns % 256, 0)
                    self.assertGreaterEqual(columns, width * di.PAGE_SIZE)
                    self.assertEqual(logits.shape, (batch, columns))
                    self.assertEqual(spec.args[1].value, columns)  # logits stride
                    self.assertEqual(spec.args[2].value, width)  # block-table stride
                    self.assertIs(spec.args[5], table)
                    self.assertEqual((spec.grid, spec.block), (148, 384))
                    self.assertEqual(spec.shared_mem, di.LOGITS_SMEM)
                    pages = cache.shape[0]
                    self.assertEqual(
                        [c[1:4] for c in encoder.calls],
                        [
                            (q.data_ptr(), [128, batch * 64], [128]),
                            (cache.data_ptr(), [128, 64, pages], [128, 8448]),
                            (cache.data_ptr() + 8192, [64, pages], [8448]),
                            (weights.data_ptr(), [64, batch], [256]),
                        ],
                    )
                    reference = workload.get_reference(inputs)[0]
                    self.assertEqual(reference.shape, logits.shape)
                    # Written columns: whole 256-token splits of each row.
                    for row, length in enumerate(lengths.tolist()):
                        written = (~torch.isnan(reference[row])).sum().item()
                        self.assertEqual(written, math.ceil(length / 256) * 256)
        finally:
            workload.tensor_map_encoder = staticmethod(
                cuda_driver.encode_tensor_map_tiled
            )

    def test_rejects_uncovered_sequence(self):
        workload = di.DSAIndexerLogits.from_arch("sm_100a", device="cuda")
        case = di.smoke_cases()[0]
        q, cache, weights, lengths, table, schedule = workload.get_inputs(case)
        with self.assertRaises(ValueError):
            workload.configure_launch(
                (q, cache, weights, lengths, table[:, :1].contiguous(), schedule)
            )


class Cases(unittest.TestCase):
    def test_smoke_paths(self):
        smoke = [c.params for c in di.smoke_cases()]
        lengths = [n for p in smoke for n in p["lengths"]]
        self.assertTrue({0, 1, 64, 256, 257} <= set(lengths))
        self.assertTrue(any(len(p["lengths"]) > 1024 for p in smoke))
        self.assertTrue(any(p.get("shared") for p in smoke))
        # Some case gives every scheduling partition more splits than the
        # five KV pipeline stages; some table is wider than its sequences.
        splits = [sum(math.ceil(n / 256) for n in p["lengths"]) for p in smoke]
        self.assertGreater(max(splits), 5 * di.NUM_SCHEDULING_PARTITIONS)
        self.assertTrue(
            any(p.get("width", 0) * 64 > max(p["lengths"]) + 256 for p in smoke)
        )
        self.assertTrue(any((p.get("width", 0) * 64) % 256 for p in smoke))

    def test_official_widths(self):
        """Trace rows keep the recorded block-table width (e.g. 43, 91)."""
        for case in di.throughput_cases():
            axes = case.source.get("axes")
            if axes:
                self.assertEqual(case.params["width"], axes["max_num_pages"])


class UpstreamCoverage(unittest.TestCase):
    def test_paged_mqa_logits_tests(self):
        tree = ast.parse(DEEPGEMM_TEST.read_text())
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "enumerate_paged_mqa_logits"
        )
        namespace = {"get_arch_major": lambda: 10, "torch": torch}
        exec(  # trusted pinned upstream test source
            compile(ast.Module([node], []), str(DEEPGEMM_TEST), "exec"), namespace
        )
        wanted = set()
        for (
            varlen,
            fmt,
            logits_dtype,
            weights_dtype,
            block_kv,
            context_2d,
            clean,
            batch,
            next_n,
            _,
            heads,
            head_dim,
            avg_kv,
        ) in namespace["enumerate_paged_mqa_logits"]():
            if (
                not varlen
                and fmt == "fp8"
                and logits_dtype == weights_dtype == torch.float
                and block_kv == di.PAGE_SIZE
                and context_2d
                and not clean
                and next_n == 1
                and (heads, head_dim) == (di.NUM_HEADS, di.HEAD_DIM)
            ):
                wanted.add((batch, avg_kv))
        self.assertEqual(wanted, set(di.UPSTREAM_PAGED))
        served = set()
        for case in di.upstream_cases():
            lengths = case.params["lengths"]
            for batch, avg in wanted:
                if len(lengths) == batch and all(
                    int(0.7 * avg) <= n < int(1.3 * avg) for n in lengths
                ):
                    served.add((batch, avg))
        self.assertEqual(served, wanted)
        names = {c.name for c in di.DSAIndexerLogits(None, device="cpu").get_cases()}
        self.assertLessEqual({c.name for c in di.upstream_cases()}, names)


if __name__ == "__main__":
    unittest.main()

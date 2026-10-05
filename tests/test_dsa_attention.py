"""dsa_attention: MLA launch arguments, case coverage and upstream subset.

``LaunchArguments`` builds the complete CUTLASS MLA launch for real tensors
with a recording TMA encoder (the sm_86 driver refuses
``cuTensorMapEncodeTiled``) and checks the five descriptors' encode inputs,
the by-value ``Params`` fields that depend on the tensors and the grid; the
``Params`` layout itself is byte-compared against the probe by
``check_param_layouts`` (``tests/test_workloads.py``). Nothing is launched.

``UpstreamCoverage`` extracts FlashInfer's ``test_cutlass_mla``
parametrizations from the pinned source and requires every one this pipeline
serves among the cases, and every distinct official inventory row among the
throughput cases.
"""

from __future__ import annotations

import ctypes
import struct
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.throughput import inventory  # noqa: E402
from harness.workload import ctypes_value  # noqa: E402
from harness.workloads import dsa_attention as da  # noqa: E402
from test_gqa_decode import upstream_parametrizations  # noqa: E402

CUDA = torch.cuda.is_available()
FLASHINFER = ROOT / "resources" / f"flashinfer-{da.FLASHINFER_REVISION}"


class RecordingEncoder:
    """Stands in for the driver's TMA encoding: records each recorded spec
    with the pointers and sizes it is resolved against."""

    def __init__(self):
        self.calls: list[tuple[dict, dict, dict]] = []

    def __call__(self, spec, pointers, sizes) -> bytes:
        self.calls.append((spec, dict(pointers), dict(sizes)))
        return bytes(128)


def field(layout: da.ParamLayout, params: bytes, name: str):
    entry = layout.fields[name]
    fmt = da.ParamLayout._formats[entry["kind"]]
    value = struct.unpack_from(fmt, params, entry["offset"])
    return value[0] if len(value) == 1 else value


@unittest.skipUnless(CUDA, "launch arguments are built from CUDA tensors")
class LaunchArguments(unittest.TestCase):
    def test_mla(self):
        workload = da.DSAAttentionMLA.from_arch("sm_100a", device="cuda")
        layout = workload.param_layout
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        own = {c.name for c in da.smoke_cases()}
        try:
            for case in [c for c in workload.get_cases() if c.name in own]:
                with self.subTest(case=case.name):
                    encoder = RecordingEncoder()
                    workload.tensor_map_encoder = encoder
                    inputs = workload.get_inputs(case)
                    padded_q, padded_qp, ckv, kpe, table, lengths, scale = inputs
                    (out, lse), spec = workload.setup_launch(inputs)
                    sizes = [ctypes.sizeof(ctypes_value(a)) for a in spec.args]
                    self.assertEqual(sizes, [p.size for p in workload.kernel_params])
                    tokens = case.params["tokens"]
                    self.assertEqual(spec.grid, (min(2 * tokens, sms), 1, 1))
                    self.assertEqual(spec.grid[0] % 2, 0)
                    self.assertEqual(
                        (spec.block, spec.cluster), ((256, 1, 1), (2, 1, 1))
                    )
                    self.assertEqual(spec.shared_mem, layout.constants["shared_mem"])
                    pointers = {
                        "q_latent": padded_q.data_ptr(),
                        "q_rope": padded_qp.data_ptr(),
                        "ckv": ckv.data_ptr(),
                        "kpe": kpe.data_ptr(),
                        "lengths": lengths.data_ptr(),
                        "page_table": table.data_ptr(),
                        "out": out.data_ptr(),
                        "lse": lse.data_ptr(),
                    }
                    expected_sizes = {
                        "batch": tokens,
                        "page_count": case.params["pages"] * da.PAGE,
                    }
                    self.assertEqual(len(encoder.calls), len(layout.raw["tensor_maps"]))
                    for (recorded, ptrs, size), tensor_map in zip(
                        encoder.calls, layout.raw["tensor_maps"].values()
                    ):
                        self.assertIs(recorded, tensor_map["encode"])
                        self.assertEqual(ptrs, pointers)
                        self.assertEqual(size, expected_sizes)
                    params = spec.args[0]
                    for name, value in (
                        ("problem_shape.B", tokens),
                        ("mainloop.ptr_q_latent", pointers["q_latent"]),
                        ("mainloop.ptr_c_latent", pointers["ckv"]),
                        ("mainloop.ptr_page_table", pointers["page_table"]),
                        ("mainloop.ptr_seq", pointers["lengths"]),
                        ("epilogue.ptr_o", pointers["out"]),
                        ("epilogue.ptr_lse", pointers["lse"]),
                        ("mainloop.page_count", expected_sizes["page_count"]),
                        ("tile_scheduler.num_blocks", 2 * tokens),
                        ("tile_scheduler.hw_info.sm_count", sms),
                    ):
                        self.assertEqual(field(layout, params, name), value, name)
                    self.assertEqual(
                        field(layout, params, "mainloop.softmax_scale"),
                        da._f32(float(scale)),
                    )
        finally:
            workload.tensor_map_encoder = staticmethod(da.encode_tensor_map)


class Cases(unittest.TestCase):
    def test_stage_limits(self):
        """sm_86 stages serve at most pack_sparse's 1024 tokens; together
        the stages serve every throughput case."""
        for cls in (da.DSAAttentionPackSparse, da.DSAAttentionDecode):
            tokens = [c.params["tokens"] for c in cls(None, device="cpu").get_cases()]
            self.assertLessEqual(max(tokens), da.PACK_CAPACITY)
        mla = {c.name for c in da.DSAAttentionMLA(None, device="cpu").get_cases()}
        self.assertLessEqual({c.name for c in da.throughput_cases()}, mla)

    def test_smoke_paths(self):
        """Scattered -1, all-invalid rows, duplicates, peaked queries, both
        scales and more MLA tiles than SMs are exercised by smoke inputs."""
        workload = da.DSAAttentionPack(None, device="cpu")
        smoke = [c for c in da.all_cases() if c.suite == "smoke"]
        interior = empty = duplicate = False
        for case in smoke:
            indices = workload.sparse_inputs(case)[4]
            valid = indices >= 0
            after_padding = (~valid).int().cummax(1).values.bool()
            interior |= bool((valid & after_padding).any())  # -1 before a valid ID
            empty |= bool((~valid.any(1)).any())
            for row in indices:
                ids = row[row >= 0]
                duplicate |= ids.unique().numel() < ids.numel()
        self.assertTrue(interior and empty and duplicate)
        params = [c.params for c in smoke]
        self.assertTrue(any(p["tokens"] > 74 for p in params))
        self.assertTrue(any(p.get("q_scale", 1) >= 3 for p in params))
        self.assertTrue(any(p.get("sm_scale") == da.YARN_SM_SCALE for p in params))
        self.assertTrue(any("sm_scale" not in p for p in params))


class UpstreamCoverage(unittest.TestCase):
    def test_cutlass_mla_tests(self):
        path = FLASHINFER / da.MLA_TEST.split("::")[0]
        wanted = {
            (p["batch_size"], p["max_seq_len"])
            for p in upstream_parametrizations(path, "test_cutlass_mla")
            if p["page_size"] == 1
            and p["dtype"] == torch.bfloat16
            and p["max_seq_len"] <= da.TOPK
        }
        self.assertEqual(wanted, {(b, n) for b in (1, 2, 4) for n in (128, 1024)})
        served = {
            (c.params["tokens"], c.params["valid"])
            for c in da.all_cases()
            if c.source.get("kind") == "upstream_test"
            and c.params.get("q_scale") == 100.0
            and c.params.get("duplicates")
        }
        self.assertLessEqual(wanted, served)

    def test_official_rows(self):
        """Every distinct inventory row is a throughput case with its scale."""
        rows, _ = inventory("dsa_attention")
        cases = {c.source.get("uuid"): c for c in da.throughput_cases()}
        for axes in {tuple(sorted(r["axes"].items())) for r in rows}:
            row = next(r for r in rows if tuple(sorted(r["axes"].items())) == axes)
            self.assertIn(row["uuid"], cases)
            params = cases[row["uuid"]].params
            self.assertEqual(params["sm_scale"], row["inputs"]["sm_scale"]["value"])


if __name__ == "__main__":
    unittest.main()

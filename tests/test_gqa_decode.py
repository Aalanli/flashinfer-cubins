"""gqa_decode: Python-built Params and launch arguments versus upstream host code.

``impls/gqa_decode/compiler.py`` records, next to the two upstream kernels'
cubins, the by-value ``Params`` bytes the upstream host code produced for
several problems (FlashInfer ``BatchDecodeParams`` on sm_86, CUTLASS
``Sm100FmhaFwdKernelTmaWarpspecialized::Params`` on sm_100a). The FMHA probe
replaces the opaque CUtensorMap encoding with a deterministic packing of the
encode arguments, mirrored by ``harness.workloads.gqa_decode.fake_encode``, so
the byte comparison covers every Params field and every descriptor argument.
None of this launches an sm_100a kernel: it runs on any machine; the complete
launch-argument checks for real tensors need a CUDA device (of any arch).

``UpstreamCoverage`` extracts the upstream test parametrizations (FlashInfer
paged decode and CUTLASS Blackwell FMHA tests) from the pinned sources and
requires every one this definition's kernels serve among the cases;
``upstream_parametrizations`` is shared with the other attention packages'
tests.
"""

from __future__ import annotations

import ast
import ctypes
import itertools
import math
import sys
import unittest
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import get_workload  # noqa: E402
from harness.workload import ctypes_value, device_matches  # noqa: E402
from harness.workloads import gqa_decode as g  # noqa: E402

CUDA = torch.cuda.is_available()
FLASHINFER = ROOT / "resources" / f"flashinfer-{g.FLASHINFER_REVISION}"


def upstream_function(path: Path, name: str) -> ast.FunctionDef:
    """The (possibly nested) function ``name`` of a pinned upstream file."""
    tree = ast.parse(path.read_text())
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def upstream_parametrizations(path: Path, name: str) -> list[dict[str, Any]]:
    """Every parameter combination of a ``@pytest.mark.parametrize``-decorated
    upstream test (the cartesian product of its decorators). Values are
    evaluated from the trusted pinned source with ``torch`` dtypes and
    ``math`` in scope."""
    axes = []
    for decorator in upstream_function(path, name).decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and getattr(decorator.func, "attr", "") == "parametrize"
        ):
            continue
        names = [n.strip() for n in ast.literal_eval(decorator.args[0]).split(",")]
        values = eval(  # trusted pinned upstream test source
            compile(ast.Expression(decorator.args[1]), str(path), "eval"),
            {"torch": torch, "math": math},
        )
        axes.append([dict(zip(names, v if len(names) > 1 else (v,))) for v in values])
    return [
        {k: v for combo in product for k, v in combo.items()}
        for product in itertools.product(*axes)
    ]


def sizes(args) -> list[int]:
    return [ctypes.sizeof(ctypes_value(a)) for a in args]


class RecordingEncoder:
    """Stands in for cuTensorMapEncodeTiled (refused with 801 on sm_86)."""

    def __init__(self):
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)
        return g.fake_encode(*args)


class ParamLayouts(unittest.TestCase):
    def test_params_match_upstream_host_code(self):
        self.assertEqual(
            g.check_param_layouts(),
            ["gqa_decode_batch_decode/sm_86", "gqa_decode_fmha/sm_100a"],
        )

    def test_comparison_is_sensitive(self):
        layout = g.ParamLayout.load(
            g.GQADecodeFMHA.cubin_path("sm_100a").with_suffix(".json")
        )
        example = layout.raw["examples"][0]
        built, _ = g.GQADecodeFMHA.build_params(
            layout,
            layout.sentinels(),
            batch=example["batch"] + 1,
            total_kv=example["total_kv"],
            sm_count=example["sm_count"],
            sm_scale=g._f32(example["sm_scale"]),
            encode=g.fake_encode,
            driver=example["driver_version"],
        )
        self.assertNotEqual(built, bytes.fromhex(example["bytes"]))

    def test_probe_runs_cover_driver_fixup(self):
        """Some recorded descriptors carry CUTLASS's bit-21 fix-up, some do not."""
        layout = g.ParamLayout.load(
            g.GQADecodeFMHA.cubin_path("sm_100a").with_suffix(".json")
        )
        fixed = unfixed = 0
        for example in layout.raw["examples"]:
            data = bytes.fromhex(example["bytes"])
            sizes_ = {"batch": example["batch"], "total_kv": example["total_kv"]}
            for name, spec in layout.raw["tensor_maps"].items():
                offset = layout.fields[name]["offset"]
                actual = data[offset : offset + 128]
                plain = g.encode_descriptor(
                    spec, layout.sentinels(), sizes_, g.fake_encode, 13020
                )
                if actual != plain:
                    fixed += 1
                    self.assertEqual(example["driver_version"], 13010)
                else:
                    unfixed += 1
        self.assertGreater(fixed, 0)
        self.assertGreater(unfixed, 0)
        drivers = {e["driver_version"] for e in layout.raw["examples"]}
        self.assertEqual(drivers, {13010, 13020})


@unittest.skipUnless(CUDA, "launch arguments are built from CUDA tensors")
class LaunchArguments(unittest.TestCase):
    """Complete launch arguments for real tensors; nothing is launched."""

    def smoke(self, workload):
        """The package's own smoke cases (upstream ones add no argument
        layout the own cases lack: same builders, other sizes)."""
        own = {c.name for c in g.smoke_cases()}
        return [c for c in workload.get_cases() if c.name in own]

    def check_spec(self, workload, spec):
        self.assertEqual(sizes(spec.args), [p.size for p in workload.kernel_params])
        self.assertTrue(all(d >= 1 for d in (*spec.grid, *spec.block)))

    def test_simple_kernels(self):
        expected = {
            "gqa_decode_plan": lambda b, sms: ((1, 1, 1), (1, 1, 1)),
            "gqa_decode_gather": lambda b, sms: ((b, 1, 1), (256, 1, 1)),
            "gqa_decode_finish": lambda b, sms: ((b, 1, 1), (256, 1, 1)),
        }
        for name, launch in expected.items():
            cls = get_workload(name)
            workload = cls.from_arch("sm_100a", device="cuda")
            self.assertEqual(workload.is_supported(), device_matches("sm_100a"))
            for case in self.smoke(workload):
                with self.subTest(workload=name, case=case.name):
                    _, spec = workload.setup_launch(workload.get_inputs(case))
                    self.check_spec(workload, spec)
                    batch = len(case.params["lengths"])
                    grid, block = launch(batch, workload.plan_sms())
                    self.assertEqual((spec.grid, spec.block), (grid, block))
                    self.assertEqual(spec.shared_mem, 0)

    def test_batch_decode(self):
        workload = g.GQADecodeBatchDecode.from_arch("sm_86", device="cuda")
        for case in self.smoke(workload):
            with self.subTest(case=case.name):
                inputs = workload.get_inputs(case)
                (out, lse), spec = workload.setup_launch(inputs)
                self.check_spec(workload, spec)
                batch = inputs[0].shape[0]
                self.assertEqual(spec.grid, (batch, g.KV_HEADS, 1))
                self.assertEqual(spec.block, (16, 8, 1))
                self.assertEqual(spec.shared_mem, 9216)
                params = spec.args[0]
                layout = workload.param_layout
                field = layout.fields
                for name, tensor in (("q", inputs[0]), ("o", out), ("lse", lse)):
                    offset = field[name]["offset"]
                    self.assertEqual(
                        int.from_bytes(params[offset : offset + 8], "little"),
                        tensor.data_ptr(),
                    )
                self.assertEqual(len(spec.keep), 4)

    def test_fmha(self):
        workload = g.GQADecodeFMHA.from_arch("sm_100a", device="cuda")
        layout = workload.param_layout
        for case in self.smoke(workload):
            for driver in (13010, 13020):
                with self.subTest(case=case.name, driver=driver):
                    encoder = RecordingEncoder()
                    workload.tensor_map_encoder = encoder
                    inputs = workload.get_inputs(case)
                    q, packed_k, packed_v, *_, scale = inputs
                    (out, lse), spec = workload.setup_launch(inputs, driver=driver)
                    self.check_spec(workload, spec)
                    sms = inputs[5].numel() - 1
                    self.assertEqual(spec.grid, (sms, 1, 1))
                    self.assertEqual(spec.block, (512, 1, 1))
                    self.assertEqual(spec.cluster, (1, 1, 1))
                    self.assertEqual(spec.shared_mem, layout.constants["shared_mem"])
                    self.assertEqual(len(spec.args[0]), 1920)
                    batch, total_kv = q.shape[0], packed_k.shape[0]
                    # Q, K, V, O descriptors with the actual tensor addresses.
                    addresses = [c[1] for c in encoder.calls]
                    self.assertEqual(
                        addresses,
                        [
                            q.data_ptr(),
                            packed_k.data_ptr(),
                            packed_v.data_ptr(),
                            out.data_ptr() - g.HEADS * g.DIM * 2,
                        ],
                    )
                    dims = [list(c[2]) for c in encoder.calls]
                    self.assertEqual(
                        dims,
                        [
                            [g.DIM, batch, g.GROUP, g.KV_HEADS],
                            [g.DIM, total_kv, g.KV_HEADS],
                            [g.DIM, total_kv, g.KV_HEADS],
                            [g.DIM, 1, g.GROUP, g.KV_HEADS, batch + 1],
                        ],
                    )
                    self.assertEqual(
                        out.data_ptr() - out.untyped_storage().data_ptr(), 8192
                    )
                    params = spec.args[0]
                    f = layout.fields
                    off = f["mainloop.scale_softmax"]["offset"]
                    self.assertEqual(
                        params[off : off + 4],
                        torch.tensor(float(scale)).numpy().tobytes(),
                    )
                    off = f["tile_scheduler.num_sm"]["offset"]
                    self.assertEqual(
                        int.from_bytes(params[off : off + 4], "little"), sms
                    )
        workload.tensor_map_encoder = g.driver_encode


class References(unittest.TestCase):
    def test_plan_independent_of_sm_count(self):
        """A plan for an SM count other than the device's (7) still covers
        every (head, request) once, and the stages compose to the result."""
        workload = g.GQADecodeFMHA(None, device="cpu")
        for case in g.smoke_cases()[:5]:
            with self.subTest(case=case.name):
                q, k, v, indptr, indices, scale = workload.official_inputs(case)
                qo, kv, work, tiles, heads, batches = g.plan_reference(indptr, 7)
                self.assertEqual(int(work[0]), 0)
                self.assertEqual(int(work[-1]), tiles.numel())
                self.assertTrue(bool((work[1:] >= work[:-1]).all()))
                packed_k, packed_v = g.gather_reference(k, v, indptr, indices, kv)
                out, lse = g.fmha_reference(q, packed_k, packed_v, qo, kv, float(scale))
                out, lse = g.finish_reference(out, indptr, lse)
                ref_out, ref_lse = g.gqa_reference(q, k, v, indptr, indices, scale)
                torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)

    def test_plan_reference(self):
        indptr = torch.tensor([0, 3, 3, 10], dtype=torch.int32)
        qo, kv, work, tiles, heads, batches = g.plan_reference(indptr, 5)
        self.assertEqual(qo.tolist(), [0, 1, 2, 3])
        self.assertEqual(kv.tolist(), [0, 3, 4, 11])
        self.assertEqual(work.tolist(), [math.floor(i * 96 / 5) for i in range(6)])
        self.assertEqual(heads[:33].tolist(), list(range(32)) + [0])
        self.assertEqual(batches[[0, 31, 32, 95]].tolist(), [0, 0, 1, 2])
        self.assertFalse(bool(tiles.any()))


class Cases(unittest.TestCase):
    def test_fmha_layouts_fit_int32(self):
        """Every case the FMHA stage serves fits its int32 packed-row layouts."""
        for case in g.GQADecodeFMHA(None, device="cpu").get_cases():
            lengths = case.params["lengths"]
            self.assertLessEqual(sum(lengths) + len(lengths), g.INT32_ROWS, case.name)

    def test_smoke_paths(self):
        """Smoke cases hold empty and length-1 requests, shared and identity
        page tables, peaked queries and more work items than SMs."""
        smoke = [c.params for c in g.all_cases() if c.suite == "smoke"]
        self.assertTrue(any(0 in p["lengths"] for p in smoke))
        self.assertTrue(any(p["lengths"][0] == 0 for p in smoke))
        self.assertTrue(any(p["lengths"][-1] == 0 for p in smoke))
        self.assertTrue(any(1 in p["lengths"] for p in smoke))
        self.assertTrue(any(p.get("shared") for p in smoke))
        self.assertTrue(any(p.get("contiguous") for p in smoke))
        self.assertTrue(any(p.get("q_scale", 1) >= 4 for p in smoke))
        self.assertTrue(any(len(p["lengths"]) * g.HEADS > 4 * 148 for p in smoke))


class UpstreamCoverage(unittest.TestCase):
    """Upstream test parametrizations served by these kernels are cases."""

    def served(self, test_file: str) -> set[tuple[int, int]]:
        return {
            (len(c.params["lengths"]), c.params["lengths"][0])
            for c in g.all_cases()
            if c.source.get("kind") == "upstream_test"
            and c.source["test"].startswith(test_file)
            and c.params.get("contiguous")
            and len(set(c.params["lengths"])) == 1
        }

    def test_batch_decode_tests(self):
        path = FLASHINFER / g.DECODE_TEST
        wanted = set()
        for name in (
            "test_batch_decode_with_paged_kv_cache",
            "test_batch_decode_with_paged_kv_cache_with_fast_plan",
            "test_batch_decode_with_tuple_paged_kv_cache",
            "test_cuda_graph_batch_decode_with_paged_kv_cache",
        ):
            for p in upstream_parametrizations(path, name):
                if (
                    p["page_size"] == 1
                    and (p["num_qo_heads"], p["num_kv_heads"]) == (g.HEADS, g.KV_HEADS)
                    and p["head_dim"] == g.DIM
                    and p["kv_layout"] == "NHD"
                    and p["pos_encoding_mode"] == "NONE"
                    and p.get("logits_soft_cap", 0.0) == 0.0
                ):
                    wanted.add((p["batch_size"], p["kv_len"]))
        self.assertEqual(len(wanted), 15)
        self.assertLessEqual(wanted, self.served(g.DECODE_TEST))

    def test_extreme_negative_logits_fixture(self):
        path = FLASHINFER / g.DECODE_TEST
        name = "test_paged_decode_extreme_negative_logits"
        dtypes = {p["dtype"] for p in upstream_parametrizations(path, name)}
        self.assertIn(torch.bfloat16, dtypes)
        source = ast.get_source_segment(path.read_text(), upstream_function(path, name))
        assert source is not None
        for fragment in (
            "num_qo_heads, num_kv_heads, head_dim = 32, 4, 128",
            "page_size, num_pages = 1, 17",
            "64.0",
            "-64.0",
            "[15, 16]",
        ):
            self.assertIn(fragment, source)
        (case,) = [c for c in g.all_cases() if c.source.get("test", "").count(name)]
        self.assertEqual(case.params["indices"], [15, 16])
        self.assertEqual(case.params["pages"], 17)
        self.assertEqual(case.params["fill"], {"q": 64.0, "k": -64.0, "v": 1.0})

    def test_blackwell_fmha_tests(self):
        path = FLASHINFER / g.FMHA_TEST
        wanted = {
            (p["batch_size"], p["kv_len"])
            for p in upstream_parametrizations(path, "test_blackwell_cutlass_fmha")
            if p["qo_len"] == 1
            and not p["causal"]
            and (p["head_dim_qk"], p["head_dim_vo"]) == (g.DIM, g.DIM)
            and p["dtype"] == torch.bfloat16
        }
        self.assertEqual(len(wanted), 25)
        self.assertLessEqual(wanted, self.served(g.FMHA_TEST))

    def test_official_latency_rows(self):
        """Every batch-1 inventory row is a throughput case."""
        from harness.throughput import inventory

        rows, _ = inventory("gqa_decode")
        wanted = {r["uuid"] for r in rows if r["axes"]["batch_size"] == 1}
        traced = {c.source.get("uuid") for c in g.throughput_cases()}
        self.assertEqual(len(wanted), 16)
        self.assertLessEqual(wanted, traced)


if __name__ == "__main__":
    unittest.main()

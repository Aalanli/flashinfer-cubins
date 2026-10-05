"""batched_gemm package checks that need no GPU.

* The trtllm-gen ``KernelParams`` (17408 / 1024 B), grids, launch attributes
  and every ``cuTensorMapEncodeTiled`` call rebuilt in Python match, byte for
  byte, the probes' recordings of FlashInfer v0.6.9's host code (the MoE
  runners themselves for the batched GEMMs) for every case of every
  registered sm_100a variant; the sm_86 ``GemmGrouped::Params`` match the
  CUTLASS probe's reference structs.
* Upstream coverage: every FlashInfer v0.6.9 test problem that may launch a
  kernel (the compile-time probe ran FlashInfer's tile selection and MoE
  runners on it) has a case of that kernel in the same regime -- the exact
  problem for one of each regime, every exact shape for the dense GEMMs and
  the segment GEMM -- and the recorded inventory is what the pinned tests
  parametrize.
* Hard paths: every kernel has smoke cases whose K loop wraps its pipeline
  and, with a persistent scheduler, more tiles than resident CTAs; every MoE
  kernel FlashInfer's tile selection can reach has a case it offers the tile
  for; throughput cases are model shapes within the memory budget.
* Weight preparation (gated interleave, epilogue shuffle, BlockMajorK) equals
  FlashInfer's own functions, executed from the pinned v0.6.9 sources.
* The GeGlu cubins compute the tanh form of GELU that the reference uses.
* Scale-factor layouts follow ``SfLayoutDecl.h``; E2m1 coding is exact.
* Routing tables satisfy the routing kernels' invariants.
* Every output format's validator accepts an ideally quantized output and
  rejects corrupted ones (wrong expert scale, swapped rows, zeroed block).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import enumerate_workloads, get_workload  # noqa: E402
from harness.workloads.batched_gemm import formats as fmt  # noqa: E402
from harness.workloads.batched_gemm import host, segment, trtllm, upstream  # noqa: E402

RESOURCES = ROOT / "resources"
FLASHINFER = RESOURCES / f"flashinfer-{upstream.REVISION}"
UPSTREAM_FILE = trtllm.PACKAGE_DIR / "variants" / "upstream.json"


def compiler():
    spec = importlib.util.spec_from_file_location(
        "impls_batched_gemm_compiler", ROOT / "impls/batched_gemm/compiler.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inventory() -> dict:
    return json.loads(UPSTREAM_FILE.read_text())


def moe_variants():
    return {
        n: e for n, e in trtllm.variant_index().items() if e["role"] in ("fc1", "fc2")
    }


def dense_variants():
    return {
        n: e for n, e in trtllm.variant_index().items() if e["role"].startswith("gemm")
    }


def flashinfer_functions(path: Path, names: set[str]) -> dict:
    """Execute the named top-level definitions of a pinned FlashInfer file."""
    tree = ast.parse(path.read_text())
    keep: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        )
    ]
    for node in keep:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    namespace: dict = {"torch": torch}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ParamLayouts(unittest.TestCase):
    def test_sm100a_fixtures(self):
        self.assertTrue(trtllm.variant_index(), "no sm_100a variants")
        self.assertTrue(trtllm.check_param_layouts())

    def test_sm86_fixtures(self):
        self.assertTrue(segment.variant_index(), "no sm_86 variants")
        self.assertTrue(segment.check_param_layouts())

    def test_registration(self):
        roles = Counter(e["role"] for e in trtllm.variant_index().values())
        # Every live kernel (compiler.py's table) is registered.
        self.assertEqual(roles["fc1"], 881)
        self.assertEqual(roles["fc2"], 482)
        self.assertEqual(
            {k: v for k, v in roles.items() if k.startswith("gemm")},
            {"gemm_fp4": 34, "gemm_mxfp8": 12, "gemm_fp8_blockscale": 4,
             "gemm_low_latency": 8},
        )  # fmt: skip
        names = {cls.name for cls in enumerate_workloads("sm_86")}
        self.assertTrue(set(segment.variant_index()) <= names)


def moe_params(problem: dict, entry: dict) -> dict:
    """An upstream MoE problem as a case of ``entry`` (the compiler's
    mapping, restated)."""
    fc1 = entry["role"] == "fc1"
    o = entry["options"]
    params = {
        "tokens": problem["tokens"],
        "top_k": problem["top_k"],
        "experts": problem["experts"],
        "hidden": problem["hidden"],
        "intermediate": problem["inter"],
    }
    if (problem["bias1"] if fc1 else problem["bias2"]) and host.moe_supports(o, "bias"):
        params["bias"] = 1
    if fc1 and problem["routing_scales"] and host.moe_supports(o, "routing_scales"):
        params["routing_scales"] = 1
    if problem["zero"]:
        params["zero"] = 1
    return params


class UpstreamCoverage(unittest.TestCase):
    """Upstream test problems are a subset of the harness cases (regimes)."""

    def test_moe_problems_have_a_case_in_their_regime(self):
        problems = inventory()["moe"]
        reached = set()
        for name, entry in moe_variants().items():
            regimes = {
                json.dumps(host.moe_regime(entry, c["params"])) for c in entry["cases"]
            }
            for i in entry["upstream"]:
                reached.add(i)
                regime = json.dumps(
                    host.moe_regime(entry, moe_params(problems[i], entry))
                )
                self.assertIn(regime, regimes, f"{name}: {problems[i]['tests'][0]}")
            for case in entry["cases"]:
                source = case["source"]
                if source["kind"] != "upstream_test":
                    continue
                problem = problems[source["inventory"]]
                self.assertIn(source["test"], problem["tests"])
                self.assertIn(source["inventory"], entry["upstream"])
                self.assertEqual(case["params"], moe_params(problem, entry), name)
        # Problems FlashInfer launches some registered kernel for.
        self.assertGreater(len(reached), 0.9 * len(problems))

    def test_dense_problems_are_cases(self):
        problems = inventory()["dense"]
        for name, entry in dense_variants().items():
            shapes = {(c["params"]["m"], c["params"]["n"], c["params"]["k"])
                      for c in entry["cases"]}  # fmt: skip
            for i in entry["upstream"]:
                p = problems[i]
                self.assertEqual(p["role"], entry["role"])
                self.assertIn(
                    (p["m"], p["n"], p["k"]), shapes, f"{name}: {p['tests'][0]}"
                )

    def test_segment_problems_are_cases(self):
        recorded = [
            c for c in next(iter(segment.variant_index().values()))["cases"]
            if c["source"]["kind"] == "upstream_test"
        ]  # fmt: skip
        self.assertEqual(len(recorded), 72)
        for name, entry in segment.variant_index().items():
            self.assertEqual(
                entry["cases"], next(iter(segment.variant_index().values()))["cases"]
            )
        if FLASHINFER.is_dir():
            # Exact problems; over 2 GiB of weights the same segments and dims
            # with fewer segments (the 6 GB sm_86 budget).
            got = {c["source"]["test"]: c for c in recorded}
            for problem in upstream.segment_problems(RESOURCES):
                case = got[problem["test"]]
                params = case["params"]
                self.assertEqual(
                    (params["n"], params["k"]), (problem["n"], problem["k"])
                )
                batch = case["source"].get("batch_reduced_from", len(params["lengths"]))
                self.assertEqual(len(problem["lengths"]), batch)
                self.assertEqual(set(params["lengths"]), set(problem["lengths"]))

    @unittest.skipUnless(
        FLASHINFER.is_dir(), "pinned FlashInfer v0.6.9 sources missing"
    )
    def test_inventory_is_current(self):
        """variants/upstream.json is what the pinned tests parametrize (the
        compiler wrote it from the same extraction)."""
        c = compiler()
        recorded = inventory()
        self.assertEqual(recorded["revision"], upstream.REVISION)
        self.assertEqual(
            recorded["moe"], c.merge_problems(upstream.moe_problems(RESOURCES))
        )
        self.assertEqual(
            recorded["dense"], c.merge_problems(upstream.dense_problems(RESOURCES))
        )

    @unittest.skipUnless(
        FLASHINFER.is_dir(), "pinned FlashInfer v0.6.9 sources missing"
    )
    def test_extraction(self):
        """The decorator evaluation and skip_checks port on known values."""
        path = FLASHINFER / "tests/moe/test_trtllm_gen_fused_moe.py"
        grid = dict(upstream.decorator_grid(path, "test_renormalize_routing"))
        self.assertEqual([v for _, (v,) in grid[("num_tokens",)]], [8, 768, 3072])
        problems = upstream.moe_problems(RESOURCES)
        tests = Counter(p["test"].split("::")[1].split("[")[0] for p in problems)
        # skip_checks keeps Geglu to NvFP4 x TopK routing with <= 128 tokens.
        geglu = [p for p in problems if p["act"] == "Geglu"]
        self.assertTrue(geglu)
        self.assertTrue(all(p["quant"] == "FP4_NVFP4_NVFP4" and p["tokens"] <= 128
                            for p in geglu))  # fmt: skip
        # SwiGlu problems of run_moe_test never exceed 1024 hidden/intermediate.
        self.assertFalse(
            [p for p in problems if p["act"] == "Swiglu" and "routed" not in p["test"]
             and "autotuner" not in p["test"] and max(p["hidden"], p["inter"]) > 1024]
        )  # fmt: skip
        self.assertNotIn("test_correctness_dpsk_fp8_fused_moe", tests)
        self.assertEqual(tests["test_llama4_routing"], 3)
        self.assertTrue(
            all(p["routing_scales"] for p in problems if "llama4" in p["test"])
        )
        self.assertEqual(tests["test_nvfp4_moe_gemm_bias"], 27)


class HardPaths(unittest.TestCase):
    def test_moe_smoke_paths(self):
        for name, entry in moe_variants().items():
            o = entry["options"]
            smoke = [c for c in entry["cases"] if c["suite"] == "smoke"]
            regimes = [host.moe_regime(entry, c["params"]) for c in smoke]
            with self.subTest(workload=name):
                self.assertTrue(
                    any(r[1] for r in regimes), "no smoke case wraps the K loop"
                )
                if o["mTileScheduler"] in host.SCHEDULERS_PERSISTENT:
                    self.assertTrue(
                        any(r[2] for r in regimes), "no CTA gets a second tile"
                    )
                for feature in ("bias", "gated_act", "routing_scales"):
                    if host.moe_supports(o, feature):
                        on = [c for c in smoke if c["params"].get(feature)]
                        self.assertTrue(on, f"{feature} never switched on")
                        self.assertTrue(len(on) < len(smoke), f"{feature} never off")

    def test_moe_dispatch_regime(self):
        """A case FlashInfer's tile selection offers this tile for, wherever
        the launcher's ladder holds the tile; the probe's selections equal
        the Python port's."""
        unreachable = json.loads(
            trtllm.PACKAGE_DIR.joinpath("provenance.json").read_text()
        )
        unreachable = unreachable["builds"]["sm_100a"]["unreachable_registered"]
        for name, entry in moe_variants().items():
            tile = entry["options"]["mTileN"]
            offered = []
            for c in entry["cases"]:
                p = c["params"]
                selected = host.selected_tiles(entry["ladder"], p["tokens"], p["top_k"],
                                               p["experts"])  # fmt: skip
                self.assertEqual(
                    selected, c["source"]["dispatch"]["selected_tiles"], name
                )
                if tile in selected and c["source"]["dispatch"]["moe_valid"]:
                    offered.append(c["name"])
            if tile in entry["ladder"]:
                self.assertIn("smoke_dispatch", offered, name)
                self.assertNotIn(name, unreachable)
            else:
                self.assertIn(name, unreachable)
        # The E4m3 per-tensor configs with tileN 192/256 (16 FC1, 8 FC2).
        per_tensor = [
            n for n, e in moe_variants().items()
            if e["launcher"] == "Fp8PerTensorLauncher" and e["options"]["mTileN"] > 128
        ]  # fmt: skip
        self.assertEqual(len(per_tensor), 24)
        self.assertLessEqual(set(per_tensor), set(unreachable))

    def test_dense_smoke_paths(self):
        for name, entry in dense_variants().items():
            o = entry["options"]
            deep = [c for c in entry["cases"] if c["name"] == "smoke_deep"]
            with self.subTest(workload=name):
                self.assertEqual(len(deep), 1)
                p = deep[0]["params"]
                k_tiles = host.ceil_div(p["k"], o["mTileK"] * o["mNumSlicesForSplitK"])
                self.assertGreater(k_tiles, host.k_stages(o))
                if o["mTileScheduler"] in host.SCHEDULERS_PERSISTENT:
                    tiles = host.ceil_div(p["n"], o["mTileM"]) * host.ceil_div(
                        p["m"], o["mTileN"]
                    )
                    self.assertGreater(tiles, host.resident_ctas(entry))

    def test_throughput_cases(self):
        c = compiler()
        for name, entry in trtllm.variant_index().items():
            with self.subTest(workload=name):
                kinds = {x["source"]["kind"] for x in entry["cases"]
                         if x["suite"] == "throughput"}  # fmt: skip
                # Model shapes; a K-offset model shape where no model K can
                # launch the config (dense tileK-128 configs).
                self.assertTrue(kinds & {"model_shape", "synthetic_stress"})
                for case in entry["cases"]:
                    if entry["role"].startswith("gemm"):
                        size = c.dense_case_bytes(case["params"])
                    else:
                        size = c.moe_case_bytes(entry, case["params"], host)
                    limit = (
                        c.SMOKE_BUDGET if case["suite"] == "smoke" else c.CASE_BUDGET
                    )
                    self.assertLessEqual(size, limit, case["name"])


@unittest.skipUnless(shutil.which("cuobjdump"), "cuobjdump not on PATH")
class GeGluForm(unittest.TestCase):
    def test_tanh_gelu(self):
        """Every GeGlu cubin evaluates phi as 0.5 (1 + tanh(sqrt(2/pi) (x +
        0.044715 x^3))): MUFU.TANH with both constants, as gelu_tanh_phi."""
        names = [n for n, e in moe_variants().items()
                 if e["options"]["mFusedAct"] and e["options"]["mActType"] == host.ACT_GEGLU]  # fmt: skip
        self.assertTrue(names)
        for name in names:
            path = trtllm.PACKAGE_DIR / "cubins/sm_100a" / f"{name}.cubin"
            sass = subprocess.run(
                ["cuobjdump", "-sass", str(path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            with self.subTest(workload=name):
                self.assertIn("MUFU.TANH", sass)
                self.assertIn("0x3f4c422a", sass)  # 0.7978845608 = sqrt(2 / pi)
                self.assertIn("0x3d372713", sass)  # 0.044715
                self.assertIsNone(re.search(r"\berff?\b", sass))
        x = torch.linspace(-4, 4, 101)
        phi = trtllm.gelu_tanh_phi(x)
        self.assertTrue(
            torch.allclose(x * phi, torch.nn.functional.gelu(x, approximate="tanh"))
        )


@unittest.skipUnless(FLASHINFER.is_dir(), "pinned FlashInfer v0.6.9 sources missing")
class WeightPreparation(unittest.TestCase):
    def test_shuffle_and_gated_rows(self):
        ns = flashinfer_functions(
            FLASHINFER / "flashinfer/utils.py",
            {
                "srcToDstBlk16RowMap",
                "srcToDstBlk32RowMap",
                "get_shuffle_block_size",
                "get_shuffle_matrix_a_row_indices",
            },
        )
        core = flashinfer_functions(
            FLASHINFER / "flashinfer/fused_moe/core.py",
            {
                "get_reorder_rows_for_gated_act_gemm_row_indices",
                "convert_to_block_layout",
            },
        )
        for rows in (64, 256, 512):
            x = torch.zeros(rows, 4)
            for epilogue in (64, 128):
                expected = ns["get_shuffle_matrix_a_row_indices"](x, epilogue)
                self.assertTrue(torch.equal(fmt.shuffle_rows(rows, epilogue), expected))
            expected = core["get_reorder_rows_for_gated_act_gemm_row_indices"](x)
            self.assertTrue(torch.equal(fmt.gated_rows(rows), expected))
            # _maybe_get_cached_w3_w1_permute_indices: permute0[permute1]
            o = {"mFusedAct": 1, "mRouteImpl": host.ROUTE_TMA,
                 "mUseShuffledMatrix": 1, "mEpilogueTileM": 128}  # fmt: skip
            combined = expected[ns["get_shuffle_matrix_a_row_indices"](x, 128)]
            self.assertTrue(torch.equal(trtllm.weight_rows(o, rows), combined))
        w = torch.arange(2 * 64 * 256).reshape(2, 64, 256)
        for block in (64, 128):
            blocked = fmt.to_block_major_k(w, block)
            for e in range(2):
                ref = core["convert_to_block_layout"](w[e], block)
                self.assertTrue(torch.equal(blocked[e], ref))
            self.assertTrue(torch.equal(fmt.from_block_major_k(blocked), w))


class Formats(unittest.TestCase):
    def test_sf_layout_offsets(self):
        # SfLayoutDecl.h: R128c4 (i/128, j/4, i%32, (i%128)/32, j%4) and
        # R8c4 (i/8, j/4, i%8, j%4) in [rows padded, cols padded to 4].
        off = fmt.sf_offsets(256, 8, fmt.SF_R128C4, torch.device("cpu"))
        self.assertEqual(int(off[0, 0]), 0)
        self.assertEqual(int(off[1, 0]), 16)
        self.assertEqual(int(off[32, 0]), 4)
        self.assertEqual(int(off[0, 1]), 1)
        self.assertEqual(int(off[0, 4]), 512)
        self.assertEqual(int(off[128, 0]), 1024)
        off = fmt.sf_offsets(16, 8, fmt.SF_R8C4, torch.device("cpu"))
        self.assertEqual(int(off[1, 0]), 4)
        self.assertEqual(int(off[0, 4]), 32)
        self.assertEqual(int(off[8, 0]), 64)
        for layout in (fmt.SF_LINEAR, fmt.SF_R8C4, fmt.SF_R128C4):
            sf = torch.randint(0, 255, (200, 12), dtype=torch.uint8)
            flat = fmt.sf_to_layout(sf, layout)
            self.assertTrue(torch.equal(fmt.sf_from_layout(flat, 200, 12, layout), sf))
            self.assertEqual(
                len(set(fmt.sf_offsets(200, 12, layout, sf.device).flatten().tolist())),
                2400,
            )
        # Per-expert R128c4 buffers concatenate (rows a multiple of 128).
        sf = torch.randint(0, 255, (3 * 256, 12), dtype=torch.uint8)
        flat = fmt.sf_to_layout(sf, fmt.SF_R128C4)
        part = trtllm.expert_slice(flat, 1, 256, 12, fmt.SF_R128C4)
        self.assertTrue(
            torch.equal(fmt.sf_from_layout(part, 256, 12, fmt.SF_R128C4), sf[256:512])
        )

    def test_e2m1(self):
        codes = torch.arange(16, dtype=torch.uint8)
        values = fmt.e2m1_decode(codes)
        back = fmt.e2m1_encode(values)
        self.assertTrue(torch.equal(fmt.e2m1_decode(back), values))
        # round to nearest, ties to the even code; saturation at 6
        x = torch.tensor([0.25, 0.75, 1.25, 2.5, 3.5, 5.0, 7.0, -0.74])
        self.assertEqual(fmt.e2m1_decode(fmt.e2m1_encode(x)).tolist(),
                         [0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 6.0, -0.5])  # fmt: skip
        self.assertTrue(torch.equal(fmt.unpack_nibbles(fmt.pack_nibbles(codes)), codes))
        self.assertEqual(
            int(fmt.pack_nibbles(torch.tensor([1, 2], dtype=torch.uint8))), 0x21
        )


class Routing(unittest.TestCase):
    def test_tables(self):
        for tokens, top_k, experts, tile, skew, empty in (
            (3, 2, 4, 8, 0.0, 0),
            (37, 2, 4, 16, 0.0, 0),
            (512, 4, 16, 128, 0.5, 0),
            (900, 2, 8, 8, 1.0, 1),
        ):
            r = trtllm.make_routing(tokens, top_k, experts, tile, 5, skew, empty)
            ctas = int(r.num_non_exiting_ctas[0])
            self.assertEqual(int(r.total_num_padded_tokens[0]), ctas * tile)
            self.assertLessEqual(
                ctas, host.max_num_ctas_in_batch_dim(tokens, top_k, experts, tile)
            )
            slots = r.expanded_idx_to_permuted_idx.long()
            self.assertEqual(len(set(slots.tolist())), tokens * top_k)
            batch = r.cta_idx_xy_to_batch_idx.long()
            limit = r.cta_idx_xy_to_mn_limit.long()
            for i, slot in enumerate(slots.tolist()):
                self.assertEqual(int(r.permuted_idx_to_token_idx[slot]), i // top_k)
                self.assertLess(slot, int(limit[slot // tile]))
                self.assertGreaterEqual(slot, (slot // tile) * tile)
            self.assertTrue(bool((batch[1:ctas] >= batch[: ctas - 1]).all()))
            if empty:
                self.assertFalse(bool((batch[:ctas] >= experts - empty).any()))


class Validators(unittest.TestCase):
    """Ideal quantization passes, corrupted outputs fail, per output kind."""

    def samples(self):
        seen: dict = {}
        for name, entry in moe_variants().items():
            key = (trtllm.output_kind(entry["options"]), entry["options"]["mSfLayoutC"])
            seen.setdefault(key, name)
        return seen.values()

    def test_moe_outputs(self):
        kinds = set()
        for name in self.samples():
            cls = get_workload(name)
            assert issubclass(cls, trtllm.MoEGemm)
            w = cls(None, device="cpu")
            kinds.add(trtllm.output_kind(cls.entry["options"]))
            case = next(c for c in w.get_cases() if c.name == "smoke_ragged")
            ref = w.get_reference(w.get_inputs(case))
            expected, total = ref[-2], int(ref[-1][0])
            with self.subTest(workload=name):
                w.validate(ref, ref)
                rows = (~torch.isnan(expected).any(-1)).nonzero().flatten()
                # Swapped rows (a wrong permutation slot).
                bad = expected.clone()
                bad[rows[0]], bad[rows[-1]] = expected[rows[-1]], expected[rows[0]]
                with self.assertRaises(AssertionError):
                    w.validate(ref, w.encode_output(bad, total))
                # One row scaled by 2 (a wrong per-expert scale).
                bad = expected.clone()
                bad[rows[0]] *= 2
                with self.assertRaises(AssertionError):
                    w.validate(ref, w.encode_output(bad, total))
                # A zeroed scale-factor block / element block.
                bad = expected.clone()
                bad[rows[0], :32] = 0
                if expected[rows[0], :32].abs().max() > 0:
                    with self.assertRaises(AssertionError):
                        w.validate(ref, w.encode_output(bad, total))
                if trtllm.output_kind(cls.entry["options"]) == "mxfp8":
                    # The split-K kernels' OCP MX floor scales (saturating).
                    w.validate(ref, w.encode_output(expected, total, mx_floor=True))
        self.assertEqual(kinds, {"bf16", "e4m3", "ds_e4m3", "nvfp4", "mxfp8"})

    def test_block_scaled_edge_cases(self):
        """Patterns seen on the B200: a block whose E4m3 scale underflows is
        flushed, one whose subnormal scale rounds down clips at 6; both are
        correct rounding at the kernel's scale. Wrong scales and elements
        more than one code off still fail."""
        check = trtllm.check_block_scaled
        e = torch.zeros(2, 32)
        e[:, 11] = 0.00106  # amax / 6 < 2**-10: scale e4m3 0
        e[:, 16 + 12] = 0.0164  # amax / 6 = 0.0027 -> 2**-9, 8.4 codes -> 6
        a = torch.zeros_like(e)
        a[:, 28] = 6 * 2.0**-9
        s = torch.tensor([[0.0, 2.0**-9]] * 2)
        check(e, a, s, "nvfp4", "edge")
        with self.assertRaises(AssertionError):  # scale two ulps off
            check(e, a * 3, s * 3, "nvfp4", "edge")
        x = torch.linspace(-3, 3, 64).reshape(2, 32)
        scale = torch.full((2, 1), 2.0**-6)
        q = trtllm.round_at_scale(x, scale.repeat_interleave(32, -1), "e4m3")
        check(x, q, scale, "mxfp8", "mx")  # ceil scale of amax 3
        check(x, trtllm.round_at_scale(x, (scale / 2).repeat_interleave(32, -1), "e4m3"),
              scale / 2, "mxfp8", "mx")  # fmt: skip
        with self.assertRaises(AssertionError):  # scale outside one binade
            check(x, trtllm.round_at_scale(x, (scale * 4).repeat_interleave(32, -1), "e4m3"),
                  scale * 4, "mxfp8", "mx")  # fmt: skip
        bad = q.clone()
        bad[0, 31] += 2 * 2.0**-6 * 2.0**-3 * 2**7  # two codes at |x| ~ 3
        with self.assertRaises(AssertionError):
            check(x, bad, scale, "mxfp8", "mx")


if __name__ == "__main__":
    unittest.main()

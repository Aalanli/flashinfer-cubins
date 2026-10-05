"""MoE package: parameter builders against the upstream host code, upstream
dispatch, case coverage of the upstream tests and inventory, and the stripped
artifacts.

* Every workload's launch (grid, block, shared memory, cooperative flag and
  every parameter byte upstream defines) is rebuilt in Python from the
  compile-time probe's fake pointers and compared with the probe's recording
  of FlashInfer's own host code (``cubins/<arch>/<workload>.fixtures.json``).
  Bytes upstream leaves indeterminate and struct padding are excluded; TMA
  descriptors use the probe's deterministic stand-in encoding. The launch a
  workload builds from real case inputs matches the same recording except
  for addresses, which must be those of the case's tensors.
* Each workload serves exactly the token counts upstream launched it for, and
  every case's (tokens, offset) is probed, so its kernel choice is upstream's.
* The cases contain FlashInfer's own test parametrizations of this pipeline
  and every official inventory row; the smoke cases reach the hard paths
  (multi-tile and empty experts, partial tiles, offset range ends, persistent
  grids larger than the GPU, a rank without tokens, exact ties).
* The routing references follow upstream's tie order and hand-derived tables.
* The stripped official sm_100a GEMM cubins keep the kernel's SASS, constant
  bank, Mercury info and capsule (the ELF strip itself is proven on sm_86 in
  ``tests/test_cubin_strip.py``).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import cubin_strip, get_workload  # noqa: E402
from harness.models import MODELS  # noqa: E402
from harness.throughput import inventory  # noqa: E402
from harness.workloads import moe  # noqa: E402

PACKAGE = ROOT / "impls" / "moe"
# Kernels that together serve every token count, one dispatch family each.
FAMILIES = (
    ("moe_routing_main",),
    ("moe_routing_cluster", "moe_routing_coop"),
    ("moe_gemm1", "moe_gemm1_persistent"),
    ("moe_activation",),
    ("moe_gemm2", "moe_gemm2_persistent"),
    ("moe_finalize", "moe_finalize_vec"),
)
NAMES = (
    "moe_routing_main",
    "moe_routing_cluster",
    "moe_routing_coop",
    "moe_gemm1",
    "moe_gemm1_persistent",
    "moe_activation",
    "moe_gemm2",
    "moe_gemm2_persistent",
    "moe_finalize",
    "moe_finalize_vec",
)


def load_compiler():
    spec = importlib.util.spec_from_file_location(
        "impls_moe_compiler", PACKAGE / "compiler.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMPILER = load_compiler()


def fixtures(name: str, arch: str) -> list[dict]:
    path = get_workload(name).cubin_path(arch).with_suffix(".fixtures.json")
    return json.loads(path.read_text())["fixtures"]


def ignored_bytes(launch: dict, layout: dict) -> set[int]:
    """Indeterminate bytes of upstream's struct and padding of the layout."""
    holes = {i for a, b in launch["padding"] for i in range(a, b)}
    covered: set[int] = set()
    for f in layout["fields"].values():
        covered.update(range(f["offset"], f["offset"] + f["size"]))
    return holes | (set(range(layout["size"])) - covered)


def rebuild(workload, fixture: dict) -> moe.LaunchSpec:
    """The workload's launch for the probe's fake pointers."""
    t, offset, p = (
        fixture["tokens"],
        fixture["local_expert_offset"],
        fixture["pointers"],
    )
    (scale,) = struct.unpack("<f", bytes.fromhex(fixture["routed_scaling_factor_hex"]))
    if isinstance(workload, moe._GEMM1):
        return workload.gemm_launch(
            moe.fake_encode,
            tokens=t,
            a=p["gemm1_weights"],
            sf_a=p["gemm1_weights_scale"],
            b=p["hidden_states"],
            sf_b=p["hidden_states_scale"],
            c=p["gemm1_output"],
            sf_c=p["gemm1_output_scale"],
            route_map=p["permuted_idx_to_token_idx"],
            per_token_sf_b=p["expert_weights"],
            total_num_padded_tokens=p["total_num_padded_tokens"],
            cta_idx_xy_to_batch_idx=p["cta_idx_xy_to_batch_idx"],
            cta_idx_xy_to_mn_limit=p["cta_idx_xy_to_mn_limit"],
            num_non_exiting_ctas=p["num_non_exiting_ctas"],
        )
    if isinstance(workload, moe._GEMM2):
        return workload.gemm_launch(
            moe.fake_encode,
            tokens=t,
            a=p["gemm2_weights"],
            sf_a=p["gemm2_weights_scale"],
            b=p["activation_output"],
            sf_b=p["activation_output_scale"],
            c=p["gemm2_output"],
            sf_c=0,
            route_map=0,
            per_token_sf_b=0,
            total_num_padded_tokens=p["total_num_padded_tokens"],
            cta_idx_xy_to_batch_idx=p["cta_idx_xy_to_batch_idx"],
            cta_idx_xy_to_mn_limit=p["cta_idx_xy_to_mn_limit"],
            num_non_exiting_ctas=p["num_non_exiting_ctas"],
        )
    if isinstance(workload, (moe.MoERoutingMain, moe._RoutingIndices)):
        pointers = {k: p[k] for k in moe._routing_pointers()}
        return workload.launch_spec(t, offset, scale, pointers)
    return workload.launch_spec(t, p)


class ParamsMatchUpstream(unittest.TestCase):
    def test_launches(self):
        for name in NAMES:
            cls = get_workload(name)
            for arch in cls.supported_arches:
                workload = cls.from_arch(arch, device="cpu")
                layout = workload.sidecar["params"]
                for fixture in fixtures(name, arch):
                    with self.subTest(
                        workload=name, arch=arch, tokens=fixture["tokens"]
                    ):
                        launch = fixture["launch"]
                        spec = rebuild(workload, fixture)
                        self.assertEqual(list(spec.grid), launch["grid"])
                        self.assertEqual(list(spec.block), launch["block"])
                        self.assertEqual(spec.shared_mem, launch["shared_mem"])
                        attrs = {a["id"]: a["value"] for a in launch["attrs"]}
                        self.assertEqual(spec.cooperative, attrs.get(2) == 1)
                        if 4 in attrs:  # CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                            self.assertEqual(list(spec.cluster or ()), attrs[4])
                            self.assertEqual(attrs.get(5), 0)  # policy: default
                            self.assertEqual(attrs.get(6), 0)  # no PDL (GEMMs)
                        else:  # runtime launch: upstream enables PDL; we do not
                            self.assertEqual(attrs.get(6), 1)
                            self.assertFalse(workload.sidecar["use_pdl"])
                        want = bytes.fromhex(launch["params_hex"])
                        self.assertEqual(len(spec.params), len(want))
                        skip = ignored_bytes(launch, layout)
                        diff = [
                            i
                            for i in range(len(want))
                            if i not in skip and spec.params[i] != want[i]
                        ]
                        self.assertEqual(diff, [], f"first differing bytes {diff[:8]}")
                        # Sizes as the cubin's EIATTR_KPARAM_INFO declares.
                        self.assertEqual(
                            [len(spec.params)],
                            [p.size for p in workload.kernel_params],
                        )

    def test_gemm_unset_members_are_indeterminate(self):
        """Fields the builder leaves zero for the GEMMs are exactly the members
        upstream never sets (reported indeterminate by the probe)."""
        for name in ("moe_gemm1", "moe_gemm2"):
            workload = get_workload(name).from_arch("sm_100a", device="cpu")
            fields = workload.sidecar["params"]["fields"]
            for fixture in fixtures(name, "sm_100a"):
                holes = {
                    i for a, b in fixture["launch"]["padding"] for i in range(a, b)
                }
                unset = ["tmaSfA", "tmaSfB", "ctaIdxXyToBatchIdx", "ctaIdxXyToMnLimit"]
                if name == "moe_gemm1":
                    unset.append("tmaB")  # ldgsts route: no B descriptor
                for f in unset:
                    span = range(
                        fields[f]["offset"], fields[f]["offset"] + fields[f]["size"]
                    )
                    self.assertTrue(set(span) <= holes, f"{name}.{f}")


class UpstreamDispatch(unittest.TestCase):
    def test_serves_matches_probe(self):
        probed = {t for t, _ in COMPILER.FIXTURE_CASES}
        for name in NAMES:
            cls = get_workload(name)
            for arch in cls.supported_arches:
                with self.subTest(workload=name, arch=arch):
                    launched = {f["tokens"] for f in fixtures(name, arch)}
                    served = {t for t in probed if cls.serves(t)}
                    self.assertEqual(launched, served)

    def test_every_case_is_probed(self):
        """Each case's (tokens, offset) ran through upstream's host code, so
        the probe chose this workload's kernel for it (and recorded the
        launch as a fixture)."""
        probed = set(COMPILER.FIXTURE_CASES)
        for name in NAMES:
            cls = get_workload(name)
            for arch in cls.supported_arches:
                recorded = {
                    (f["tokens"], f["local_expert_offset"])
                    for f in fixtures(name, arch)
                }
                for case in cls(None, device="cpu").get_cases():
                    key = (case.params["tokens"], case.params["offset"])
                    with self.subTest(workload=name, arch=arch, case=case.name):
                        self.assertIn(key, probed)
                        self.assertIn(key, recorded)

    def test_families_partition_the_cases(self):
        """Within each dispatch family every smoke and throughput case belongs
        to exactly one kernel, and each kernel has a smoke case."""
        every = sorted(c.name for c in moe.smoke_cases() + moe.throughput_cases())
        self.assertEqual(len(every), len(set(every)))
        for family in FAMILIES:
            with self.subTest(family=family):
                names = []
                for name in family:
                    cases = get_workload(name)(None, device="cpu").get_cases()
                    self.assertTrue(any(c.suite == "smoke" for c in cases), name)
                    names += [c.name for c in cases]
                self.assertEqual(sorted(names), every)

    def test_gemm_configs_are_the_selected_cubins(self):
        for name in (
            "moe_gemm1",
            "moe_gemm1_persistent",
            "moe_gemm2",
            "moe_gemm2_persistent",
        ):
            sidecar = get_workload(name).from_arch("sm_100a", device="cpu").sidecar
            spec = next(s for s in COMPILER.KERNELS if s.workload == name)
            self.assertEqual(sidecar["gemm_options"]["sha256"], spec.gemm_sha256)
            self.assertEqual(sidecar["symbol"], spec.probe_kernel)
            self.assertEqual(
                sidecar["stripped_kernels"], [spec.probe_kernel + "GetSmemSize"]
            )


class HostHelpers(unittest.TestCase):
    def test_int_fast_div(self):
        for d in (1, 2, 3, 7, 8, 13, 100, 1000):
            div, magic, shift, add = moe.int_fast_div(d)
            for n in range(0, 5000, 7):
                q = ((magic * n) >> 32) + n * add
                if shift >= 0:
                    q >>= shift
                    q += (q & 0xFFFFFFFF) >> 31
                self.assertEqual(q, n // d, (d, n))

    def test_tie_order_is_upstreams(self):
        """Upstream's TopKRedType packs ``65535 - idx`` below the value bits,
        so of tied groups and tied experts the lower index wins. Groups 3, 4
        and 7 tie (top-2 logits 1, 1) for the last two kept places: 3 and 4
        win. Of the five experts at logit 1 in kept groups (97, 98, 116, 135,
        136) the four lowest are selected, after the strictly larger ones."""
        logits = torch.full((1, moe.NUM_EXPERTS), -4.0)
        for experts, value in (
            ((197, 201), 3.0),  # group 6
            ((32, 63), 2.0),  # group 1
            ((97, 98, 116, 135, 136, 224, 225), 1.0),  # groups 3, 4, 7
        ):
            logits[0, list(experts)] = value
        bias = torch.zeros(moe.NUM_EXPERTS, dtype=torch.bfloat16)
        packed, weights = moe.routing_main_reference(logits, bias, 2.5)
        idx, score = moe.unpack_expert_indexes(packed)
        self.assertEqual(idx[0].tolist(), [197, 201, 32, 63, 97, 98, 116, 135])
        torch.testing.assert_close(score, weights, rtol=0, atol=0)
        # Exact ties do not count as near-ties: the row needs no resampling.
        _, _, margin = moe._routing_choice(logits, bias)
        self.assertGreater(float(margin[0]), moe.TIE_MARGIN)

    def test_indices_reference_by_hand(self):
        """Three tokens, offset 96: local experts 0 (2 routes), 4 (3) and 31
        (1) get one CTA each, rows in expanded-index order."""
        experts = [
            [96, 100, 5, 6, 7, 8, 9, 10],
            [100, 1, 2, 3, 4, 11, 12, 13],
            [96, 127, 100, 14, 15, 16, 17, 18],
        ]
        packed = (torch.tensor(experts, dtype=torch.int32) << 16) | 0x3F80
        r = moe.routing_indices_reference(packed, 96)
        expanded = torch.full((24,), -1, dtype=torch.int32)
        for position, row in ((0, 0), (16, 1), (1, 8), (8, 9), (18, 10), (17, 16)):
            expanded[position] = row
        self.assertEqual(r.expanded_idx_to_permuted_idx.tolist(), expanded.tolist())
        to_token = r.permuted_idx_to_token_idx
        self.assertEqual(
            [int(to_token[row]) for row in (0, 1, 8, 9, 10, 16)], [0, 2, 0, 1, 2, 2]
        )
        self.assertEqual(int(r.total_num_padded_tokens[0]), 24)
        self.assertEqual(int(r.num_non_exiting_ctas[0]), 3)
        self.assertEqual(r.cta_idx_xy_to_batch_idx[:3].tolist(), [0, 4, 31])
        self.assertEqual(r.cta_idx_xy_to_mn_limit[:3].tolist(), [2, 11, 17])


class StrippedArtifacts(unittest.TestCase):
    def _sidecars(self):
        for name in NAMES:
            cls = get_workload(name)
            for arch in cls.supported_arches:
                path = cls.cubin_path(arch)
                yield name, arch, path, json.loads(
                    path.with_suffix(".json").read_text()
                )

    def test_sidecars_and_provenance_match_cubins(self):
        provenance = json.loads((PACKAGE / "provenance.json").read_text())
        for name, arch, path, sidecar in self._sidecars():
            with self.subTest(workload=name, arch=arch):
                digest = COMPILER._sha256(path.read_bytes())
                self.assertEqual(digest, sidecar["cubin_sha256"])
                record = provenance["builds"][arch]["kernels"][name]
                self.assertEqual(digest, record["sha256"])

    def test_gemm_kernel_bytes_unchanged(self):
        compiler = COMPILER.ImplCompiler("sm_100a")
        for spec in COMPILER.KERNELS:
            if spec.gemm_cubin is None:
                continue
            path = compiler.artifacts / spec.gemm_cubin
            if not path.is_file():
                self.skipTest(f"official cubin not in {compiler.artifacts}")
            with self.subTest(workload=spec.workload):
                original = compiler.gemm_image(spec)
                stripped = (
                    get_workload(spec.workload).cubin_path("sm_100a").read_bytes()
                )
                self.assertEqual(
                    cubin_strip.strip_cubin(original, spec.probe_kernel), stripped
                )
                cubin_strip.check_strip(original, stripped, spec.probe_kernel)
                keep = spec.probe_kernel
                # .nv.info: only EIATTR_PARAM_CBANK's symbol word differs, and it
                # names the kept constant bank in both.
                info_a = cubin_strip.section_bytes(original, ".nv.info." + keep)
                info_b = cubin_strip.section_bytes(stripped, ".nv.info." + keep)
                diff = [i for i in range(len(info_a)) if info_a[i] != info_b[i]]
                self.assertEqual(len(info_a), len(info_b))
                self.assertLessEqual(len(diff), 4)
                capsule = cubin_strip.section_bytes(
                    stripped, ".nv.capmerc.text." + keep
                )
                elf = cubin_strip._parse_elf(stripped)
                text_index = next(
                    s.index for s in elf.sections if s.name == ".text." + keep
                )
                self.assertEqual(struct.unpack_from("<I", capsule)[0], text_index)
                if shutil.which("nvdisasm"):
                    with tempfile.TemporaryDirectory() as tmp:
                        a, b = Path(tmp) / "a.cubin", Path(tmp) / "b.cubin"
                        a.write_bytes(original)
                        b.write_bytes(stripped)
                        sass = [
                            subprocess.run(
                                ["cuobjdump", "-sass", str(p)],
                                capture_output=True,
                                text=True,
                                check=True,
                            ).stdout
                            for p in (a, b)
                        ]
                        functions = [re.split(r"\n\s*Function : ", s) for s in sass]
                        pick = lambda parts: next(  # noqa: E731
                            p for p in parts if p.startswith(keep + "\n")
                        )
                        self.assertEqual(pick(functions[0][1:]), pick(functions[1][1:]))

    def test_nvcc_kernels_line_tables_kept(self):
        """-lineinfo kernel cubins: the cub EmptyKernel was stripped, the kept
        kernel's DWARF line program survives and the tools parse it."""
        if not shutil.which("nvdisasm"):
            self.skipTest("nvdisasm not on PATH")
        for name, arch, path, sidecar in self._sidecars():
            if name.startswith("moe_gemm"):
                continue
            with self.subTest(workload=name, arch=arch):
                self.assertEqual(len(sidecar["stripped_kernels"]), 1)
                self.assertIn("EmptyKernel", sidecar["stripped_kernels"][0])
                listing = subprocess.run(
                    ["nvdisasm", "-g", str(path)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                self.assertIn("//## File", listing)
                self.assertNotIn("EmptyKernel", listing.split(".nv_debug_ptx_txt")[0])


UPSTREAM_TESTS = (
    ROOT
    / "resources"
    / COMPILER.FLASHINFER_DIR
    / "tests"
    / "test_trtllm_gen_fused_moe.py"
)


class UpstreamCoverage(unittest.TestCase):
    """The official definition, every inventory row and FlashInfer's own test
    parametrizations of this pipeline are cases of the kernels serving them
    (which kernel serves a case is checked against the probe above)."""

    def test_official_definition(self):
        axes = json.loads((ROOT / "resources" / "moe.json").read_text())["axes"]
        const = {k: v["value"] for k, v in axes.items() if v["type"] == "const"}
        self.assertEqual(
            const,
            {
                "num_experts": moe.NUM_EXPERTS,
                "num_local_experts": moe.LOCAL_EXPERTS,
                "hidden_size": moe.HIDDEN,
                "intermediate_size": moe.INTERMEDIATE,
                "gemm1_out_size": moe.GEMM1_N,
                "num_hidden_blocks": moe.HIDDEN // moe.BLOCK,
                "num_intermediate_blocks": moe.INTERMEDIATE // moe.BLOCK,
                "num_gemm1_out_blocks": moe.GEMM1_N // moe.BLOCK,
            },
        )
        # The model the definition (and the model_shape cases) is taken from.
        model = MODELS["deepseek_v3"]
        self.assertEqual(
            (model.hidden, model.moe_intermediate, model.experts, model.top_k),
            (moe.HIDDEN, moe.INTERMEDIATE, moe.NUM_EXPERTS, moe.TOP_K),
        )
        self.assertEqual(
            (
                model.get("n_group"),
                model.get("topk_group"),
                model.get("routed_scaling_factor"),
            ),
            (moe.N_GROUP, moe.TOPK_GROUP, moe.ROUTED_SCALING_FACTOR),
        )

    def test_every_inventory_row_is_a_case(self):
        rows, _ = inventory("moe")
        self.assertEqual(len(rows), 19)
        traced = {
            c.source["uuid"]: c.params
            for c in moe.throughput_cases()
            if c.source.get("inventory") == "moe"
        }
        for row in rows:
            with self.subTest(uuid=row["uuid"]):
                params = traced[row["uuid"]]
                self.assertEqual(params["tokens"], row["axes"]["seq_len"])
                self.assertEqual(
                    params["offset"], row["inputs"]["local_expert_offset"]["value"]
                )
                scale = row["inputs"]["routed_scaling_factor"]["value"]
                self.assertEqual(scale, moe.ROUTED_SCALING_FACTOR)

    def test_upstream_test_parametrizations(self):
        """tests/test_trtllm_gen_fused_moe.py of the pinned FlashInfer runs
        trtllm_fp8_block_scale_moe (FP8_Block, NoShuffle_MajorK) with this
        definition's routing config for every ``num_tokens``, offset 0."""
        if not UPSTREAM_TESTS.is_file():
            self.skipTest(f"{UPSTREAM_TESTS} not fetched")
        text = UPSTREAM_TESTS.read_text()
        (tokens,) = re.findall(
            r'@pytest\.mark\.parametrize\("num_tokens", \[([^\]]*)\]\)', text
        )
        self.assertEqual(tuple(int(t) for t in tokens.split(",")), moe.UPSTREAM_TOKENS)
        dsv3 = re.search(r"\{([^{}]*)\},\s*id=\"DSv3\"", text, re.S)
        assert dsv3 is not None
        config = dsv3.group(1)
        for key, value in (
            ("num_experts", moe.NUM_EXPERTS),
            ("top_k", moe.TOP_K),
            ("padding", moe.TILE),
            ("n_groups", moe.N_GROUP),
            ("top_k_groups", moe.TOPK_GROUP),
            ("routed_scaling", moe.ROUTED_SCALING_FACTOR),
            ("has_routing_bias", True),
        ):
            self.assertIn(f'"{key}": {value},', config)
        self.assertIn("FP8BlockScaleMoe", config)
        self.assertRegex(
            text, r"intermediate_size,\s*0,\s*num_experts,\s*routed_scaling"
        )  # local_expert_offset = 0
        self.assertIn(
            "routing_bias = torch.randn(num_experts, "
            'device="cuda", dtype=torch.bfloat16)',
            text,
        )
        for t in moe.UPSTREAM_TOKENS:
            test = moe.UPSTREAM_TEST % t
            for family in FAMILIES:
                with self.subTest(tokens=t, family=family):
                    found = [
                        c
                        for name in family
                        for c in get_workload(name)(None, device="cpu").get_cases()
                        if c.source.get("test") == test
                    ]
                    self.assertEqual(len(found), 1)
                    self.assertEqual(found[0].suite, "smoke")
                    self.assertEqual(
                        found[0].params,
                        {"tokens": t, "offset": 0, "routing": "upstream"},
                    )

    def test_upstream_distribution(self):
        """``upstream`` cases sample as the upstream test: logits and BF16
        bias ~ N(0, 1), hidden states E4M3(2 N(0, 1)) with scales 2."""
        device = torch.device("cpu")
        logits, bias = moe.official_routing_inputs(device, 512, 1, 0, "upstream")
        self.assertAlmostEqual(float(logits.std()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(bias.float().std()), 1.0, delta=0.2)
        hidden, scale = moe.official_activations(device, 4, 1, "upstream")
        self.assertTrue(bool((scale == 2).all()))
        self.assertAlmostEqual(float(hidden.float().std()), 2.0, delta=0.1)


def smoke_routing() -> dict[str, tuple[moe.CaseSpec, torch.Tensor, moe.Routing]]:
    """Per smoke case: (case, logits, reference routing) as the GEMM, SwiGLU
    and finalize workloads generate them."""
    out = {}
    for case in moe.smoke_cases():
        logits, bias = moe.case_routing_inputs(
            torch.device("cpu"), case, require_local=True
        )
        packed, _ = moe.routing_main_reference(logits, bias, 2.5)
        routing = moe.routing_indices_reference(packed, case.params["offset"])
        out[case.name] = case, logits, routing
    return out


class SmokeHardPaths(unittest.TestCase):
    """The smoke cases each kernel runs on every test pass reach its hard
    paths, measured on the routing the cases actually produce."""

    routing: dict

    @classmethod
    def setUpClass(cls):
        cls.routing = smoke_routing()

    def counts(self, routing: moe.Routing) -> torch.Tensor:
        n = int(routing.num_non_exiting_ctas[0])
        batch = routing.cta_idx_xy_to_batch_idx[:n].long()
        limit = routing.cta_idx_xy_to_mn_limit[:n].long()
        sizes = limit - torch.arange(n) * moe.TILE
        return torch.zeros(moe.LOCAL_EXPERTS, dtype=torch.long).index_add_(
            0, batch, sizes
        )

    def smoke(self, name: str):
        cases = get_workload(name)(None, device="cpu").get_cases()
        return [self.routing[c.name] for c in cases if c.suite == "smoke"]

    def test_gemm_and_permutation_paths(self):
        for name in (
            "moe_routing_cluster",
            "moe_routing_coop",
            "moe_gemm1",
            "moe_gemm1_persistent",
            "moe_gemm2",
            "moe_gemm2_persistent",
            "moe_activation",
        ):
            with self.subTest(workload=name):
                smoke = self.smoke(name)
                counts = [self.counts(r) for _, _, r in smoke]
                # An expert over several tiles ending in a partial tile (an
                # mnLimit cut-off) beside empty experts.
                self.assertTrue(
                    any(
                        bool(((c > moe.TILE) & (c % moe.TILE != 0)).any())
                        and bool((c == 0).any())
                        for c in counts
                    )
                )
                offsets = {case.params["offset"] for case, _, _ in smoke}
                self.assertTrue({0, 224} <= offsets or name == "moe_routing_coop")
                if name.endswith("persistent"):
                    m = get_workload(name).m
                    tiles = max(
                        m // moe.BLOCK * int(r.num_non_exiting_ctas[0])
                        for _, _, r in smoke
                    )
                    # Each of the 148 resident CTAs takes several tiles.
                    self.assertGreater(tiles, 8 * 148)

    def test_rank_without_tokens(self):
        """A case routes no token here: zero CTAs (GEMM early exit) and an
        all-zero finalize output."""
        for name in (
            "moe_routing_cluster",
            "moe_gemm1_persistent",
            "moe_gemm2_persistent",
            "moe_activation",
            "moe_finalize_vec",
        ):
            with self.subTest(workload=name):
                self.assertIn(
                    0, [int(r.num_non_exiting_ctas[0]) for _, _, r in self.smoke(name)]
                )

    def test_finalize_paths(self):
        for name in ("moe_finalize", "moe_finalize_vec"):
            with self.subTest(workload=name):
                local = [
                    (r.expanded_idx_to_permuted_idx.view(-1, moe.TOP_K) >= 0).sum(-1)
                    for _, _, r in self.smoke(name)
                ]
                # Tokens without local experts (zero rows) and with several.
                self.assertTrue(any(bool((n == 0).any()) for n in local))
                self.assertTrue(any(bool((n >= 3).any()) for n in local))

    def test_exact_ties(self):
        """The ``ties`` case has exact ties across the kept/dropped group
        boundary and between the 8th and 9th expert."""
        (case,) = [c for c in moe.smoke_cases() if c.params["routing"] == "ties"]
        _, logits, _ = self.routing[case.name]
        scores = 0.5 * torch.tanh(0.5 * logits.double()) + 0.5
        grouped = scores.view(-1, moe.N_GROUP, moe.EXPERTS_PER_GROUP)
        groups = grouped.topk(2, dim=-1).values.sum(-1).sort(-1, descending=True)
        self.assertTrue(bool((groups.values[:, 3] == groups.values[:, 4]).any()))
        kept = groups.indices[:, : moe.TOPK_GROUP]
        mask = torch.zeros_like(groups.values, dtype=torch.bool).scatter_(1, kept, True)
        masked = scores.masked_fill(~mask.repeat_interleave(32, -1), -1)
        top = masked.sort(-1, descending=True).values
        self.assertTrue(bool((top[:, 7] == top[:, 8]).any()))


class BuildMatchesUpstream(unittest.TestCase):
    """The launch a workload builds from real case inputs (``_build``) equals
    upstream's recording for the same (tokens, offset), except addresses,
    and every address is one of the case's tensors: token counts, buffer
    sizes and TMA shapes are derived from the inputs as upstream derives
    them. (Descriptors use the fake encoder; nothing is launched.)"""

    def test_build(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name in NAMES:
            cls = get_workload(name)
            smoke = [
                c for c in cls(None, device=device).get_cases() if c.suite == "smoke"
            ]
            for case in (smoke[0], smoke[-1]) if device == "cuda" else smoke[:1]:
                inputs = cls(None, device=device).get_inputs(case)
                for arch in cls.supported_arches:
                    with self.subTest(workload=name, arch=arch, case=case.name):
                        self.check(cls.from_arch(arch, device=device), case, inputs)
                del inputs

    def check(self, workload, case, inputs):
        if isinstance(workload, moe._GEMM):
            outputs, spec, _ = workload._build(inputs, encode=moe.fake_encode)
        else:
            outputs, spec, _ = workload._build(inputs)
        (fixture,) = [
            f
            for f in fixtures(workload.name, workload.arch)
            if (f["tokens"], f["local_expert_offset"])
            == (case.params["tokens"], case.params["offset"])
        ]
        launch = fixture["launch"]
        self.assertEqual(list(spec.grid), launch["grid"])
        self.assertEqual(list(spec.block), launch["block"])
        self.assertEqual(spec.shared_mem, launch["shared_mem"])
        attrs = {a["id"]: a["value"] for a in launch["attrs"]}
        self.assertEqual(spec.cooperative, attrs.get(2) == 1)
        layout = workload.sidecar["params"]
        want = bytes.fromhex(launch["params_hex"])
        self.assertEqual(len(spec.params), len(want))
        addresses = {
            t.data_ptr()
            for t in (*inputs, *outputs)
            if isinstance(t, torch.Tensor) and t.numel()
        }
        skip = ignored_bytes(launch, layout)
        for field, f in layout["fields"].items():
            if f["kind"] not in ("ptr", "tensor_map"):
                continue
            # A descriptor's global address is its bytes 8..16.
            at = f["offset"] + (8 if f["kind"] == "tensor_map" else 0)
            skip.update(range(at, at + 8))
            (built,) = struct.unpack_from("<Q", spec.params, at)
            (upstream,) = struct.unpack_from("<Q", want, at)
            if not built:  # null pointer or a descriptor upstream leaves unset
                if f["kind"] == "ptr" and isinstance(workload, moe._GEMM):
                    self.assertEqual(upstream, 0, field)
                continue
            self.assertNotEqual(upstream, 0, field)
            # The coop workload's own scratch: its histogram, and the weights
            # buffer upstream passes but routingIndicesCoopKernel never writes.
            scratch = ("mPtrExpertCounts", "mPtrExpertWeights")
            if not (isinstance(workload, moe.MoERoutingCoop) and field in scratch):
                self.assertIn(built, addresses, field)
        diff = [
            i for i in range(len(want)) if i not in skip and spec.params[i] != want[i]
        ]
        self.assertEqual(diff, [], f"first differing bytes {diff[:8]}")


if __name__ == "__main__":
    unittest.main()

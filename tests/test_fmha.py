"""fmha package: by-value structs, TMA descriptors and dispatch versus the probes.

sm_86 (FlashInfer FA2 of the 0.7.0 sm80 JIT cache; ``tests.test_workloads``
runs ``check_param_layouts``: every by-value struct, planner result and
launch constant against the probes):

* every launch FlashInfer's own tests make on these kernels
  (``impls/fmha/fa2_upstream.py``) is in the regime of one of its kernel's
  cases, and the recorded upstream case table is current;
* the smoke cases' features change the reference beyond the validation
  tolerance (soft cap, window, sink, custom mask, ``v_scale``);
* no two registered kernels have identical code; removed duplicates are
  served by their twin with and without a window.

sm_100a (trtllm-gen; ``tests.test_workloads`` runs ``check_param_layouts``,
the SHA256 of every recorded case's launch and ``KernelParams``):

* the kept examples' full ``KernelParams`` bytes equal the probe's run of
  FlashInfer v0.6.9's ``TllmGenFmhaKernel::run``;
* registered kernels have distinct selection hashes per dtype triple and no
  two have identical code; removed duplicates are served by their twin;
* every FlashInfer v0.6.9 test parametrization that selects a registered
  kernel is one of its cases, with every upstream flag value;
* skip-softmax tile-pattern inputs put the skipped tiles below the kernel's
  threshold and the kept ones above it;
* ``prepare`` maps every pointer role of every smoke case (feature profile).

Both: ``kernels.json`` lists exactly the registered workloads, and every
stripped cubin holds its one recorded kernel with the recorded SHA256.
Nothing here launches a kernel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import enumerate_workloads  # noqa: E402
from harness.workload import cubin_info  # noqa: E402
from harness.workloads.fmha import fa2, trtllm  # noqa: E402

PACKAGE = ROOT / "impls" / "fmha"


def provenance(arch: str) -> dict:
    return json.loads((PACKAGE / "provenance.json").read_text())["builds"][arch]


def load_build(name: str):
    path = PACKAGE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_fmha_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fa2(unittest.TestCase):
    def setUp(self):
        if not fa2.load_index()["kernels"]:
            self.skipTest(
                "impls/fmha/variants/sm_86.json missing; run compile_kernels.py"
            )

    def test_upstream_tests_are_cases(self):
        """Every launch of FlashInfer's own tests on these kernels has the
        regime (``fa2.regime``) of one of its kernel's cases, and the
        recorded upstream case table is what the build derives today."""
        if not (ROOT / "resources" / upstream_module().FLASHINFER).is_dir():
            self.skipTest(
                "resources/flashinfer-<rev> missing; scripts/fetch_resources.py"
            )
        up = upstream_module()
        regimes: dict[str, set[str]] = {}
        launches = 0
        for name, params, test in up.launches(ROOT / "resources", fa2):
            if name not in regimes:
                cases = fa2.probe(name).get_cases()
                regimes[name] = {fa2.regime_key(name, c.params) for c in cases}
            launches += 1
            with self.subTest(test=test, workload=name):
                self.assertIn(fa2.regime_key(name, params), regimes[name])
        self.assertGreater(launches, 40000)
        self.assertGreater(len(regimes), 100)
        recorded = json.loads(fa2.UPSTREAM.read_text())["kernels"]
        self.assertEqual(recorded, up.representatives(ROOT / "resources", fa2))

    def test_features_change_the_reference(self):
        """Smoke cases switch every runtime feature on with values that move
        the reference beyond the validation tolerance: switching the feature
        off must fail ``validate`` (a soft cap of 30 over unit-variance logits,
        for example, would not)."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        toggles = {
            "fmha_fa2_prefill_paged_bf16_bf16_hd128_causal_softcap_cta128": (
                "params", {"logits_soft_cap": 0.0}),
            "fmha_fa2_prefill_ragged_f16_e4m3_hd64_none_swa_cta64": (
                "params", {"window_left": -1}),
            "fmha_fa2_prefill_paged_f16_f16_hd256_custom_cta16": ("mask", None),
            "fmha_fa2_sink_paged_bf16_bf16_hd64_causal_cta64": ("sink", None),
            "fmha_fa2_decode_f16_f16_hd128_g4_swa_softcap": (
                "params", {"logits_soft_cap": 0.0, "window_left": -1}),
            "fmha_fa2_persistent_f16_f16_hd64_causal_softcap": (
                "params", {"logits_soft_cap": 0.0, "v_scale": 1.0}),
        }  # fmt: skip
        for name, (field, change) in toggles.items():
            workload = type(fa2.probe(name))(None, device=device)
            smoke = [c for c in workload.get_cases() if c.name.startswith("smoke_")]
            with self.subTest(workload=name):
                detected = 0
                for case in smoke:
                    inputs = workload.get_inputs(case)
                    ref = workload.get_reference(inputs)
                    if field == "params":
                        p = inputs.params
                        if all(p.get(k, v) == v for k, v in change.items()):
                            continue  # feature already off in this case
                        if "window_left" in change and (
                            fa2.regime(name, case.params)["window"] != "active"
                        ):
                            continue  # a window wider than every request
                        off = inputs._replace(params={**p, **change})
                    elif field == "sink":
                        off = inputs._replace(sink=None)
                    else:  # custom mask -> attend to every key
                        everything = torch.full_like(inputs.custom_mask, 255)
                        off = inputs._replace(custom_mask=everything)
                    with self.assertRaises(AssertionError, msg=case.name):
                        workload.validate(ref, workload.get_reference(off))
                    detected += 1
                self.assertGreater(detected, 0)

    def test_no_duplicate_kernels(self):
        """No two registered kernels have identical code (``fa2_build.
        code_hash``); every removed duplicate's twin is registered and has
        windowed and unwindowed cases."""
        build = load_build("fa2_build")
        from harness import workload as harness_workload

        manifest = json.loads((PACKAGE / "kernels.json").read_text())["sm_86"]
        owners: dict[str, str] = {}
        for name, path in sorted(manifest.items()):
            digest = build.code_hash(harness_workload, (PACKAGE / path).read_bytes())
            self.assertNotIn(digest, owners, f"{name} == {owners.get(digest)}")
            owners[digest] = name
        removed = provenance("sm_86")["duplicates_removed"]["kept"]
        self.assertEqual(len(removed), 24)
        for gone, kept in removed.items():
            self.assertNotIn(gone, manifest)
            windows = {
                c.params.get("window_left", -1) >= 0
                for c in fa2.probe(kept).get_cases()
            }
            self.assertEqual(windows, {False, True}, kept)


def upstream_module():
    return load_build("fa2_upstream")


class Trtllm(unittest.TestCase):
    def setUp(self):
        if not trtllm.load_index()["kernels"]:
            self.skipTest(
                "impls/fmha/variants/sm_100a.json missing; run compile_kernels.py"
            )

    def test_examples_full_bytes(self):
        """The kept examples' full KernelParams bytes (not only their hash)
        equal the Python port's."""
        index = trtllm.load_index()
        fixtures = json.loads(trtllm.FIXTURES.read_text())
        metas = {row[0]: trtllm.meta_of(row) for row in index["kernels"]}
        pointers = {role: trtllm.sentinel(role) for role in trtllm.ROLES}
        self.assertTrue(fixtures["examples"])
        for kernel, raw in fixtures["examples"].items():
            m = metas[kernel]
            case = index["cases"][kernel][fixtures["records"][kernel][0]["case"]]
            r = trtllm.runner_params(m, case)
            data, _ = trtllm.build_params(
                m, r, {role: pointers[role] for role in r["ptrs"]}, trtllm.fake_encode
            )
            self.assertEqual(data, bytes.fromhex(raw), kernel)

    def test_selection_hashes_are_distinct(self):
        build = load_build("trtllm_build")
        seen: dict[tuple, str] = {}
        for row in trtllm.load_index()["kernels"]:
            m = trtllm.meta_of(row)
            key = (m["dtq"], m["dtkv"], m["dto"], build.hash_id(m))
            self.assertNotIn(key, seen, f"{m['name']} / {seen.get(key)}")
            seen[key] = m["name"]

    def test_no_identical_kernel_code(self):
        """No two registered kernels have identical code (the removed
        PackedQkv twins are served by their SeparateQkv kernel)."""
        from harness.workload import _sections

        build = load_build("trtllm_build")
        manifest = json.loads((PACKAGE / "kernels.json").read_text())["sm_100a"]
        seen: dict[str, str] = {}
        for name, path in sorted(manifest.items()):
            digest = build.code_hash(_sections((PACKAGE / path).read_bytes()))
            self.assertNotIn(digest, seen, f"{name} == {seen.get(digest)}")
            seen[digest] = name
        removed = provenance("sm_100a")["removed_duplicates"]
        cases = trtllm.load_index()["cases"]
        for packed, separate in removed.items():
            with self.subTest(removed=packed):
                self.assertIn("duplicate", provenance("sm_100a")["dead"][packed])
                suites = {c["suite"] for c in cases[separate] if c.get("packed")}
                self.assertEqual(suites, {"smoke", "throughput"})

    def test_upstream_parametrizations_are_cases(self):
        """Every FlashInfer v0.6.9 test parametrization that selects a
        registered kernel (selection recorded by the probe per upstream
        shape) is a case of that kernel with the same shape, and every
        upstream flag value of the kernel (KV layout, sinks, shared page
        indices, non-contiguous query, device scales, separate K/V tensors,
        cumulative query lengths) appears among its upstream cases."""
        build = load_build("trtllm_build")
        upstream = build.load_upstream(PACKAGE)
        selection = json.loads(trtllm.FIXTURES.read_text())["upstream"]
        index = trtllm.load_index()
        cases = {row[0]: index["cases"][row[0]] for row in index["kernels"]}
        entries = upstream.upstream_params()
        keys: dict[str, dict] = {}
        needed: dict[str, set] = {}
        for entry in entries:
            um = build.upstream_meta(trtllm, entry)
            case = build.finish_case(trtllm, um, dict(entry["case"]))
            skey = upstream.shape_key(case)
            full = json.dumps([entry["dtypes"], entry["hd"], entry["layout"], skey])
            key = hashlib.sha256(full.encode()).hexdigest()[:16]
            keys.setdefault(key, case)
            self.assertIn(key, selection, entry["id"])
            kernel = selection[key]
            if kernel not in cases:
                self.assertTrue(
                    kernel.startswith(("excluded: ", "error: ")), (entry["id"], kernel)
                )
                continue
            needed.setdefault(kernel, set()).update(
                (f, v)
                for f, v in entry["flags"].items()
                if f in upstream.FLAG_KEYS and f != "sf_start"
            )
        self.assertEqual(set(keys), set(selection))
        live = 0
        for key, kernel in selection.items():
            if kernel not in cases:
                continue
            live += 1
            mirrors = [
                c
                for c in cases[kernel]
                if (c.get("source") or {}).get("upstream_key") == key
            ]
            with self.subTest(kernel=kernel, upstream=key):
                self.assertTrue(mirrors)
                for c in mirrors:
                    self.assertEqual(
                        upstream.shape_key(c), upstream.shape_key(keys[key])
                    )
        self.assertGreater(live, 1000)
        for kernel, flags in needed.items():
            present = set()
            for c in cases[kernel]:
                if (c.get("source") or {}).get("kind") != "upstream_test":
                    continue
                present |= {
                    ("kv_layout", c.get("kv_layout", "HND")),
                    ("shared_idx", c.get("shared_idx", True)),
                    *(
                        (f, bool(c.get(f)))
                        for f in (
                            "sinks",
                            "q_noncontig",
                            "device_scales",
                            "kv_tuple",
                            "cum_q",
                        )
                    ),
                }
            with self.subTest(kernel=kernel):
                self.assertLessEqual(flags, present)

    def test_skip_softmax_data_skips_tiles(self):
        """Skip-softmax tile-pattern inputs: in every row that sees a high
        tile, the visible keys of low KV tiles peak >= 25 below the row's
        maximum visible logit
        (exp < 2e-11, below the threshold 1e-3 / KV length the kernel compares
        with) while every high tile with a visible key peaks within 9 of it
        (exp > 1e-4 > threshold), so the kernel skips exactly the low tiles;
        their exact weight is < 1e-10 of the row."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checked = 0
        kernels = [
            cls
            for cls in enumerate_workloads("sm_100a")
            if issubclass(cls, trtllm.TrtllmFmha) and cls.meta["skips"]
        ]
        # An even sample over dtypes, head dims, layouts and kernel types.
        for cls in kernels[:: max(1, len(kernels) // 24)]:
            workload = cls(None, device=device)
            cases = [
                c
                for c in workload.get_cases()
                if c.suite == "smoke" and c.params.get("skip_data")
            ]
            for case in cases:
                p, t = workload.get_inputs(case)
                with self.subTest(workload=cls.name, case=case.name):
                    self._check_skip_rows(workload, p, t)
            checked += 1
        self.assertGreater(checked, 0)

    def _check_skip_rows(self, workload, p, t) -> None:  # noqa: ANN001
        import torch

        tile, dqk = workload.meta["tileKv"], workload.meta["hdQk"]
        if "qkv" in t:
            q_all = t["qkv"][:, : p["hq"] * dqk].reshape(-1, p["hq"], dqk).float()
        else:
            q_all = t["q"].float()
        causal = p["runner"]["mask"] != trtllm.DENSE
        start = 0
        for b, (lq, lk) in enumerate(zip(p["q_lens"], p["kv_lens"])):
            k, _ = workload._request_kv(p, t, b)
            k = k.repeat_interleave(p["hq"] // p["hkv"], dim=1)
            q = q_all[start : start + lq]
            start += lq
            logits = torch.einsum("qhd,khd->hqk", q, k) * p["bmm1_scale"]
            pos = torch.arange(lq, device=q.device).view(-1, 1) + (lk - lq)
            key = torch.arange(lk, device=q.device).view(1, -1)
            visible = (key <= pos) | (not causal)
            if p["window_left"] >= 0:
                visible &= key >= pos - p["window_left"]
            logits = logits.masked_fill(~visible, float("-inf"))
            row_max = logits.amax(-1)
            high_tiles = ((key // tile) % 2 == 0) & visible  # [q, k]
            # Rows whose window holds no high tile see only low keys: there
            # is nothing to skip (and nothing is skipped).
            rows = high_tiles.any(-1)
            for j in range(0, lk, tile):
                seen = visible[:, j : j + tile].any(-1) & rows  # [q]
                peak = logits[..., j : j + tile].amax(-1)  # [h, q]
                gap = (row_max - peak)[:, seen]
                if (j // tile) % 2:
                    self.assertTrue(bool((gap >= 25).all()), (b, j))
                else:
                    self.assertTrue(bool((gap <= 9).all()), (b, j))

    def test_prepare_on_cuda_tensors(self):
        """``prepare`` maps every pointer role to an input or output tensor
        and builds the recorded launch (recording TMA encoder: the driver
        refuses TMA descriptors without a TMA-capable GPU); one workload per
        (layout, kernel type, dtypes, MLA/sparse/FP4) family, every smoke
        case (each runtime-feature profile)."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("needs CUDA tensors")
        families: dict[tuple, type[trtllm.TrtllmFmha]] = {}
        for cls in enumerate_workloads("sm_100a"):
            if not issubclass(cls, trtllm.TrtllmFmha):
                continue
            m = cls.meta
            key = (
                m["layout"],
                m["ktype"],
                m["dtq"],
                m["dtkv"],
                m["dto"],
                m["hdQk"],
                m["sparse"],
            )
            families.setdefault(key, cls)
        self.assertGreater(len(families), 20)
        launches: list[dict] = []

        def record(self, grid, block, args, **kwargs):  # noqa: ANN001
            launches.append(
                {"grid": list(grid), "block": block, "args": args, **kwargs}
            )

        with (
            mock.patch.object(trtllm, "driver_encode", trtllm.fake_encode),
            mock.patch.object(trtllm.TrtllmFmha, "launch", record),
            mock.patch.object(
                trtllm.TrtllmFmha, "function", new_callable=mock.PropertyMock
            ),
            mock.patch.object(
                trtllm.cuda_driver, "max_active_clusters", return_value=1 << 20
            ),
        ):
            for key, cls in sorted(families.items(), key=lambda kv: str(kv[0])):
                workload = cls(None, device="cuda")
                for case in workload.get_cases():
                    if case.suite != "smoke":
                        continue
                    with self.subTest(workload=cls.name, case=case.name):
                        launches.clear()
                        inputs = workload.get_inputs(case)
                        launch, outputs = workload.prepare(inputs)
                        self.assertIs(launch(), outputs)
                        (call,) = launches
                        r = trtllm.runner_params(cls.meta, case.params)
                        expected = trtllm.launch_config(cls.meta, r)
                        self.assertEqual(call["grid"], expected["grid"])
                        self.assertEqual(call["cluster"], expected["cluster"])
                        self.assertEqual(
                            len(call["args"][0]), trtllm.load_index()["layout"]["size"]
                        )


class Artifacts(unittest.TestCase):
    def test_manifest_lists_registered_workloads(self):
        manifest = json.loads((PACKAGE / "kernels.json").read_text())
        for arch, prefix in (("sm_86", "fmha_fa2_"), ("sm_100a", "fmha_trtllm_")):
            registered = {
                c.name for c in enumerate_workloads(arch) if c.name.startswith(prefix)
            }
            with self.subTest(arch=arch):
                self.assertEqual(set(manifest.get(arch, {})), registered)

    def test_stripped_cubins_match_provenance(self):
        manifest = json.loads((PACKAGE / "kernels.json").read_text())
        for arch in ("sm_86", "sm_100a"):
            records = provenance(arch)["kernels"]
            for name, path in sorted(manifest.get(arch, {}).items()):
                data = (PACKAGE / path).read_bytes()
                record = records[name]
                with self.subTest(workload=name):
                    self.assertEqual(hashlib.sha256(data).hexdigest(), record["sha256"])
                    _, kernels, _ = cubin_info(data)
                    symbol = record.get("symbol") or record.get("kernel")
                    self.assertEqual(kernels, [symbol])


if __name__ == "__main__":
    unittest.main()

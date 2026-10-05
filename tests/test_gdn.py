"""gdn: launch builder versus the upstream host code, and upstream test coverage.

``impls/gdn/compiler.py`` runs a host-only probe built from the upstream
TVM-FFI shim's ``EncodeTma_{Q,K,V,O}`` and ``Run`` kargs construction
(verbatim excerpts) with fake pointers and a recording
``cuTensorMapEncodeTiled``, for decode (T = N) and ragged prefill shapes; the
sidecar ``cubins/sm_100a/gdn_chunk.json`` holds the encode arguments and the
packed kernel parameter bytes per probe run. The Python builder, with a
recording encoder writing the same marker bytes, must reproduce both exactly.
None of this executes Blackwell code.

``UpstreamCoverage`` reads the pinned FlashInfer GDN tests and checks that
every parametrization the chunk kernel serves (projected onto the
definitions' fixed BF16 / FP32-state / 4-4-8-head configuration) is a case.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.throughput import split_lengths  # noqa: E402
from harness.workload import CaseSpec, ctypes_value  # noqa: E402
from harness.workloads import gdn  # noqa: E402

ARCH = "sm_100a"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
UPSTREAM = ROOT / "resources" / f"flashinfer-{gdn.UPSTREAM_REVISION}" / "tests/gdn"


def sidecar(cls, arch: str = ARCH) -> dict:
    return json.loads(cls.cubin_path(arch).with_suffix(".json").read_text())


def load_chunk(device: str) -> gdn.GDNChunk:
    workload = gdn.GDNChunk.from_arch(ARCH, device=device)
    assert isinstance(workload, gdn.GDNChunk)
    return workload


class Recorder:
    """Records encode calls; returns 128 marker bytes like the probe's stub."""

    def __init__(self, marker: int):
        self.marker = marker
        self.calls: list[dict] = []

    def __call__(self, data_type, address, dims, strides, box, element_strides, *e):
        interleave, swizzle, l2_promotion, oob_fill = e
        self.calls.append(
            dict(
                data_type=data_type,
                rank=len(dims),
                address=address,
                dims=list(dims),
                strides=list(strides),
                box=list(box),
                element_strides=list(element_strides),
                interleave=interleave,
                swizzle=swizzle,
                l2_promotion=l2_promotion,
                oob_fill=oob_fill,
            )
        )
        marker = self.marker | (len(self.calls) - 1)
        return marker.to_bytes(8, "little") + bytes(120)


class ChunkLaunch(unittest.TestCase):
    def setUp(self):
        self.meta = sidecar(gdn.GDNChunk)
        self.probe = self.meta["probe"]

    def build(self, run: dict, device: str):
        workload = load_chunk(device)
        workload.sm_count = run["sm_count"]
        recorder = Recorder(self.probe["tma_marker"])
        workload.encode = recorder
        tokens, seqs = run["tokens"], run["seqs"]
        params: dict[str, Any] = (
            {"batch": seqs}
            if tokens == seqs
            else {"lengths": split_lengths(tokens, seqs)}
        )
        inputs = workload.get_inputs(CaseSpec("probe", params, seqs))
        inputs = inputs[:-1] + (workload.scalar(run["scale"]),)
        spec, outputs = workload.configure_launch(inputs)
        return workload, recorder, inputs, spec, outputs

    def test_matches_upstream_host_code(self):
        roles = (
            "Q K V O gate beta cu_seqlens state_indices initial_state output_state "
            "checkpoint_state cu_checkpoints tensormap_workspace"
        ).split()
        sentinel = self.probe["sentinel_stride"]
        runs = self.probe["runs"]
        self.assertTrue(any(r["tokens"] == r["seqs"] for r in runs))  # decode
        self.assertTrue(any(r["tokens"] > r["seqs"] > 1 for r in runs))  # ragged
        for run in runs:
            for device in DEVICES:
                key = dict(t=run["tokens"], n=run["seqs"], sms=run["sm_count"])
                with self.subTest(**key, device=device):
                    workload, recorder, inputs, spec, outputs = self.build(run, device)
                    q, k, v, state = inputs[:4]
                    out, new_state = outputs
                    pointers = {
                        "Q": q,
                        "K": k,
                        "V": v,
                        "O": out,
                        "gate": inputs[4],
                        "beta": inputs[5],
                        "cu_seqlens": inputs[6],
                        "initial_state": state,
                        "output_state": new_state,
                    }
                    # Encode arguments, addresses as (role, offset).
                    self.assertEqual(len(recorder.calls), len(run["tma_calls"]))
                    for got, want in zip(recorder.calls, run["tma_calls"]):
                        role = want["address"]["role"]
                        address = pointers[role].data_ptr() + want["address"]["offset"]
                        self.assertEqual(got["address"], address, role)
                        self.assertEqual(dict(got, address=want["address"]), want)
                    # Packed parameter bytes, pointers as the probe's sentinels.
                    params = workload.kernel_params
                    self.assertEqual(len(spec.args), len(params))
                    buffer = bytearray(params[-1].offset + params[-1].size)
                    for index, (arg, param) in enumerate(zip(spec.args, params)):
                        if isinstance(arg, torch.Tensor):
                            # Argument i is upstream kargs[i]: role i's sentinel.
                            role = roles[index]
                            if role in pointers:
                                self.assertIs(arg, pointers[role], role)
                            value = (sentinel * (index + 1)).to_bytes(8, "little")
                        else:
                            value = bytes(ctypes_value(arg))
                        self.assertEqual(len(value), param.size, index)
                        buffer[param.offset : param.offset + param.size] = value
                    self.assertEqual(buffer.hex(), run["bytes"])
                    # Launch configuration and scratch size.
                    self.assertEqual([spec.grid, 1, 1], run["grid"])
                    self.assertEqual(spec.args[12].numel(), run["workspace_bytes"])
                    self.assertEqual(spec.block, self.meta["constants"]["threads"])
                    self.assertEqual(
                        spec.shared_mem, self.meta["constants"]["dynamic_smem"]
                    )
                    if not workload.is_supported():
                        # Never execute sm_100a code on another GPU.
                        with self.assertRaises(RuntimeError):
                            workload.function  # noqa: B018

    def test_rejects_bad_operands(self):
        workload = load_chunk("cpu")
        workload.encode = Recorder(self.probe["tma_marker"])
        case = next(c for c in workload.get_cases() if c.name == "prefill_ragged")
        inputs = workload.get_inputs(case)
        q, k, v, state, gate, beta, cu, scale = inputs
        bad = [
            (q.float(), k, v, state, gate, beta, cu, scale),
            (q, k[:, :2], v, state, gate, beta, cu, scale),
            (q, k, v, state.transpose(2, 3), gate, beta, cu, scale),
            (q, k, v, state, gate.t().contiguous().t(), beta, cu, scale),
            (q, k, v, state, gate, beta, cu.long(), scale),
            (q, k, v, state[:1], gate, beta, cu, scale),
            (q, k, v, state, gate, beta, cu, 0.5),
        ]
        for args in bad:
            with self.assertRaises(ValueError):
                workload.configure_launch(args)


class Tolerance(unittest.TestCase):
    def test_rejects_small_uniform_errors(self):
        """The scale-relative bounds reject a uniform 2% error of either
        output (the former absolute 1e-2 bounds admitted x1.3 on model-scale
        outputs) and accept BF16 re-rounding noise."""
        workload = gdn.GDNChunk(None, device="cpu")
        case = next(c for c in workload.get_cases() if c.name == "prefill_ragged")
        output, state = workload.get_reference(workload.get_inputs(case))
        workload.validate((output, state), (output, state))
        noisy = (output.float() * (1 + 2e-3 * torch.randn_like(output.float()))).to(
            output.dtype
        )
        workload.validate((output, state), (noisy, state))
        for bad in (
            ((output.float() * 1.02).to(output.dtype), state),
            (output, state * 1.02),
        ):
            with self.assertRaises(AssertionError):
                workload.validate((output, state), bad)


class Artifacts(unittest.TestCase):
    def test_generated_kernel_is_reproducible(self):
        """kernels/ holds exactly what ImplCompiler.generate() would write."""
        path = ROOT / "impls/gdn/compiler.py"
        spec = importlib.util.spec_from_file_location("gdn_compiler", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        compiler = module.ImplCompiler(ARCH)
        if not compiler.flashinfer.is_dir():
            self.skipTest("pinned FlashInfer checkout is not in resources/")
        checked_in = (module.KERNELS / module.GENERATED).read_text()
        self.assertEqual(checked_in, compiler.generated_source())
        meta = sidecar(gdn.GDNChunk)
        self.assertEqual(
            meta["source_sha256"], module._sha256(module.KERNELS / module.GENERATED)
        )


# -- upstream coverage ---------------------------------------------------------


class _Torch:
    """Stand-in for ``torch`` in upstream parametrize literals."""

    def __getattr__(self, name: str) -> str:
        return name


def parametrizations(path: Path) -> dict[str, dict[str, list]]:
    """``{test function: {argnames: values}}`` of every literal
    ``@pytest.mark.parametrize`` in an upstream test file."""
    result: dict[str, dict[str, list]] = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not (
                isinstance(decorator, ast.Call)
                and getattr(decorator.func, "attr", None) == "parametrize"
            ):
                continue
            names = ast.literal_eval(decorator.args[0])
            if not isinstance(names, str):
                names = ",".join(names)
            values = eval(  # trusted pinned upstream literal
                compile(ast.Expression(decorator.args[1]), str(path), "eval"),
                {"torch": _Torch(), "list": list, "zip": zip},
            )
            result.setdefault(node.name, {})[names.replace(" ", "")] = list(values)
    return result


class UpstreamCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not UPSTREAM.is_dir():
            raise unittest.SkipTest("pinned FlashInfer checkout is not in resources/")
        cls.cases = [
            c
            for c in gdn.GDNChunk(None, device="cpu").get_cases()
            if c.suite == "smoke"
        ]

    def matching(self, **want: Any) -> list[CaseSpec]:
        """Smoke cases whose (defaulted) params equal ``want``; ``lengths``
        also matches a decode ``batch`` of one-token sequences."""
        found = []
        for case in self.cases:
            params = {**gdn.DEFAULTS, **case.params}
            params["lengths"] = gdn.case_lengths(case.params)
            if all(
                params.get(key) == (list(value) if key == "lengths" else value)
                for key, value in want.items()
            ):
                found.append(case)
        return found

    def require(self, test: str, **want: Any) -> list[CaseSpec]:
        found = self.matching(**want)
        self.assertTrue(found, f"{test}: no case with {want}")
        return found

    def test_recorded_parametrizations_are_cases(self):
        """Every UPSTREAM_TESTS entry is a smoke case of gdn_chunk, and of
        gdn_gates when its gates come from raw inputs."""
        gates = {c.name for c in gdn.GDNGates(None, device="cpu").get_cases()}
        for test, params in gdn.UPSTREAM_TESTS:
            with self.subTest(test=test):
                found = self.require(test, **params)
                if params.get("gates", "model") in gdn.RAW_GATES:
                    self.assertTrue(any(c.name in gates for c in found))
                self.assertTrue(
                    any(c.source.get("kind") == "upstream_test" for c in found)
                )

    def test_prefill_delta_rule(self):
        tests = parametrizations(UPSTREAM / "test_prefill_delta_rule.py")
        heads = "num_q_heads,num_k_heads,num_v_heads"
        for name in ("test_prefill_kernel_basic", "test_prefill_kernel_nonfull"):
            grid = tests[name]
            self.assertIn((2, 2, 4), grid[heads])  # GVA ratio 2, as the kernel
            self.assertIn("bfloat16", grid["dtype"])
            # initial_state=None: zero state; alpha/beta on/off and both
            # scales appear across the length sets (both off is skipped).
            modes, scales = set(), set()
            for lengths in grid["seq_lens"]:
                found = self.require(name, lengths=lengths, zero_state=True)
                modes |= {gdn.param(c, "gates") for c in found}
                scales |= {gdn.param(c, "scale") for c in found}
            self.assertLessEqual({"uniform", "unit_alpha", "unit_beta"}, modes)
            self.assertLessEqual({0.0, 1.0}, scales)
        grid = tests["test_chunked_prefill"]
        self.assertIn((2, 2, 4), grid[heads])
        modes = set()
        for _, lengths in grid["seq_lens1,seq_lens2"]:
            # The second call continues from the first call's final state.
            found = self.require(
                "test_chunked_prefill", lengths=lengths, zero_state=False
            )
            modes |= {gdn.param(c, "gates") for c in found}
        self.assertLessEqual({"uniform", "unit_alpha", "unit_beta"}, modes)
        for length in tests["test_prefill_kernel_zero_length_sequence"]["seq_len"]:
            self.require("zero_length_sequence", lengths=[length, 0])
        self.require("state_untouched", lengths=[256, 0])
        source = (UPSTREAM / "test_prefill_delta_rule.py").read_text()
        self.assertIn("seq_lens = [64, 111, 192]", source)
        self.assertIn("alpha = 0.99 + 0.01 * torch.rand", source)
        self.require("block_end_decay", lengths=[64, 111, 192], gates="near_one")

    def test_cake_prefill(self):
        source = (UPSTREAM / "test_cake_gdn_prefill_gpu.py").read_text()
        literals = re.findall(r"seq_lens\"?\s*[=:]\s*\(([\d,\s]+)\)", source)
        self.assertGreaterEqual(len(set(literals)), 5)
        for literal in set(literals):
            lengths = [int(x) for x in literal.replace(" ", "").split(",") if x]
            self.require("test_cake_gdn_prefill_gpu", lengths=lengths, zero_state=False)

    def test_decode(self):
        tests = parametrizations(UPSTREAM / "test_decode_delta_rule.py")
        batches = set(tests["test_decode_kernel_basic_pretranspose"]["batch_size"])
        batches |= set(tests["test_decode_kernel_basic_nontranspose"]["batch_size"])
        for name in (
            "test_decode_kernel_pretranspose_pool",
            "test_output_state_indices",
        ):
            self.assertIn("float32", tests[name]["state_dtype"])
            batches |= set(tests[name]["batch_size"])
        cake = parametrizations(UPSTREAM / "test_cake_gdn_decode_gpu.py")
        mtp = set()
        for name, grid in cake.items():
            if "fp32" not in name:
                continue
            for key, values in grid.items():
                if key == "batch_size":
                    batches |= set(values)
                elif key == "batch_size,seq_len":
                    mtp |= set(values)
        self.assertLessEqual({1, 4, 16, 32, 512}, batches)  # parsed, not vacuous
        for batch in sorted(batches):
            self.require("decode T=1", lengths=[1] * batch, zero_state=False)
        grid = tests["test_mtp_fp32_state_with_cache_and_state_update"]
        mtp |= {(b, t) for b in grid["batch_size"] for t in grid["seq_len"]}
        self.assertGreaterEqual(len(mtp), 15)
        for batch, tokens in sorted(mtp):
            self.require("MTP", lengths=[tokens] * batch, zero_state=False)


if __name__ == "__main__":
    unittest.main()

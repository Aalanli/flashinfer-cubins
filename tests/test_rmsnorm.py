"""rmsnorm: Python-built launch versus FlashInfer's own host launcher.

``impls/rmsnorm/compiler.py`` runs upstream ``norm::RMSNorm`` in a host-only
probe with intercepted ``cudaLaunchKernelEx``/``cudaFuncSetAttribute`` and
records, per architecture, the launch constants and the exact kernel parameter
bytes for several (batch, stride, PDL) combinations. These tests rebuild every
recorded launch in Python and compare it byte for byte. No kernel is launched,
so they run on any machine; they cover sm_100a without executing it.
"""

from __future__ import annotations

import ast
import ctypes
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.workload import ctypes_value  # noqa: E402
from harness.workloads.rmsnorm import (  # noqa: E402
    HIDDEN,
    UPSTREAM_GRID,
    UPSTREAM_REVISION,
    RMSNorm,
    build_launch,
    flashinfer_enable_pdl,
)

# cudaLaunchAttributeProgrammaticStreamSerialization
PDL_ATTRIBUTE = 6


def pack(args, params) -> bytes:
    """Kernel parameter buffer of ``args`` at the cubin's parameter offsets."""
    buffer = bytearray(params[-1].offset + params[-1].size)
    for value, param in zip(args, params, strict=True):
        raw = bytes(memoryview(ctypes_value(value)).cast("B"))
        assert len(raw) == param.size
        buffer[param.offset : param.offset + param.size] = raw
    return bytes(buffer)


def load(arch: str, device: str) -> RMSNorm:
    workload = RMSNorm.from_arch(arch, device=device)
    assert isinstance(workload, RMSNorm)
    return workload


class RMSNormLaunch(unittest.TestCase):
    def sidecar(self, arch: str) -> dict:
        return json.loads(RMSNorm.cubin_path(arch).with_suffix(".json").read_text())

    def test_launch_matches_flashinfer(self):
        for arch in RMSNorm.supported_arches:
            workload = load(arch, "cpu")
            record = workload.sidecar  # also checks the recorded constants
            self.assertEqual(workload.image_name(), record["kernel"])
            params = workload.kernel_params
            self.assertEqual(
                record["kernel_params"],
                [{"offset": p.offset, "size": p.size} for p in params],
            )
            self.assertGreaterEqual(len(record["examples"]), 4)
            for example in record["examples"]:
                with self.subTest(arch=arch, batch=example["batch"]):
                    p = example["pointers"]
                    spec = build_launch(
                        arch,
                        example["batch"],
                        p["input"],
                        p["weight"],
                        p["output"],
                        example["stride_input"],
                        example["stride_output"],
                    )
                    self.assertEqual(
                        pack(spec.args, params).hex(), example["params_hex"]
                    )
                    self.assertEqual(list(spec.grid), example["grid"])
                    self.assertEqual(list(spec.block), record["launch"]["block"])
                    self.assertEqual(spec.shared_mem, record["launch"]["shared_mem"])
                    self.assertEqual(
                        example["launch_attributes"],
                        [{"id": PDL_ATTRIBUTE, "value": int(example["enable_pdl"])}],
                    )
                    self.assertEqual(spec.pdl, record["launch"]["enable_pdl"])

    def test_rejects_u32_offset_overflow(self):
        """The kernel indexes rows as u32 ``blockIdx.x * stride``."""
        rows = (2**32 - 7168) // 7168 + 1  # last row ends exactly at 2^32
        build_launch("sm_86", rows, 0, 0, 0, 7168, 7168)
        for stride in ((7168, 7168 * 2), (7168 * 2, 7168)):
            with self.assertRaises(ValueError):
                build_launch("sm_86", rows, 0, 0, 0, *stride)

    def test_pdl_policy(self):
        self.assertFalse(flashinfer_enable_pdl("sm_86"))
        self.assertTrue(flashinfer_enable_pdl("sm_100a"))

    def test_rejects_bad_inputs(self):
        if not torch.cuda.is_available():
            self.skipTest("input checks use CUDA tensors")
        workload = load("sm_100a", "cuda")  # never launched
        x = torch.zeros((4, 7168), dtype=torch.bfloat16, device="cuda")
        w = torch.zeros((7168,), dtype=torch.bfloat16, device="cuda")
        (out,), spec = workload.setup_launch((x, w))
        self.assertEqual(tuple(out.shape), (4, 7168))
        self.assertEqual([ctypes.sizeof(a) for a in spec.args], [8] * 3 + [4] * 5)
        for bad in (
            (x.float(), w),
            (x[:, :-8], w),
            (x[:0], w),
            (x.t().contiguous().t(), w),
            (x, w[:-1]),
            (x.cpu(), w),
            (torch.zeros((4, 7176), dtype=torch.bfloat16, device="cuda")[:, 1:], w),
        ):
            with self.assertRaises(ValueError):
                workload.setup_launch(bad)
        workload.close()


class Tolerance(unittest.TestCase):
    def test_rejects_systematic_one_step_error(self):
        workload = RMSNorm(None, device="cpu")
        case = next(c for c in workload.get_cases() if c.name == "batch7")
        (ref,) = workload.get_reference(workload.get_inputs(case))
        workload.validate((ref,), (ref,))
        bumped = ref.view(torch.int16) + 1  # every element one BF16 step away
        with self.assertRaises(AssertionError):
            workload.validate((ref,), (bumped.view(torch.bfloat16),))


class UpstreamCoverage(unittest.TestCase):
    def test_norm_grid_is_covered(self):
        """FlashInfer's test_norm batch x contiguity grid (parsed from the
        pinned test) is a subset of the smoke cases."""
        path = (
            ROOT / f"resources/flashinfer-{UPSTREAM_REVISION}/tests/utils/test_norm.py"
        )
        if not path.is_file():
            self.skipTest("pinned FlashInfer checkout is not in resources/")
        (function,) = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.FunctionDef) and node.name == "test_norm"
        ]
        grid = {
            ast.literal_eval(d.args[0]): d.args[1]
            for d in function.decorator_list
            if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize"
        }
        batches = ast.literal_eval(grid["batch_size"])
        contiguity = ast.literal_eval(grid["contiguous"])
        self.assertEqual(
            set(UPSTREAM_GRID), {(b, c) for b in batches for c in contiguity}
        )
        smoke = [
            c.params
            for c in RMSNorm(None, device="cpu").get_cases()
            if c.suite == "smoke"
        ]
        for batch, contiguous in UPSTREAM_GRID:
            stride = HIDDEN if contiguous else 2 * HIDDEN
            self.assertTrue(
                any(
                    p["batch"] == batch and p.get("stride", HIDDEN) == stride
                    for p in smoke
                ),
                (batch, contiguous),
            )


if __name__ == "__main__":
    unittest.main()

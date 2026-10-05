"""The single-kernel ELF strip (``harness/cubin_strip.py``), proven on sm_86.

A cubin with three kernels, constant banks, device globals, printf and a
cloned helper function is stripped to each kernel in turn: ``check_strip``
accepts every result, the CUDA tools parse it, and it loads and runs on this
GPU with outputs, device globals and printf output identical to the
original's (compiled at test time; skipped without nvcc or an sm_86 GPU).
The packages that strip precompiled cubins (moe, batched_gemm, deep_gemm)
check their own artifacts in their package tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.cubin_strip import check_strip, cubin_kernels, strip_cubin  # noqa: E402
from harness.workload import cubin_info, device_matches  # noqa: E402

PROOF_SOURCE = r"""
#include <cstdio>
__constant__ float coeffs[4] = {1.5f, -2.0f, 0.25f, 3.0f};
__device__ int counter;
__device__ float table[64];
__device__ __noinline__ float helper(float x, int i) { return x * coeffs[i & 3] + table[i & 63]; }
extern "C" __global__ void kernel_a(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = helper(in[i], i);
  if (i == 0) { atomicAdd(&counter, 1); printf("kernel_a n=%d c0=%f\n", n, coeffs[0]); }
}
extern "C" __global__ void kernel_b(float* out, int n, float scale) {
  __shared__ float s[128];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  s[threadIdx.x] = i < n ? scale * (float)i : 0.f;
  __syncthreads();
  if (i < n) { table[i & 63] = s[(threadIdx.x + 1) & 127]; out[i] = s[127 - threadIdx.x] + coeffs[1]; }
  if (i == 0) printf("kernel_b scale=%f\n", scale);
}
extern "C" __global__ void kernel_c(int* out) {
  out[threadIdx.x] = counter + (int)helper(1.f, threadIdx.x);
}
"""

PROOF_RUNNER = r"""
import ctypes, sys, torch
sys.path.insert(0, sys.argv[1])
from harness import cuda_driver
image = open(sys.argv[2], "rb").read()
name = sys.argv[3]
dev = torch.device("cuda")
cuda_driver.ensure_context(dev)
module = cuda_driver.Module(image)
f = module.function(name)
n = 200
x = torch.arange(n, device=dev, dtype=torch.float32) * 0.5 - 3
out = torch.zeros(256, device=dev)
iout = torch.zeros(128, device=dev, dtype=torch.int32)
if name == "kernel_a":
    f.launch(2, 128, [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(out.data_ptr()), ctypes.c_int(n)])
elif name == "kernel_b":
    f.launch(2, 128, [ctypes.c_void_p(out.data_ptr()), ctypes.c_int(n), ctypes.c_float(1.25)])
else:
    f.launch(1, 128, [ctypes.c_void_p(iout.data_ptr())])
torch.cuda.synchronize()
lib = cuda_driver.lib()
state = []
for symbol in ("counter", "table"):
    ptr, size = ctypes.c_void_p(), ctypes.c_size_t()
    cuda_driver.check(lib.cuModuleGetGlobal_v2(ctypes.byref(ptr), ctypes.byref(size), module.handle, symbol.encode()), "global")
    buf = (ctypes.c_char * size.value)()
    cuda_driver.check(lib.cuMemcpyDtoH_v2(buf, ptr, size), "copy")
    state.append(bytes(buf).hex())
sys.stdout.flush()
print("RESULT", out.cpu().numpy().tobytes().hex(), iout.cpu().numpy().tobytes().hex(), *state, flush=True)
module.unload()
"""


def _nvcc() -> str | None:
    return (
        os.environ.get("NVCC")
        or shutil.which("nvcc")
        or (
            "/usr/local/cuda/bin/nvcc"
            if Path("/usr/local/cuda/bin/nvcc").exists()
            else None
        )
    )


class ElfStripSm86(unittest.TestCase):
    """The strip technique, proven by running stripped cubins on sm_86."""

    tmp: tempfile.TemporaryDirectory
    image: bytes

    @classmethod
    def setUpClass(cls):
        nvcc = _nvcc()
        if nvcc is None:
            raise unittest.SkipTest("nvcc is required to build the proof cubin")
        cls.tmp = tempfile.TemporaryDirectory(prefix="cubin-strip-")
        tmp = Path(cls.tmp.name)
        (tmp / "multi.cu").write_text(PROOF_SOURCE)
        # No -lineinfo: like the official trtllm-gen cubins. (Line tables are
        # covered by the moe kernel cubins built with -lineinfo, tests/test_moe.py.)
        subprocess.run(
            [
                nvcc,
                "-cubin",
                "-gencode=arch=compute_86,code=sm_86",
                "multi.cu",
                "-o",
                "multi.cubin",
            ],
            check=True,
            cwd=tmp,
        )
        cls.image = (tmp / "multi.cubin").read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_structure(self):
        kernels = cubin_kernels(self.image)
        self.assertEqual(sorted(kernels), ["kernel_a", "kernel_b", "kernel_c"])
        self.assertEqual(strip_cubin(self.image, None), self.image)
        for keep in kernels:
            with self.subTest(keep=keep):
                stripped = strip_cubin(self.image, keep)
                check_strip(self.image, stripped, keep)
                arch, names, _ = cubin_info(stripped)
                self.assertEqual((arch, names), ("sm_86", [keep]))
                path = Path(self.tmp.name) / f"{keep}.cubin"
                path.write_bytes(stripped)
                for tool in (
                    ["cuobjdump", "-elf"],
                    ["nvdisasm"],
                    ["cuobjdump", "-sass"],
                ):
                    if shutil.which(tool[0]):
                        subprocess.run(
                            tool + [str(path)], check=True, capture_output=True
                        )

    def _run(self, image: bytes, name: str) -> tuple[str, list[str]]:
        path = Path(self.tmp.name) / f"run_{name}_{len(image)}.cubin"
        path.write_bytes(image)
        result = subprocess.run(
            [sys.executable, "-c", PROOF_RUNNER, str(ROOT), str(path), name],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = result.stdout.splitlines()
        values = next(line for line in lines if line.startswith("RESULT "))
        prints = sorted(line for line in lines if line.startswith("kernel_"))
        return values, prints

    def test_runs_identically(self):
        if not device_matches("sm_86"):
            self.skipTest("needs an sm_86 GPU")
        for keep in ("kernel_a", "kernel_b", "kernel_c"):
            with self.subTest(keep=keep):
                original = self._run(self.image, keep)
                stripped = self._run(strip_cubin(self.image, keep), keep)
                self.assertEqual(stripped, original)
                if keep != "kernel_c":
                    self.assertTrue(stripped[1], "printf output missing")


if __name__ == "__main__":
    unittest.main()

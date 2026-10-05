"""Compile the RMSNorm workload into one single-kernel cubin per architecture.

Workload: DeepSeek-V3/R1 ``rmsnorm_h7168`` (``resources/rmsnorm.json``):
``out = x * rsqrt(mean(x^2) + 1e-6) * weight`` over hidden size 7168, BF16.
FlashInfer's CUDA path (``flashinfer.norm.rmsnorm`` -> ``csrc/norm.cu`` ->
``norm::RMSNorm`` in ``include/flashinfer/norm.cuh``) launches one kernel::

    RMSNormKernel<8, __nv_bfloat16>  <<<batch, (32, 28), 112 bytes>>>

``norm::RMSNorm`` computes ``vec_size = gcd(16 / sizeof(T), d) = 8``,
``block_size = min(1024, d / vec_size) = 896`` threads = 28 warps, dynamic
shared memory ``28 * sizeof(float)``, ``weight_bias = 0`` and sets the
programmatic-stream-serialization (PDL) launch attribute to ``enable_pdl``,
which ``flashinfer.norm.rmsnorm(enable_pdl=None)`` resolves to
``compute capability >= 9`` (on for sm_100a, off for sm_86; the kernel's
``griddepcontrol`` instructions are only compiled for ``__CUDA_ARCH__ >= 900``).

Kernel inventory and decisions. Only *live* kernels get a cubin and a workload:

=========================================  =======  ===========================================
kernel                                     status   reason
=========================================  =======  ===========================================
norm::RMSNormKernel<8, __nv_bfloat16>      live     The only kernel ``norm::RMSNorm`` launches
(sm_86, sm_100a)                                    for d = 7168 / BF16; one launch per call.
RMSNormKernel<1|2|4|16, __nv_bfloat16>     dead     Other ``DISPATCH_ALIGNED_VEC_SIZE`` cases;
                                                    ``gcd(8, 7168) = 8`` always selects 8 (the
                                                    probe confirms the selected instantiation).
QKRMSNorm / FusedAddRMSNorm / Gemma* /     dead     Other FlashInfer norm entry points (3-D
RMSNormQuant kernels                                input, residual, Gemma bias, FP8 quant); not
                                                    on the 2-D ``rmsnorm`` path. Not instantiated.
CuTe DSL ``rmsnorm_cute`` (Python path)    n/a      At this revision ``flashinfer.norm.rmsnorm``
                                                    prefers the CuTe DSL kernel when the
                                                    installed DSL targets the GPU, and falls
                                                    back to (or with ``FLASHINFER_USE_CUDA_NORM``
                                                    uses) the CUDA JIT kernel above. The DSL
                                                    kernel is JIT-generated Python without a
                                                    pinned cubin; this package keeps the CUDA
                                                    kernel, as the previous package did.
legacy ``CUBIN/rmsnorm__native_module_*``  dup      Exports of the same kernel from the previous
(2x sm_100a, 3x sm_86)                              ``.so`` builds; SASS identical to the cubins
                                                    built here. Not used.
=========================================  =======  ===========================================

One kernel per cubin is achieved at compile time: ``kernels/rmsnorm.cu`` holds
exactly one explicit instantiation of the unmodified upstream kernel, built
with ``nvcc -cubin -lineinfo`` for a single ``-gencode``; ``compile()`` checks
the architecture, the kernel count and the exact mangled symbol.

The kernel takes eight scalar/pointer parameters (no by-value struct). A
host-only probe (``kernels/probe_rmsnorm_launch.cu``, compiled and run here, no
GPU needed) runs upstream ``norm::RMSNorm`` with ``cudaFuncSetAttribute`` and
``cudaLaunchKernelEx`` redirected to recorders and fake pointers, and records
in the sidecar ``cubins/<arch>/rmsnorm.json``:

* ``launch``: the constants every probe run agrees on (selected ``vec_size``,
  ``block``, ``shared_mem``, the max-dynamic-smem function attribute) plus the
  arch's PDL policy (``enable_pdl``);
* ``examples``: per probe run, the grid, the launch attributes and the exact
  kernel parameter bytes, so the Python launch builder can be checked byte for
  byte without a GPU (``tests/test_rmsnorm.py``).

Upstream coverage (``UPSTREAM_COVERAGE``; enforced by ``tests/test_rmsnorm.py``):
FlashInfer's ``tests/utils/test_norm.py::test_norm`` runs ``flashinfer.norm.
rmsnorm`` over batch {1, 19, 99, 989} x contiguous {True, False (rows of a
``[batch, 2 * hidden]`` tensor)} x specify_out x enable_pdl, in FP16 at its
own hidden sizes. With the definition's BF16 / hidden 7168 (which selects
this kernel), every batch x contiguity pair is a smoke case of the workload
(row stride 14336 for the non-contiguous rows); specify_out only decides who
allocates the contiguous output and PDL is a launch attribute (on for sm_100a
as FlashInfer's default, off for sm_86). Not served by this kernel: hidden
sizes that are not multiples of 8 (other ``VEC_SIZE`` instantiations), the
quant/fused/Gemma variants, and the int64-stride / ``M * H > 2^31`` tests
(the kernel's strides and row offsets are u32; ``build_launch`` rejects
offsets past ``2^32``, and 300k-row inputs exceed the sm_86 case budget).

Nothing in this module runs at benchmark time; ``harness/workloads/rmsnorm.py``
loads the cubin listed in ``kernels.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
KERNELS = PACKAGE / "kernels"
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"

FLASHINFER_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer"
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
FLASHINFER_DIR = f"flashinfer-{FLASHINFER_REVISION}"

WORKLOAD = "rmsnorm"
SOURCE = "rmsnorm.cu"
PROBE = "probe_rmsnorm_launch.cu"
SYMBOL = "_ZN10flashinfer4norm13RMSNormKernelILj8E13__nv_bfloat16EEvPT0_S4_S4_jjjff"
HIDDEN = 7168
EPS = "1e-6"
VEC_SIZE = 8
ADAPTATION = (
    "Unmodified upstream FlashInfer norm::RMSNormKernel; explicit instantiation"
    " <8, __nv_bfloat16> selected by norm::RMSNorm for hidden 7168."
)
# (batch, stride_input, stride_output, enable_pdl); distinct values so each
# field of the recorded launch is attributable.
PROBE_RUNS: tuple[tuple[int, int, int, int], ...] = (
    (1, 7168, 7168, 0),
    (7, 7168, 7168, 1),
    (128, 8192, 7168, 0),
    (19, 14336, 7168, 1),  # upstream test_norm contiguous=False rows
    (14521, 7176, 7184, 1),
)
UPSTREAM_COVERAGE = {
    "enforced_by": "tests/test_rmsnorm.py (harness.workloads.rmsnorm.UPSTREAM_GRID "
    "must each match a smoke case)",
    "tests/utils/test_norm.py::test_norm": "batch {1, 19, 99, 989} x contiguous "
    "{True, False: row stride 2 * hidden} at the definition's BF16 / hidden 7168; "
    "specify_out (output allocation) and enable_pdl (launch attribute) do not "
    "change the kernel's work",
    "not_served": "hidden % 8 != 0 (other VEC_SIZE instantiations), quant/fused/"
    "Gemma kernels, int64 strides and M * H > 2^31 (u32 strides and offsets)",
}
INVENTORY: tuple[dict[str, str], ...] = (
    {
        "kernel": "norm::RMSNormKernel<8, __nv_bfloat16>",
        "status": "live",
        "reason": "The only kernel norm::RMSNorm launches for d=7168, BF16.",
    },
    {
        "kernel": "norm::RMSNormKernel<1|2|4|16, __nv_bfloat16>",
        "status": "dead",
        "reason": "Other DISPATCH_ALIGNED_VEC_SIZE cases; gcd(8, 7168) = 8.",
    },
    {
        "kernel": "QKRMSNorm/FusedAddRMSNorm/Gemma*/RMSNormQuant kernels",
        "status": "dead",
        "reason": "Other norm entry points, not on the 2-D rmsnorm path.",
    },
    {
        "kernel": "CuTe DSL rmsnorm_cute",
        "status": "n/a",
        "reason": "JIT Python DSL path preferred by flashinfer.norm.rmsnorm when"
        " the DSL supports the GPU; the CUDA JIT kernel (FLASHINFER_USE_CUDA_NORM"
        " or fallback) is the one packaged, as before.",
    },
    {
        "kernel": "legacy CUBIN/rmsnorm__native_module_*",
        "status": "dup",
        "reason": "Previous .so exports of the same kernel; identical SASS.",
    },
)


def _run(command: Sequence[str], cwd: Path) -> str:
    result = subprocess.run(
        list(command), cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


class ImplCompiler:
    """Build the single live RMSNorm kernel for one architecture."""

    supported_arches: tuple[str, ...] = ("sm_86", "sm_100a")

    def __init__(
        self,
        arch: str,
        *,
        optimization: str = "-O3",
        flags: Sequence[str] = (),
        nvcc: str | None = None,
        resources: Path | None = None,
    ):
        arch = "sm_" + str(arch).removeprefix("-arch=").removeprefix("sm_")
        if arch not in self.supported_arches:
            raise ValueError(f"rmsnorm supports {self.supported_arches}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = tuple(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        # Not resolved: keep repo-relative paths in line tables.
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.out_dir = PACKAGE / "cubins" / arch

    # -- paths ---------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        path = Path(os.path.abspath(path))
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def _includes(self) -> list[str]:
        flashinfer = self.resources / FLASHINFER_DIR
        if not flashinfer.is_dir():
            raise FileNotFoundError(
                f"{flashinfer} is missing (scripts/fetch_resources.py)"
            )
        return ["-I" + self._rel(flashinfer / "include")]

    def _gencode(self) -> str:
        number = self.arch.removeprefix("sm_")
        return f"-gencode=arch=compute_{number},code=sm_{number}"

    def _enable_pdl(self) -> bool:
        # flashinfer.utils.device_support_pdl: compute capability major >= 9.
        return int(self.arch.removeprefix("sm_").rstrip("af")) >= 90

    # -- compile -------------------------------------------------------------

    def _cubin_command(self, source: Path, output: Path, depfile: Path) -> list[str]:
        return [
            self.nvcc,
            "-std=c++17",
            self.optimization,
            "-cubin",
            self._gencode(),
            "-lineinfo",
            "--expt-relaxed-constexpr",
            *self._includes(),
            *self.flags,
            "-MD",
            "-MF",
            str(depfile),
            self._rel(source),
            "-o",
            str(output),
        ]

    def _probe_command(self, source: Path, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-std=c++17",
            "-O1",
            self._gencode(),
            "--expt-relaxed-constexpr",
            *self._includes(),
            self._rel(source),
            "-o",
            str(output),
        ]

    def _recorded(self, command: list[str], output: Path) -> list[str]:
        recorded = [Path(command[0]).name]
        skip = False
        for arg in command[1:]:
            if skip:
                skip = False
            elif arg == "-MF":
                skip = True
            elif arg != "-MD":
                recorded.append(self._rel(output) if arg == str(output) else arg)
        return recorded

    def _dependencies(self, depfile: Path) -> dict[str, str]:
        text = depfile.read_text().replace("\\\n", " ")
        tree = self.resources / FLASHINFER_DIR
        files: dict[str, str] = {}
        for token in text.split(":", 1)[1].split():
            path = Path(os.path.abspath(ROOT / token))
            if path.is_relative_to(tree):
                files[f"flashinfer/{path.relative_to(tree)}"] = _sha256(path)
        return dict(sorted(files.items()))

    def _check_cubin(self, image: bytes) -> list[dict[str, int]]:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from harness.workload import cubin_info

        arch, names, params = cubin_info(image)
        if arch != self.arch:
            raise RuntimeError(f"{WORKLOAD}: cubin declares {arch}, not {self.arch}")
        if names != [SYMBOL]:
            raise RuntimeError(f"{WORKLOAD}: expected [{SYMBOL}], found {names}")
        return [{"offset": p.offset, "size": p.size} for p in params]

    def _version(self) -> str:
        return _run([self.nvcc, "--version"], ROOT).strip()

    def compile(self) -> dict[str, str]:
        version = self._version()
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)
        source = KERNELS / SOURCE
        cubin = self.out_dir / f"{WORKLOAD}.cubin"
        with tempfile.TemporaryDirectory(prefix="rmsnorm-") as tmp:
            tmpdir = Path(tmp)
            depfile = tmpdir / f"{WORKLOAD}.d"
            command = self._cubin_command(source, cubin, depfile)
            _run(command, ROOT)
            image = cubin.read_bytes()
            params = self._check_cubin(image)
            files = self._dependencies(depfile)
            launch, examples = self._probe(tmpdir, params)
        sidecar: dict[str, Any] = {
            "workload": WORKLOAD,
            "arch": self.arch,
            "kernel": SYMBOL,
            "kernel_params": params,
            "cubin_sha256": hashlib.sha256(image).hexdigest(),
            "source": self._rel(source),
            "source_sha256": _sha256(source),
            "build": {
                "command": self._recorded(command, cubin),
                "nvcc_version": version,
            },
            "adaptation": ADAPTATION,
            "upstream_files": files,
            "probe": {
                "source": self._rel(KERNELS / PROBE),
                "source_sha256": _sha256(KERNELS / PROBE),
            },
            "launch": launch,
            "examples": examples,
        }
        _write_json(self.out_dir / f"{WORKLOAD}.json", sidecar)
        mapping = {WORKLOAD: str(cubin.relative_to(PACKAGE))}
        self._merge_manifest(mapping)
        self._merge_provenance(files)
        return mapping

    # -- launch probe ----------------------------------------------------------

    def _probe(
        self, tmpdir: Path, params: list[dict[str, int]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        exe = tmpdir / PROBE.removesuffix(".cu")
        _run(self._probe_command(KERNELS / PROBE, exe), ROOT)
        examples = []
        for batch, stride_in, stride_out, pdl in PROBE_RUNS:
            out = _run(
                [
                    str(exe),
                    str(batch),
                    str(HIDDEN),
                    str(stride_in),
                    str(stride_out),
                    str(pdl),
                    EPS,
                ],
                ROOT,
            )
            examples.append(json.loads(out))
        constant_keys = ("vec_size", "block", "shared_mem", "func_attributes")
        first = examples[0]
        for example in examples:
            for key in constant_keys:
                if example[key] != first[key]:
                    raise RuntimeError(f"probe {key} differs between runs")
            if example["params"] != params:
                raise RuntimeError(
                    f"probe parameter layout {example['params']} != cubin {params}"
                )
            if not example["stream_passed"]:
                raise RuntimeError("probe: the launch did not use the given stream")
            if example["grid"] != [example["batch"], 1, 1]:
                raise RuntimeError(f"probe grid {example['grid']} != (batch, 1, 1)")
        if first["vec_size"] != VEC_SIZE:
            raise RuntimeError(
                f"norm::RMSNorm selected VEC_SIZE {first['vec_size']}, the cubin"
                f" instantiates {VEC_SIZE}"
            )
        if any(not a["same_kernel"] for a in first["func_attributes"]):
            raise RuntimeError("probe: attribute set on a different kernel")
        launch = {
            "hidden": HIDDEN,
            "eps": float(EPS),
            "weight_bias": 0.0,
            "vec_size": first["vec_size"],
            "block": first["block"],
            "shared_mem": first["shared_mem"],
            "func_attributes": [
                {"attr": a["attr"], "value": a["value"]}
                for a in first["func_attributes"]
            ],
            "enable_pdl": self._enable_pdl(),
        }
        for example in examples:
            for key in (*constant_keys, "params", "stream_passed"):
                example.pop(key)
        return launch, examples

    # -- manifests -------------------------------------------------------------

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        _write_json(MANIFEST, dict(sorted(manifest.items())))

    def _merge_provenance(self, files: dict[str, str]) -> None:
        entry: dict[str, Any] = {
            "arch": self.arch,
            "repository": FLASHINFER_REPOSITORY,
            "revision": FLASHINFER_REVISION,
            "kernels": {
                WORKLOAD: {
                    "symbol": SYMBOL,
                    "source": f"kernels/{SOURCE}",
                    "probe": f"kernels/{PROBE}",
                    "adaptation": ADAPTATION,
                }
            },
            "kernel_inventory": list(INVENTORY),
            "upstream_coverage": UPSTREAM_COVERAGE,
            "files": dict(sorted(files.items())),
        }
        try:
            existing = json.loads(PROVENANCE.read_text())
        except (OSError, ValueError):
            existing = []
        if not isinstance(existing, list):
            existing = []
        merged = [e for e in existing if e.get("arch") != self.arch] + [entry]
        order = {a: i for i, a in enumerate(self.supported_arches)}
        merged.sort(key=lambda e: order.get(e.get("arch"), len(order)))
        _write_json(PROVENANCE, merged)

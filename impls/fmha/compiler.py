"""Single-kernel FMHA cubins: FlashInfer FA2 (sm_86) and trtllm-gen (sm_100a).

Two independent libraries share the ``fmha`` package; each architecture has
its own build module next to this file, loaded by path (``compile_kernels.py``
loads this file by path, so plain sibling imports are unavailable):

* ``sm_86`` -> ``fa2_build.py``: FlashAttention-2 kernels of the FlashInfer
  0.7.0 sm80 JIT-cache wheel (``resources/flashinfer-jit-cache-sm80``), one
  kernel per cubin via the shared ELF strip (sm_80 SASS runs on sm_86).
* ``sm_100a`` -> ``trtllm_build.py``: the trtllm-gen FMHA cubins of
  ``cubins/fmha`` (flashinfer_cubin 0.6.8/0.6.9, artifact
  ``55bba559.../fmha/trtllm-gen``), with their never-launched
  ``<kernel>GetSmemSize`` helper stripped.

See those modules' docstrings for the kernel inventories, the live/dead
decisions and the probes. Each build module writes

* ``cubins/<arch>/<workload>.cubin`` (generated, not checked in),
* ``variants/<arch>.json``: the compact variant index the registry reads at
  import (no cubin is parsed at import time),
* its ``kernels.json`` and ``provenance.json`` sections (only its own arch),
* sidecars/fixtures under ``cubins/<arch>/`` or ``fixtures/``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"
VARIANTS = PACKAGE / "variants"

BUILD_MODULES = {"sm_86": "fa2_build.py", "sm_100a": "trtllm_build.py"}


def run(command: Sequence[str], cwd: Path = ROOT, stdin: str | None = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
        )
    return result.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=indent, sort_keys=False) + "\n")
    os.replace(tmp, path)


def merge_arch(path: Path, arch: str, value: Any) -> None:
    """Replace only ``arch`` in the JSON object at ``path``."""
    data = json.loads(path.read_text()) if path.is_file() else {}
    data[arch] = value
    write_json(path, dict(sorted(data.items())))


def merge_provenance(arch: str, build: dict[str, Any]) -> None:
    """Replace only ``arch``'s build record in ``provenance.json``."""
    data = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
    builds = data.get("builds", {})
    builds[arch] = build
    write_json(
        PROVENANCE,
        {
            "package": "fmha",
            "license": "LICENSE (FlashInfer, Apache-2.0); the trtllm-gen cubins and "
            "meta-info are NVIDIA's (Apache-2.0 headers)",
            "scope": "sm_86: FlashInfer FA2 prefill/decode/merge kernels of the "
            "FlashInfer 0.7.0 sm80 JIT cache; sm_100a: trtllm-gen FMHA cubins of "
            "cubins/fmha (flashinfer_cubin 0.6.8/0.6.9)",
            "builds": dict(sorted(builds.items())),
        },
    )


def rel(path: Path) -> str:
    path = Path(os.path.abspath(path))
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_workload_module(root: Path, name: str) -> Any:
    """``harness.workloads.fmha.<name>`` without importing every package of
    ``harness.workloads`` (the compile stage must not depend on them)."""
    import types

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for package in ("harness.workloads", "harness.workloads.fmha"):
        if package not in sys.modules:
            stub = types.ModuleType(package)
            stub.__path__ = [str(root.joinpath(*package.split(".")))]
            sys.modules[package] = stub
    return importlib.import_module(f"harness.workloads.fmha.{name}")


def load_build_module(arch: str) -> Any:
    path = PACKAGE / BUILD_MODULES[arch]
    name = f"impls_fmha_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ImplCompiler:
    """Build (strip/extract) the fmha single-kernel cubins of one arch."""

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
            raise ValueError(f"fmha supports {self.supported_arches}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        # Precompiled kernels are not recompiled; optimization/flags only apply
        # to the host-only layout probes.
        self.optimization = optimization
        self.flags = tuple(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.out_dir = PACKAGE / "cubins" / arch

    def compile(self) -> dict[str, str]:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = load_build_module(self.arch)
        mapping: dict[str, str] = module.build(self)
        merge_arch(MANIFEST, self.arch, dict(sorted(mapping.items())))
        return mapping

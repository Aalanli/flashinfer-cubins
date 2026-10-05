#!/usr/bin/env python3
"""Compile every single-kernel workload cubin for every supported architecture.

Each ``impls/<package>/compiler.py`` defines ``ImplCompiler`` with a
``supported_arches`` tuple; ``ImplCompiler(arch, ...).compile()`` writes the
package's cubins and merges ``{arch: {workload: cubin}}`` into its
``kernels.json``. This is the only stage that invokes a compiler: workloads
load the resulting cubins through libcuda at run time.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
IMPLS = ROOT / "impls"


def compiler_packages() -> dict[str, Path]:
    return {path.parent.name: path for path in sorted(IMPLS.glob("*/compiler.py"))}


def load_compiler(package: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"impls_{package}_compiler", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses etc. resolve their module
    spec.loader.exec_module(module)
    return module.ImplCompiler


def main(argv: list[str] | None = None) -> int:
    packages = compiler_packages()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        choices=sorted(packages),
        help="Compile only these packages (repeatable; default: all)",
    )
    parser.add_argument(
        "--arch",
        action="append",
        help="Compile only these architectures, e.g. sm_100a (repeatable)",
    )
    parser.add_argument(
        "--optimization", default="-O3", choices=("-O0", "-O1", "-O2", "-O3")
    )
    parser.add_argument("--nvcc", help="nvcc to use (default: $NVCC or PATH)")
    parser.add_argument(
        "--resources", type=Path, help="Pinned upstream sources (default: resources/)"
    )
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from harness.workload import normalize_arch

    wanted = {normalize_arch(a) for a in args.arch} if args.arch else None
    failed = False
    for package in args.package or sorted(packages):
        compiler = load_compiler(package, packages[package])
        for arch in compiler.supported_arches:
            if wanted is not None and arch not in wanted:
                continue
            start = time.perf_counter()
            try:
                mapping = compiler(
                    arch,
                    optimization=args.optimization,
                    nvcc=args.nvcc,
                    resources=args.resources,
                ).compile()
            except Exception as exc:
                failed = True
                print(
                    f"{package} {arch} FAILED: {type(exc).__name__}: {exc}", flush=True
                )
                continue
            elapsed = time.perf_counter() - start
            print(f"{package} {arch} ok ({elapsed:.1f}s)", flush=True)
            for name, cubin in sorted(mapping.items()):
                print(f"  {name}: {cubin}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

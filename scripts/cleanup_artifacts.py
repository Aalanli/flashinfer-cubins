#!/usr/bin/env python3
"""Remove what ``scripts/fetch_resources.py`` downloads and what the compilers
generate, returning the tree to a fresh clone.

Downloads (``resources/``; paths from ``fetch_resources.py``'s constants):

- the pinned source trees (``SOURCE_TREES``: FlashInfer, CUTLASS, FlashInfer
  v0.2.10 for moe), the DeepGEMM checkout and the FlashInfer v0.6.9 tree;
- the FlashInfer sm80 JIT-cache wheel and its extracted cubins;
- the trtllm-gen export headers (``trtllm-gen-artifacts/``);
- the source tarballs and ``downloads.json`` of a full fetch;
- ``moe-artifacts/``, the official GEMM cubins ``impls/moe/compiler.py``
  downloads.

The definitions a full fetch also writes (``resources/*.json``, the gpu-mode
task files, ``flashinfer_revision.txt``) are checked in and kept.

Generated (everything ``impls/<package>/compiler.py`` writes; ``.gitignore``
lists the same paths):

- ``impls/*/cubins/``: cubins, sidecars and probe fixtures;
- ``impls/*/kernels.json`` and ``impls/*/provenance.json``;
- ``impls/batched_gemm/{variants,fixtures}/``, ``impls/fmha/{variants,fixtures}/``
  and ``impls/deep_gemm/variants.json``;
- ``impls/gdn/kernels/gdn_chunk.cu`` (generated from the pinned upstream
  kernel);
- ``impls/**/*.tmp`` left by an interrupted atomic write.

Every target must be ignored by git and hold no tracked file; otherwise
nothing is removed. A symlink (``resources/`` may link trees shared with
another checkout) is unlinked, never followed. Dry run by default; pass
``--apply`` to delete, ``--downloads`` or ``--generated`` to select one group.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "resources"
IMPLS = ROOT / "impls"

# Compiler outputs beyond each package's cubins/, kernels.json, provenance.json.
GENERATED = (
    "batched_gemm/variants",
    "batched_gemm/fixtures",
    "deep_gemm/variants.json",
    "fmha/variants",
    "fmha/fixtures",
    "gdn/kernels/gdn_chunk.cu",
)


def fetch_resources() -> ModuleType:
    """``scripts/fetch_resources.py``, loaded by path (``scripts`` is not a
    package)."""
    path = Path(__file__).resolve().with_name("fetch_resources.py")
    spec = importlib.util.spec_from_file_location("fetch_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def downloads() -> list[Path]:
    fetch = fetch_resources()
    candidates = [RESOURCES / name for name in fetch.SOURCE_TREES]
    candidates += [
        fetch.DEEPGEMM_DIR,
        fetch.FLASHINFER_V069_DIR,
        fetch.JIT_CACHE_DIR,
        RESOURCES / fetch.JIT_CACHE_WHEEL,
        fetch.TRTLLM_GEN_DIR,
        RESOURCES / "flashinfer.tar.gz",
        RESOURCES / "cutlass.tar.gz",
        RESOURCES / "downloads.json",
        RESOURCES / "moe-artifacts",
    ]
    return [p for p in candidates if p.exists() or p.is_symlink()]


def generated() -> list[Path]:
    candidates: list[Path] = []
    for package in sorted(p for p in IMPLS.iterdir() if p.is_dir()):
        candidates += [
            package / "cubins",
            package / "kernels.json",
            package / "provenance.json",
        ]
    candidates += [IMPLS / name for name in GENERATED]
    found = [p for p in candidates if p.exists() or p.is_symlink()]
    found += [
        p
        for p in sorted(IMPLS.rglob("*.tmp"))
        if not any(p.is_relative_to(q) for q in found)
    ]
    return found


def git(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    # check-ignore exits 1 when nothing is ignored.
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return set(result.stdout.splitlines())


def check(paths: list[Path]) -> list[str]:
    """Reasons to refuse: a target outside the tree, not ignored by git, or
    holding a tracked file."""
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    if not names:
        return []
    # --no-index: match the patterns even for tracked paths (reported below).
    ignored = git("check-ignore", "--no-index", "--", *names)
    tracked = git("ls-files", "--", *names)
    problems = []
    for path, name in zip(paths, names):
        if not path.parent.resolve().is_relative_to(ROOT):
            problems.append(f"{name}: outside the repository")
        elif name not in ignored:
            problems.append(f"{name}: not ignored by .gitignore")
        elif any(t == name or t.startswith(name + "/") for t in tracked):
            problems.append(f"{name}: holds files tracked by git")
    return problems


def remove(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="delete (default: list)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--downloads", action="store_true", help="only fetch_resources downloads"
    )
    group.add_argument("--generated", action="store_true", help="only compiler outputs")
    args = parser.parse_args(argv)
    paths: list[Path] = []
    if not args.generated:
        paths += downloads()
    if not args.downloads:
        paths += generated()
    problems = check(paths)
    if problems:
        print("refusing to delete anything:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        kind = " (symlink)" if path.is_symlink() else ""
        print(("removing " if args.apply else "would remove ") + name + kind)
        if args.apply:
            remove(path)
    if not args.apply:
        print(f"{len(paths)} paths; rerun with --apply to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

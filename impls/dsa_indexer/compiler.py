"""Compile the DSA indexer's live DeepGEMM kernels into single-kernel cubins.

The DSA indexer workload (DeepSeek-V3.2 ``dsa_topk_indexer_fp8_h64_d128_topk2048_ps64``
without its top-k stage, see ``missing_workloads.md``) is DeepGEMM's paged
FP8 MQA logits path at the pinned revision below. Its real execution path is::

    deep_gemm.get_paged_mqa_logits_metadata(context_lens, 64, num_sms)
        -> sm100_paged_mqa_logits_metadata   (1 CTA, 1024 threads)
    deep_gemm.fp8_paged_mqa_logits(q, kv_cache, weights, context_lens,
                                   block_table, schedule_meta, max_len, False)
        -> sm100_paged_mqa_logits            (num_sms CTAs, 384 threads)

Kernel inventory of the previous multi-kernel package and the decision taken
here (only *live* kernels get a cubin and a workload):

==============================  ======  ==============================================
kernel                          status  reason
==============================  ======  ==============================================
sm100_paged_mqa_logits_metadata live    Launched by DeepGEMM on every call of
                                        ``get_paged_mqa_logits_metadata``; its output
                                        (per-SM ``(q_token, kv_split)`` starts) is a
                                        required input of the logits kernel.
sm100_paged_mqa_logits          live    The workload's Tensor Core (tcgen05) kernel.
mask_padding                    dead    Harness glue from the old C++ adapter: it
                                        overwrote logits at/after each sequence
                                        length with -inf because DeepGEMM leaves them
                                        undefined (``clean_logits`` is unsupported for
                                        paged logits). Neither DeepGEMM nor FlashInfer
                                        launches it; downstream top-k reads only the
                                        first ``seq_len`` logits of a row.
mask_padding (``dsa_indexer__   dead    The same ``_Z12mask_paddingPfPKii`` kernel,
legacy__*.cubin``,                      exported from older builds of the C++ adapter
``native_module_1__*.cubin``)           (verified: each holds only that symbol).
top-k / page-table transform    absent  Excluded at the user's request
                                        (``missing_workloads.md``).
==============================  ======  ==============================================

One kernel per cubin is achieved at compile time: each source under
``kernels/`` is a translation unit with exactly one explicit template
instantiation, compiled with ``nvcc -cubin -lineinfo`` for a single
``-gencode``. The result is a genuine nvcc cubin (no post-hoc ELF surgery);
``compile()`` checks the architecture, the kernel count and the exact mangled
symbol of every emitted cubin.

``num_scheduling_partitions`` (DeepGEMM's ``num_sms``) is a template argument
of the metadata kernel and fixed here to 148, the B200 SM count. The logits
kernel has no SM-count template argument: its grid is the number of schedule
partitions (``schedule_meta.shape[0] - 1``).

Harness path versus upstream: the same two kernels with DeepGEMM's
arguments; the logits row stride is ``align(block_table.shape[1] * 64, 256)``
(``max_context_len = width * 64`` as DeepGEMM's test passes it) and the
block-table stride is the table's own, so any table width is served.
Upstream test of these kernels and the cases covering it
(``tests/test_dsa_indexer.py::UpstreamCoverage`` runs the pinned test's own
case enumerator):

==========================================  =====================================
upstream                                    harness cases
==========================================  =====================================
DeepGEMM test_paged_mqa_logits (SM100):     ``upstream_paged_b256_avg8192``,
non-varlen FP8, FP32 logits/weights,        ``_b256_avg65536``, ``_b4096_avg8192``
block_kv 64, 2D context lens, next_n 1,     (throughput; lengths randint(0.7,
64 heads, d128: batch 256/4096 x avg_kv     1.3) x avg_kv as upstream)
8192/65536 (<= 32Mi pool tokens)
official inventory rows                     10 ``throughput_batch*_pages*`` rows
                                            with their table widths
==========================================  =====================================

Not served (outside the definition or other kernel instantiations): varlen,
MXFP4/MXFP8, BF16 logits/weights, block_kv 32/128, next_n 6, other head counts
and dims, ``test_mqa_logits`` (non-paged) and the sparse MQA logits tests.

Nothing in this module runs at benchmark time; the workloads in
``harness/workloads/dsa_indexer.py`` load the cubins listed in ``kernels.json``.
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
from typing import NamedTuple

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"

DEEPGEMM_REPOSITORY = "https://github.com/deepseek-ai/DeepGEMM"
DEEPGEMM_REVISION = "78b69000794d0937b47ae3387eff7663410264d1"
CUTLASS_DIR = "cutlass-b46b16d003484063bca4ed365e44095c4c6ed633"
# sha256 over (relative path + sha256(content)) of every file in include/.
CUTLASS_INCLUDE_TREE_SHA256 = (
    "def1849ca143942f68179aeb664a6cb9bf46001082d7d4f5c63531ea03866e6d"
)
NUM_SCHEDULING_PARTITIONS = 148

GENCODE = {"sm_100a": "-gencode=arch=compute_100a,code=sm_100a"}


class Kernel(NamedTuple):
    # A NamedTuple, not a dataclass: compile_kernels.py loads this file without
    # registering it in sys.modules, which dataclasses require.
    workload: str
    source: str  # relative to PACKAGE
    symbol: str
    upstream: str


LIVE_KERNELS = (
    Kernel(
        "dsa_indexer_metadata",
        "kernels/get_paged_mqa_logits_metadata.cu",
        "_ZN9deep_gemm5sched31sm100_paged_mqa_logits_metadataILj1ELb1ELb0ELj256ELj148EEEvjjPKjS3_Pj",
        "deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh",
    ),
    Kernel(
        "dsa_indexer_logits",
        "kernels/fp8_paged_mqa_logits.cu",
        "_ZN9deep_gemm22sm100_paged_mqa_logitsILj1ELj64ELj128ELj64ELb0ELb1ELb0ELj3ELj5ELj256ELj16ELj128ELj256EN7cutlass12float_e4m3_tEffLj2EEEvjjjPKjPT13_S4_S4_S4_14CUtensorMap_stS7_S7_S7_S7_",
        "deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh",
    ),
)

DEAD_KERNELS = {
    "_Z12mask_paddingPfPKii": "harness glue (-inf padding normalization), not part "
    "of DeepGEMM/FlashInfer; also the sole kernel of the legacy "
    "dsa_indexer__legacy__*.cubin and native_module_1__*.cubin exports",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cutlass_tree_sha256(cutlass: Path) -> str:
    include = cutlass / "include"
    return hashlib.sha256(
        b"".join(
            str(path.relative_to(cutlass)).encode()
            + hashlib.sha256(path.read_bytes()).digest()
            for path in sorted(include.rglob("*"))
            if path.is_file()
        )
    ).hexdigest()


def _cubin_info(image: bytes) -> tuple[str, list[str]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness.workload import cubin_info

    arch, names, _ = cubin_info(image)
    return arch, names


class ImplCompiler:
    """Builds ``cubins/<arch>/<workload>.cubin`` and merges ``kernels.json``."""

    supported_arches: tuple[str, ...] = tuple(GENCODE)

    def __init__(
        self,
        arch: str,
        *,
        optimization: str = "-O3",
        flags: Sequence[str] = (),
        nvcc: str | None = None,
        resources: Path | None = None,
    ):
        if arch not in self.supported_arches:
            raise ValueError(
                f"dsa_indexer supports {self.supported_arches}, not {arch}"
            )
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = list(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(resources or ROOT / "resources").resolve()
        self.deepgemm = self.resources / "DeepGEMM"
        self.cutlass = self.resources / CUTLASS_DIR

    # -- pinned sources ---------------------------------------------------------

    def verify_sources(self) -> None:
        revision = subprocess.check_output(
            ["git", "-C", str(self.deepgemm), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != DEEPGEMM_REVISION:
            raise ValueError(f"expected DeepGEMM {DEEPGEMM_REVISION}, got {revision}")
        dirty = subprocess.run(
            ["git", "-C", str(self.deepgemm), "diff", "--quiet", "HEAD", "--"]
            + ["deep_gemm/include"],
            check=False,
        )
        if dirty.returncode:
            raise ValueError(
                "DeepGEMM deep_gemm/include differs from the pinned revision"
            )
        tree = _cutlass_tree_sha256(self.cutlass)
        if tree != CUTLASS_INCLUDE_TREE_SHA256:
            raise ValueError(f"unexpected CUTLASS include tree hash {tree}")

    def command(self, source: Path, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-std=c++20",
            self.optimization,
            "-lineinfo",
            "-cubin",
            GENCODE[self.arch],
            "--expt-relaxed-constexpr",
            "-I" + str(self.deepgemm / "deep_gemm/include"),
            "-I" + str(self.cutlass / "include"),
            *self.flags,
            str(source),
            "-o",
            str(output),
        ]

    def _portable(self, command: list[str]) -> list[str]:
        """The command with repository paths made relative to the repo root."""
        root = str(ROOT) + os.sep
        return [part.replace(root, "") for part in command]

    # -- build -------------------------------------------------------------------

    def compile(self) -> dict[str, str]:
        self.verify_sources()
        out_dir = PACKAGE / "cubins" / self.arch
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("*.cubin"):
            stale.unlink()
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        mapping: dict[str, str] = {}
        records: dict[str, dict] = {}
        with tempfile.TemporaryDirectory(prefix="dsa_indexer-") as tmp:
            for kernel in LIVE_KERNELS:
                source = PACKAGE / kernel.source
                target = out_dir / f"{kernel.workload}.cubin"
                staged = Path(tmp) / target.name
                command = self.command(source, staged)
                subprocess.run(command, check=True)
                image = staged.read_bytes()
                arch, names = _cubin_info(image)
                if arch != self.arch:
                    raise ValueError(f"{kernel.source}: cubin declares {arch}")
                if names != [kernel.symbol]:
                    raise ValueError(
                        f"{kernel.source}: expected only {kernel.symbol}, found {names}"
                    )
                shutil.move(staged, target)
                relative = target.relative_to(PACKAGE).as_posix()
                mapping[kernel.workload] = relative
                records[kernel.workload] = {
                    "cubin": relative,
                    "symbol": kernel.symbol,
                    "sha256": hashlib.sha256(image).hexdigest(),
                    "source": kernel.source,
                    "source_sha256": _sha256(source),
                    "upstream": kernel.upstream,
                    # Compiled to a staging file, then moved to ``cubin``.
                    "command": self._portable(self.command(source, target)),
                }
        self._merge_manifest(mapping)
        self._write_provenance(records, version.strip().splitlines()[-1])
        return mapping

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        MANIFEST.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")

    def _write_provenance(self, records: dict[str, dict], nvcc_version: str) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {})
        builds[self.arch] = {"nvcc": nvcc_version, "kernels": records}
        files = {
            str(path.relative_to(self.deepgemm)): _sha256(path)
            for path in sorted((self.deepgemm / "deep_gemm/include").rglob("*.cuh"))
        }
        provenance = {
            "repository": DEEPGEMM_REPOSITORY,
            "revision": DEEPGEMM_REVISION,
            "license": "LICENSE (DeepGEMM, MIT); CUTLASS_LICENSE (BSD-3-Clause)",
            "cutlass": f"resources/{CUTLASS_DIR}",
            "cutlass_include_tree_sha256": CUTLASS_INCLUDE_TREE_SHA256,
            "num_scheduling_partitions": NUM_SCHEDULING_PARTITIONS,
            "scope": "metadata + paged FP8 MQA logits, one kernel per cubin; "
            "top-k excluded; logits after each sequence's last 256-token split "
            "are left unwritten as upstream",
            "builds": dict(sorted(builds.items())),
            "excluded_kernels": DEAD_KERNELS,
            "files": files,
        }
        PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")

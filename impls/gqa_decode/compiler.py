"""Compile the GQA paged decode kernels into single-kernel cubins.

Workload: Qwen3-30B-A3B ``gqa_paged_decode_h32_kv4_d128_ps1`` (32 query heads,
4 KV heads, head dim 128, page size 1, BF16, see ``resources/gqa_decode.json``);
outputs ``(output [B, 32, 128] bf16, lse [B, 32] f32 base 2)``. Every live
kernel becomes its own cubin and its own workload:

sm_86 (one upstream FlashInfer kernel)::

    BatchDecodeWithPagedKVCacheKernel<kNone, 2, 1, 8, 16, 8, 1, ...>
                     <<<(batch, 4), (16, 8)>>>   non-partitioned paged decode

sm_100a (CUTLASS SM100 FMHA over gathered ragged K/V, all 32 query heads)::

    gqa_plan         <<<1, 1>>>                 tile-scheduler / varlen metadata
    gqa_gather       <<<batch, 256>>>           size-1 pages -> ragged packed K/V
    device_kernel<Sm100FmhaFwdKernelTmaWarpspecialized<...>>
                     <<<sm_count, 512, cluster (1, 1, 1)>>>
    gqa_finish       <<<batch, 256>>>           copy output, empty requests

Kernel inventory of the previous multi-kernel package (``implementation.json``
and its exported ``CUBIN/*.cubin``) and the decision taken here. Only *live*
kernels get a cubin and a workload:

==========================================  =======  ==========================================
kernel (previous cubin)                     status   reason
==========================================  =======  ==========================================
BatchDecodeWithPagedKVCacheKernel           live     The only kernel the sm_86 launcher ran (one
<kNone,2,1,8,16,8,1,...> (sm_86)                     launch per call; metadata precomputed on
                                                     the host in ``setup``).
MergeStates / VariableLengthMergeStates     dead     Upstream split-KV reduction; only launched
(FlashInfer, sm_86)                                  with ``partition_kv = true``. The launcher
                                                     always ran the non-partitioned kernel
                                                     (``partition_kv = false``, one CTA per
                                                     (request, KV head)), so they were never
                                                     instantiated. Not built.
gqa_plan (sm_100a)                          live     Launched on every sm_100a run; writes the
                                                     varlen offsets and the host-precomputed
                                                     tile scheduler's work lists.
gqa_gather (sm_100a)                        live     Launched on every sm_100a run; the FMHA
                                                     reads K/V through dense TMA descriptors.
device_kernel<Sm100FmhaFwdKernelTma...>     live     The sm_100a Tensor Core (tcgen05) kernel.
(sm_100a)
gqa_finish (sm_100a)                        live     Launched on every sm_100a run; the FMHA
                                                     attends empty requests to one zero row.
BatchDecode... built for sm_100a            dead     ``CUBIN/*__sm_100a__004fe83ca63c9b97.cubin``
                                                     is a stale export of the sm_86 template
                                                     compiled for sm_100a; the sm_100a package
                                                     entry always used the CUTLASS path.
duplicate exports (``CUBIN/*__sm_86__*``,   dup      Several builds of the same kernels; covered
``*__sm_100a__6a63...``/``d9a7...``)                 by the live rows above.
==========================================  =======  ==========================================

One kernel per cubin is achieved at compile time: every source in ``kernels/``
is a translation unit holding exactly one ``__global__`` function (the three
project-local adapters, moved verbatim) or exactly one explicit template
instantiation of an upstream kernel (FlashInfer decode, CUTLASS FMHA), compiled
with ``nvcc -cubin -lineinfo`` for a single ``-gencode``; ``compile()`` checks
the architecture, the kernel count and the exact mangled symbol of every cubin.
Upstream code is included unmodified from the pinned checkouts in
``resources/``.

Every cubin gets a sidecar ``cubins/<arch>/<workload>.json`` with its build
record. The two upstream kernels take one by-value ``Params`` struct; for them
a host-only probe (``kernels/probe_*.cu``, compiled and run here, needs no GPU)
builds that struct with the upstream host code exactly as the previous native
launcher did and records into the sidecar ``param_layout``:

* ``param_size`` and ``fields`` (``{name: {offset, size, kind}}``; kinds
  ``ptr``, ``i32``, ``u32``, ``f32``, ``bool``, ``bytes``, ``tma``) covering
  every non-padding byte; FMHA layout fields are the dynamic leaves of the cute
  shapes/strides (``layout_Q.shape.2.1`` is ``get<2,1>(shape)``);
* ``constants``: launch constants (dynamic shared memory, block, cluster,
  FlashInfer's opaque ``uint_fastdiv(1)`` bytes, ...);
* ``examples``: probe runs with fake pointers (``pointer_roles`` names their
  sentinel indices) and the struct's raw bytes, so the Python builders can be
  checked byte for byte without a GPU. The FMHA probe interposes
  ``cuTensorMapEncodeTiled`` with a deterministic packing of its arguments
  (mirrored by ``harness.workloads.gqa_decode.fake_encode``) and
  ``cudaDriverGetVersion`` (CUTLASS's driver <= 13.1 descriptor fix-up), so the
  comparison covers every encode argument too;
* ``tensor_maps`` (FMHA only): for each TMA descriptor, the encode arguments
  symbolized over the probe runs: the address as ``{pointer, offset}`` (the
  output descriptor starts one row before ``o``), sizes as constants or
  ``{var, scale, offset}`` affine in ``batch`` / ``total_kv``.

Harness path versus upstream. The definition's upstream API is FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper``. On sm_86 the harness launches that
wrapper's FA2 decode kernel, but always non-partitioned (``partition_kv =
false``, one CTA per (request, KV head)) where upstream's planner may split
long KV across CTAs and merge (``MergeStates``, not built). On sm_100a the
harness pipeline is not upstream's decode path at all: it gathers the paged
K/V and runs the CUTLASS SM100 *prefill* FMHA (256-row Q tiles, one query row
per request) over all 32 heads. Both are kept deliberately; what must hold is
that every regime upstream tests is a harness case, which
``tests/test_gqa_decode.py::UpstreamCoverage`` checks against the pinned
sources:

==============================================  ==================================
upstream test (FlashInfer aa7c67f)              harness cases
==============================================  ==================================
test_batch_decode_with_paged_kv_cache (+ fast   ``upstream_decode_b{12,17,128}_kv
plan, tuple cache, CUDA graph variants):        {54,97,512,2048,16384}``: uniform
page_size 1, 4/32 heads, d128, NHD, no RoPE     lengths, identity page table
(fp16/fp8 dtypes there; BF16 per definition)    (b128 x 16384 in throughput)
test_paged_decode_extreme_negative_logits       ``upstream_decode_extreme_negative
(bf16)                                          _logits``: q 64, k -64, v 1,
                                                pages 15, 16 of 17
test_blackwell_cutlass_fmha: qo_len 1,          ``upstream_fmha_b{1,2,3,9,17}_kv
non-causal, head dim 128, bf16 (8/32 KV         {1,17,544,977,1999}`` with the
heads there; 4 per definition)                  definition's 32/4 heads
==============================================  ==================================

Not served (outside the definition): RoPE, soft cap, page sizes 8/16, head
dims 64/192/256/512, FP8/NVFP4 KV, multi-token queries (``qo_len > 1``,
causal masks, ``test_blackwell_cutlass_qo_kv_varlen``). Beyond upstream the
cases add empty and length-1 requests, skewed lengths, random and shared page
tables, peaked attention, all 16 batch-1 inventory rows and Qwen3-30B-A3B
long-context (B 1-4 at 32k-128k) and stress batches (B 128-256 at 8k-32k,
up to 4.1M KV tokens, the FMHA's int32 layout limit). sm_86 serves the
cases within 6 GB.

Nothing in this module runs at benchmark time; the workloads in
``harness/workloads/gqa_decode.py`` load the cubins listed in ``kernels.json``.
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
from typing import Any, NamedTuple

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
KERNELS = PACKAGE / "kernels"
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"

FLASHINFER_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer"
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
FLASHINFER_DIR = f"flashinfer-{FLASHINFER_REVISION}"
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
CUTLASS_REVISION = "b46b16d003484063bca4ed365e44095c4c6ed633"
CUTLASS_DIR = f"cutlass-{CUTLASS_REVISION}"

# Fake device addresses used by the probes: 0x100000000000 * (index + 1).
SENTINEL_STRIDE = 0x100000000000


class KernelSpec(NamedTuple):
    arch: str
    workload: str
    source: str  # under kernels/
    symbol: str
    probe: str | None  # host-only parameter-layout probe under kernels/
    adaptation: str


SPECS: tuple[KernelSpec, ...] = (
    KernelSpec(
        "sm_86",
        "gqa_decode_batch_decode",
        "sm86_batch_decode.cu",
        "_ZN10flashinfer33BatchDecodeWithPagedKVCacheKernelILNS_15PosEncodingModeE0ELj2"
        "ELj1ELj8ELj16ELj8ELj1ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_17BatchDecode"
        "ParamsI13__nv_bfloat16S5_S5_iEEEEvT7_",
        "probe_sm86_decode_params.cu",
        "Unmodified upstream FlashInfer paged decode; explicit instantiation of the"
        " previously launched non-partitioned configuration (h32/kv4/d128/ps1).",
    ),
    KernelSpec(
        "sm_100a",
        "gqa_decode_plan",
        "sm100a_plan.cu",
        "_Z8gqa_planPKiPiS1_S1_S1_S1_S1_ii",
        None,
        "Project-local serial planner: varlen offsets (empty request -> one dummy row)"
        " and a balanced split of batch*32 (head, request) work items over the SMs.",
    ),
    KernelSpec(
        "sm_100a",
        "gqa_decode_gather",
        "sm100a_gather.cu",
        "_Z10gqa_gatherPKN7cutlass10bfloat16_tES2_PKiS4_S4_PS0_S5_",
        None,
        "Project-local gather: page-size-1 K/V -> ragged contiguous [rows, 4, 128].",
    ),
    KernelSpec(
        "sm_100a",
        "gqa_decode_fmha",
        "sm100a_fmha.cu",
        "_ZN7cutlass13device_kernelINS_4fmha6kernel36Sm100FmhaFwdKernelTmaWarpspecialized"
        "IN4cute5tupleIJNS1_10collective14VariableLengthES7_iNS5_IJNS5_IJiiEEEiEEEEEENS6_"
        "38Sm100FmhaFwdMainloopTmaWarpspecializedINS_10bfloat16_tEffNS5_IJNS4_1CILi256EE"
        "ENSD_ILi128EEESF_EEESG_NS5_IJiNSD_ILi1EEES8_EEENS5_IJiSH_NS5_IJNSD_ILi0EEEiEEEE"
        "EENS5_IJSH_iSK_EEENS6_12ResidualMaskENS5_IJNSD_ILi2EEESH_SH_EEEEENS6_38Sm100Fmha"
        "FwdEpilogueTmaWarpspecializedISC_fNS5_IJSF_SF_SF_EEEEENS2_28HostPrecomputedTile"
        "SchedulerENS2_41Sm100FmhaCtxKernelWarpspecializedScheduleEEEEEvNT_6ParamsE",
        "probe_sm100a_fmha_params.cu",
        "Unmodified upstream CUTLASS SM100 FMHA forward (via FlashInfer FwdRunner):"
        " 256x128x128 tiles, ResidualMask, varlen Q/KV, host-precomputed scheduler"
        " over all 32 query heads.",
    ),
    KernelSpec(
        "sm_100a",
        "gqa_decode_finish",
        "sm100a_finish.cu",
        "_Z10gqa_finishPKN7cutlass10bfloat16_tEPKiPS0_Pf",
        None,
        "Project-local fixup: copy FMHA output; zero output and -inf LSE for empty"
        " requests.",
    ),
)

DEAD_KERNELS: dict[str, list[dict[str, str]]] = {
    "sm_86": [
        {
            "kernel": "flashinfer::MergeStates / VariableLengthMergeStates",
            "reason": "split-KV reduction, launched only with partition_kv = true;"
            " the launcher always ran the non-partitioned decode.",
        }
    ],
    "sm_100a": [
        {
            "kernel": "BatchDecodeWithPagedKVCacheKernel built for sm_100a"
            " (CUBIN/*__sm_100a__004fe83ca63c9b97.cubin)",
            "reason": "stale export of the sm_86 template; the sm_100a package entry"
            " always used the CUTLASS FMHA path.",
        }
    ],
}

# Probe argument sets. Values are distinct so sizes can be symbolized.
DECODE_PROBE_RUNS: tuple[dict[str, Any], ...] = (
    {"batch": 7, "sm_scale": 0.0883883476},
    {"batch": 300, "sm_scale": 0.5},
)
# Both driver versions (CUTLASS clears bit 21 of descriptor word 1 for tensors
# under 128 KiB on drivers <= 13.1) and both small and large tensors.
FMHA_PROBE_RUNS: tuple[dict[str, Any], ...] = (
    {"batch": 3, "total_kv": 40, "sm_count": 148, "sm_scale": 0.0883883476},
    {"batch": 5, "total_kv": 1000, "sm_count": 132, "sm_scale": 0.5},
    {"batch": 64, "total_kv": 61583, "sm_count": 148, "sm_scale": 0.125},
    {"batch": 7, "total_kv": 9, "sm_count": 82, "sm_scale": 1.0},
    {"batch": 128, "total_kv": 262272, "sm_count": 148, "sm_scale": 0.0883883476},
)
FMHA_PROBE_DRIVERS = (13010, 13020)
DECODE_POINTER_ROLES = (
    "q",
    "k_cache",
    "v_cache",
    "kv_indices",
    "kv_indptr",
    "last_page_len",
    "o",
    "lse",
    "request_indices",
    "kv_tile_indices",
    "kv_chunk_size",
)
FMHA_POINTER_ROLES = (
    "q",
    "packed_k",
    "packed_v",
    "qo_offsets",
    "kv_offsets",
    "work_indptr",
    "qo_tile_indices",
    "qo_head_indices",
    "batch_indices",
    "o",
    "lse",
)
FMHA_SIZE_VARS = ("batch", "total_kv")


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
    """Build the live GQA decode kernels of one architecture."""

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
            raise ValueError(f"gqa_decode supports {self.supported_arches}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = tuple(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        # Not resolved: resources/ entries may be symlinks, and line tables should
        # record the repo-relative path rather than the link target.
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.out_dir = PACKAGE / "cubins" / arch

    # -- paths ---------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        """Path relative to the repo root when possible (keeps cubins portable)."""
        path = Path(os.path.abspath(path))
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def _includes(self) -> list[str]:
        flashinfer = self.resources / FLASHINFER_DIR
        cutlass = self.resources / CUTLASS_DIR
        trees = [flashinfer] + ([cutlass] if self.arch == "sm_100a" else [])
        for tree in trees:
            if not tree.is_dir():
                raise FileNotFoundError(
                    f"{tree} is missing (scripts/fetch_resources.py)"
                )
        dirs = [flashinfer / "include"]
        if self.arch == "sm_100a":
            dirs += [cutlass / "include", cutlass / "tools/util/include"]
        return ["-I" + self._rel(d) for d in dirs]

    def _gencode(self) -> str:
        number = self.arch.removeprefix("sm_")
        return f"-gencode=arch=compute_{number},code=sm_{number}"

    def _version(self) -> str:
        return _run([self.nvcc, "--version"], ROOT).strip()

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
        # Host-only executable: no kernel is instantiated; NDEBUG keeps CUTLASS's
        # host asserts quiet. The FMHA probe interposes cuTensorMapEncodeTiled
        # (direct driver call) and, through the shared cudart,
        # cudaDriverGetVersion.
        extra = (
            ["--cudart=shared", "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL"]
            if self.arch == "sm_100a"
            else []
        )
        return [
            self.nvcc,
            "-std=c++17",
            "-O1",
            "-DNDEBUG",
            self._gencode(),
            "--expt-relaxed-constexpr",
            *extra,
            *self._includes(),
            self._rel(source),
            "-o",
            str(output),
        ]

    def _recorded(self, command: list[str], output: Path) -> list[str]:
        """The build command without machine-specific paths."""
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
        """sha256 of every upstream header the translation unit included."""
        text = depfile.read_text().replace("\\\n", " ")
        files: dict[str, str] = {}
        for token in text.split(":", 1)[1].split():
            path = Path(os.path.abspath(ROOT / token))
            for label, tree in (
                ("flashinfer", self.resources / FLASHINFER_DIR),
                ("cutlass", self.resources / CUTLASS_DIR),
            ):
                if path.is_relative_to(tree):
                    files[f"{label}/{path.relative_to(tree)}"] = _sha256(path)
        return dict(sorted(files.items()))

    def _check_cubin(self, spec: KernelSpec, image: bytes) -> list[dict[str, int]]:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from harness.workload import cubin_info

        arch, names, params = cubin_info(image)
        if arch != self.arch:
            raise RuntimeError(
                f"{spec.workload}: cubin declares {arch}, not {self.arch}"
            )
        if names != [spec.symbol]:
            raise RuntimeError(
                f"{spec.workload}: expected [{spec.symbol}], found {names}"
            )
        return [{"offset": p.offset, "size": p.size} for p in params]

    def compile(self) -> dict[str, str]:
        specs = [s for s in SPECS if s.arch == self.arch]
        version = self._version()
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)
        mapping: dict[str, str] = {}
        upstream: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="gqa_decode-") as tmp:
            tmpdir = Path(tmp)
            for spec in specs:
                source = KERNELS / spec.source
                cubin = self.out_dir / f"{spec.workload}.cubin"
                depfile = tmpdir / f"{spec.workload}.d"
                command = self._cubin_command(source, cubin, depfile)
                _run(command, ROOT)
                image = cubin.read_bytes()
                params = self._check_cubin(spec, image)
                files = self._dependencies(depfile)
                upstream.update(files)
                sidecar: dict[str, Any] = {
                    "workload": spec.workload,
                    "arch": self.arch,
                    "kernel": spec.symbol,
                    "kernel_params": params,
                    "cubin_sha256": hashlib.sha256(image).hexdigest(),
                    "source": self._rel(source),
                    "source_sha256": _sha256(source),
                    "build": {
                        "command": self._recorded(command, cubin),
                        "nvcc_version": version,
                    },
                    "adaptation": spec.adaptation,
                    "upstream_files": files,
                }
                if spec.probe:
                    layout = self._probe(spec, tmpdir)
                    if layout["param_size"] != sum(p["size"] for p in params):
                        raise RuntimeError(
                            f"{spec.workload}: probe sizeof(Params) "
                            f"{layout['param_size']} != cubin {params}"
                        )
                    exe = tmpdir / spec.probe.removesuffix(".cu")
                    probe_command = self._recorded(
                        self._probe_command(KERNELS / spec.probe, exe), exe
                    )
                    sidecar["probe"] = {
                        "source": self._rel(KERNELS / spec.probe),
                        "source_sha256": _sha256(KERNELS / spec.probe),
                        "command": [*probe_command[:-1], exe.name],
                    }
                    sidecar["param_layout"] = layout
                _write_json(self.out_dir / f"{spec.workload}.json", sidecar)
                mapping[spec.workload] = str(cubin.relative_to(PACKAGE))
        self._merge_manifest(mapping)
        self._merge_provenance(specs, upstream)
        return mapping

    # -- parameter-layout probes ----------------------------------------------

    def _probe(self, spec: KernelSpec, tmpdir: Path) -> dict[str, Any]:
        assert spec.probe is not None
        exe = tmpdir / spec.probe.removesuffix(".cu")
        _run(self._probe_command(KERNELS / spec.probe, exe), ROOT)
        if spec.workload == "gqa_decode_batch_decode":
            runs = [
                json.loads(_run([str(exe), str(r["batch"]), repr(r["sm_scale"])], ROOT))
                for r in DECODE_PROBE_RUNS
            ]
            layout = self._common_layout(runs)
            layout["pointer_roles"] = list(DECODE_POINTER_ROLES)
            layout["examples"] = [run["example"] for run in runs]
            return layout
        runs = []
        for r in FMHA_PROBE_RUNS:
            for driver in FMHA_PROBE_DRIVERS:
                args = [r["batch"], r["total_kv"], r["sm_count"]]
                command = [str(exe), *map(str, args), repr(r["sm_scale"]), str(driver)]
                runs.append(json.loads(_run(command, ROOT)))
        return self._fmha_layout(runs)

    @staticmethod
    def _common_layout(runs: list[dict[str, Any]]) -> dict[str, Any]:
        first = runs[0]
        for run in runs[1:]:
            for key in ("param_size", "fields", "constants"):
                if run[key] != first[key]:
                    raise RuntimeError(f"probe {key} differs between runs")
        return {
            "param_size": first["param_size"],
            "constants": first["constants"],
            "fields": first["fields"],
            "sentinel_stride": SENTINEL_STRIDE,
        }

    def _fmha_layout(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        layout = self._common_layout(runs)
        layout["pointer_roles"] = list(FMHA_POINTER_ROLES)
        examples = [run["example"] for run in runs]
        tensor_maps: dict[str, Any] = {}
        for name, field in layout["fields"].items():
            if field["kind"] != "tma":
                continue
            calls = [ex["tma_calls"][ex["tma_fields"][name]] for ex in examples]
            tensor_maps[name] = self._symbolize(name, calls, examples)
        if set(tensor_maps) != set(examples[0]["tma_fields"]):
            raise RuntimeError("TMA fields and probe descriptors disagree")
        layout["tensor_maps"] = tensor_maps
        for example in examples:
            example.pop("tma_calls")  # descriptors are fake packings in "bytes"
            example.pop("tma_fields")
        layout["examples"] = examples
        return layout

    @staticmethod
    def _symbolize(
        name: str, calls: list[dict[str, Any]], examples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Express one encode call's arguments in terms of the probe variables."""
        result: dict[str, Any] = {}
        for key, values in calls[0].items():
            column = [call[key] for call in calls]
            if key == "address":
                roles = [
                    (a + SENTINEL_STRIDE // 2) // SENTINEL_STRIDE - 1 for a in column
                ]
                offsets = [a - SENTINEL_STRIDE * (r + 1) for a, r in zip(column, roles)]
                if len(set(roles)) != 1 or len(set(offsets)) != 1:
                    raise RuntimeError(f"{name}: address is not a fixed pointer offset")
                result[key] = {
                    "pointer": FMHA_POINTER_ROLES[roles[0]],
                    "offset": offsets[0],
                }
            elif isinstance(values, list):
                result[key] = [
                    ImplCompiler._symbolize_value(
                        f"{name}.{key}[{i}]", [c[key][i] for c in calls], examples
                    )
                    for i in range(len(values))
                ]
            else:
                if len(set(column)) != 1:
                    raise RuntimeError(f"{name}.{key} varies between probe runs")
                result[key] = values
        return result

    @staticmethod
    def _symbolize_value(
        what: str, column: list[int], examples: list[dict[str, Any]]
    ) -> int | dict[str, Any]:
        # A value equal in every run is a constant; otherwise it must be affine,
        # scale * var + offset, in exactly one probe variable.
        if len(set(column)) == 1:
            return column[0]
        for var in FMHA_SIZE_VARS:
            xs = [ex[var] for ex in examples]
            pairs = [(x, y) for x, y in zip(xs, column) if x != xs[0]]
            if not pairs:
                continue
            x1, y1 = pairs[0]
            scale, rem = divmod(y1 - column[0], x1 - xs[0])
            if rem:
                continue
            offset = column[0] - scale * xs[0]
            if all(y == scale * x + offset for x, y in zip(xs, column)):
                return {"var": var, "scale": scale, "offset": offset}
        raise RuntimeError(f"{what}: {column} is not affine in {FMHA_SIZE_VARS}")

    # -- manifests -------------------------------------------------------------

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        _write_json(MANIFEST, dict(sorted(manifest.items())))

    def _merge_provenance(self, specs: list[KernelSpec], files: dict[str, str]) -> None:
        entry: dict[str, Any] = {
            "arch": self.arch,
            "repository": FLASHINFER_REPOSITORY,
            "revision": FLASHINFER_REVISION,
        }
        if any(f.startswith("cutlass/") for f in files):
            entry["cutlass_repository"] = CUTLASS_REPOSITORY
            entry["cutlass_revision"] = CUTLASS_REVISION
        entry["kernels"] = {
            s.workload: {
                "symbol": s.symbol,
                "source": f"kernels/{s.source}",
                "status": "live",
                "adaptation": s.adaptation,
            }
            for s in specs
        }
        entry["dead_kernels"] = DEAD_KERNELS[self.arch]
        entry["files"] = dict(sorted(files.items()))
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

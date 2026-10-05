"""Compile the DSA sparse attention pipeline into single-kernel cubins.

Workload: DeepSeek-V3.2 ``dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64``
(16 query heads, 512 latent + 64 RoPE dims, 2048 sparse token IDs per query,
64-token pages, see ``resources/dsa_attention.json``). Each architecture runs a
three-kernel pipeline; every kernel becomes its own cubin and its own workload:

sm_86 (FlashInfer MLA decode, sparse token IDs as size-1 pages)::

    pack_sparse      <<<1, 1>>>               token IDs -> CSR pages metadata
    BatchDecodeWithPagedKVCacheKernelMLA      upstream FlashInfer decode
                     <<<batch, (32, 8)>>>
    fix_empty        <<<batch, 256>>>         zero output / -inf LSE of empty rows

sm_100a (CUTLASS SM100 MLA, 16 heads padded to the 128-head tile)::

    dsa_pack         <<<batch, 256>>>         pad heads, compact IDs into a table
    device_kernel<Sm100FmhaMlaKernelTmaWarpspecialized<...>>
                     <<<min(2*batch, sms), 256, cluster (2,1,1)>>>
    dsa_unpack       <<<batch, 256>>>         compact heads, base-2 LSE, empty rows

Kernel inventory of the previous multi-kernel package (``implementation.json``
and its exported ``CUBIN/*.cubin``) and the decision taken here. Only *live*
kernels get a cubin and a workload:

=========================================  =======  ===========================================
kernel (previous cubin)                    status   reason
=========================================  =======  ===========================================
pack_sparse (sm_86)                        live     Launched on every sm_86 run; produces the
                                                    CSR indptr/indices/last_page_len and the
                                                    request/tile metadata read by the decode.
BatchDecodeWithPagedKVCacheKernelMLA       live     The sm_86 attention kernel (one launch,
<2,16,2,32,8,1,2,...> (sm_86)                       ``partition_kv = false``).
fix_empty (sm_86)                          live     Launched on every sm_86 run; the decode
                                                    leaves zero-length rows undefined.
dsa_pack (sm_100a)                         live     Launched on every sm_100a run; the MLA tile
                                                    needs 128 heads and a dense page table.
device_kernel<Sm100FmhaMlaKernelTma...>    live     The sm_100a Tensor Core (tcgen05) kernel.
(sm_100a)
dsa_unpack (sm_100a)                       live     Launched on every sm_100a run.
device_kernel<Sm100FmhaMlaReductionKernel  dead     ``cutlass::fmha::device::MLA::run`` launches
<bf16, float, float, 128, 512, 256>>                it only when ``split_kv > 1``; the pipeline
(sm_100a)                                           always sets ``split_kv = 1`` (one CTA pair
                                                    per query covers all 2048 sparse tokens),
                                                    so it is never launched. Not instantiated.
pack_sparse / BatchDecode...MLA /          dead     ``CUBIN/*__sm_100a__8d29ec0f6f5aee0f.cubin``
fix_empty built for sm_100a                         is a stale export of the sm_86 template
                                                    compiled for sm_100a. The sm_100a package
                                                    entry always used the CUTLASS path, so this
                                                    FlashInfer path never ran on Blackwell.
sm_86 duplicates (``CUBIN/*__sm_86__*``)   dup      Four exports of the same three kernels from
                                                    different builds; covered by the live rows.
=========================================  =======  ===========================================

One kernel per cubin is achieved at compile time: every source in ``kernels/``
is a translation unit holding exactly one ``__global__`` function (the three
project-local adapters) or exactly one explicit template instantiation of an
upstream kernel (FlashInfer decode, CUTLASS MLA), compiled with
``nvcc -cubin -lineinfo`` for a single ``-gencode``. The result is a genuine
nvcc cubin (no post-hoc ELF surgery); ``compile()`` checks the architecture,
the kernel count and the exact mangled symbol of every cubin. Upstream code is
included unmodified from the pinned checkouts under ``resources/``.

Every cubin gets a sidecar ``cubins/<arch>/<workload>.json`` with its build
record. The two upstream kernels take one by-value ``Params`` struct; for them
a host-only probe (``kernels/probe_*.cu``, compiled and run here, needs no
GPU) builds that struct with the upstream host code exactly as the previous
native launcher did and records into the sidecar ``param_layout``:

* ``param_size`` and ``fields``: ``{name: {offset, size, kind}}`` of every
  member the launcher sets (kinds: ``ptr``, ``i32``, ``u32``, ``i64``, ``f32``,
  ``bool``, ``bytes``, ``fast_divmod``, ``dim3``, ``tma_atom``);
* ``constants``: launch constants (dynamic shared memory, block, cluster) and
  opaque constant bytes (FlashInfer's ``uint_fastdiv(1)``);
* ``examples``: probe runs with fake pointers (``pointer_roles`` names their
  sentinel indices) and the struct's raw bytes, so the Python builder can be
  checked byte-for-byte without a GPU;
* ``tensor_maps`` (MLA only): for each TMA atom, the ``cuTensorMapEncodeTiled``
  arguments CUTLASS's ``to_underlying_arguments`` passes, recorded by a stub
  and symbolized over several probe runs: addresses as ``{pointer, offset}``,
  sizes as constants or ``{var, scale}``.

Harness path versus upstream. sm_100a always runs the CUTLASS MLA with
``split_kv = 1`` (each 2-SM CTA pair walks all 16 KV tiles of its query)
where FlashInfer's ``cutlass_mla`` lets ``MLA::set_split_kv`` split small
batches and launches the split-KV reduction; sm_86 runs FlashInfer's FA2 MLA
decode non-partitioned on size-1 pages. Upstream tests of these kernels and
the cases covering them (``tests/test_dsa_attention.py::UpstreamCoverage``
checks against the pinned source):

==========================================  =====================================
upstream                                    harness cases
==========================================  =====================================
FlashInfer test_cutlass_mla: batch 1/2/4,   ``upstream_cutlass_mla_b{1,2,4}_seq
max_seq_len 128/1024, page_size 1, bf16     {128,1024}``: queries x 100,
                                            ``randint`` (duplicate) page table
official definition + inventory rows        ``throughput_tokens{1,2,6,7,8}_pages
(1-8 tokens, 8462 pages, sm_scale 0.13523)  8462`` with each row's sm_scale
==========================================  =====================================

Not served: page sizes 16/128 and ``max_seq_len`` 4096 of test_cutlass_mla
(the definition's sparse token IDs fix page size 1 and at most 2048 tokens);
no upstream test at the pinned revision drives the FA2 MLA decode kernel
(``BatchDecodeWithPagedKVCacheKernelMLA``) or a sparse MLA through these
kernels (the sparse MLA tests target trtllm-gen, XQA, cute-dsl and SM120
kernels). The TMA atoms of the MLA ``Params`` are encoded as upstream does
though the cp.async kernel never reads them;
``tests/test_dsa_attention.py::LaunchArguments`` checks their encode inputs
with a recording encoder on GPUs without TMA.

Nothing in this module runs at benchmark time; the workloads in
``harness/workloads/dsa_attention.py`` load the cubins listed in
``kernels.json``.
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
        "dsa_attention_pack_sparse",
        "sm86_pack_sparse.cu",
        "_Z11pack_sparsePKiPii",
        None,
        "Project-local serial packer: sparse token IDs -> FlashInfer CSR size-1 pages.",
    ),
    KernelSpec(
        "sm_86",
        "dsa_attention_decode",
        "sm86_decode_mla.cu",
        "_ZN10flashinfer36BatchDecodeWithPagedKVCacheKernelMLAILj2ELj16ELj2ELj32ELj8"
        "ELj1ELj2ENS_16DefaultAttentionILb0ELb0ELb0ELb0EEENS_20BatchDecodeParamsMLAI"
        "13__nv_bfloat16S4_S4_iEEEEvT7_",
        "probe_sm86_decode_params.cu",
        "Unmodified upstream FlashInfer MLA decode; explicit instantiation of the"
        " previously launched configuration.",
    ),
    KernelSpec(
        "sm_86",
        "dsa_attention_fix_empty",
        "sm86_fix_empty.cu",
        "_Z9fix_emptyPKiP13__nv_bfloat16Pf",
        None,
        "Project-local fixup: zero output and -inf LSE for empty sparse rows.",
    ),
    KernelSpec(
        "sm_100a",
        "dsa_attention_pack",
        "sm100a_pack.cu",
        "_Z8dsa_packPKN7cutlass10bfloat16_tES2_PKiPS0_S5_PiS6_",
        None,
        "Project-local packer: pad 16 heads to 128, compact IDs into a page table.",
    ),
    KernelSpec(
        "sm_100a",
        "dsa_attention_mla",
        "sm100a_mla.cu",
        "_ZN7cutlass13device_kernelINS_4fmha6kernel36Sm100FmhaMlaKernelTmaWarpspecialized"
        "IN4cute5tupleIJNS4_1CILi128EEES7_NS5_IJNS6_ILi512EEENS6_ILi64EEEEEEEEENS_"
        "10bfloat16_tEfSC_fNS2_31Sm100MlaPersistentTileSchedulerELb1EEEEEvNT_6ParamsE",
        "probe_sm100a_mla_params.cu",
        "Unmodified upstream CUTLASS SM100 MLA (via FlashInfer MlaSm100); split_kv=1,"
        " page size 1, separate latent/RoPE strides; reduction kernel not instantiated.",
    ),
    KernelSpec(
        "sm_100a",
        "dsa_attention_unpack",
        "sm100a_unpack.cu",
        "_Z10dsa_unpackPKN7cutlass10bfloat16_tEPKfPKiPS0_Pf",
        None,
        "Project-local unpacker: 16 live heads, natural->base-2 LSE, empty rows.",
    ),
)

# Probe argument sets. Values are distinct so sizes can be symbolized.
DECODE_PROBE_RUNS: tuple[dict[str, Any], ...] = (
    {"batch": 7, "sm_scale": 0.0721687836},
    {"batch": 300, "sm_scale": 0.5},
)
MLA_PROBE_RUNS: tuple[dict[str, Any], ...] = (
    {"batch": 3, "page_count": 541696, "sm_count": 148, "device_id": 0},
    {"batch": 5, "page_count": 1000, "sm_count": 132, "device_id": 1},
    {"batch": 512, "page_count": 64, "sm_count": 148, "device_id": 0},
)
MLA_PROBE_SCALES = (0.0721687836, 0.5, 0.125)
DECODE_POINTER_ROLES = (
    "q_nope",
    "q_pe",
    "ckv",
    "kpe",
    "o",
    "lse",
    "indices",
    "indptr",
    "last_page_len",
    "request_indices",
    "kv_tile_indices",
    "kv_chunk_size",
)
MLA_POINTER_ROLES = (
    "q_latent",
    "q_rope",
    "ckv",
    "kpe",
    "lengths",
    "page_table",
    "out",
    "lse",
)
MLA_SIZE_VARS = ("batch", "page_count")
TMA_MARKER = 0x7E5A000000000000


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
    """Build the live DSA attention kernels of one architecture."""

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
            raise ValueError(f"dsa_attention supports {self.supported_arches}")
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
        for tree in (flashinfer, cutlass):
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
        # host asserts from requiring a CUDA driver when the probe runs.
        return [
            self.nvcc,
            "-std=c++17",
            "-O1",
            "-DNDEBUG",
            self._gencode(),
            "--expt-relaxed-constexpr",
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
        with tempfile.TemporaryDirectory(prefix="dsa_attention-") as tmp:
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
                    sidecar["probe"] = {
                        "source": self._rel(KERNELS / spec.probe),
                        "source_sha256": _sha256(KERNELS / spec.probe),
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
        if spec.workload == "dsa_attention_decode":
            runs = [
                json.loads(_run([str(exe), str(r["batch"]), repr(r["sm_scale"])], ROOT))
                for r in DECODE_PROBE_RUNS
            ]
            return self._decode_layout(runs)
        runs = [
            json.loads(
                _run(
                    [
                        str(exe),
                        str(r["batch"]),
                        str(r["page_count"]),
                        str(r["sm_count"]),
                        str(r["device_id"]),
                        repr(scale),
                    ],
                    ROOT,
                )
            )
            for r, scale in zip(MLA_PROBE_RUNS, MLA_PROBE_SCALES)
        ]
        return self._mla_layout(runs)

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

    def _decode_layout(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        layout = self._common_layout(runs)
        layout["pointer_roles"] = list(DECODE_POINTER_ROLES)
        layout["examples"] = [run["example"] for run in runs]
        return layout

    def _mla_layout(self, runs: list[dict[str, Any]]) -> dict[str, Any]:
        layout = self._common_layout(runs)
        layout["pointer_roles"] = list(MLA_POINTER_ROLES)
        examples = [run["example"] for run in runs]
        tensor_maps: dict[str, Any] = {}
        for name, field in layout["fields"].items():
            if field["kind"] != "tma_atom":
                continue
            calls = []
            for example in examples:
                data = bytes.fromhex(example["bytes"])
                atom = data[field["offset"] : field["offset"] + field["size"]]
                # The 128-byte CUtensorMap is the first member of the atom; the
                # rest (auxiliary gmem strides) must be static (all zero).
                marker = int.from_bytes(atom[:8], "little")
                if marker & ~0xFFFF != TMA_MARKER or any(atom[8:]):
                    raise RuntimeError(f"{name}: unexpected TMA atom bytes")
                calls.append(example["tma_calls"][marker & 0xFFFF])
            tensor_maps[name] = {
                "descriptor_offset": 0,
                "encode": self._symbolize(name, calls, examples),
            }
        layout["tensor_maps"] = tensor_maps
        for example in examples:
            example.pop("tma_calls")  # descriptors are marker bytes in "bytes"
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
                roles = [a // SENTINEL_STRIDE - 1 for a in column]
                offsets = [a % SENTINEL_STRIDE for a in column]
                if len(set(roles)) != 1 or len(set(offsets)) != 1:
                    raise RuntimeError(f"{name}: address is not a fixed pointer offset")
                result[key] = {
                    "pointer": MLA_POINTER_ROLES[roles[0]],
                    "offset": offsets[0],
                }
            elif isinstance(values, list):
                entries = []
                for i in range(len(values)):
                    entries.append(
                        ImplCompiler._symbolize_value(
                            f"{name}.{key}[{i}]", [c[i] for c in column], examples
                        )
                    )
                result[key] = entries
            else:
                if len(set(column)) != 1:
                    raise RuntimeError(f"{name}.{key} varies between probe runs")
                result[key] = values
        return result

    @staticmethod
    def _symbolize_value(
        what: str, column: list[int], examples: list[dict[str, Any]]
    ) -> int | dict[str, Any]:
        # The probe variables differ between runs, so a value that is equal in
        # every run is a constant and one that differs must scale a variable.
        if len(set(column)) == 1:
            return column[0]
        for var in MLA_SIZE_VARS:
            base = examples[0][var]
            if column[0] % base:
                continue
            scale = column[0] // base
            if all(v == ex[var] * scale for v, ex in zip(column, examples)):
                return {"var": var, "scale": scale}
        raise RuntimeError(f"{what}: {column} is not a multiple of {MLA_SIZE_VARS}")

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
                "adaptation": s.adaptation,
            }
            for s in specs
        }
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

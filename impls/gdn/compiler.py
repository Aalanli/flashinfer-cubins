"""Compile the GDN (Gated DeltaNet) kernels into single-kernel cubins.

One package serves both official Qwen3-Next definitions (TP = 4 shard of the
linear-attention layers: 16 key / 32 value heads become Hq = Hk = 4, Hv = 8,
head size 128, BF16 q/k/v/output, FP32 k-last state ``[N, Hv, V, K]``):

* ``gdn_decode_qk4_v8_d128_k_last`` (``resources/gdn_decode.json``): one token
  per sequence, ``[B, 1, H, 128]`` tensors;
* ``gdn_prefill_qk4_v8_d128_k_last`` (``resources/gdn_prefill.json``): ragged
  sequences ``[T, H, 128]`` with int64 ``cu_seqlens``.

Both run the same two-kernel pipeline, so each kernel is one cubin and one
workload (``harness/workloads/gdn.py``):

``gdn_gates`` (sm_86, sm_100a)
    ``prepare_gates<<<ceil(max(T*8, N+1)/256), 256>>>``: project-local glue
    (``kernels/gdn_gates.cu``) computing FlashInfer's FP32 ``g``/``beta``
    ``[T, 8]`` and int32 ``cu_seqlens`` from the definitions' raw inputs
    (int64 ``cu_seqlens`` for prefill, ``0..B`` when ``cu`` is null for decode).
``gdn_chunk`` (sm_100a)
    ``kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial<<<min(SMs, 16N),
    384, 226048>>>``: FlashInfer's generated ("Cake GDN") Blackwell tcgen05
    chunked delta-rule prefill, DV-split schedule, FP32 initial + final state;
    decode is B one-token sequences.

Merged duplicates. The former ``gdn_decode`` and ``gdn_prefill`` packages each
shipped both kernels: ``gdn_decode_dvsplit`` and ``gdn_prefill_chunk`` were
built from the same upstream file with the same substitution (the generated
``.cu`` files differed only in comments; SASS byte-identical), and
``gdn_decode_gates`` / ``gdn_prefill_gates`` were the same ``prepare_gates``
(same symbol and parameters; the decode copy additionally handles
``cu == nullptr``). They are merged here: ``gdn_chunk`` and ``gdn_gates``
serve the union of both packages' cases (``provenance.json`` records the
removed names).

How the upstream kernel is produced. FlashInfer ships *generated CUDA C++*
(``csrc/gdn/cake/cuda/*.cu``, one ``extern "C" __global__`` per file, produced
upstream by the Cake schedule generator, which is not part of the repository)
plus a TVM-FFI host shim per variant (``csrc/gdn/cake/host/*.cc``);
``flashinfer/jit/cake_gdn.py`` compiles the ``.cu`` with ``nvcc --cubin
--std=c++17 -O3 --use_fast_math`` (the manifest's ``compile_options``) and the
include roots ``csrc/gdn/cake/cuda`` and ``csrc/gdn``. This compiler
reproduces that build from the pinned checkout (manifest and every file
hash-checked) with one specialization: variant ``373d87a0efe5`` (GVA head
ratio 2, ``HEAD_GROUP_LOG2 = 1``, ``IS_GQA = 0``, FP32 initial/final state, no
checkpoints or state indices) is generated for 4 output heads
(``NUM_O_HEADS_LOG2 2``); the definitions have Hv = 8, so
``kernels/gdn_chunk.cu`` is the upstream file with ``NUM_O_HEADS_LOG2 3``. The
generated code uses the macro only to split the persistent tile index into
``(sequence, output head)`` (``tile >> NUM_O_HEADS_LOG2``, ``tile & (Hv -
1)``), so 3 is the exact specialization for 8 output heads with the same GVA
ratio. Every other byte is unchanged and line numbers match upstream (the
provenance note is appended at the end); ``compile()`` regenerates the file.

Kernel inventory and decisions (only *live* kernels get a cubin):

==============================================  ======  =========================================
kernel                                          status  reason
==============================================  ======  =========================================
prepare_gates (project glue)                    live    The definitions supply raw A_log/a/dt_bias
                                                        /b (and int64 cu_seqlens); the upstream
                                                        kernel needs FP32 g/beta [T, 8] and int32
                                                        offsets. Plain CUDA, also built for sm_86.
kernel_flashinfer_blackwell_gdn_prefill_        live    The Tensor Core (tcgen05/UMMA, TMA) kernel
dvsplit_initial                                         of both definitions.
gdn_decode_dvsplit / gdn_prefill_chunk          dup     Former per-package copies of the chunk
                                                        kernel (identical SASS); merged.
gdn_decode_gates / gdn_prefill_gates            dup     Former per-package copies of
                                                        prepare_gates; merged (decode's superset).
kernel_gdn_decode_pretranspose_splitv8          dead    Stale export of an older decode package;
                                                        upstream has no Hq=4/Hv=8 FP32 variant
                                                        (the Cake router fails closed) and it
                                                        L2-normalizes Q/K, unlike the definition.
cudaMemcpyAsync state copy (splitv8 adapter)    dead    In-place workaround of that adapter; the
                                                        chunk kernel writes a separate state.
legacy gdn_prefill native_module cubins         dup     Exports of the same two kernels.
TVM-FFI host shim ``Run``                       host    Argument checks, descriptor encoding and
                                                        the launch move to Python; its
                                                        ``EncodeTma_*`` and ``kargs`` construction
                                                        run in the host-only probe below.
==============================================  ======  =========================================

Scope: upstream routes the DV-split schedule only when ``2 * num_seqs * Hv <=
148`` (``num_seqs <= 9`` for Hv = 8) and otherwise picks a full-DV variant;
this package uses the DV-split kernel for every shape, as both former packages
did (its tile loop is persistent and admits any ``total_tiles``), with
upstream's grid ``min(SMs, total_tiles)``. Empty sequences are supported by the
kernel (``num_chunks == 0`` copies the initial state to the output state).

Launch parameters. The kernel takes four ``__grid_constant__ CUtensorMap``
descriptors and plain pointers/scalars. A host-only probe
(``kernels/probe_chunk_launch.cu``) compiles the upstream shim's
``EncodeTma_{Q,K,V,O}`` and the ``kargs`` construction of its ``Run``
verbatim (extracted at compile time) against a recording
``cuTensorMapEncodeTiled`` and fake pointers, for the arguments
``flashinfer/gdn_prefill.py`` passes (``total_tiles = N * Hv * 2``, grid
``min(sm_count, total_tiles)``, workspace ``grid * 512`` bytes) for decode
(T = N) and ragged prefill shapes. The sidecar ``cubins/sm_100a/gdn_chunk.json``
records every encode argument and the packed parameter bytes per probe run;
``tests/test_gdn.py`` compares the Python launch builder byte for byte.

Upstream coverage (enforced by ``tests/test_gdn.py``; the list is
``harness.workloads.gdn.UPSTREAM_TESTS``). No upstream test launches this
exact kernel: FlashInfer only reaches the Cake path with ``backend="cake_gdn"``
and its Cake prefill tests use other variants (FP16 I/O, BF16/indexed state,
checkpoints); its generic GDN tests run the CuTe DSL kernels. Projected onto
the definitions' fixed configuration (BF16 I/O, FP32 state, Hq/Hv = 4/8, the
TP4 shard of the upstream 16/32-head tests), every upstream parametrization
maps to a harness case with the same sequence lengths (or decode batch /
MTP batch x tokens), initial-state presence, gate regime and scale:
``tests/gdn/test_prefill_delta_rule.py`` (basic, nonfull, chunked prefill,
zero-length sequence, block-end decay), ``tests/gdn/test_cake_gdn_prefill_gpu.py``
(the Hq=4/Hv=8 length sets b7_t421 and b5_t4296, single 128/64/1 token
sequences), ``tests/gdn/test_decode_delta_rule.py`` (FP32-state T = 1 batches
1/4/16/32/512 and FP32 MTP batches 1/4/8/16/64 x 2/4/8 tokens) and
``tests/gdn/test_cake_gdn_decode_gpu.py`` (FP32 T = 1 and MTP rows).
Checkpoints, state indices and low-precision states are other kernels.

One kernel per cubin: each source under ``kernels/`` holds exactly one
``__global__`` function and no host launch code, compiled with ``nvcc -cubin
-lineinfo`` for one ``-gencode``; ``compile()`` checks arch, kernel count,
exact symbol and parameter sizes. Nothing here runs at benchmark time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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
CAKE = "csrc/gdn/cake"
# flashinfer/jit/cake_gdn.py pins the manifest; the manifest pins every file.
CAKE_MANIFEST_SHA256 = (
    "836fa2ab739f31c9c037f2277b079a4593947363408caa350972be88d1d2517a"
)
VARIANT = "prefill_flashinfer_blackwell_gdn_prefill_dvsplit_initial_373d87a0efe5"
UPSTREAM_KERNEL = f"{CAKE}/cuda/cake_gdn_prefill_dvsplit_initial_373d87a0efe5.cu"
UPSTREAM_HOST = f"{CAKE}/host/cake_gdn_prefill_dvsplit_initial_373d87a0efe5.cc"
UPSTREAM_SHA256 = {
    UPSTREAM_KERNEL: "e1e74e35781bb6b43d1fa34bb2aa26072716d0b0534b45cc250e9b56ea35b4bc",
    UPSTREAM_HOST: "270eb89984edb7bc4ca3320117088df6741a6f695a219414f22c878d1d9746c1",
}
CHUNK_SYMBOL = "kernel_flashinfer_blackwell_gdn_prefill_dvsplit_initial"
GATES_SYMBOL = "_Z13prepare_gatesPKfPK13__nv_bfloat16S0_S3_PKlPfS6_Piii"
GENERATED = "gdn_chunk.cu"
SPECIALIZE_FROM = "#define NUM_O_HEADS_LOG2 2\n"
SPECIALIZE_TO = "#define NUM_O_HEADS_LOG2 3\n"
PROBE = "probe_chunk_launch.cu"
PROBE_INCLUDE = "chunk_upstream.inc"
TMA_MARKER = 0x7E5A000000000000
SENTINEL_STRIDE = 0x100000000000
DEFAULT_SCALE = 0.08838834764831845  # 1/sqrt(128)
# (tokens, seqs, scale, sm_count): decode rows have tokens == seqs; B200 has
# 148 SMs; others cover grid < / = total_tiles and another SM count.
PROBE_RUNS: tuple[tuple[int, int, float, int], ...] = (
    (1, 1, DEFAULT_SCALE, 148),
    (3, 3, 0.5, 148),
    (16, 16, DEFAULT_SCALE, 148),
    (64, 64, 1.0, 148),
    (7, 7, 0.25, 132),
    (25, 3, DEFAULT_SCALE, 148),
    (421, 7, 1.0, 148),
    (2107, 1, 0.5, 148),
    (8192, 57, DEFAULT_SCALE, 148),
    (4296, 5, DEFAULT_SCALE, 132),
)

# Kernel signature, in order: (name, size in bytes).
CHUNK_PARAMS: tuple[tuple[str, int], ...] = (
    ("Q", 128),
    ("K", 128),
    ("V", 128),
    ("O", 128),
    ("gate", 8),
    ("beta", 8),
    ("cu_seqlens", 8),
    ("state_indices", 8),
    ("initial_state", 8),
    ("output_state", 8),
    ("checkpoint_state", 8),
    ("cu_checkpoints", 8),
    ("tensormap_workspace", 8),
    ("initial_state_stride_slot", 8),
    ("output_state_stride_slot", 8),
    ("checkpoint_every_n_tokens", 4),
    ("scale", 4),
    ("num_seqs", 4),
    ("num_q_heads", 4),
    ("num_v_heads", 4),
    ("total_tiles", 4),
)
GATES_PARAMS: tuple[tuple[str, int], ...] = (
    ("A_log", 8),
    ("a", 8),
    ("dt_bias", 8),
    ("b", 8),
    ("cu_seqlens", 8),
    ("gate", 8),
    ("beta", 8),
    ("offsets", 8),
    ("tokens", 4),
    ("seqs", 4),
)


class KernelSpec(NamedTuple):
    # A NamedTuple, not a dataclass: compile_kernels.py loads this file without
    # registering it in sys.modules, which dataclasses require.
    workload: str
    arches: tuple[str, ...]
    source: str  # under kernels/
    symbol: str
    params: tuple[tuple[str, int], ...]
    upstream: bool
    adaptation: str


SPECS: tuple[KernelSpec, ...] = (
    KernelSpec(
        "gdn_gates",
        ("sm_86", "sm_100a"),
        "gdn_gates.cu",
        GATES_SYMBOL,
        GATES_PARAMS,
        False,
        "Project-local glue: FP32 g = exp(-exp(A_log) * softplus(a + dt_bias)), "
        "beta = sigmoid(b) [T, 8] and int32 cu_seqlens (0..B when cu is null).",
    ),
    KernelSpec(
        "gdn_chunk",
        ("sm_100a",),
        GENERATED,
        CHUNK_SYMBOL,
        CHUNK_PARAMS,
        True,
        "Upstream generated Cake GDN DV-split prefill (variant 373d87a0efe5) with "
        "NUM_O_HEADS_LOG2 2 -> 3 for Hv = 8; HEAD_GROUP_LOG2 = 1 (Hv / Hq = 2) as "
        "upstream; --use_fast_math as the manifest's compile_options. Serves "
        "decode as B one-token sequences.",
    ),
)

MERGED_DUPLICATES = {
    "gdn_decode_dvsplit (impls/gdn_decode)": "same upstream file and substitution "
    "as gdn_prefill_chunk (generated .cu differed only in comments, SASS "
    "byte-identical); merged into gdn_chunk",
    "gdn_prefill_chunk (impls/gdn_prefill)": "merged into gdn_chunk",
    "gdn_decode_gates (impls/gdn_decode)": "prepare_gates, same symbol and "
    "parameters as gdn_prefill_gates plus the cu == nullptr decode branch; "
    "merged into gdn_gates",
    "gdn_prefill_gates (impls/gdn_prefill)": "prepare_gates without the decode "
    "branch; merged into gdn_gates",
}

DEAD_KERNELS = {
    "kernel_gdn_decode_pretranspose_splitv8": "stale export of an older decode "
    "package (CUBIN/*__16a0dcd961f48505.cubin, *__9d18db14903bcb05.cubin); not "
    "launched by the current pipeline; upstream has no Hq=4/Hv=8 FP32 variant "
    "(router fails closed) and it L2-normalizes Q/K, unlike the definition",
    "cudaMemcpyAsync(state) of the splitv8 adapter": "in-place-update workaround of "
    "the stale adapter; the chunk kernel writes a separate output state",
    "gdn_prefill__native_module_1__sm_100a__{55a870e30a0bb011,ce0083041c1b22ad,"
    "d5cb174f041f47a5}.cubin": "legacy exports of the same two live kernels "
    "(prepare_gates + the chunk kernel) from different builds of the C++ adapter",
    "TVM-FFI host shim Run (csrc/gdn/cake/host/...373d87a0efe5.cc)": "host code; "
    "replaced by the Python launch, EncodeTma_* and kargs probed at compile time",
}

UPSTREAM_COVERAGE = {
    "enforced_by": "tests/test_gdn.py::UpstreamCoverage (harness.workloads.gdn."
    "UPSTREAM_TESTS must each match a gdn_chunk case)",
    "projection": "upstream head configurations (2/2/4, 16/16/32, ...) map to the "
    "definitions' fixed Hq/Hk/Hv = 4/4/8 (Qwen3-Next TP4 shard); BF16 I/O, FP32 "
    "state; same sequence lengths, initial-state presence, gate regime, scale",
    "tests": [
        "tests/gdn/test_prefill_delta_rule.py::test_prefill_kernel_basic "
        "(zero initial state)",
        "tests/gdn/test_prefill_delta_rule.py::test_prefill_kernel_nonfull "
        "(zero initial state)",
        "tests/gdn/test_prefill_delta_rule.py::test_chunked_prefill (second call: "
        "nonzero initial state; alpha/beta on/off, scale 1 / auto)",
        "tests/gdn/test_prefill_delta_rule.py::test_prefill_kernel_zero_length_"
        "sequence / test_prefill_zero_length_sequence_state_untouched",
        "tests/gdn/test_prefill_delta_rule.py::test_prefill_block_end_decay",
        "tests/gdn/test_cake_gdn_prefill_gpu.py (b7_t421, b5_t4296, 128/64/1-token "
        "sequences; FP32-state analogues)",
        "tests/gdn/test_decode_delta_rule.py (FP32-state T=1 batches and FP32 MTP "
        "batch x tokens grid)",
        "tests/gdn/test_cake_gdn_decode_gpu.py (FP32 T=1 and MTP rows)",
    ],
    "not_served": "checkpoints, state indices/pools, BF16/FP16/FP8 state, FP16 "
    "I/O and other head configurations select other kernels",
}


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
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def _write_text_if_changed(path: Path, text: str) -> None:
    if not path.is_file() or path.read_text() != text:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)


def _macro(text: str, name: str) -> int:
    values = re.findall(rf"^#define {name} (-?\d+)$", text, flags=re.MULTILINE)
    if len(values) != 1:
        raise RuntimeError(f"expected one #define {name}, found {values}")
    return int(values[0])


def _extract_block(text: str, start: str, what: str) -> str:
    """The text from ``start`` through the next line that is exactly ``}``."""
    begin = text.find(start)
    if begin < 0 or text.find(start, begin + 1) >= 0:
        raise RuntimeError(f"expected exactly one {what} in the upstream host shim")
    end = text.find("\n}\n", begin)
    if end < 0:
        raise RuntimeError(f"unterminated {what} in the upstream host shim")
    return text[begin : end + 3]


class ImplCompiler:
    """Build the live GDN kernels of one architecture."""

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
            raise ValueError(f"gdn supports {self.supported_arches}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = tuple(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        # Not resolved: resources/ entries may be symlinks, and line tables should
        # record the repo-relative path rather than the link target.
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.flashinfer = self.resources / FLASHINFER_DIR
        self.cake = self.flashinfer / CAKE
        self.out_dir = PACKAGE / "cubins" / arch

    # -- paths ---------------------------------------------------------------

    def _rel(self, path: Path) -> str:
        path = Path(os.path.abspath(path))
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def _gencode(self) -> str:
        number = self.arch.removeprefix("sm_")
        return f"-gencode=arch=compute_{number},code=sm_{number}"

    # -- pinned upstream sources --------------------------------------------------

    def manifest_record(self) -> dict[str, Any]:
        """Verify the pinned Cake manifest, the variant's files and every CUDA
        header against it; returns the variant's manifest record."""
        manifest_path = self.cake / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{manifest_path} is missing (fetch_resources.py)")
        if _sha256(manifest_path) != CAKE_MANIFEST_SHA256:
            raise ValueError(f"{manifest_path} differs from the pinned manifest")
        manifest = json.loads(manifest_path.read_text())
        (record,) = [v for v in manifest["variants"] if v["name"] == VARIANT]
        if record["cuda_symbol"] != CHUNK_SYMBOL or record["compile_options"] != [
            "--use_fast_math"
        ]:
            raise ValueError(f"unexpected manifest record for {VARIANT}")
        (cuda,) = [o for o in record["outputs"] if "sm_100a" in o["architectures"]]
        files = [
            (self.cake / cuda["path"], cuda["sha256"]),
            (
                self.cake / record["host_binding"]["path"],
                record["host_binding"]["sha256"],
            ),
        ] + [(self.cake / h["path"], h["sha256"]) for h in manifest["cuda_headers"]]
        for path, expected in files:
            if _sha256(path) != expected:
                raise ValueError(f"{path} differs from the pinned manifest")
        for relative, expected in UPSTREAM_SHA256.items():
            if _sha256(self.flashinfer / relative) != expected:
                raise ValueError(f"{relative}: sha256 differs from the pinned value")
        return record

    def generated_source(self) -> str:
        """The Hv = 8 specialization of upstream variant 373d87a0efe5."""
        text = (self.flashinfer / UPSTREAM_KERNEL).read_text()
        if (
            _sha256(self.flashinfer / UPSTREAM_KERNEL)
            != UPSTREAM_SHA256[UPSTREAM_KERNEL]
        ):
            raise ValueError(f"{UPSTREAM_KERNEL}: sha256 differs from the pinned value")
        if text.count(SPECIALIZE_FROM) != 1:
            raise RuntimeError(f"{UPSTREAM_KERNEL}: expected one {SPECIALIZE_FROM!r}")
        text = text.replace(SPECIALIZE_FROM, SPECIALIZE_TO)
        return text + (
            "// ---------------------------------------------------------------------\n"
            "// Generated by impls/gdn/compiler.py (ImplCompiler.generate) from\n"
            f"// FlashInfer {FLASHINFER_REVISION}\n"
            f"// {UPSTREAM_KERNEL}\n"
            f"// (sha256 {UPSTREAM_SHA256[UPSTREAM_KERNEL]}).\n"
            "// Only change: NUM_O_HEADS_LOG2 2 -> 3 (8 output/value heads, GVA ratio\n"
            "// 2 unchanged); line numbers match upstream. Do not edit by hand.\n"
        )

    def generate(self) -> Path:
        """(Re)write ``kernels/gdn_chunk.cu``."""
        path = KERNELS / GENERATED
        _write_text_if_changed(path, self.generated_source())
        return path

    def probe_include(self) -> str:
        """Verbatim excerpts of the upstream host shim used by the probe."""
        text = (self.flashinfer / UPSTREAM_HOST).read_text()
        encoders = [
            _extract_block(text, f"inline CUtensorMap EncodeTma_{name}(", name)
            for name in "QKVO"
        ]
        run = re.search(r"^void Run\((.*)\) \{$", text, flags=re.MULTILINE)
        if run is None or len(re.findall(r"^void Run\(", text, re.MULTILINE)) != 1:
            raise RuntimeError("expected one Run() in the upstream host shim")
        begin = text.index("  CUtensorMap p_Q = EncodeTma_Q(arg_Q);", run.end())
        kargs = re.compile(r"^  void\* kargs\[\] = \{.*\};$", re.MULTILINE)
        end_match = kargs.search(text, begin)
        if end_match is None:
            raise RuntimeError("kargs not found in the upstream Run()")
        body = text[begin : end_match.end()]
        return (
            f"// Extracted from {UPSTREAM_HOST} (FlashInfer {FLASHINFER_REVISION}).\n"
            + "\n".join(encoders)
            + f"\nstatic void upstream_kargs({run.group(1)}) {{\n"
            + body
            + "\n  probe::pack(kargs, sizeof(kargs) / sizeof(kargs[0]));\n}\n"
        )

    # -- compile -------------------------------------------------------------

    def _includes(self, spec: KernelSpec) -> list[str]:
        if not spec.upstream:
            return []
        # The include roots flashinfer/jit/cake_gdn.py passes.
        return [
            "-I" + self._rel(self.cake / "cuda"),
            "-I" + self._rel(self.cake.parent),
        ]

    def _cubin_command(
        self, spec: KernelSpec, source: Path, output: Path, depfile: Path
    ) -> list[str]:
        return [
            self.nvcc,
            "-std=c++17",
            self.optimization,
            "-cubin",
            self._gencode(),
            "-lineinfo",
            *(["--use_fast_math"] if spec.upstream else []),
            *self._includes(spec),
            *self.flags,
            "-MD",
            "-MF",
            str(depfile),
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
        """sha256 of every pinned upstream file the translation unit included."""
        text = depfile.read_text().replace("\\\n", " ")
        files: dict[str, str] = {}
        for token in text.split(":", 1)[1].split():
            path = Path(os.path.abspath(ROOT / token))
            if path.is_relative_to(self.flashinfer):
                files[str(path.relative_to(self.flashinfer))] = _sha256(path)
        return dict(sorted(files.items()))

    def _check_cubin(self, spec: KernelSpec, image: bytes) -> list[dict[str, Any]]:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from harness.workload import cubin_info

        arch, names, params = cubin_info(image)
        if arch != self.arch:
            raise RuntimeError(f"{spec.workload}: cubin declares {arch}")
        if names != [spec.symbol]:
            raise RuntimeError(
                f"{spec.workload}: expected [{spec.symbol}], found {names}"
            )
        if [p.size for p in params] != [size for _, size in spec.params]:
            raise RuntimeError(f"{spec.workload}: parameter sizes {params}")
        return [
            {"name": name, "offset": p.offset, "size": p.size}
            for (name, _), p in zip(spec.params, params)
        ]

    def compile(self) -> dict[str, str]:
        specs = [s for s in SPECS if self.arch in s.arches]
        record = None
        if any(s.upstream for s in specs):
            record = self.manifest_record()
            self.generate()
        version = _run([self.nvcc, "--version"], ROOT).strip()
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True)
        mapping: dict[str, str] = {}
        upstream: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory(prefix="gdn-") as tmp:
            tmpdir = Path(tmp)
            for spec in specs:
                source = KERNELS / spec.source
                cubin = self.out_dir / f"{spec.workload}.cubin"
                depfile = tmpdir / f"{spec.workload}.d"
                command = self._cubin_command(spec, source, cubin, depfile)
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
                if spec.upstream:
                    assert record is not None
                    sidecar["generated_from"] = {
                        "file": UPSTREAM_KERNEL,
                        "sha256": UPSTREAM_SHA256[UPSTREAM_KERNEL],
                        "variant": VARIANT,
                        "substitution": [
                            SPECIALIZE_FROM.strip(),
                            SPECIALIZE_TO.strip(),
                        ],
                    }
                    sidecar["constants"] = self._constants(source, record)
                    sidecar["probe"] = self._probe(params, tmpdir)
                else:
                    sidecar["constants"] = {"block": 256, "num_v_heads": 8}
                _write_json(self.out_dir / f"{spec.workload}.json", sidecar)
                mapping[spec.workload] = str(cubin.relative_to(PACKAGE))
                records[spec.workload] = {
                    "cubin": mapping[spec.workload],
                    "symbol": spec.symbol,
                    "sha256": sidecar["cubin_sha256"],
                    "source": f"kernels/{spec.source}",
                    "source_sha256": sidecar["source_sha256"],
                    "command": sidecar["build"]["command"],
                    "adaptation": spec.adaptation,
                }
        self._merge_manifest(mapping)
        self._merge_provenance(records, upstream, version.splitlines()[-1])
        return mapping

    def _constants(self, source: Path, record: dict[str, Any]) -> dict[str, int]:
        """Launch constants and specializations of the generated kernel,
        cross-checked with the upstream manifest record."""
        text = source.read_text()
        constants = {
            "threads": _macro(text, "THREADS"),
            "dynamic_smem": _macro(text, "SMEM_TOTAL"),
            "tensormap_workspace_bytes_per_cta": 512,
            "num_q_heads": 4,
            "num_v_heads": 8,
            "head_size": 128,
            "value_splits": 2,
            "chunk": 64,
        }
        for name, value in record["specializations"].items():
            constants[name.lower()] = _macro(text, name)
            expected = 3 if name == "NUM_O_HEADS_LOG2" else value
            if constants[name.lower()] != expected:
                raise RuntimeError(f"{source.name}: {name} differs from {expected}")
        if (
            record["threads"] != constants["threads"]
            or record["dynamic_smem_bytes"] != constants["dynamic_smem"]
        ):
            raise RuntimeError(f"{VARIANT}: launch constants differ from the manifest")
        return constants

    def _probe(self, params: list[dict[str, Any]], tmpdir: Path) -> dict[str, Any]:
        (tmpdir / PROBE_INCLUDE).write_text(self.probe_include())
        exe = tmpdir / "probe_chunk_launch"
        _run(
            [
                self.nvcc,
                "-std=c++17",
                "-O1",
                "-I" + str(tmpdir),
                self._rel(KERNELS / PROBE),
                "-o",
                str(exe),
            ],
            ROOT,
        )
        layout = ",".join(f"{p['offset']}:{p['size']}" for p in params)
        runs = [
            json.loads(
                _run([str(exe), str(t), str(n), repr(s), str(sms), layout], ROOT)
            )
            for t, n, s, sms in PROBE_RUNS
        ]
        return {
            "source": self._rel(KERNELS / PROBE),
            "source_sha256": _sha256(KERNELS / PROBE),
            "upstream_host": UPSTREAM_HOST,
            "upstream_host_sha256": UPSTREAM_SHA256[UPSTREAM_HOST],
            "tma_marker": TMA_MARKER,
            "sentinel_stride": SENTINEL_STRIDE,
            "runs": runs,
        }

    # -- manifests -------------------------------------------------------------

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        _write_json(MANIFEST, dict(sorted(manifest.items())))

    def _merge_provenance(
        self, records: dict[str, dict[str, Any]], files: dict[str, str], nvcc: str
    ) -> None:
        entry: dict[str, Any] = {"nvcc": nvcc, "kernels": records, "files": files}
        if any(s.upstream for s in SPECS if s.workload in records):
            entry["files"] = dict(
                sorted(
                    {
                        **files,
                        **UPSTREAM_SHA256,
                        f"{CAKE}/manifest.json": CAKE_MANIFEST_SHA256,
                    }.items()
                )
            )
            entry["generated"] = {
                f"kernels/{GENERATED}": {
                    "from": UPSTREAM_KERNEL,
                    "from_sha256": UPSTREAM_SHA256[UPSTREAM_KERNEL],
                    "substitution": [SPECIALIZE_FROM.strip(), SPECIALIZE_TO.strip()],
                }
            }
        try:
            existing = json.loads(PROVENANCE.read_text())
        except (OSError, ValueError):
            existing = {}
        builds = existing.get("builds", {}) if isinstance(existing, dict) else {}
        builds[self.arch] = entry
        _write_json(
            PROVENANCE,
            {
                "package": "gdn",
                "definitions": [
                    "gdn_decode_qk4_v8_d128_k_last",
                    "gdn_prefill_qk4_v8_d128_k_last",
                ],
                "model": "Qwen3-Next-80B-A3B linear attention, TP=4 shard (16/32 "
                "key/value heads -> 4/8)",
                "repository": FLASHINFER_REPOSITORY,
                "revision": FLASHINFER_REVISION,
                "license": "LICENSE (FlashInfer, Apache-2.0)",
                "cake_manifest_sha256": CAKE_MANIFEST_SHA256,
                "variant": VARIANT,
                "kernels": {
                    GATES_SYMBOL: {
                        "status": "live",
                        "workload": "gdn_gates",
                        "reason": "definitions supply raw A_log/a/dt_bias/b (and "
                        "int64 cu_seqlens); the upstream kernel needs FP32 g/beta "
                        "and int32 offsets",
                    },
                    CHUNK_SYMBOL: {
                        "status": "live",
                        "workload": "gdn_chunk",
                        "reason": "the Tensor Core (tcgen05) kernel of both "
                        "definitions (decode = B one-token sequences)",
                    },
                },
                "merged_duplicates": MERGED_DUPLICATES,
                "excluded_kernels": DEAD_KERNELS,
                "scope": "Upstream routes the DV-split schedule only when "
                "2 * num_seqs * Hv <= 148 (num_seqs <= 9); this package uses it for "
                "every shape, as both former packages did (its tile loop is "
                "persistent and admits any total_tiles), with upstream's grid "
                "min(SMs, total_tiles). Empty sequences copy the initial state.",
                "upstream_coverage": UPSTREAM_COVERAGE,
                "builds": dict(sorted(builds.items())),
            },
        )

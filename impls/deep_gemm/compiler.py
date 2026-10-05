"""deep_gemm package: index the DeepGEMM sm_100a cubins, strip the sm_86 ones.

sm_100a -- FlashInfer's published DeepGEMM FP8 m-grouped GEMM cubins
====================================================================

``cubins/deep_gemm/kernel.fp8_m_grouped_gemm.<hash>.cubin`` (317 files) are
byte-identical to the DeepGEMM artifact of FlashInfer v0.6.9
(``flashinfer/artifacts.py``: ``ArtifactPath.DEEPGEMM =
"a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm/"``, whose
``checksums.txt`` has the pinned ``CheckSumHash.DEEPGEMM``), shipped in
``flashinfer_cubin-0.6.9-py3-none-any.whl`` (SHA256 11e06a1e...). Each holds
exactly one kernel (declared ``sm_100f``, registered for ``sm_100a``), so
``kernels.json`` references the original files; nothing is copied or
compiled. ``compile()`` verifies every file against ``checksums.txt`` and
the artifact's ``kernel_map.json`` (cubin -> symbol, SHA256; pinned by v0.6.9's
``KernelMap.KERNEL_MAP_HASH``), fetched by ``scripts/fetch_resources.py
--only trtllm-gen`` into ``resources/trtllm-gen-artifacts/``.

Each kernel is ``deep_gemm::sm100_fp8_gemm_1d1d_impl<...>``; its mangled
symbol holds every template argument. FlashInfer names the cubin
``kernel.fp8_m_grouped_gemm.<md5(name + "$$" + code)[:12]>`` where ``code`` is
``SM100FP8GemmRuntime.generate(static kwargs)``; ``compile()`` rebuilds that
code from the decoded arguments and checks that the hash reproduces the file
name, which pins the exact static kwargs of every cubin. All 317 are K-major
A/B, ``compiled_dims="nk"`` (M = 0), BLOCK_M = BLOCK_K = 128, 128 + 128
threads, no multicast, no accumulation, bf16 output; 122 are
``GemmType::MGroupedContiguous`` and 195 ``MGroupedMasked``, over
(N, K) in {(4096, 7168), (7168, 2048), (512, 128), (128, 512)} and 1-256
groups. The kernel source is DeepGEMM 9da4a23 (identical for these kernels
at aff9da0, whose only include change is an SM90-only scheduler line); the
signature (``kNumLastStages``, no ``kNumSMs``, ``CUtensorMap`` parameters)
matches exactly that range. The resources/DeepGEMM checkout (78b6900) no
longer has this kernel; nothing is compiled from it.

Dispatch: which shapes reach a kernel is decided by FlashInfer's
``get_best_configs`` (replicated in ``harness/workloads/deep_gemm.py``,
cross-checked against FlashInfer's own code in ``tests/test_deep_gemm.py``)
from (M, N, K, num_groups, num_sms), where M is the total row count
(contiguous) or ``expected_m`` (masked) and enters only as ``ceil(M / 128)``.
For each cubin the compiler searches ``ceil(M / 128)`` in 1..1024 at the
B200's 148 SMs; 241 cubins are dispatched there. The other 76 are only
selected for other SM counts (the published set was tuned over several
devices); for those the nearest SM count that selects the kernel is
recorded (110-192) and the cases use it for the config and the persistent
grid (the kernel is correct for any grid; CTAs beyond 148 run in a later
wave). Every cubin is served; none is excluded.

Duplicates: the 317 cubins hold 236 distinct kernels. 81 contiguous cubins
have byte-identical ``.text``/``.nv.info``/``.nv.shared``/``.nv.constant``
sections to another cubin and differ only in ``NUM_GROUPS`` (the
MGroupedContiguous scheduler never reads it); e.g. ``n4096_k7168`` with
BLOCK_N 240 for 1, 2, ..., 128 groups. ``compile()`` groups cubins by that
code hash, keeps the cubin with the fewest groups as the workload
``deep_gemm_contig_n<N>_k<K>_bn<BLOCK_N>`` (41 contiguous workloads) and
records one ``dispatch`` entry per group count (SM count, ranges, the cubin
FlashInfer loads), so the workload's cases are the union of its duplicates'.
``provenance.json`` maps every former ``deep_gemm_contig_..._g<G>_...`` name
to its workload and lists each merged cubin. Masked cubins all differ.

Upstream coverage (``UPSTREAM_COVERAGE``; enforced by
``tests/test_deep_gemm.py``'s ``UpstreamCoverage`` with FlashInfer's own
selection code): every FlashInfer v0.6.9 ``test_fp8_groupwise_group_deepgemm``
parametrization (28) is a smoke case of the kernel it selects at 148 SMs;
``test_fp8_groupwise_batch_deepgemm_masked`` draws masked_m on the GPU, so
for each of its 96 parametrizations every kernel selectable by an
expected_m up to its capacity (360 (parametrization, block) pairs) has a
case with that capacity; ``bench_deepgemm_blackwell.py``'s configurations
are throughput cases. The pinned DeepGEMM checkout (78b6900) no longer
has this kernel and tests other (N, K); its m-grouped (num_groups,
expected m) regimes are replayed on the production (N, K), except masked
num_groups = 6 (no 6-group kernel).

Live/dead: every cubin is live (one kernel each, launched once per
``m_grouped_fp8_gemm_nt_{contiguous,masked}`` call). FlashInfer's layout
transform (``get_col_major_tma_aligned_packed_tensor``, torch ops) is not a
kernel of the artifact; the workloads take the scale factors already in the
kernel-native packed layout.

sm_86 -- FlashInfer CUTLASS segment GEMM, column-major weights
===============================================================

FlashInfer 0.7.0's sm80 JIT-cache wheel (``flashinfer_jit_cache_sm80-0.7.0+
cu130``, extracted by ``scripts/fetch_resources.py --only jit-cache``) holds
the ``gemm`` module; ``gemm/gemm.2.sm_80.cubin`` has the 8
``cutlass::Kernel<GemmGrouped<...>>`` instantiations of
``CutlassSegmentGEMMRun`` (``include/flashinfer/gemm/group_gemm.cuh``):
{bf16, fp16} x {B RowMajor, B ColumnMajor} x {2, 4 stages}
(``gemm.1``/``gemm.3`` hold no kernel). This package owns the four
ColumnMajor ("NT", ``weight_column_major=True``) kernels, DeepGEMM's
m-grouped contiguous operand layout; the RowMajor ones belong to
batched_gemm. Each is cut out with ``harness.cubin_strip.strip_cubin``.

==========================  ======  ==========================================
kernel (B ColumnMajor)      status  reason
==========================  ======  ==========================================
bf16 MmaPipelined (2 st.)   live    ``DISPATCH_SMEM_CONFIG``: chosen when the
fp16 MmaPipelined (2 st.)   live    SM has < 147968 B shared memory (sm_86,
                                    sm_89: 100 KB); upstream's sm_86 kernel.
bf16 MmaMultistage (4 st.)  live*   Chosen on sm_80/sm_87 (164 KB/SM). Never
fp16 MmaMultistage (4 st.)  live*   dispatched on sm_86, but its sm_80 SASS
                                    (64 KB shared memory) runs there
                                    unchanged; registered so that the local
                                    library covers both upstream mainloops.
B RowMajor kernels (4)      other   batched_gemm package.
==========================  ======  ==========================================

Upstream coverage (``SEGMENT_UPSTREAM_COVERAGE``): FlashInfer 0.7.0's
``test_segment_gemm`` grid with ``column_major=True`` on the sm80 backend
(72 shapes, float16) is a smoke case of both fp16 kernels.

``GemmGrouped::Params`` (144 bytes) is passed by value. A host-only probe
(``kernels/segment_gemm_probe.cu``, CUTLASS b46b16d as pinned by FlashInfer
0.7.0's 3rdparty/cutlass) instantiates the identical ``DefaultGemmGrouped``
types, records sizeof/offsetof of every field, the launch constants and
reference Params built as ``GemmGrouped::initialize`` builds them from
``CutlassSegmentGEMMRun``'s Arguments (fake pointers). Its embedded sm_80
device code must have exactly the stripped kernels' symbols. The stripped
cubins and their sidecars (``cubins/sm_86/<workload>.json``,
``.fixtures.json``) are generated (gitignored).
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
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"
VARIANTS = PACKAGE / "variants.json"

FLASHINFER_V069 = "a1aa676196f798435248d9ea205c67674476f473"
WHEEL = (
    "https://files.pythonhosted.org/packages/f4/b1/"
    "d1055c8bed5adb6019295c43f03b17f48d57ad10977e6e9e0cf9a70917fc/"
    "flashinfer_cubin-0.6.9-py3-none-any.whl"
)
WHEEL_SHA256 = "11e06a1eb2c61fc9f69cdbca3a6abe7d22e7244fbdcaadbb9f9e0bb65fd335ed"
ARTIFACT = "a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm"
ARTIFACT_URL = (
    "https://edge.urm.nvidia.com/artifactory/"
    "sw-kernelinferencelibrary-public-generic-local/" + ARTIFACT + "/"
)
CHECKSUMS_SHA256 = "1a2a166839042dbd2a57f48051c82cd1ad032815927c753db269a4ed10d0ffbf"
KERNEL_MAP_SHA256 = "f161e031826adb8c4f0d31ddbd2ed77e4909e4e43cdfc9728918162a62fcccfb"
DEEPGEMM_REPOSITORY = "https://github.com/deepseek-ai/DeepGEMM"
DEEPGEMM_REVISION = "9da4a23561e114d192e25e893f16358d68b04da3"
DEEPGEMM_REVISION_EQUIVALENT = "aff9da0aba5aab57a0c9d8c13c41452dc01ce646"
CUBIN_DIR = ROOT / "cubins" / "deep_gemm"
NUM_SMS = 148  # B200
MAX_M_BLOCKS = 1024

JIT_CACHE_CUBIN = "flashinfer-jit-cache-sm80/gemm/gemm.2.sm_80.cubin"
JIT_CACHE_SHA256 = "8a23d28908a7b0ee713646602dd17dfaf995c77b1410e1c36482739fa8909c3c"
FLASHINFER_V070 = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
CUTLASS_DIR = "cutlass-b46b16d003484063bca4ed365e44095c4c6ed633"
PROBE_SOURCE = "kernels/segment_gemm_probe.cu"
SEGMENT_KERNELS = {
    # workload: (probe key, dtype token in the symbol, pipelined?)
    "deep_gemm_segment_bf16_pipelined": ("bf16_pipelined", "bfloat16_t", True),
    "deep_gemm_segment_bf16_multistage": ("bf16_multistage", "bfloat16_t", False),
    "deep_gemm_segment_fp16_pipelined": ("fp16_pipelined", "half_t", True),
    "deep_gemm_segment_fp16_multistage": ("fp16_multistage", "half_t", False),
}
COLUMN_MAJOR_B = "40ColumnMajorTensorOpMultiplicandCrosswise"
CODE_SECTIONS = (".text.", ".nv.info.", ".nv.shared.", ".nv.constant")
UPSTREAM_COVERAGE = {
    "flashinfer_tests": (
        "tests/gemm/test_groupwise_scaled_gemm_fp8.py (v0.6.9): "
        "test_fp8_groupwise_group_deepgemm (m in 128..1024, all 4 (N, K), "
        "group_size with m // group_size >= 128) is a smoke case of the kernel "
        "FlashInfer selects at 148 SMs, with its exact equal groups; "
        "test_fp8_groupwise_batch_deepgemm_masked (m, (N, K), group_size) draws "
        "masked_m ~ randint(0, m) on the GPU, so expected_m can select any "
        "kernel for ceil(expected_m / 128) in 1..m / 128: every such kernel has a "
        "smoke case with capacity m (masked_m uniform below m)"
    ),
    "flashinfer_benchmarks": (
        "benchmarks/bench_deepgemm_blackwell.py (v0.6.9): every grouped and "
        "batch (masked, at the mean expected_m) configuration is a throughput "
        "case of the kernel it selects"
    ),
    "deepgemm_tests": (
        "resources/DeepGEMM (78b6900) no longer has this kernel and tests "
        "(N, K) none of these cubins is compiled for; the num_groups / "
        "expected-m regimes of enumerate_m_grouped_contiguous (4 x 8192, "
        "8 x 4096, -1 padded) and enumerate_m_grouped_masked (32 x 192, "
        "32 x 20, max_m 4096, expected_m = 1.2 x) are replayed on the "
        "production (N, K) as smoke cases; its num_groups = 6 masked rows have "
        "no 6-group kernel"
    ),
    "enforcement": "tests/test_deep_gemm.py UpstreamCoverage (selection by "
    "FlashInfer's own get_best_configs / cubin names)",
}
SEGMENT_UPSTREAM_COVERAGE = {
    "flashinfer_tests": (
        "tests/gemm/test_group_gemm.py::test_segment_gemm (0.7.0), "
        "column_major=True, backend sm80, float16: every (batch_size, "
        "num_rows_per_batch, d_in, d_out) with batch_size * num_rows_per_batch "
        "<= 8192 is a smoke case of both fp16 kernels; weights over ~3 GB "
        "(batch_size 199 at 4096 x 4096) are a smaller bank indexed modulo "
        "(same problem sizes and launch); use_weight_indices=True (identity "
        "indices into a 1024-weight bank) launches the same kernel arguments and "
        "is covered by the permuted weight-bank cases"
    ),
    "enforcement": "tests/test_deep_gemm.py UpstreamCoverage",
}

SYMBOL = re.compile(
    r"_ZN9deep_gemm24sm100_fp8_gemm_1d1d_implILN4cute4UMMA5MajorE(\d)ELS3_(\d)E"
    r"((?:Lj\d+E){15})Lb(\d)ELNS_8GemmTypeE(\d)ELb(\d)E"
    r"(N7cutlass10bfloat16_tE|f)EEvPijjj14CUtensorMap_stS8_S8_S8_S8_S8_"
)
TEMPLATE_FIELDS = (
    "M N K BLOCK_M BLOCK_N BLOCK_K NUM_GROUPS SWIZZLE_A_MODE SWIZZLE_B_MODE "
    "SWIZZLE_CD_MODE NUM_STAGES NUM_LAST_STAGES NUM_NON_EPILOGUE_THREADS "
    "NUM_EPILOGUE_THREADS NUM_MULTICAST"
).split()


def _harness():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness import cubin_strip, workload
    from harness.workloads import deep_gemm

    return workload, cubin_strip, deep_gemm


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_symbol(symbol: str) -> dict[str, Any]:
    """Template arguments of an ``sm100_fp8_gemm_1d1d_impl`` symbol."""
    match = SYMBOL.fullmatch(symbol)
    if not match:
        raise ValueError(f"not an sm100_fp8_gemm_1d1d_impl symbol: {symbol}")
    numbers = [int(x) for x in re.findall(r"Lj(\d+)E", match.group(3))]
    args: dict[str, Any] = dict(zip(TEMPLATE_FIELDS, numbers))
    args.update(
        MAJOR_A=int(match.group(1)),
        MAJOR_B=int(match.group(2)),
        IS_MULTICAST_ON_A=bool(int(match.group(4))),
        GEMM_TYPE=int(match.group(5)),
        WITH_ACCUMULATION=bool(int(match.group(6))),
        CD_DTYPE_T="cutlass::bfloat16_t" if "bfloat" in match.group(7) else "float",
    )
    return args


def flashinfer_code(args: dict[str, Any]) -> str:
    """``SM100FP8GemmRuntime.generate`` (FlashInfer v0.6.9) for ``args``."""
    major = {0: "cute::UMMA::Major::K", 1: "cute::UMMA::Major::MN"}
    gemm = {
        0: "GemmType::Normal",
        1: "GemmType::GroupedContiguous",
        2: "GemmType::GroupedMasked",
    }

    def b(value: bool) -> str:
        return "true" if value else "false"

    return f"""
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_1d1d_impl<
        {major[args["MAJOR_A"]]},
        {major[args["MAJOR_B"]]},
        {args["M"]},
        {args["N"]},
        {args["K"]},
        {args["BLOCK_M"]},
        {args["BLOCK_N"]},
        {args["BLOCK_K"]},
        {args["NUM_GROUPS"]},
        {args["SWIZZLE_A_MODE"]},
        {args["SWIZZLE_B_MODE"]},
        {args["SWIZZLE_CD_MODE"]},
        {args["NUM_STAGES"]},
        {args["NUM_LAST_STAGES"]},
        {args["NUM_NON_EPILOGUE_THREADS"]},
        {args["NUM_EPILOGUE_THREADS"]},
        {args["NUM_MULTICAST"]},
        {b(args["IS_MULTICAST_ON_A"])},
        {gemm[args["GEMM_TYPE"]]},
        {b(args["WITH_ACCUMULATION"])},
        {args["CD_DTYPE_T"]}
      >);
}};
"""


def cubin_name(args: dict[str, Any]) -> str:
    """FlashInfer's ``load("fp8_m_grouped_gemm", code)`` cubin name."""
    digest = hashlib.md5(
        ("fp8_m_grouped_gemm$$" + flashinfer_code(args)).encode()
    ).hexdigest()
    return f"kernel.fp8_m_grouped_gemm.{digest[:12]}"


def _ranges(values: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for value in values:
        if ranges and ranges[-1][1] == value - 1:
            ranges[-1][1] = value
        else:
            ranges.append([value, value])
    return ranges


class ImplCompiler:
    """Indexes the sm_100a cubins / strips the sm_86 ones; merges kernels.json."""

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
        if arch not in self.supported_arches:
            raise ValueError(f"deep_gemm supports {self.supported_arches}, not {arch}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = list(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(resources or ROOT / "resources").resolve()

    def compile(self) -> dict[str, str]:
        if self.arch == "sm_100a":
            return self._index_sm100()
        return self._strip_sm86()

    # -- sm_100a -----------------------------------------------------------------

    def _artifact_files(self) -> tuple[dict[str, str], dict[str, list[str]]]:
        base = self.resources / "trtllm-gen-artifacts" / ARTIFACT
        checksums, kernel_map = base / "checksums.txt", base / "kernel_map.json"
        if not (checksums.is_file() and kernel_map.is_file()):
            raise FileNotFoundError(
                f"{base} lacks checksums.txt/kernel_map.json; run "
                "scripts/fetch_resources.py --only trtllm-gen"
            )
        if _sha256(checksums.read_bytes()) != CHECKSUMS_SHA256:
            raise ValueError("deep-gemm checksums.txt does not match v0.6.9")
        if _sha256(kernel_map.read_bytes()) != KERNEL_MAP_SHA256:
            raise ValueError("deep-gemm kernel_map.json does not match v0.6.9")
        sums = dict(line.split()[::-1] for line in checksums.read_text().splitlines())
        return sums, json.loads(kernel_map.read_text())

    @staticmethod
    def _dispatch(
        deep_gemm, layout: str, args: dict[str, Any]
    ) -> tuple[int, list[int]]:
        """(SM count, ceil(M/128) values) for which FlashInfer selects args."""

        def blocks(sms: int) -> list[int]:
            found = []
            for m_blocks in range(1, MAX_M_BLOCKS + 1):
                config = deep_gemm.best_config(
                    layout,
                    m_blocks * 128,
                    args["N"],
                    args["K"],
                    args["NUM_GROUPS"],
                    sms,
                )
                if (
                    config.block_n,
                    config.num_stages,
                    config.num_last_stages,
                    config.swizzle_cd,
                ) == (
                    args["BLOCK_N"],
                    args["NUM_STAGES"],
                    args["NUM_LAST_STAGES"],
                    args["SWIZZLE_CD_MODE"],
                ):
                    found.append(m_blocks)
            return found

        for sms in sorted(
            range(1, 2 * NUM_SMS + 1), key=lambda s: (abs(s - NUM_SMS), s)
        ):
            found = blocks(sms)
            if found:
                return sms, found
        raise ValueError(f"no SM count selects {args}")

    @staticmethod
    def _code_key(workload, image: bytes) -> str:
        """SHA256 of the kernel's code and metadata sections (``.text.*``,
        ``.nv.info.*``, ``.nv.shared.*``, ``.nv.constant*``), names excluded:
        equal keys mean identical SASS, parameters, shared memory and
        constant banks under different (mangled) symbols."""
        sections = workload._sections(image)
        return _sha256(
            b"|".join(
                value
                for name, value in sorted(sections.items())
                if name.startswith(CODE_SECTIONS)
            )
        )

    def _index_sm100(self) -> dict[str, str]:
        workload, _, deep_gemm = _harness()
        sums, kernel_map = self._artifact_files()
        files = sorted(CUBIN_DIR.glob("kernel.fp8_m_grouped_gemm.*.cubin"))
        if len(files) != len(kernel_map) or len(sums) != len(kernel_map):
            raise ValueError(
                f"{len(files)} cubins, {len(kernel_map)} kernel_map entries, "
                f"{len(sums)} checksums"
            )
        # Every published cubin, verified; grouped by kernel code below.
        cubins: list[dict[str, Any]] = []
        for path in files:
            stem = path.name.removesuffix(".cubin")
            image = path.read_bytes()
            digest = _sha256(image)
            symbol, expected = kernel_map[stem]
            if digest != expected or sums[path.name] != expected:
                raise ValueError(f"{path.name}: SHA256 differs from the artifact")
            arch, names, _ = workload.cubin_info(image)
            if arch != "sm_100f" or names != [symbol]:
                raise ValueError(f"{path.name}: {arch} {names}")
            args = decode_symbol(symbol)
            if cubin_name(args) != stem:
                raise ValueError(f"{path.name}: FlashInfer's hash does not reproduce")
            fixed = dict(
                MAJOR_A=0, MAJOR_B=0, M=0, BLOCK_M=128, BLOCK_K=128,
                SWIZZLE_A_MODE=128, SWIZZLE_B_MODE=128,
                NUM_NON_EPILOGUE_THREADS=128, NUM_EPILOGUE_THREADS=128,
                NUM_MULTICAST=1, IS_MULTICAST_ON_A=True, WITH_ACCUMULATION=False,
                CD_DTYPE_T="cutlass::bfloat16_t",
            )  # fmt: skip
            if any(args[key] != value for key, value in fixed.items()):
                raise ValueError(f"{path.name}: unsupported template arguments {args}")
            layout = {1: "contiguous", 2: "masked"}[args["GEMM_TYPE"]]
            smem, swizzle_cd = deep_gemm.smem_size(
                128, args["BLOCK_N"], args["NUM_STAGES"]
            )
            if swizzle_cd != args["SWIZZLE_CD_MODE"]:
                raise ValueError(f"{path.name}: unexpected CD swizzle")
            sms, m_blocks = self._dispatch(deep_gemm, layout, args)
            cubins.append(
                {
                    "path": path,
                    "stem": stem,
                    "sha256": digest,
                    "symbol": symbol,
                    "args": args,
                    "layout": layout,
                    "smem": smem,
                    "code": self._code_key(workload, image),
                    "dispatch": {
                        "num_groups": args["NUM_GROUPS"],
                        "sms": sms,
                        "m_blocks": _ranges(m_blocks),
                        "cubin": stem,
                    },
                }
            )
        # Deduplicate: cubins with identical code are one kernel. Only
        # contiguous cubins that differ in NUM_GROUPS alone coincide (the
        # contiguous scheduler never reads kNumGroups); the canonical cubin is
        # the one with the fewest groups, and it serves every group count of
        # its duplicates (FlashInfer would load the duplicate's file for that
        # group count, with byte-identical code).
        by_code: dict[str, list[dict[str, Any]]] = {}
        for cubin in cubins:
            by_code.setdefault(cubin["code"], []).append(cubin)
        mapping: dict[str, str] = {}
        variants: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        removed: dict[str, str] = {}
        for group in by_code.values():
            group.sort(key=lambda c: c["args"]["NUM_GROUPS"])
            keep = group[0]
            args, layout = keep["args"], keep["layout"]
            identity = ("N", "K", "BLOCK_N", "NUM_STAGES", "NUM_LAST_STAGES")
            for other in group[1:]:
                if other["layout"] != "contiguous" or any(
                    other["args"][key] != args[key] for key in identity
                ):
                    raise ValueError(
                        f"{other['stem']} has {keep['stem']}'s code but different "
                        "template arguments other than NUM_GROUPS"
                    )
            if layout == "contiguous":
                name = f"deep_gemm_contig_n{args['N']}_k{args['K']}_bn{args['BLOCK_N']}"
            else:
                name = (
                    f"deep_gemm_masked_n{args['N']}_k{args['K']}"
                    f"_g{args['NUM_GROUPS']}_bn{args['BLOCK_N']}"
                )
            if name in variants:
                raise ValueError(f"duplicate variant name {name} (different code)")
            mapping[name] = os.path.relpath(keep["path"], PACKAGE)
            variants[name] = {
                "layout": layout,
                "n": args["N"],
                "k": args["K"],
                "block_n": args["BLOCK_N"],
                "num_stages": args["NUM_STAGES"],
                "num_last_stages": args["NUM_LAST_STAGES"],
                "swizzle_cd": args["SWIZZLE_CD_MODE"],
                "smem": keep["smem"],
                "dispatch": [c["dispatch"] for c in group],
            }
            records[name] = {
                "cubin": keep["path"].name,
                "sha256": keep["sha256"],
                "symbol": keep["symbol"],
            }
            if len(group) > 1:
                records[name]["identical_code"] = {
                    c["stem"]
                    + ".cubin": {
                        "num_groups": c["args"]["NUM_GROUPS"],
                        "sha256": c["sha256"],
                        "symbol": c["symbol"],
                    }
                    for c in group[1:]
                }
            if layout == "contiguous":
                for c in group:
                    old = (
                        f"deep_gemm_contig_n{args['N']}_k{args['K']}"
                        f"_g{c['args']['NUM_GROUPS']}_bn{args['BLOCK_N']}"
                    )
                    removed[old] = name
        self._merge_manifest(mapping)
        self._merge_variants(variants)
        dispatched = sum(1 for c in cubins if c["dispatch"]["sms"] == NUM_SMS)
        self._merge_provenance(
            {
                "source": {
                    "flashinfer_release": "v0.6.9",
                    "flashinfer_revision": FLASHINFER_V069,
                    "wheel": WHEEL,
                    "wheel_sha256": WHEEL_SHA256,
                    "artifact": ARTIFACT_URL,
                    "checksums_txt_sha256": CHECKSUMS_SHA256,
                    "kernel_map_json_sha256": KERNEL_MAP_SHA256,
                    "launcher": "flashinfer/deep_gemm.py (v0.6.9)",
                    "deepgemm_repository": DEEPGEMM_REPOSITORY,
                    "deepgemm_revision": DEEPGEMM_REVISION,
                    "deepgemm_revision_equivalent": DEEPGEMM_REVISION_EQUIVALENT,
                    "kernel_source": "deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh",
                    "nvcc": "Cuda compilation tools, release 12.9, V12.9.86 (-arch sm_100f)",
                },
                "verification": [
                    "SHA256 of every cubin equals checksums.txt and kernel_map.json",
                    "each cubin has exactly the kernel_map symbol (sm_100f)",
                    (
                        "md5(fp8_m_grouped_gemm$$generate(kwargs decoded from the "
                        "symbol)) reproduces every file name"
                    ),
                    (
                        "cubins merged into one kernel have byte-identical .text, "
                        ".nv.info, .nv.shared and .nv.constant sections"
                    ),
                ],
                "dispatch": (
                    f"{dispatched} of {len(cubins)} cubins are selected by FlashInfer's"
                    f" get_best_configs at {NUM_SMS} SMs (B200) for some ceil(M/128) <= "
                    f"{MAX_M_BLOCKS}; {len(cubins) - dispatched} only at other SM "
                    "counts (dispatch[].sms in variants.json)"
                ),
                "deduplication": (
                    f"{len(cubins)} cubins hold {len(variants)} distinct kernels: "
                    f"{len(cubins) - len(variants)} contiguous cubins differ from "
                    "another only in NUM_GROUPS, which the MGroupedContiguous "
                    "scheduler never reads (identical code sections). Each set is "
                    "one workload deep_gemm_contig_n<N>_k<K>_bn<BLOCK_N> (cubin of "
                    "the smallest group count) whose cases cover every group count "
                    "of the set; masked kernels depend on NUM_GROUPS and keep it in "
                    "their names"
                ),
                "renamed_or_removed": dict(sorted(removed.items())),
                "upstream_coverage": UPSTREAM_COVERAGE,
                "live": "all cubins (one kernel each, no dead kernels)",
                "excluded": {},
                "kernels": dict(sorted(records.items())),
            }
        )
        return mapping

    # -- sm_86 -------------------------------------------------------------------

    def probe_command(self, output: Path) -> list[str]:
        cutlass = self.resources / CUTLASS_DIR
        return [
            self.nvcc,
            "-std=c++17",
            self.optimization,
            "-gencode=arch=compute_80,code=sm_80",
            "--expt-relaxed-constexpr",
            "-diag-suppress=20012",
            "-I" + str(cutlass / "include"),
            *self.flags,
            str(PACKAGE / PROBE_SOURCE),
            "-o",
            str(output),
        ]

    @staticmethod
    def _portable(command: list[str]) -> list[str]:
        root = str(ROOT) + os.sep
        return [part.replace(root, "") for part in command]

    def _strip_sm86(self) -> dict[str, str]:
        workload, cubin_strip, _ = _harness()
        source_path = self.resources / JIT_CACHE_CUBIN
        source = source_path.read_bytes()
        if _sha256(source) != JIT_CACHE_SHA256:
            raise ValueError(f"{JIT_CACHE_CUBIN}: unexpected SHA256")
        kernels = cubin_strip.cubin_kernels(source)
        if len(kernels) != 8:
            raise ValueError(f"expected 8 GemmGrouped kernels, found {len(kernels)}")
        out_dir = PACKAGE / "cubins" / self.arch
        out_dir.mkdir(parents=True, exist_ok=True)
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        with tempfile.TemporaryDirectory(prefix="deep_gemm-") as tmp:
            probe = Path(tmp) / "segment_gemm_probe"
            subprocess.run(self.probe_command(probe), check=True, cwd=ROOT)
            report = json.loads(subprocess.check_output([str(probe)], text=True))
            subprocess.run(
                ["cuobjdump", "-xelf", "all", str(probe)],
                check=True,
                cwd=tmp,
                capture_output=True,
            )
            probe_kernels = {
                name
                for image in Path(tmp).glob("*.cubin")
                for name in cubin_strip.cubin_kernels(image.read_bytes())
            }
        mapping: dict[str, str] = {}
        records: dict[str, Any] = {}
        variants: dict[str, dict[str, Any]] = {}
        column_major = [k for k in kernels if COLUMN_MAJOR_B in k]
        if len(column_major) != 4 or set(column_major) - probe_kernels:
            raise ValueError("the probe's kernels differ from the JIT-cache kernels")
        for name, (key, dtype, pipelined) in SEGMENT_KERNELS.items():
            (symbol,) = [
                k
                for k in column_major
                if f"NS_{len(dtype)}{dtype}E" in k
                and ("12MmaPipelined" in k) == pipelined
            ]
            layout = report[key]
            stripped = cubin_strip.strip_cubin(source, symbol)
            cubin_strip.check_strip(source, stripped, symbol)
            arch, names, params = workload.cubin_info(stripped)
            if arch != "sm_80" or names != [symbol]:
                raise ValueError(f"{name}: stripped cubin {arch} {names}")
            if [p.size for p in params] != [layout["params_size"]]:
                raise ValueError(f"{name}: Params size differs from the probe")
            target = out_dir / f"{name}.cubin"
            target.write_bytes(stripped)
            sidecar = {
                "workload": name,
                "arch": self.arch,
                "cubin": target.name,
                "cubin_sha256": _sha256(stripped),
                "symbol": symbol,
                **{k: v for k, v in layout.items() if k != "examples"},
                "threadblock_count": 4,
            }
            fixtures = {"workload": name, "examples": layout["examples"]}
            self._write_json(out_dir / f"{name}.json", sidecar)
            self._write_json(out_dir / f"{name}.fixtures.json", fixtures)
            relative = target.relative_to(PACKAGE).as_posix()
            mapping[name] = relative
            variants[name] = {
                "dtype": "bf16" if dtype == "bfloat16_t" else "fp16",
                "mainloop": "MmaPipelined" if pipelined else "MmaMultistage",
                "stages": layout["stages"],
                "upstream_dispatch": (
                    "sm_86/sm_89 (< 147968 B shared memory per SM)"
                    if pipelined
                    else "sm_80/sm_87 (>= 147968 B shared memory per SM)"
                ),
                "sidecar": f"cubins/{self.arch}/{name}.json",
            }
            records[name] = {
                "cubin": relative,
                "sha256": _sha256(stripped),
                "symbol": symbol,
                "sidecar": f"cubins/{self.arch}/{name}.json",
            }
        self._merge_manifest(mapping)
        self._merge_variants(variants)
        self._merge_provenance(
            {
                "source": {
                    "wheel": "flashinfer_jit_cache_sm80-0.7.0+cu130-cp39-abi3-"
                    "manylinux_2_28_x86_64.whl (scripts/fetch_resources.py)",
                    "cubin": f"resources/{JIT_CACHE_CUBIN} (from gemm/gemm.so)",
                    "cubin_sha256": JIT_CACHE_SHA256,
                    "flashinfer_revision": FLASHINFER_V070,
                    "host_code": "include/flashinfer/gemm/group_gemm.cuh "
                    "(CutlassSegmentGEMMRun), csrc/group_gemm.cu, "
                    "flashinfer/gemm/gemm_base.py (SegmentGEMMWrapper, sm80 backend), "
                    "flashinfer/triton/gemm.py (compute_sm80_group_gemm_args)",
                    "cutlass": f"resources/{CUTLASS_DIR}",
                },
                "strip": "harness.cubin_strip.strip_cubin (+ check_strip)",
                "probe": PROBE_SOURCE,
                "probe_sha256": _sha256((PACKAGE / PROBE_SOURCE).read_bytes()),
                "probe_command": self._portable(
                    self.probe_command(Path("segment_gemm_probe"))
                ),
                "nvcc": version.strip().splitlines()[-1],
                "upstream_coverage": SEGMENT_UPSTREAM_COVERAGE,
                "live_dead": {
                    "B ColumnMajor MmaPipelined (bf16, fp16)": "live: upstream's "
                    "choice when the SM has < 147968 B shared memory (sm_86)",
                    "B ColumnMajor MmaMultistage (bf16, fp16)": "live on sm_80/sm_87 "
                    "(>= 147968 B); registered for sm_86, where its sm_80 SASS "
                    "(64 KiB shared memory) runs unchanged",
                    "B RowMajor (4 kernels)": "owned by the batched_gemm package",
                },
                "kernels": records,
            }
        )
        return mapping

    # -- outputs -----------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n")

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        self._write_json(MANIFEST, dict(sorted(manifest.items())))

    def _merge_variants(self, variants: dict[str, dict[str, Any]]) -> None:
        index = json.loads(VARIANTS.read_text()) if VARIANTS.is_file() else {}
        index[self.arch] = dict(sorted(variants.items()))
        VARIANTS.write_text(
            "{\n"
            + ",\n".join(
                f"  {json.dumps(arch)}: {{\n"
                + ",\n".join(
                    f"    {json.dumps(name)}: {json.dumps(entry)}"
                    for name, entry in entries.items()
                )
                + "\n  }"
                for arch, entries in sorted(index.items())
            )
            + "\n}\n"
        )

    def _merge_provenance(self, build: dict[str, Any]) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {})
        builds[self.arch] = build
        provenance = {
            "package": "deep_gemm",
            "license": "LICENSE (DeepGEMM, MIT); FLASHINFER_LICENSE (Apache-2.0); "
            "CUTLASS_LICENSE (BSD-3-Clause)",
            "scope": "sm_100a: every DeepGEMM FP8 m-grouped GEMM cubin of FlashInfer "
            "v0.6.9 (contiguous and masked layouts), one workload each; sm_86: "
            "FlashInfer CUTLASS segment GEMM with column-major weights",
            "builds": dict(sorted(builds.items())),
        }
        self._write_json(PROVENANCE, provenance)

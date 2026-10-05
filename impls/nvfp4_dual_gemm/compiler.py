"""Compile the nvfp4_dual_gemm pipeline's live kernels into single-kernel cubins.

The workload is ``out = half(silu(A @ B1^T) * (A @ B2^T))`` with NVFP4 (E2M1,
two values per byte) A [m, k, l], B1/B2 [n, k, l] and E4M3 block scales per 16
K elements (``resources/nvfp4_dual_gemm/{task.yml,reference.py}``). The
previous package (``implementation_sm_100a.cpp`` = amalgamated FlashInfer
``fp4_gemm_template_sm100.h`` + ``impls/templates/nvfp4.cu`` with
``WORKLOAD_KIND == 1``) ran, per call, on one stream::

    pack_scales(SFA)  -> packed SFA    <<<ceil(|SFA'| / 256), 256>>>  \\ gemm(B1)
    pack_scales(SFB1) -> packed SFB1   <<<ceil(|SFB'| / 256), 256>>>  /
    GemmUniversal<float out>(A, B1)  -> FP32 h1 [m, n, l]  (cluster 1x1x1, PDL)
    pack_scales(SFA)  -> packed SFA    (again, identical result)      \\ gemm(B2)
    pack_scales(SFB2) -> packed SFB2                                   /
    GemmUniversal<float out>(A, B2)  -> FP32 h2 [m, n, l]
    silu_product(h1, h2) -> FP16 out   <<<ceil(m n l / 256), 256>>>

Both GEMMs use the same kernel (``genericFp4GemmKernelLauncher<float, 128,
128, 256, 1, 1, 1, _1SM>``); they differ only in their Params. Each live
kernel becomes one cubin and one registered workload:

=============================================  ======  =====================================
kernel (previous cubins ``CUBIN/*.cubin``)     status  reason
=============================================  ======  =====================================
``device_kernel<...DeviceGemmFp4GemmSm100_     live    Both GEMMs of every call (all smoke
float_128_128_256_1_1_1_1SM::Sm10x11xOnly<             and throughput shapes): 128x128x256
GemmUniversal<...BlockScaled<5,2,2,...>,               1-SM tcgen05 block-scaled MMA, 5
Sm100TmaWarpSpecialized<4,2,32,...>,                   stages, dynamic cluster (1,1,1), CLC
PersistentScheduler>>>`` (sm_100a)                     persistent scheduler, FP32 output.
                                                       Workload ``nvfp4_dual_gemm_gemm``.
``pack_scales`` (sm_100a)                      live    Four launches per call (SFA twice,
                                                       SFB1, SFB2): relayouts the logical
                                                       scales into CUTLASS's swizzled
                                                       Sm1xxBlkScaledConfig layout. Workload
                                                       ``nvfp4_dual_gemm_pack_scales``.
``silu_product`` (sm_100a)                     live    One launch per call: FP32 h1, h2 ->
                                                       FP16. Workload
                                                       ``nvfp4_dual_gemm_silu_product``.
``device_kernel<...DeviceGemmFp4GemmSm100_     dead    The shared template instantiates the
half_128_128_256_1_1_1_1SM...>`` (sm_100a)             FP16-output GEMM for nvfp4_gemm /
                                                       nvfp4_group_gemm; the dual GEMM
                                                       (``WORKLOAD_KIND == 1``) never calls
                                                       it (both GEMMs write FP32
                                                       intermediates). Not instantiated.
``CUBIN/*__{00ea3d23,7a383261,93ac03c7,        dup     Four exports of successive builds,
e23966ea}*.cubin``                                     each with the same four kernels.
=============================================  ======  =====================================

``pack_scales`` and ``silu_product`` are plain CUDA (no sm_100a features) and
are also built for sm_86 so they run natively on the RTX 3090 as glue-kernel
checks; that says nothing about the sm_100a GEMM. The GEMM workload's inputs
are already in the kernel-native packed scale layout (``get_inputs`` uses the
torch version of ``pack_scales``), so benchmarking the GEMM alone excludes the
relayout that the previous package timed; one previous dual-GEMM call is
``4 x pack_scales + 2 x gemm + silu_product`` (the second SFA pack being
redundant).

One kernel per cubin
--------------------

``kernels/nvfp4_dual_gemm_sm100.cuh`` reproduces the device part of
FlashInfer's ``INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128, 128, 256, 1,
1, 1, _1SM)`` verbatim (same namespace and struct name, hence the same mangled
symbol as the previous cubin) and ``kernels/nvfp4_dual_gemm_gemm.cu`` holds
exactly one explicit ``cutlass::device_kernel<GemmKernel>`` instantiation and
no host code. ``kernels/pack_scales.cu`` and ``kernels/silu_product.cu`` hold
the previous glue kernels verbatim. Each TU is compiled with ``nvcc -cubin
-lineinfo`` for a single ``-gencode``; ``compile()`` checks architecture,
kernel count, exact symbol and parameter sizes.

Host-side setup without native code
-----------------------------------

The GEMM takes one by-value ``GemmKernel::Params`` (3072 bytes: nine TMA
descriptors (A, B, SFA, SFB, their dynamic-cluster fallbacks, D; C is void
and stays zero), problem shape, scale-factor layouts, fusion params, tile
scheduler and hardware info). ``kernels/nvfp4_dual_gemm_probe.cu`` is a
host-only program built and run here, at compile time only. Its ``layout``
mode records offsets/sizes of every field plus the launch constants in the
sidecar ``cubins/sm_100a/nvfp4_dual_gemm_gemm.json``; its ``params`` mode
builds the Arguments exactly as FlashInfer's ``prepareGemmArgs_*`` and runs
CUTLASS's own ``to_underlying_arguments``/``get_grid_shape`` for several
problems (fake aligned addresses; ``cuTensorMapEncodeTiled`` and
``cudaDriverGetVersion`` interposed), recording Params bytes and every encode
call in ``nvfp4_dual_gemm_gemm.fixtures.json`` (regime edges for both driver
versions plus every case shape of the GEMM workload, ``case_problems``).
Params are rebuilt in Python from the sidecar by ``harness/workloads/
nvfp4_gemm.py``'s ``CutlassFp4Gemm`` (shared with nvfp4_gemm, the FP16-output
instantiation: same field offsets, the probes only name the cute leaves
differently); ``tests/test_nvfp4_dual_gemm.py`` checks it byte-for-byte
against the fixtures.

``pack_scales`` is the one glue kernel shared by the three NVFP4 packages'
previous pipelines; nvfp4_gemm and nvfp4_group_gemm fold it into input
generation (``quantization.to_blocked_scales``) and its single cubin is the
one registered here.

Upstream tests and harness cases
--------------------------------

Every ``resources/nvfp4_dual_gemm/task.yml`` test line (smoke cases
``task_*`` of all three stages, with the task generator's distributions --
0xBB-masked E2M1, U[0, 1) scales -- and seed; the duplicated line once) and
benchmark line (throughput) is a case
(``tests/test_nvfp4_dual_gemm.py::test_upstream_parametrizations_are_cases``).
FlashInfer has no dual-GEMM test.

Nothing in this module runs at benchmark time.
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
# Upstream files: the template whose struct the kernel header reproduces, and
# the header it includes (for flashinfer::arch::is_major_v).
FLASHINFER_FILES = {
    "include/flashinfer/gemm/fp4_gemm_template_sm100.h": (
        "868fad7f668ed07a3569e49691126da48f1f070b97085792ce84de3ef8fb4f86"
    ),
    "include/flashinfer/arch_condition.h": (
        "99d6b54df9bd2b9095bfe64401581122542ea65d84aa609e24e2ffc1be522418"
    ),
}
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
CUTLASS_REVISION = "b46b16d003484063bca4ed365e44095c4c6ed633"
CUTLASS_DIR = f"cutlass-{CUTLASS_REVISION}"
# sha256 over (relative path + sha256(content)) of every file in include/.
CUTLASS_INCLUDE_TREE_SHA256 = (
    "def1849ca143942f68179aeb664a6cb9bf46001082d7d4f5c63531ea03866e6d"
)

GENCODE = {
    "sm_86": "-gencode=arch=compute_86,code=sm_86",
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
}
GEMM = "nvfp4_dual_gemm_gemm"
PACK = "nvfp4_dual_gemm_pack_scales"
# Same wording in all three NVFP4 packages' provenance.
PACK_SCALES = (
    "previous-package harness glue (impls/templates/nvfp4.cu, not FlashInfer "
    "code): logical -> blocked scale relayout. Its single cubin is kept once, "
    "registered as nvfp4_dual_gemm_pack_scales (a stage of the previous dual "
    "GEMM pipeline); every GEMM workload takes scales in the kernel-native "
    "blocked layout (quantization.to_blocked_scales, as the NVIDIA tasks supply "
    "them), so no GEMM timing includes it. Here: live (4 launches per previous "
    "call)."
)
SILU = "nvfp4_dual_gemm_silu_product"
GEMM_HEADER = "kernels/nvfp4_dual_gemm_sm100.cuh"
PROBE_SOURCE = "kernels/nvfp4_dual_gemm_probe.cu"
GEMM_SYMBOL = (
    "_ZN7cutlass13device_kernelIN10flashinfer4gemm50DeviceGemmFp4GemmSm100_float_128_"
    "128_256_1_1_1_1SM12Sm10x11xOnlyINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiii"
    "EEENS5_10collective13CollectiveMmaINS5_46MainloopSm100TmaUmmaWarpSpecializedBlock"
    "ScaledILi5ELi2ELi2ENS9_IJiiNS8_1CILi1EEEEEENS_4arch5Sm100EEENS9_IJNSE_ILi128EEESK"
    "_NSE_ILi256EEEEEENS9_IJNS_12float_e2m1_tENS_13float_ue4m3_tEEEENS9_IJNS9_IJlSF_lE"
    "EENS8_6LayoutINS9_IJNS9_IJNS9_IJNSE_ILi32EEENSE_ILi4EEEEEEiEEENS9_IJNS9_IJNSE_ILi"
    "16EEEST_EEEiEEENS9_IJSF_iEEEEEENS9_IJSY_NS9_IJNS9_IJNSE_ILi0EEESF_EEENSE_ILi512EE"
    "EEEENS9_IJS11_iEEEEEEEEEEESP_S18_NS8_8TiledMMAINS8_8MMA_AtomIJNS8_17SM100_MMA_MXF"
    "4_SSISN_SN_fSO_Li128ELi128ELi16ELNS8_4UMMA5MajorE0ELS1D_0ELNS1C_7ScaleInE0ELS1E_0"
    "EEEEEENSR_INS9_IJSF_SF_SF_EEENS9_IJS11_S11_S11_EEEEENS9_IJNS8_10UnderscoreES1K_S1"
    "K_EEEEENS9_IJNS8_23SM90_TMA_LOAD_MULTICASTES1N_EEENS9_IJNS8_14ComposedLayoutINS8_"
    "7SwizzleILi3ELi4ELi3EEENS8_18smem_ptr_flag_bitsILi4EEENSR_INS9_IJNSE_ILi8EEESL_EE"
    "ENS9_IJSL_SF_EEEEEEENSR_INS9_IJNS9_IJNS9_IJSU_SF_EEESX_EEESF_NS9_IJSF_ST_EEEEEENS"
    "9_IJNS9_IJNS9_IJSX_S13_EEES12_EEES11_NS9_IJST_S13_EEEEEEEEEEEvNS8_8identityES1O_S"
    "28_vS29_EENS_8epilogue10collective18CollectiveEpilogueINS2B_23Sm100TmaWarpSpecial"
    "izedILi4ELi2ELi32ELb0ELb1EEEJSM_NS9_IJNSR_ISK_SF_EENSR_ISS_SF_EEEEEvSQ_fSQ_NS2B_6"
    "fusion15FusionCallbacksIS2F_NS2J_17LinearCombinationIffvfLNS_15FloatRoundStyleE2E"
    "EESM_S2I_JEEENS8_5SM1004TMEM4LOAD26SM100_TMEM_LOAD_32dp32b32xENS8_13SM90_TMA_LOAD"
    "ENS1P_IS1R_NS1S_ILi32EEENSR_INS9_IJS1U_SS_EEENS9_IJSS_SF_EEEEEEENS8_39AutoVectori"
    "zingCopyWithAssumedAlignmentILi128EEENS8_14SM90_TMA_STOREES2Y_S30_S30_EEENS5_19Pe"
    "rsistentSchedulerEvEEEEEEvNT_6ParamsE"
)


class KernelSpec(NamedTuple):
    workload: str
    arches: tuple[str, ...]
    source: str  # relative to the package
    symbol: str
    param_sizes: tuple[int, ...] | None  # None: checked against the probe
    adaptation: str


SPECS: tuple[KernelSpec, ...] = (
    KernelSpec(
        GEMM,
        ("sm_100a",),
        "kernels/nvfp4_dual_gemm_gemm.cu",
        GEMM_SYMBOL,
        None,
        "Device part of FlashInfer's INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(float, 128,"
        " 128, 256, 1, 1, 1, _1SM) verbatim; one explicit device_kernel instantiation.",
    ),
    KernelSpec(
        PACK,
        ("sm_86", "sm_100a"),
        "kernels/pack_scales.cu",
        "_Z11pack_scalesPKhPhiii",
        (8, 8, 4, 4, 4),
        "Previous package's pack_scales glue kernel (impls/templates/nvfp4.cu), verbatim.",
    ),
    KernelSpec(
        SILU,
        ("sm_86", "sm_100a"),
        "kernels/silu_product.cu",
        "_Z12silu_productPKfS0_P6__halfi",
        (8, 8, 8, 4),
        "Previous package's silu_product glue kernel (impls/templates/nvfp4.cu),"
        " verbatim.",
    ),
)

# Problems (m, n, k, l) whose Params CUTLASS builds at compile time (test
# fixtures) for both driver versions: partial M/N/K tiles, batches, both
# rasterization orders (incl. a transposed grid) and tiny tensors that trigger
# CUTLASS's driver-version-dependent descriptor fix-up. Every case shape of the
# GEMM workload is added for CASE_DRIVER_VERSION (case_problems).
FIXTURE_PROBLEMS = (
    (128, 128, 128, 1),
    (128, 256, 256, 2),
    (256, 128, 512, 1),
    (256, 4096, 7168, 1),
    (512, 4096, 7168, 1),
    (256, 3072, 4096, 1),
    (512, 3072, 7168, 1),
    (200, 264, 512, 2),  # partial M and N tiles, batched
    (300, 136, 384, 3),  # partial tiles, AlongN, k not a multiple of 256
    (1, 8, 128, 1),
    (129, 1032, 256, 1),
    (4096, 128, 64, 1),  # AlongN with a transposed grid, one K block
    (64, 68, 160, 1),  # k / 16 not a multiple of 4 (padded scale columns)
)
FIXTURE_DRIVER_VERSIONS = (13020, 13010)  # without / with the bit-21 fix-up
CASE_DRIVER_VERSION = 13010
# Fake device addresses (aligned, bit 21 set so that the fix-up is visible in
# the descriptor stand-in; never dereferenced).
FIXTURE_POINTERS = {
    "a": 0x7F0000200000,
    "b": 0x7F1000200000,
    "sfa": 0x7F2000200000,
    "sfb": 0x7F3000200000,
    "d": 0x7F4000200000,
    "alpha": 0x7F5000000040,
}
# Params slot -> operand whose descriptor it holds, in CUTLASS's encode order.
DESCRIPTOR_SLOTS = {
    "mainloop.tma_load_a": "a",
    "mainloop.tma_load_b": "b",
    "mainloop.tma_load_a_fallback": "a",
    "mainloop.tma_load_b_fallback": "b",
    "mainloop.tma_load_sfa": "sfa",
    "mainloop.tma_load_sfb": "sfb",
    "mainloop.tma_load_sfa_fallback": "sfa",
    "mainloop.tma_load_sfb_fallback": "sfb",
    "epilogue.tma_store_d": "d",
}
# Never encoded (ElementC = void): default-constructed, all zero.
ZERO_SLOTS = ("epilogue.tma_load_c",)
THREAD_DATA = (
    "alpha",
    "beta",
    "alpha_ptr",
    "beta_ptr",
    "alpha_stride_l",
    "beta_stride_l",
)
STATIC_TMA_KEYS = (
    "data_type",
    "rank",
    "box",
    "element_strides",
    "interleave",
    "swizzle",
    "l2_promotion",
    "oob_fill",
)


def case_problems(workload: str) -> list[tuple[int, int, int, int]]:
    """Every (m, n, k, l) of the registered workload's cases (smoke, task
    tests, throughput), so that each case shape has a CUTLASS-built fixture."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness import get_workload

    cases = get_workload(workload)(None, device="cpu").get_cases()
    return sorted({tuple(c.params[key] for key in "mnkl") for c in cases})


def write_fixtures(path: Path, header: dict[str, Any], problems: list[Any]) -> None:
    """Fixture file with one compact JSON line per problem."""
    lines = ",\n".join(json.dumps(p, separators=(",", ":")) for p in problems)
    head = json.dumps(header, indent=2)[:-2]
    path.write_text(f'{head},\n  "problems": [\n{lines}\n  ]\n}}\n')


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


def _relative(path: Path) -> str:
    """``path`` relative to the repository root when inside it (compilers run
    with ``cwd=ROOT``), so recorded commands are portable."""
    path = Path(os.path.abspath(path))  # keep symlinks (resources/ may hold some)
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _cubin_info(image: bytes) -> tuple[str, list[str], list[Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness.workload import cubin_info

    return cubin_info(image)


def fake_tensor_map(call: dict[str, Any]) -> bytes:
    """The probe's deterministic 128-byte CUtensorMap stand-in (see the probe)."""
    out = bytearray(128)
    out[0:6] = bytes(
        call[key]
        for key in (
            "data_type",
            "rank",
            "interleave",
            "swizzle",
            "l2_promotion",
            "oob_fill",
        )
    )
    out[8:16] = call["address"].to_bytes(8, "little")
    for i, value in enumerate(call["dims"]):
        out[16 + 8 * i : 24 + 8 * i] = value.to_bytes(8, "little")
    for i, value in enumerate(call["strides"]):
        out[56 + 8 * i : 64 + 8 * i] = value.to_bytes(8, "little")
    for i, value in enumerate(call["box"]):
        out[88 + 4 * i : 92 + 4 * i] = value.to_bytes(4, "little")
    for i, value in enumerate(call["element_strides"]):
        out[108 + 4 * i : 112 + 4 * i] = value.to_bytes(4, "little")
    return bytes(out)


class ImplCompiler:
    """Builds ``cubins/<arch>/<workload>.cubin`` (+ sidecars), merges ``kernels.json``."""

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
            raise ValueError(
                f"nvfp4_dual_gemm supports {self.supported_arches}, not {arch}"
            )
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = list(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.cutlass = self.resources / CUTLASS_DIR
        self.flashinfer = self.resources / FLASHINFER_DIR

    @property
    def specs(self) -> list[KernelSpec]:
        return [spec for spec in SPECS if self.arch in spec.arches]

    # -- pinned sources ---------------------------------------------------------

    def verify_sources(self) -> None:
        for name, digest in FLASHINFER_FILES.items():
            path = self.flashinfer / name
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(
                    f"{path} is missing or differs from FlashInfer {FLASHINFER_REVISION}"
                )
        tree = _cutlass_tree_sha256(self.cutlass)
        if tree != CUTLASS_INCLUDE_TREE_SHA256:
            raise ValueError(f"unexpected CUTLASS include tree hash {tree}")

    def _common_flags(self, gemm: bool) -> list[str]:
        flags = [
            "-std=c++17",
            self.optimization,
            "-lineinfo",
            GENCODE[self.arch],
            "--expt-relaxed-constexpr",
        ]
        if gemm:
            flags += [
                "-I" + _relative(self.cutlass / "include"),
                "-I" + _relative(self.cutlass / "tools/util/include"),
                "-I" + _relative(self.flashinfer / "include"),
            ]
        return flags + self.flags

    def kernel_command(self, spec: KernelSpec, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-cubin",
            *self._common_flags(spec.workload == GEMM),
            _relative(PACKAGE / spec.source),
            "-o",
            str(output),
        ]

    def probe_command(self, output: Path) -> list[str]:
        # Host executable; the same flags guarantee the same Params layout as the
        # kernel's device compilation. The shared cudart lets the probe
        # interpose cudaDriverGetVersion.
        return [
            self.nvcc,
            *self._common_flags(True),
            "--cudart=shared",
            "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL",
            _relative(PACKAGE / PROBE_SOURCE),
            "-o",
            str(output),
        ]

    @staticmethod
    def _portable(command: list[str]) -> list[str]:
        """The command with repository paths made relative to the repo root."""
        root = str(ROOT) + os.sep
        return [part.replace(root, "") for part in command]

    # -- build -------------------------------------------------------------------

    def compile(self) -> dict[str, str]:
        if any(spec.workload == GEMM for spec in self.specs):
            self.verify_sources()
        out_dir = PACKAGE / "cubins" / self.arch
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        nvcc_version = version.strip().splitlines()[-1]
        Fixtures = list[dict[str, Any]] | None
        staged: dict[str, tuple[Path, bytes, dict[str, Any], Fixtures]] = {}
        with tempfile.TemporaryDirectory(prefix="nvfp4_dual_gemm-") as tmp:
            for spec in self.specs:
                cubin = Path(tmp) / f"{spec.workload}.cubin"
                subprocess.run(self.kernel_command(spec, cubin), check=True, cwd=ROOT)
                image = cubin.read_bytes()
                arch, names, params = _cubin_info(image)
                if arch != self.arch:
                    raise ValueError(f"{spec.source}: cubin declares {arch}")
                if names != [spec.symbol]:
                    raise ValueError(f"{spec.source}: expected {spec.symbol}, {names}")
                sizes = tuple(p.size for p in params)
                sidecar: dict[str, Any] = {
                    "workload": spec.workload,
                    "arch": self.arch,
                    "cubin": cubin.name,
                    "cubin_sha256": hashlib.sha256(image).hexdigest(),
                    "symbol": spec.symbol,
                    "kernel_params": [
                        {"offset": p.offset, "size": p.size} for p in params
                    ],
                    "source": spec.source,
                    "source_sha256": _sha256(PACKAGE / spec.source),
                    "adaptation": spec.adaptation,
                }
                fixtures: Fixtures = None
                if spec.param_sizes is not None:
                    if sizes != spec.param_sizes:
                        raise ValueError(f"{spec.source}: parameter sizes {sizes}")
                    sidecar["block"] = [256, 1, 1]  # the previous launches
                else:
                    layout, fixtures = self._probe(Path(tmp), params)
                    sidecar.update(layout)
                staged[spec.workload] = (cubin, image, sidecar, fixtures)

            # Only now replace the previous artifacts.
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            mapping: dict[str, str] = {}
            records: dict[str, Any] = {}
            for spec in self.specs:
                cubin, image, sidecar, fixtures = staged[spec.workload]
                target = out_dir / cubin.name
                shutil.move(cubin, target)
                relative = target.relative_to(PACKAGE).as_posix()
                mapping[spec.workload] = relative
                self._write_json(out_dir / f"{spec.workload}.json", sidecar)
                if fixtures is not None:
                    write_fixtures(
                        out_dir / f"{spec.workload}.fixtures.json",
                        {
                            "workload": spec.workload,
                            "arch": self.arch,
                            "tensor_map_encoder": "fake (nvfp4_dual_gemm_probe.cu "
                            "fake_tensor_map)",
                        },
                        fixtures,
                    )
                record = {
                    "cubin": relative,
                    "symbol": spec.symbol,
                    "sha256": sidecar["cubin_sha256"],
                    "source": spec.source,
                    "source_sha256": sidecar["source_sha256"],
                    # Compiled to a staging file, then moved to ``cubin``.
                    "command": self._portable(self.kernel_command(spec, target)),
                }
                if spec.workload == GEMM:
                    record["header"] = GEMM_HEADER
                    record["header_sha256"] = _sha256(PACKAGE / GEMM_HEADER)
                    record["probe"] = PROBE_SOURCE
                    record["probe_sha256"] = _sha256(PACKAGE / PROBE_SOURCE)
                records[spec.workload] = record
        self._merge_manifest(mapping)
        self._write_provenance(records, nvcc_version)
        return mapping

    def _probe(
        self, tmp: Path, params: list[Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        probe = tmp / "nvfp4_dual_gemm_probe"
        subprocess.run(self.probe_command(probe), check=True, cwd=ROOT)
        layout = json.loads(subprocess.check_output([str(probe), "layout"], text=True))
        if [(p.offset, p.size) for p in params] != [(0, layout["params_size"])]:
            raise ValueError(
                f"kernel parameters {params} do not match sizeof(Params) "
                f"{layout['params_size']}"
            )
        runs = [(p, d) for p in FIXTURE_PROBLEMS for d in FIXTURE_DRIVER_VERSIONS]
        runs += [
            (p, CASE_DRIVER_VERSION)
            for p in case_problems(GEMM)
            if p not in FIXTURE_PROBLEMS
        ]
        fixtures = [
            self._run_probe(probe, layout, problem, driver) for problem, driver in runs
        ]
        tensor_maps = self._static_tensor_maps(layout, fixtures)
        self._check_padding(layout, fixtures)
        sidecar = {
            "params_size": layout["params_size"],
            "params_align": layout["params_align"],
            "fields": layout["fields"],
            "constants": layout["constants"],
            "tensor_maps": tensor_maps,
            "probe": self._portable(self.probe_command(Path(probe.name))),
        }
        return sidecar, fixtures

    def _run_probe(
        self,
        probe: Path,
        layout: dict[str, Any],
        problem: tuple[int, int, int, int],
        driver: int,
    ) -> dict[str, Any]:
        p = FIXTURE_POINTERS
        argv = [str(probe), "params", *map(str, problem)]
        argv += [hex(p[key]) for key in ("a", "b", "sfa", "sfb", "d", "alpha")]
        argv += [str(driver)]
        result = json.loads(subprocess.check_output(argv, text=True))
        if not result["can_implement"] or result["workspace_size"] != 0:
            raise ValueError(f"unexpected CUTLASS arguments for {problem}: {result}")
        # Bytes the kernel never reads: those the probe found indeterminate (they
        # differ between 0x00/0xff stack poisoning), the fusion params' padding
        # and each TMA atom's bytes outside its descriptor (its aux strides are
        # static, i.e. empty, and the atom is padded to 128-byte alignment).
        holes = set(result.pop("indeterminate_bytes"))
        fields = layout["fields"]
        thread = fields["epilogue.thread"]
        holes |= set(range(thread["offset"], thread["offset"] + thread["size"]))
        for name in THREAD_DATA:
            member = fields["epilogue.thread." + name]
            holes -= set(range(member["offset"], member["offset"] + member["size"]))
        for slot in (*DESCRIPTOR_SLOTS, *ZERO_SLOTS):
            if fields[slot + ".aux_g_stride"]["size"]:
                raise ValueError(f"{slot}: dynamic TMA aux strides are not supported")
            atom, desc = fields[slot], fields[slot + ".desc"]
            holes |= set(range(atom["offset"], atom["offset"] + atom["size"]))
            holes -= set(range(desc["offset"], desc["offset"] + desc["size"]))
        ranges: list[list[int]] = []  # half-open [start, end)
        for index in sorted(holes):
            if ranges and ranges[-1][1] == index:
                ranges[-1][1] += 1
            else:
                ranges.append([index, index + 1])
        result["padding"] = ranges
        return result

    @staticmethod
    def _static_tensor_maps(
        layout: dict[str, Any], fixtures: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Static encode parameters of each operand's descriptor, cross-checked
        against every recorded encode call (matched to its Params slot)."""
        static: dict[str, dict[str, Any]] = {}
        fields = layout["fields"]
        for fixture in fixtures:
            params = bytes.fromhex(fixture["params_hex"])
            calls = fixture["tensor_maps"]
            if len(calls) != len(DESCRIPTOR_SLOTS):
                raise ValueError("unexpected number of cuTensorMapEncodeTiled calls")
            for (slot, operand), call in zip(DESCRIPTOR_SLOTS.items(), calls):
                offset = fields[slot + ".desc"]["offset"]
                descriptor = params[offset : offset + 128]
                expected = fake_tensor_map(call)
                # The bit-21 fix-up may have changed the stored descriptor.
                word = int.from_bytes(expected[8:16], "little") & ~(1 << 21)
                patched = expected[:8] + word.to_bytes(8, "little") + expected[16:]
                if descriptor not in (expected, patched):
                    raise ValueError(f"{slot}: descriptor does not match encode call")
                values = {key: call[key] for key in STATIC_TMA_KEYS}
                if static.setdefault(operand, values) != values:
                    raise ValueError(f"{operand}: static TMA parameters vary")
            for slot in ZERO_SLOTS:
                desc = fields[slot + ".desc"]
                if any(params[desc["offset"] : desc["offset"] + desc["size"]]):
                    raise ValueError(f"{slot}: expected an all-zero descriptor")
        return static

    @staticmethod
    def _check_padding(layout: dict[str, Any], fixtures: list[dict[str, Any]]) -> None:
        """Indeterminate bytes must be identical everywhere and never overlap a
        leaf field the Python builder writes (aggregate fields contain padding)."""
        padding = {tuple(map(tuple, fixture["padding"])) for fixture in fixtures}
        if len(padding) != 1:
            raise ValueError("indeterminate Params bytes vary between problems")
        holes = {i for start, end in padding.pop() for i in range(start, end)}
        fields = layout["fields"]
        for name, field in fields.items():
            if any(other.startswith(name + ".") for other in fields):
                continue
            if name == "problem_shape":
                continue
            span = set(range(field["offset"], field["offset"] + field["size"]))
            if span & holes:
                raise ValueError(f"field {name} overlaps indeterminate Params bytes")

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2) + "\n")

    def _merge_manifest(self, mapping: dict[str, str]) -> None:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
        manifest[self.arch] = dict(sorted(mapping.items()))
        self._write_json(MANIFEST, dict(sorted(manifest.items())))

    def _write_provenance(self, records: dict[str, Any], nvcc_version: str) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {}) if isinstance(provenance, dict) else {}
        builds[self.arch] = {"nvcc": nvcc_version, "kernels": records}
        provenance = {
            "repository": FLASHINFER_REPOSITORY,
            "revision": FLASHINFER_REVISION,
            "files": FLASHINFER_FILES,
            "license": "LICENSE (FlashInfer, Apache-2.0); CUTLASS_LICENSE (BSD-3-Clause)",
            "cutlass_repository": CUTLASS_REPOSITORY,
            "cutlass_revision": CUTLASS_REVISION,
            "cutlass": f"resources/{CUTLASS_DIR}",
            "cutlass_include_tree_sha256": CUTLASS_INCLUDE_TREE_SHA256,
            "task": "resources/nvfp4_dual_gemm/{task.yml,reference.py} (gpu-mode "
            "reference-kernels nvidia/nvfp4_dual_gemm)",
            "scope": "silu(A @ B1^T) * (A @ B2^T) with NVFP4 inputs as the previous "
            "package ran it: pack_scales x4 -> FlashInfer genericFp4GemmKernelLauncher"
            "<float,128,128,256,1,1,1,_1SM> x2 (FP32 intermediates) -> silu_product. "
            "One kernel per cubin and per workload; GEMM Params built in Python from "
            "the probe's layout (alpha_ptr -> the alpha input, 1 in the task, "
            "beta=0, C=void) by the builder shared with nvfp4_gemm (the FP16-output "
            "instantiation of the same template).",
            "workloads": {
                GEMM: "sm_100a: both GEMMs (same kernel, different B/SFB/output)",
                PACK: "sm_86 + sm_100a: " + PACK_SCALES,
                SILU: "sm_86 + sm_100a: half(silu(h1) * h2)",
            },
            "adaptations": [
                (
                    "GEMM kernel type copied from the device part of FlashInfer's "
                    "INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER macro (same symbol); host "
                    "setup (prepareGemmArgs_* -> Params, grid, smem, launch "
                    "attributes) reimplemented in Python and checked byte-for-byte "
                    "against CUTLASS's to_underlying_arguments at compile time."
                ),
                (
                    "No workspace: CUTLASS reports 0 bytes for this kernel (the "
                    "previous package reserved 32 MiB that was never used)."
                ),
                (
                    "The GEMM is launched as upstream GemmUniversalAdapter::run does: "
                    "cluster dimension (1,1,1) and programmatic stream serialization "
                    "(enablePDL=true); the preferred-cluster attribute it adds equals "
                    "the cluster dimension and is omitted."
                ),
                (
                    "The previous fixed scratch buffers (packed scales, FP32 "
                    "intermediates, device alpha) are torch tensors allocated per "
                    "call, so their capacity limits (8 MiB per packed scale tensor, "
                    "4 Mi intermediate elements) no longer apply; the int32 index "
                    "ranges of the kernels and of FlashInfer's argument setup are "
                    "checked instead."
                ),
            ],
            "builds": dict(sorted(builds.items())),
            "upstream_coverage": {
                "cases": "every resources/nvfp4_dual_gemm/task.yml test line (smoke "
                "cases task_* of all three stages, task distribution and seed; the "
                "duplicated line once) and benchmark line (throughput)",
                "not_served": "none: FlashInfer has no dual-GEMM test; the GEMM "
                "template's mm_fp4 tests are covered by nvfp4_gemm",
                "check": "tests/test_nvfp4_dual_gemm.py::"
                "test_upstream_parametrizations_are_cases",
            },
            "excluded_kernels": {
                "DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM": "FP16-output "
                "GEMM of the shared template, never called by the dual GEMM",
                "CUBIN/*.cubin": "stale exports of the previous package; each holds the "
                "same four kernels",
            },
        }
        self._write_json(PROVENANCE, provenance)

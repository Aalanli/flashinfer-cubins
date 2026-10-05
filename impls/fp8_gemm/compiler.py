"""Compile the fp8_gemm package's live CUTLASS kernels into single-kernel cubins.

The package is FlashInfer's groupwise-scaled FP8 GEMM on Blackwell
(``gemm_fp8_nt_groupwise(..., backend="cutlass")``, pinned revision below):
``D[m, n] = (A[m, k] * SFA) @ (B[n, k] * SFB)^T`` in FP32, stored as BF16, with
A scales per (1, 128) and B scales per (128, 128) blocks, both K-major.

Upstream dispatch
-----------------

``csrc/gemm_groupwise_sm100.cu`` (``CutlassGemmGroupwiseScaledSM100``) picks
one of two kernel templates of ``include/flashinfer/gemm/gemm_groupwise_sm100.cuh``
per call, for scale granularity (1, 128, 128)::

    if (SCALE_GRANULARITY_M == 1 && m <= 32)
        CutlassGroupwiseScaledGEMMSM100LowLatency<1, 128, 128, K, MmaSM>  -> fp8_gemm_small_m
    else
        CutlassGroupwiseScaledGEMMSM100<1, 128, 128, K, MmaSM>            -> fp8_gemm

Each workload of this package wraps exactly the kernel upstream dispatches to
for the shapes it serves (``FP8GEMM_SMALL_M_MAX_M = 32``).

Kernel inventory and the live/dead decision
-------------------------------------------

=================================================  ======  ===================================
kernel                                             status  reason
=================================================  ======  ===================================
``device_kernel<GemmUniversal<..., MainloopSm100   live    ``CutlassGroupwiseScaledGEMMSM100
TmaUmmaWarpSpecializedBlockwiseScaling<5,5,4,             <1,128,128,true,1>``: 128x128x128
1x1x1>, ..., Sm100TmaWarpSpecialized<4,2,32>...>>``       MMA tile, static 1x1x1 cluster, CLC
                                                          tile scheduler, 5 stages. Upstream's
                                                          kernel for m > 32. Workload
                                                          ``fp8_gemm``.
``device_kernel<GemmUniversal<..., MainloopSm100   live    ``CutlassGroupwiseScaledGEMMSM100
TmaUmmaWarpSpecializedBlockwiseScaling<12,33,32,          LowLatency<1,128,128,true,1>``:
(int,int,1)>, ..., Sm100TmaWarpSpecialized<1,1,           swap-AB (computes D^T = B A^T),
16,false,true>...>>``                                     128x16x128 MMA tile, dynamic cluster
                                                          launched as 1x1x1, no C source, CLC
                                                          tile scheduler, 12 stages.
                                                          Upstream's kernel for m <= 32.
                                                          Workload ``fp8_gemm_small_m``.
``impls/fp8_gemm/CUBIN/*__2e40f877*.cubin``,       dead    Not distinct kernels: all three hold
``*__5646fde6*.cubin``, ``*__fe565169*.cubin``             the ``fp8_gemm`` symbol (stale
                                                           exports of successive builds of the
                                                           previous package; its CUBIN
                                                           manifest lists only 2e40f877).
``CutlassGroupwiseScaledGEMMSM100<..., MmaSM=2>``  absent  Upstream 2-SM (256x128, 2x1x1
                                                           cluster) variant, selected only by
                                                           FlashInfer's non-default
                                                           ``mma_sm=2`` option (m > 32).
``ScaleMajorK=false`` / (128,128,128) scales       absent  Other upstream template options;
                                                           (128,128,128) never takes the
                                                           low-latency path upstream.
=================================================  ======  ===================================

The previous package (``implementation_sm_100a.cpp``) dispatched every shape,
also m <= 32, to the ``fp8_gemm`` kernel; the low-latency kernel is added
here so that each benchmarked shape runs the kernel upstream would run.
``MmaSM`` is a template parameter of the low-latency function but unused by
its kernel type.

One kernel per cubin
--------------------

``kernels/<workload>_sm100.cuh`` reproduces the kernel type of the FlashInfer
function template (its type aliases verbatim, template parameters fixed) and
``kernels/<workload>_sm100.cu`` contains exactly one explicit instantiation of
``cutlass::device_kernel<GemmKernel>`` and no host code. ``nvcc -cubin
-lineinfo`` for a single ``-gencode`` therefore emits a genuine single-kernel
cubin; ``compile()`` checks its architecture, kernel count, symbol and
parameter size. Rebuilt cubins are not bit-identical (their SASS is): nvcc
records absolute directories in the line tables and a per-run hash in
internal-linkage symbol names. Commands use repo-relative paths (run from the
repository root) so the recorded provenance is portable.

Host-side setup without native code
-----------------------------------

Each kernel takes one by-value ``GemmKernel::Params`` (2048 bytes: six TMA
descriptor slots, problem shape, scale-factor pointers/layouts, fusion params,
tile-scheduler params and hardware info). ``kernels/fp8_gemm_probe.cu`` is a
host-only program built and run here, at compile time only. Its ``layout``
mode records offsets and sizes of every field plus the launch constants in the
sidecar ``cubins/<arch>/<workload>.json``; its ``params`` mode runs the
upstream host function's Arguments construction (for ``fp8_gemm_small_m``
including the (m, n)/(A, B)/(SFA, SFB) swap and ``make_kernel_hardware_info``)
and CUTLASS's own ``to_underlying_arguments``/``get_grid_shape`` for several
problems (fake, aligned device addresses; ``cuTensorMapEncodeTiled``,
``cudaDriverGetVersion`` and the hardware queries interposed, see the probe's
header) and records the Params bytes, every descriptor's encode arguments,
the hardware queries and the launch cluster in
``cubins/<arch>/<workload>.fixtures.json``. The static TMA parameters (data
type, box, swizzle, L2 promotion, ...), the occupancy query and the launch
cluster in the sidecar are taken from those recordings.
``harness/workloads/fp8_gemm.py`` rebuilds Params in Python from the sidecar
and is checked byte-for-byte against the fixtures by ``tests/test_fp8_gemm.py``.
Fixtures cover regime edges (``KernelSpec.problems``, both driver versions)
and every case shape of the workload (``case_problems``, read from the
registered workload at compile time, driver version with the fix-up).

Upstream tests and harness cases
--------------------------------

Every upstream parametrization a kernel here serves is one of its cases
(``tests/test_fp8_gemm.py::test_upstream_parametrizations_are_cases``):

* FlashInfer ``tests/gemm/test_groupwise_scaled_gemm_fp8.py``:
  ``test_fp8_groupwise_gemm`` with the cutlass backend and
  ``scale_major_mode="K"``, all 125 (m, n, k) in {128, 256, 512, 4096,
  8192}^3 (m > 32: ``fp8_gemm`` smoke cases ``groupwise_*``), and
  ``test_fp8_groupwise_gemm_small_batch_size`` ("K"), m in {1, 4, 16} x n in
  {128, 256}, k = 256 (``fp8_gemm_small_m`` cases ``small_batch_*``).
* DeepGEMM ``tests/generators.py::enumerate_normal(float8_e4m3fn)`` with the
  legacy (1,128)/(128,128) quantization, forward, BF16 output: m in {1, 128,
  4096} x 7 (n, k) (throughput cases ``deepgemm_*``, m = 1 on
  ``fp8_gemm_small_m``).

Not served (other kernels or a restricted range): the small-batch test's
m = 32 cases (``fp8_gemm_small_m`` is restricted to m <= 16 after a race on
a B200; they are not moved to ``fp8_gemm``, which upstream never runs for
m <= 32), ``scale_major_mode="MN"`` (ScaleMajorK=false template),
``test_fp8_blockscale_gemm`` ((128,128,128) granularity), the trtllm/cuTile
backends and group-GEMM tests, and DeepGEMM's BF16-accumulation, FP4/UE8M0
and MN-major (backward) configurations.

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
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"

FLASHINFER_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer"
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
FLASHINFER_DIR = f"flashinfer-{FLASHINFER_REVISION}"
# The header whose type definitions kernels/*_sm100.cuh reproduce, and the
# dispatcher that selects between its two kernels.
FLASHINFER_FILES = {
    "include/flashinfer/gemm/gemm_groupwise_sm100.cuh": (
        "2970418dcbab8f6779ec777d9d579c454ab737103279c45b456e6beb8dc49b18"
    ),
    "csrc/gemm_groupwise_sm100.cu": (
        "d1f66c6c7c09ae23a40868bd9e10dfd5bbf06a2a84f0f8ac7143864deb5f9457"
    ),
}
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
CUTLASS_REVISION = "b46b16d003484063bca4ed365e44095c4c6ed633"
CUTLASS_DIR = f"cutlass-{CUTLASS_REVISION}"
# sha256 over (relative path + sha256(content)) of every file in include/.
CUTLASS_INCLUDE_TREE_SHA256 = (
    "def1849ca143942f68179aeb664a6cb9bf46001082d7d4f5c63531ea03866e6d"
)

# csrc/gemm_groupwise_sm100.cu: ``can_use_small_batch && m <= 32``.
SMALL_M_MAX_M = 32

GENCODE = {"sm_100a": "-gencode=arch=compute_100a,code=sm_100a"}
PROBE_SOURCE = "kernels/fp8_gemm_probe.cu"
_MANGLED_PREFIX = (
    "_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEE"
    "NS1_10collective13CollectiveMmaINS1_51"
    "MainloopSm100TmaUmmaWarpSpecializedBlockwiseScalingI"
)

# Fake device addresses (16-byte aligned, bit 21 set so that the fix-up is
# visible in the descriptor stand-in, never dereferenced).
FIXTURE_POINTERS = {
    "a": 0x7F0000200000,
    "b": 0x7F1000200000,
    "sfa": 0x7F2000000100,
    "sfb": 0x7F3000000200,
    "d": 0x7F4000200000,
}
# (driver version, SM count, max active clusters) of each fixture run: the
# driver versions without / with CUTLASS's bit-21 descriptor fix-up; the
# hardware values only reach Params through fp8_gemm_small_m's
# make_kernel_hardware_info (-1: the occupancy query fails, CUTLASS stores 0).
FIXTURE_RUNS = ((13020, 148, -1), (13010, 160, 37))
# The run used for the workloads' case shapes (the one with the fix-up).
CASE_RUN = FIXTURE_RUNS[1]
TMA_SLOTS = (
    "mainloop.tma_load_a",
    "mainloop.tma_load_b",
    "mainloop.tma_load_a_fallback",
    "mainloop.tma_load_b_fallback",
    "epilogue.tma_load_c",
    "epilogue.tma_store_d",
)
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


class KernelSpec(NamedTuple):
    """One workload's kernel: sources, expected symbol and Params fixtures.

    (A NamedTuple, not a dataclass: compile_kernels.py loads this module by
    path without registering it in sys.modules.)
    """

    workload: str
    upstream: str
    source: str
    header: str
    symbol_prefix: str
    # Params slot -> kernel operand whose descriptor it holds (encoded slots).
    descriptor_slots: dict[str, str]
    # Original (unswapped) problems (m, n, k) whose Params CUTLASS builds for
    # every FIXTURE_RUNS entry (regime edges); every case shape of the
    # workload is added for CASE_RUN (``case_problems``).
    problems: tuple[tuple[int, int, int], ...]


KERNELS = (
    KernelSpec(
        workload="fp8_gemm",
        upstream="CutlassGroupwiseScaledGEMMSM100<1,128,128,ScaleMajorK=true,MmaSM=1>",
        source="kernels/fp8_gemm_sm100.cu",
        header="kernels/fp8_gemm_sm100.cuh",
        symbol_prefix=_MANGLED_PREFIX + "Li5ELi5ELi4E",
        descriptor_slots={
            "mainloop.tma_load_a": "a",
            "mainloop.tma_load_b": "b",
            "mainloop.tma_load_a_fallback": "a",
            "mainloop.tma_load_b_fallback": "b",
            "epilogue.tma_load_c": "c",
            "epilogue.tma_store_d": "d",
        },
        # Partial M and N tiles, both rasterization orders, the descriptor
        # fix-up on both sides of 128 KiB (plus small-m problems the kernel
        # also supports).
        problems=(
            (4, 128, 512),
            (20, 256, 1024),
            (128, 512, 4096),
            (128, 7168, 16384),
            (4096, 4096, 7168),
            (4096, 7168, 2048),
            (200, 264, 512),  # partial M and N tiles, AlongM
            (300, 136, 384),  # partial M and N tiles, AlongN
            (1, 8, 128),
            (129, 1032, 256),
            (33, 1032, 256),  # smallest m upstream sends to this kernel
        ),
    ),
    KernelSpec(
        workload="fp8_gemm_small_m",
        upstream=(
            "CutlassGroupwiseScaledGEMMSM100LowLatency<1,128,128,"
            "ScaleMajorK=true,MmaSM=1>"
        ),
        source="kernels/fp8_gemm_small_m_sm100.cu",
        header="kernels/fp8_gemm_small_m_sm100.cuh",
        symbol_prefix=_MANGLED_PREFIX + "Li12ELi33ELi32E",
        # No C source: tma_load_c is value-initialized, never encoded. The
        # kernel's A is the problem's B and vice versa (swap-AB).
        descriptor_slots={
            "mainloop.tma_load_a": "a",
            "mainloop.tma_load_b": "b",
            "mainloop.tma_load_a_fallback": "a",
            "mainloop.tma_load_b_fallback": "b",
            "epilogue.tma_store_d": "d",
        },
        # m <= 32 only (upstream's dispatch), k % 128 == 0; partial tiles of
        # the transposed problem in both of its dimensions (n % 128, m % 16),
        # both rasterization orders (also m in (16, 32], built by CUTLASS
        # although the harness does not run it).
        problems=(
            (1, 8, 128),
            (1, 136, 256),
            (4, 128, 512),
            (20, 256, 1024),
            (24, 120, 256),  # one 128-row tile, two 16-column tiles: AlongM
            (32, 1032, 384),
            (7, 264, 512),
            (16, 7168, 2048),
            (32, 4096, 7168),
            (17, 512, 4096),
        ),
    ),
)


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


def _ranges(indices: set[int]) -> list[list[int]]:
    """Sorted half-open [start, end) ranges covering ``indices``."""
    ranges: list[list[int]] = []
    for index in sorted(indices):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] += 1
        else:
            ranges.append([index, index + 1])
    return ranges


def case_problems(workload: str) -> list[tuple[int, int, int]]:
    """Every (m, n, k) of the registered workload's cases (smoke, upstream,
    throughput), so that each case shape has a CUTLASS-built fixture."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness import get_workload

    cases = get_workload(workload)(None, device="cpu").get_cases()
    return sorted({(c.params["m"], c.params["n"], c.params["k"]) for c in cases})


def write_fixtures(path: Path, header: dict[str, Any], problems: list[Any]) -> None:
    """Fixture file with one compact JSON line per problem."""
    lines = ",\n".join(json.dumps(p, separators=(",", ":")) for p in problems)
    head = json.dumps(header, indent=2)[:-2]
    path.write_text(f'{head},\n  "problems": [\n{lines}\n  ]\n}}\n')


class ImplCompiler:
    """Builds ``cubins/<arch>/<workload>.cubin`` (+ sidecars) for both
    workloads and merges them into ``kernels.json``."""

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
            raise ValueError(f"fp8_gemm supports {self.supported_arches}, not {arch}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        self.optimization = optimization
        self.flags = list(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.cutlass = self.resources / CUTLASS_DIR
        self.flashinfer = self.resources / FLASHINFER_DIR

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

    def _common_flags(self) -> list[str]:
        return [
            "-std=c++17",
            self.optimization,
            "-lineinfo",
            GENCODE[self.arch],
            "--expt-relaxed-constexpr",
            "-I" + _relative(self.cutlass / "include"),
            "-I" + _relative(self.cutlass / "tools/util/include"),
            *self.flags,
        ]

    def kernel_command(self, spec: KernelSpec, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-cubin",
            *self._common_flags(),
            _relative(PACKAGE / spec.source),
            "-o",
            str(output),
        ]

    def probe_command(self, output: Path) -> list[str]:
        # Host executable; the same flags guarantee the same Params layout as the
        # kernels' device compilation. The shared cudart lets the probe
        # interpose cudaDriverGetVersion and the hardware queries.
        return [
            self.nvcc,
            *self._common_flags(),
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
        self.verify_sources()
        out_dir = PACKAGE / "cubins" / self.arch
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        built: list[
            tuple[KernelSpec, bytes, str, dict[str, Any], list[dict[str, Any]]]
        ] = []
        with tempfile.TemporaryDirectory(prefix="fp8_gemm-") as tmp:
            probe = Path(tmp) / "fp8_gemm_probe"
            subprocess.run(self.probe_command(probe), check=True, cwd=ROOT)
            for spec in KERNELS:
                staged = Path(tmp) / f"{spec.workload}.cubin"
                subprocess.run(self.kernel_command(spec, staged), check=True, cwd=ROOT)
                image = staged.read_bytes()
                arch, names, params = _cubin_info(image)
                if arch != self.arch:
                    raise ValueError(f"{spec.source}: cubin declares {arch}")
                if len(names) != 1 or not names[0].startswith(spec.symbol_prefix):
                    raise ValueError(
                        f"{spec.source}: expected one GemmUniversal, {names}"
                    )
                layout = json.loads(
                    subprocess.check_output(
                        [str(probe), "layout", spec.workload], text=True
                    )
                )
                if [(p.offset, p.size) for p in params] != [(0, layout["params_size"])]:
                    raise ValueError(
                        f"{spec.workload}: kernel parameters {params} do not match "
                        f"sizeof(Params) {layout['params_size']}"
                    )
                runs = [(p, run) for p in spec.problems for run in FIXTURE_RUNS]
                runs += [
                    (p, CASE_RUN)
                    for p in case_problems(spec.workload)
                    if p not in spec.problems
                ]
                fixtures = [
                    self._run_probe(probe, spec, layout, problem, run)
                    for problem, run in runs
                ]
                sidecar = {
                    "workload": spec.workload,
                    "arch": self.arch,
                    "upstream": spec.upstream,
                    "cubin": f"{spec.workload}.cubin",
                    "cubin_sha256": hashlib.sha256(image).hexdigest(),
                    "symbol": names[0],
                    "params_size": layout["params_size"],
                    "params_align": layout["params_align"],
                    "fields": layout["fields"],
                    "constants": layout["constants"],
                    "tensor_maps": self._static_tensor_maps(spec, layout, fixtures),
                    **self._launch(spec, fixtures),
                    "probe": self._portable(self.probe_command(Path("fp8_gemm_probe"))),
                }
                self._check_padding(spec, layout, fixtures)
                built.append((spec, image, names[0], sidecar, fixtures))

            # Only now replace the previous artifacts.
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            for spec, *_ in built:
                shutil.move(
                    Path(tmp) / f"{spec.workload}.cubin",
                    out_dir / f"{spec.workload}.cubin",
                )
        mapping: dict[str, str] = {}
        records: dict[str, dict[str, Any]] = {}
        for spec, _image, symbol, sidecar, fixtures in built:
            target = out_dir / f"{spec.workload}.cubin"
            relative = target.relative_to(PACKAGE).as_posix()
            mapping[spec.workload] = relative
            self._write_json(out_dir / f"{spec.workload}.json", sidecar)
            write_fixtures(
                out_dir / f"{spec.workload}.fixtures.json",
                {
                    "workload": spec.workload,
                    "arch": self.arch,
                    "tensor_map_encoder": "fake (fp8_gemm_probe.cu fake_tensor_map)",
                },
                fixtures,
            )
            records[spec.workload] = {
                "cubin": relative,
                "upstream": spec.upstream,
                "symbol": symbol,
                "sha256": sidecar["cubin_sha256"],
                "source": spec.source,
                "source_sha256": _sha256(PACKAGE / spec.source),
                "header": spec.header,
                "header_sha256": _sha256(PACKAGE / spec.header),
                "probe": PROBE_SOURCE,
                "probe_sha256": _sha256(PACKAGE / PROBE_SOURCE),
                # Compiled to a staging file, then moved to ``cubin``.
                "command": self._portable(self.kernel_command(spec, target)),
            }
        self._merge_manifest(mapping)
        self._write_provenance(records, version.strip().splitlines()[-1])
        return mapping

    def _run_probe(
        self,
        probe: Path,
        spec: KernelSpec,
        layout: dict[str, Any],
        problem: tuple[int, int, int],
        run: tuple[int, int, int],
    ) -> dict[str, Any]:
        p = FIXTURE_POINTERS
        argv = [str(probe), "params", spec.workload, *map(str, problem)]
        argv += [hex(p[key]) for key in ("a", "b", "sfa", "sfb", "d")]
        argv += [str(value) for value in run]
        result = json.loads(subprocess.check_output(argv, text=True))
        if not result["can_implement"] or result["workspace_size"] != 0:
            raise ValueError(f"unexpected CUTLASS arguments for {problem}: {result}")
        if len(result["tensor_maps"]) != len(spec.descriptor_slots):
            raise ValueError("unexpected number of cuTensorMapEncodeTiled calls")
        # Bytes the kernel never reads: those the probe found indeterminate (they
        # differ between 0x00/0xff stack poisoning), the fusion params' padding
        # and each TMA atom's bytes outside its descriptor (its aux strides are
        # static, i.e. empty, and the atom is padded to 128-byte alignment).
        holes = set(result.pop("indeterminate_bytes"))
        fields = layout["fields"]
        params = bytes.fromhex(result["params_hex"])
        # Likewise the fusion params outside the six data members the probe located.
        thread = fields["epilogue.thread"]
        holes |= set(range(thread["offset"], thread["offset"] + thread["size"]))
        for name in THREAD_DATA:
            member = fields["epilogue.thread." + name]
            holes -= set(range(member["offset"], member["offset"] + member["size"]))
        for slot in TMA_SLOTS:
            if fields[slot + ".aux_g_stride"]["size"]:
                raise ValueError(f"{slot}: dynamic TMA aux strides are not supported")
            atom, desc = fields[slot], fields[slot + ".desc"]
            holes |= set(range(atom["offset"], atom["offset"] + atom["size"]))
            holes -= set(range(desc["offset"], desc["offset"] + desc["size"]))
            if slot not in spec.descriptor_slots and any(
                params[desc["offset"] : desc["offset"] + desc["size"]]
            ):
                raise ValueError(f"{slot}: unencoded descriptor is not zero")
        result["padding"] = _ranges(holes)
        return result

    @staticmethod
    def _static_tensor_maps(
        spec: KernelSpec, layout: dict[str, Any], fixtures: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Static encode parameters of each operand's descriptor, cross-checked
        against every recorded encode call (matched to its Params slot)."""
        static: dict[str, dict[str, Any]] = {}
        for fixture in fixtures:
            params = bytes.fromhex(fixture["params_hex"])
            stand_ins = {fake_tensor_map(call): call for call in fixture["tensor_maps"]}
            for slot, operand in spec.descriptor_slots.items():
                offset = layout["fields"][slot + ".desc"]["offset"]
                descriptor = params[offset : offset + 128]
                # The bit-21 fix-up may have changed the stored descriptor.
                word = int.from_bytes(descriptor[8:16], "little") | (1 << 21)
                patched = descriptor[:8] + word.to_bytes(8, "little") + descriptor[16:]
                call = stand_ins.get(descriptor) or stand_ins.get(patched)
                if call is None:
                    raise ValueError(f"{slot}: descriptor matches no encode call")
                values = {key: call[key] for key in STATIC_TMA_KEYS}
                if static.setdefault(operand, values) != values:
                    raise ValueError(f"{operand}: static TMA parameters vary")
        return static

    @staticmethod
    def _launch(spec: KernelSpec, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
        """Launch cluster and hardware queries, identical for every problem."""
        launch = {
            "cluster": fixtures[0]["cluster"],
            "preferred_cluster": fixtures[0]["preferred_cluster"],
            # The occupancy query make_kernel_hardware_info issues (none for a
            # kernel whose host function passes no KernelHardwareInfo).
            "occupancy_query": next(
                (
                    {key: q[key] for key in ("grid", "block", "shared_mem", "attrs")}
                    for q in fixtures[0]["hardware_queries"]
                    if q["call"] == "cudaOccupancyMaxActiveClusters"
                ),
                None,
            ),
        }
        for fixture in fixtures:
            if (fixture["cluster"], fixture["preferred_cluster"]) != (
                launch["cluster"],
                launch["preferred_cluster"],
            ) or fixture["hardware_queries"] != fixtures[0]["hardware_queries"]:
                raise ValueError(f"{spec.workload}: launch setup varies by problem")
        return launch

    @staticmethod
    def _check_padding(
        spec: KernelSpec, layout: dict[str, Any], fixtures: list[dict[str, Any]]
    ) -> None:
        """Indeterminate bytes must be identical everywhere and never overlap a
        leaf field the Python builder writes (aggregate fields contain padding)."""
        padding = {tuple(map(tuple, fixture["padding"])) for fixture in fixtures}
        if len(padding) != 1:
            raise ValueError(
                f"{spec.workload}: indeterminate Params bytes vary between problems"
            )
        holes = {i for start, end in padding.pop() for i in range(start, end)}
        fields = layout["fields"]
        for name, field in fields.items():
            if any(other.startswith(name + ".") for other in fields):
                continue
            if name in TMA_SLOTS or name == "problem_shape":
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

    def _write_provenance(
        self, records: dict[str, dict[str, Any]], nvcc_version: str
    ) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {}) if isinstance(provenance, dict) else {}
        builds[self.arch] = {
            "nvcc": nvcc_version,
            "kernels": dict(sorted(records.items())),
        }
        provenance = {
            "repository": FLASHINFER_REPOSITORY,
            "revision": FLASHINFER_REVISION,
            "files": FLASHINFER_FILES,
            "license": "LICENSE (FlashInfer, Apache-2.0); CUTLASS_LICENSE (BSD-3-Clause)",
            "cutlass_repository": CUTLASS_REPOSITORY,
            "cutlass_revision": CUTLASS_REVISION,
            "cutlass": f"resources/{CUTLASS_DIR}",
            "cutlass_include_tree_sha256": CUTLASS_INCLUDE_TREE_SHA256,
            "scope": "FlashInfer gemm_fp8_nt_groupwise (cutlass backend, FP8 E4M3 -> "
            "BF16, scale granularity (1,128,128), K-major scales), one kernel per "
            "cubin, dispatched as upstream: fp8_gemm_small_m = "
            "CutlassGroupwiseScaledGEMMSM100LowLatency for m <= "
            f"{SMALL_M_MAX_M}, fp8_gemm = CutlassGroupwiseScaledGEMMSM100"
            "<...,MmaSM=1> otherwise; Params built in Python from the probe's "
            "layout (alpha=1, beta=0 as upstream)",
            "dispatch": {
                "source": "csrc/gemm_groupwise_sm100.cu (CutlassGemmGroupwiseScaledSM100)",
                "rule": f"SCALE_GRANULARITY_M == 1 && m <= {SMALL_M_MAX_M} -> "
                "fp8_gemm_small_m, else fp8_gemm",
            },
            "live_kernels": {
                "fp8_gemm": "CutlassGroupwiseScaledGEMMSM100<1,128,128,true,1>: "
                "128x128x128 tile, static 1x1x1 cluster, C = D; m > 32",
                "fp8_gemm_small_m": "CutlassGroupwiseScaledGEMMSM100LowLatency"
                "<1,128,128,true,1>: swap-AB D^T = B A^T, 128x16x128 tile, dynamic "
                "cluster launched 1x1x1, no C; m <= 32",
            },
            "adaptations": [
                (
                    "Kernel types copied from the upstream host function templates; "
                    "host setup (Arguments -> Params including the low-latency "
                    "swap-AB and KernelHardwareInfo, grid, cluster, smem) "
                    "reimplemented in Python and checked byte-for-byte against "
                    "CUTLASS's to_underlying_arguments at compile time."
                ),
                (
                    "No workspace: CUTLASS reports 0 bytes for both kernels (the "
                    "previous package reserved 32 MiB that was never used)."
                ),
                (
                    "The previous package ran every shape on the fp8_gemm kernel; "
                    "m <= 32 shapes now run the low-latency kernel upstream "
                    "dispatches them to."
                ),
                (
                    "fp8_gemm_small_m's launch sets the cluster dimension 1x1x1 "
                    "(upstream's fallback cluster) but not the redundant preferred "
                    "cluster dimension attribute (also 1x1x1) that CUTLASS's "
                    "ClusterLauncher adds."
                ),
            ],
            "builds": dict(sorted(builds.items())),
            "upstream_coverage": {
                "fp8_gemm": "FlashInfer tests/gemm/test_groupwise_scaled_gemm_fp8.py"
                "::test_fp8_groupwise_gemm (cutlass, scale_major_mode K): all 125 "
                "(m, n, k) in {128,256,512,4096,8192}^3 as smoke cases groupwise_*; "
                "DeepGEMM tests/generators.py::enumerate_normal FP8 legacy forward "
                "m in {128, 4096} x 7 (n, k) as throughput cases deepgemm_*",
                "fp8_gemm_small_m": "test_fp8_groupwise_gemm_small_batch_size (K): "
                "m in {1, 4, 16} x n in {128, 256}, k = 256 as smoke cases "
                "small_batch_*; DeepGEMM enumerate_normal m = 1 x 7 (n, k)",
                "not_served": "small-batch m = 32 (2 cases): fp8_gemm_small_m is "
                "restricted to m <= 16 (B200 race), and upstream never runs them on "
                "fp8_gemm; scale_major_mode MN (ScaleMajorK=false); "
                "test_fp8_blockscale_gemm ((128,128,128) granularity); trtllm/cutile "
                "backends; group GEMM tests; DeepGEMM BF16 accumulation, FP4/UE8M0 "
                "and MN-major (backward) configurations",
                "check": "tests/test_fp8_gemm.py::"
                "test_upstream_parametrizations_are_cases",
            },
            "excluded_kernels": {
                "CUBIN/*.cubin": "stale exports of the previous package; each holds "
                "only the fp8_gemm symbol",
                "CutlassGroupwiseScaledGEMMSM100<...,MmaSM=2>": "upstream 2-SM "
                "option (non-default mma_sm=2), never instantiated by the package",
                "ScaleMajorK=false, (128,128,128) granularity": "other upstream "
                "template options, never instantiated by the package",
            },
        }
        self._write_json(PROVENANCE, provenance)

"""Compile the nvfp4_group_gemm workload: one grouped CUTLASS NVFP4 GEMM cubin.

The workload multiplies G independent packed-FP4 problems
``D_g[m_g, n_g] = (A_g * SFA_g) @ (B_g * SFB_g)^T`` (E2M1 A/B, K-major, UE4M3
scale factors per 16 K elements in the blocked tcgen05 layout, FP32
accumulation, FP16 D) in **one** kernel launch.

What the previous package did
-----------------------------

``implementation_sm_100a.cpp`` (= amalgamated FlashInfer
``fp4_gemm_template_sm100.h`` + ``impls/templates/nvfp4.cu``, shared with
nvfp4_gemm and nvfp4_dual_gemm) looped over the groups on the host and, for
**each group**, launched ``pack_scales`` twice (logical (rows, K/16) scales ->
blocked layout, for A and B) and then FlashInfer's single NVFP4 GEMM
(``genericFp4GemmKernelLauncher<half, 128, 128, 256, 1, 1, 1, _1SM>``): 3 G
launches per run. "One launch per run" forbids that, and the per-group GEMM
loop is not what an NVFP4 grouped GEMM on Blackwell does upstream either.

What upstream provides, and the choice made here
------------------------------------------------

* FlashInfer's C++ NVFP4 *grouped* GEMM (``group_gemm_nvfp4_groupwise_sm120.cuh``)
  is SM120-only; its SM100 grouped GEMMs are FP8 (``group_gemm_fp8_groupwise_sm100``)
  and MXFP4 with UE8M0 scales per 32 (``group_gemm_mxfp4_groupwise_sm100``),
  i.e. other data types; the TensorRT-LLM MoE grouped GEMM bundled in
  FlashInfer is a fused-MoE contract (shared N/K, token permutation).
* CUTLASS (the revision FlashInfer pins) provides exactly this data type as a
  genuinely grouped kernel: ``KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100``
  with a ``GroupProblemShape`` and per-group pointer/stride/layout arrays,
  device-side TMA-descriptor updates and a persistent group tile scheduler
  (``examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu``).

The workload therefore uses that CUTLASS grouped kernel, configured like
FlashInfer's single GEMM the package ran per group (128x128x256 MMA tile,
1-SM MMA, 1x1x1 cluster, automatic stages/epilogue tile, sourceless
LinearCombination, FP16 D); ``kernels/nvfp4_group_gemm_sm100.cuh`` lists the
exact differences. Each output tile is still computed by the same tcgen05
block-scaled MMA mainloop with full-K accumulation, so the arithmetic per
tile is unchanged; scheduling (one persistent launch over all groups' tiles)
is what changes.

Kernel inventory and the live/dead decision
-------------------------------------------

==============================================  ========  ==================================
kernel                                          status    reason
==============================================  ========  ==================================
``device_kernel<GemmUniversal<GroupProblem      live      The workload's only kernel: one
Shape<(int,int,int)>, MainloopSm100ArrayTma               launch computes every group.
UmmaWarpSpecializedBlockScaled<5,8,2,1x1x1>,              Workload ``nvfp4_group_gemm``.
..., Sm100PtrArrayTmaWarpSpecialized ...>>``
``DeviceGemmFp4GemmSm100_half_128_128_256_      replaced  The legacy per-group GEMM; launched
1_1_1_1SM`` (FlashInfer single GEMM, FP16 D)              G times per run. Superseded by the
                                                          grouped kernel above (same tile
                                                          config); not emitted.
``pack_scales``                                 folded    Logical -> blocked scale relayout.
                                                into      The kernel consumes the blocked
                                                inputs    layout (the one ``to_blocked`` /
                                                          ``create_reordered_scale_factor_
                                                          tensor`` of the NVIDIA task build
                                                          and hand to submissions), so
                                                          ``get_inputs`` produces it
                                                          directly (``quantization.
                                                          to_blocked_scales``): input
                                                          preparation, not part of the
                                                          timed GEMM. Not emitted here; the
                                                          kernel's one cubin is registered
                                                          as ``nvfp4_dual_gemm_pack_
                                                          scales``.
``DeviceGemmFp4GemmSm100_float_128_128_256_     dead      FP32-output GEMM used only by
1_1_1_1SM``                                               nvfp4_dual_gemm's intermediates.
``silu_product``                                dead      nvfp4_dual_gemm's epilogue only.
``impls/nvfp4_group_gemm/CUBIN/*.cubin`` (4)    dead      Stale exports of the previous
                                                          package; each holds the four
                                                          kernels above.
==============================================  ========  ==================================

Benchmark implication: the previous package timed 3 G launches (2 G scale
relayouts + G GEMMs, each GEMM with its own launch/tail); this workload times
one grouped launch with pre-blocked scales, matching the NVIDIA task, whose
submissions receive the reordered scale factors as inputs.

One kernel per cubin
--------------------

``kernels/nvfp4_group_gemm_sm100.cu`` contains exactly one explicit
instantiation of ``cutlass::device_kernel<GemmKernel>`` and no host code, so
``nvcc -cubin -lineinfo`` for a single ``-gencode`` emits a genuine
single-kernel cubin; ``compile()`` checks its architecture, kernel count,
symbol and parameter size. Rebuilt cubins are not bit-identical (their SASS
is): nvcc records absolute directories in line tables and per-run hashes in
internal symbol names. Commands use repo-relative paths (run from the
repository root).

Host-side setup without native code
-----------------------------------

The kernel takes one by-value ``GemmKernel::Params`` (3328 bytes: nine TMA
descriptor slots encoded for placeholder tile-sized problems with null
addresses -- the kernel rewrites address/shape/stride per group on the device
in its tensormap workspace --, the group problem shape, pointers to the
per-group device arrays, fusion params, group tile-scheduler params and
hardware info) plus device memory Python allocates with torch: the per-group
arrays (problem shapes, A/B/SFA/SFB/D pointers, A/B/D strides, SFA/SFB
layouts) and a tensormap workspace sized per SM.
``kernels/nvfp4_group_gemm_probe.cu`` is a host-only program built and run
here, at compile time only. Its ``layout`` mode records the offsets/sizes of
every Params field and array element plus launch/workspace constants in the
sidecar ``cubins/<arch>/nvfp4_group_gemm.json``; its ``params`` mode builds
the Arguments as CUTLASS's example 75 does and runs CUTLASS's own
``can_implement``/``get_workspace_size``/``to_underlying_arguments``/
``get_grid_shape`` for many group configurations (fake aligned device
addresses; ``cuTensorMapEncodeTiled`` and ``cudaDriverGetVersion``
interposed), recording the Params bytes, the per-group array bytes, the
launch shape, the workspace size and every encode call in
``cubins/<arch>/nvfp4_group_gemm.fixtures.json``. Because the grouped
kernel's host descriptors do not depend on the problems, the sidecar records
each descriptor's complete encode call (and whether CUTLASS's small-tensor
descriptor fix-up applies to it), cross-checked against every fixture.
``harness/workloads/nvfp4_group_gemm.py`` rebuilds Params and the arrays in
Python from the sidecar; ``tests/test_nvfp4_group_gemm.py`` checks them byte
for byte against the fixtures. Fixtures cover the regime edges below (both
driver versions, fewer SMs than tiles) and every case configuration of the
workload (``case_groups``, read from the registered workload at compile time).

Upstream tests and harness cases
--------------------------------

Every ``resources/nvfp4_group_gemm/task.yml`` test line (smoke cases
``task_*``, with the task generator's distributions -- 0xBB-masked E2M1,
scales {1, 2} -- and seed) and benchmark line (throughput), and the regime of
CUTLASS example 75's ``randomize_problems`` (10 groups with m, n, k random
multiples of 32; smoke case ``cutlass_example75``, alpha = 1 and beta = 0 as
this kernel is configured) are cases
(``tests/test_nvfp4_group_gemm.py::test_upstream_parametrizations_are_cases``).
Not served: FlashInfer's ``tests/gemm/test_group_gemm_fp4.py`` runs the
SM120-only ``group_gemm_nvfp4_groupwise_sm120`` kernel, not this one; empty
groups (example 75 can draw m = 0; the harness leaves experts without tokens
out of the launch).

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
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"
WORKLOAD = "nvfp4_group_gemm"
# Same wording in all three NVFP4 packages' provenance.
PACK_SCALES = (
    "previous-package harness glue (impls/templates/nvfp4.cu, not FlashInfer "
    "code): logical -> blocked scale relayout. Its single cubin is kept once, "
    "registered as nvfp4_dual_gemm_pack_scales (a stage of the previous dual "
    "GEMM pipeline); every GEMM workload takes scales in the kernel-native "
    "blocked layout (quantization.to_blocked_scales, as the NVIDIA tasks supply "
    "them), so no GEMM timing includes it. Here: not emitted."
)

FLASHINFER_REPOSITORY = "https://github.com/flashinfer-ai/flashinfer"
FLASHINFER_REVISION = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
FLASHINFER_DIR = f"flashinfer-{FLASHINFER_REVISION}"
# The header whose kernel configuration kernels/nvfp4_group_gemm_sm100.cuh reproduces.
FLASHINFER_HEADER = "include/flashinfer/gemm/fp4_gemm_template_sm100.h"
FLASHINFER_HEADER_SHA256 = (
    "868fad7f668ed07a3569e49691126da48f1f070b97085792ce84de3ef8fb4f86"
)
CUTLASS_REPOSITORY = "https://github.com/NVIDIA/cutlass"
CUTLASS_REVISION = "b46b16d003484063bca4ed365e44095c4c6ed633"
CUTLASS_DIR = f"cutlass-{CUTLASS_REVISION}"
# The example whose grouped-GEMM construction and host setup are followed.
CUTLASS_EXAMPLE = (
    "examples/75_blackwell_grouped_gemm/75_blackwell_grouped_gemm_block_scaled.cu"
)
CUTLASS_EXAMPLE_SHA256 = (
    "52f2966221fc4f9a2e4a6acfab487ce3860376fe769dc43789d6159ea3427965"
)
# sha256 over (relative path + sha256(content)) of every file in include/.
CUTLASS_INCLUDE_TREE_SHA256 = (
    "def1849ca143942f68179aeb664a6cb9bf46001082d7d4f5c63531ea03866e6d"
)

GENCODE = {"sm_100a": "-gencode=arch=compute_100a,code=sm_100a"}
KERNEL_SOURCE = "kernels/nvfp4_group_gemm_sm100.cu"
KERNEL_HEADER = "kernels/nvfp4_group_gemm_sm100.cuh"
PROBE_SOURCE = "kernels/nvfp4_group_gemm_probe.cu"
SYMBOL_PREFIX = (
    "_ZN7cutlass13device_kernelINS_4gemm6kernel13GemmUniversalINS1_17"
    "GroupProblemShapeIN4cute5tupleIJiiiEEEEENS1_10collective13CollectiveMmaINS1_51"
    "MainloopSm100ArrayTmaUmmaWarpSpecializedBlockScaledILi5ELi8ELi2E"
)

# Group configurations whose Params CUTLASS builds at compile time (test
# fixtures): the harness smoke cases, every NVIDIA task test and benchmark
# shape, a single 1-tile partial problem and a problem with more tiles than
# SMs (persistent-grid truncation).
SMOKE_GROUPS = (
    ((128, 128, 128),),
    ((17, 128, 128), (128, 256, 256), (65, 128, 512)),
)
TASK_TESTS = (
    ((96, 128), (128, 256), (256, 512)),
    ((256, 72), (512, 384), (256, 256)),
    ((128, 128), (128, 256), (512, 256)),
    ((80, 128, 256), (384, 256, 128), (256, 512, 256)),
    ((64, 72, 96), (128, 384, 512), (512, 512, 256)),
    ((64, 256, 128), (768, 128, 256), (512, 256, 512)),
    ((128, 128, 64), (256, 512, 512), (768, 256, 768)),
    ((128, 128, 128, 128), (128, 128, 128, 128), (512, 256, 512, 256)),
    ((40, 56, 384, 512), (512, 384, 256, 128), (256, 256, 256, 256)),
    ((512, 384, 256, 128), (256, 256, 256, 256), (512, 768, 512, 768)),
)
TASK_BENCHMARKS = (
    ((80, 176, 128, 72, 64, 248, 96, 160), (4096,) * 8, (7168,) * 8),
    ((40, 76, 168, 72, 164, 148, 196, 160), (7168,) * 8, (2048,) * 8),
    ((192, 320), (3072, 3072), (4096, 4096)),
    ((128, 384), (4096, 4096), (1536, 1536)),
)
EXTRA_GROUPS = (
    ((1, 8, 32),),
    ((1024, 4096, 256),),
)
FIXTURE_GROUPS = (
    SMOKE_GROUPS
    + tuple(tuple(zip(*shape, strict=True)) for shape in TASK_TESTS + TASK_BENCHMARKS)
    + EXTRA_GROUPS
)
B200_SM_COUNT = 148
FIXTURE_DRIVER_VERSIONS = (13020, 13010)  # without / with the bit-21 fix-up
# Every case configuration of the workload is added for this driver version
# at B200_SM_COUNT (case_groups).
CASE_DRIVER_VERSION = 13010
# Additional (groups index, sm_count) runs: fewer SMs than tiles.
FIXTURE_SMALL_SM = ((1, 2), (len(SMOKE_GROUPS) + len(TASK_TESTS), 8))
FIXTURE_BASE = 0x7F0000000000  # fake device address base (see the probe)

# Params descriptor slot -> operand whose (placeholder) descriptor it holds.
DESCRIPTOR_SLOTS = {
    "mainloop.tma_load_a": "a",
    "mainloop.tma_load_b": "b",
    "mainloop.tma_load_sfa": "sfa",
    "mainloop.tma_load_sfb": "sfb",
    "mainloop.tma_load_a_fallback": "a",
    "mainloop.tma_load_b_fallback": "b",
    "mainloop.tma_load_sfa_fallback": "sfa",
    "mainloop.tma_load_sfb_fallback": "sfb",
    "epilogue.tma_store_d": "d",
}
# Sourceless epilogue: tma_load_c is value-initialized (all zero), never encoded.
UNENCODED_SLOTS = ("epilogue.tma_load_c",)
THREAD_DATA = (
    "alpha",
    "beta",
    "alpha_ptr",
    "beta_ptr",
    "alpha_ptr_array",
    "beta_ptr_array",
    "alpha_stride_l",
    "beta_stride_l",
)
TMA_KEYS = (
    "data_type",
    "rank",
    "address",
    "dims",
    "strides",
    "box",
    "element_strides",
    "interleave",
    "swizzle",
    "l2_promotion",
    "oob_fill",
)
FIXUP_BIT = 1 << 21


def case_groups(workload: str) -> list[tuple[tuple[int, int, int], ...]]:
    """Every group configuration of the registered workload's cases (smoke,
    task tests, throughput), so that each has a CUTLASS-built fixture."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness import get_workload

    cases = get_workload(workload)(None, device="cpu").get_cases()
    configurations = {tuple(map(tuple, c.params["groups"])) for c in cases}
    return sorted(configurations)


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
    out[8:16] = (call["address"] | FIXUP_BIT).to_bytes(8, "little")
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
    """Builds ``cubins/<arch>/nvfp4_group_gemm.cubin`` (+ sidecars), merges ``kernels.json``."""

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
                f"nvfp4_group_gemm supports {self.supported_arches}, not {arch}"
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

    # -- pinned sources ---------------------------------------------------------

    def verify_sources(self) -> None:
        for path, digest, origin in (
            (
                self.flashinfer / FLASHINFER_HEADER,
                FLASHINFER_HEADER_SHA256,
                f"FlashInfer {FLASHINFER_REVISION}",
            ),
            (
                self.cutlass / CUTLASS_EXAMPLE,
                CUTLASS_EXAMPLE_SHA256,
                f"CUTLASS {CUTLASS_REVISION}",
            ),
        ):
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"{path} is missing or differs from {origin}")
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

    def kernel_command(self, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-cubin",
            *self._common_flags(),
            _relative(PACKAGE / KERNEL_SOURCE),
            "-o",
            str(output),
        ]

    def probe_command(self, output: Path) -> list[str]:
        # Host executable; the same flags guarantee the same Params layout as the
        # kernel's device compilation. The shared cudart lets the probe
        # interpose cudaDriverGetVersion.
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
        with tempfile.TemporaryDirectory(prefix="nvfp4_group_gemm-") as tmp:
            staged = Path(tmp) / f"{WORKLOAD}.cubin"
            subprocess.run(self.kernel_command(staged), check=True, cwd=ROOT)
            image = staged.read_bytes()
            arch, names, params = _cubin_info(image)
            if arch != self.arch:
                raise ValueError(f"{KERNEL_SOURCE}: cubin declares {arch}")
            if len(names) != 1 or not names[0].startswith(SYMBOL_PREFIX):
                raise ValueError(
                    f"{KERNEL_SOURCE}: expected one grouped GemmUniversal, {names}"
                )
            probe = Path(tmp) / "nvfp4_group_gemm_probe"
            subprocess.run(self.probe_command(probe), check=True, cwd=ROOT)
            layout = json.loads(
                subprocess.check_output([str(probe), "layout"], text=True)
            )
            if [(p.offset, p.size) for p in params] != [(0, layout["params_size"])]:
                raise ValueError(
                    f"kernel parameters {params} do not match sizeof(Params) "
                    f"{layout['params_size']}"
                )
            runs = [
                (groups, B200_SM_COUNT, driver)
                for groups in FIXTURE_GROUPS
                for driver in FIXTURE_DRIVER_VERSIONS
            ] + [
                (FIXTURE_GROUPS[index], sm_count, FIXTURE_DRIVER_VERSIONS[0])
                for index, sm_count in FIXTURE_SMALL_SM
            ]
            runs += [
                (groups, B200_SM_COUNT, CASE_DRIVER_VERSION)
                for groups in case_groups(WORKLOAD)
                if groups not in FIXTURE_GROUPS
            ]
            fixtures = [
                self._run_probe(probe, layout, groups, sm_count, driver)
                for groups, sm_count, driver in runs
            ]
            tensor_maps = self._tensor_maps(layout, fixtures)
            self._check_padding(layout, fixtures)

            # Only now replace the previous artifacts.
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            target = out_dir / f"{WORKLOAD}.cubin"
            shutil.move(staged, target)
        relative = target.relative_to(PACKAGE).as_posix()
        sidecar = {
            "workload": WORKLOAD,
            "arch": self.arch,
            "cubin": target.name,
            "cubin_sha256": hashlib.sha256(image).hexdigest(),
            "symbol": names[0],
            "params_size": layout["params_size"],
            "params_align": layout["params_align"],
            "fields": layout["fields"],
            "elements": layout["elements"],
            "element_align": layout["element_align"],
            "constants": layout["constants"],
            "tensor_maps": tensor_maps,
            "probe": self._portable(self.probe_command(Path("nvfp4_group_gemm_probe"))),
        }
        self._write_json(out_dir / f"{WORKLOAD}.json", sidecar)
        write_fixtures(
            out_dir / f"{WORKLOAD}.fixtures.json",
            {
                "workload": WORKLOAD,
                "arch": self.arch,
                "tensor_map_encoder": "fake (nvfp4_group_gemm_probe.cu fake_tensor_map)",
            },
            fixtures,
        )
        self._merge_manifest({WORKLOAD: relative})
        self._write_provenance(
            {
                "cubin": relative,
                "symbol": names[0],
                "sha256": sidecar["cubin_sha256"],
                "source": KERNEL_SOURCE,
                "source_sha256": _sha256(PACKAGE / KERNEL_SOURCE),
                "header": KERNEL_HEADER,
                "header_sha256": _sha256(PACKAGE / KERNEL_HEADER),
                "probe": PROBE_SOURCE,
                "probe_sha256": _sha256(PACKAGE / PROBE_SOURCE),
                # Compiled to a staging file, then moved to ``cubin``.
                "command": self._portable(self.kernel_command(target)),
            },
            version.strip().splitlines()[-1],
        )
        return {WORKLOAD: relative}

    def _run_probe(
        self,
        probe: Path,
        layout: dict[str, Any],
        groups: Sequence[tuple[int, int, int]],
        sm_count: int,
        driver: int,
    ) -> dict[str, Any]:
        argv = [str(probe), "params", str(sm_count), str(driver), hex(FIXTURE_BASE)]
        argv += [",".join(map(str, problem)) for problem in groups]
        result = json.loads(subprocess.check_output(argv, text=True))
        if not result["can_implement"]:
            raise ValueError(f"CUTLASS cannot implement {groups}")
        # Bytes the kernel never reads: those the probe found indeterminate (they
        # differ between 0x00/0xff stack poisoning), the fusion params outside
        # their data members and each TMA atom's bytes outside its descriptor
        # (its aux strides are static, i.e. empty, and the atom is padded to
        # 128-byte alignment).
        holes = set(result.pop("indeterminate_bytes"))
        fields = layout["fields"]
        thread = fields["epilogue.thread"]
        holes |= set(range(thread["offset"], thread["offset"] + thread["size"]))
        for name in THREAD_DATA:
            member = fields["epilogue.thread." + name]
            holes -= set(range(member["offset"], member["offset"] + member["size"]))
        for slot in (*DESCRIPTOR_SLOTS, *UNENCODED_SLOTS):
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
    def _tensor_maps(
        layout: dict[str, Any], fixtures: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """The complete encode call behind each operand's placeholder descriptor
        and whether CUTLASS's small-tensor fix-up (drivers <= 13.1 clear bit 21
        of descriptor word 1) applies to it, cross-checked against every
        recorded encode call (matched to its Params slot)."""
        maps: dict[str, dict[str, Any]] = {}
        for fixture in fixtures:
            params = bytes.fromhex(fixture["params_hex"])
            stand_ins = {fake_tensor_map(call): call for call in fixture["tensor_maps"]}
            if len(fixture["tensor_maps"]) != len(DESCRIPTOR_SLOTS):
                raise ValueError("unexpected number of cuTensorMapEncodeTiled calls")
            for slot in UNENCODED_SLOTS:
                desc = layout["fields"][slot + ".desc"]
                if any(params[desc["offset"] : desc["offset"] + desc["size"]]):
                    raise ValueError(f"{slot}: expected an all-zero descriptor")
            for slot, operand in DESCRIPTOR_SLOTS.items():
                offset = layout["fields"][slot + ".desc"]["offset"]
                descriptor = params[offset : offset + 128]
                word = int.from_bytes(descriptor[8:16], "little")
                patched = (
                    descriptor[:8]
                    + (word | FIXUP_BIT).to_bytes(8, "little")
                    + descriptor[16:]
                )
                call = stand_ins.get(descriptor)
                fixup = call is None
                if fixup:
                    call = stand_ins.get(patched)
                if call is None:
                    raise ValueError(f"{slot}: descriptor matches no encode call")
                if fixup and fixture["driver_version"] > 13010:
                    raise ValueError(f"{slot}: unexpected descriptor fix-up")
                values = {key: call[key] for key in TMA_KEYS}
                if values["address"] != 0:
                    raise ValueError(f"{slot}: grouped placeholder must be null")
                if fixture["driver_version"] <= 13010:
                    values["small_tensor_fixup"] = fixup
                known = maps.setdefault(operand, values)
                if "small_tensor_fixup" in values:
                    known.setdefault("small_tensor_fixup", fixup)
                if {k: known[k] for k in values} != values:
                    raise ValueError(f"{operand}: placeholder descriptor varies")
        for operand, values in maps.items():
            if "small_tensor_fixup" not in values:
                raise ValueError(f"{operand}: no fixture exercises the fix-up")
        return maps

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
            if name in DESCRIPTOR_SLOTS or name in UNENCODED_SLOTS:
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

    def _write_provenance(self, record: dict[str, Any], nvcc_version: str) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {}) if isinstance(provenance, dict) else {}
        builds[self.arch] = {"nvcc": nvcc_version, "kernels": {WORKLOAD: record}}
        provenance = {
            "repository": FLASHINFER_REPOSITORY,
            "revision": FLASHINFER_REVISION,
            "files": {FLASHINFER_HEADER: FLASHINFER_HEADER_SHA256},
            "license": "LICENSE (FlashInfer, Apache-2.0); CUTLASS_LICENSE (BSD-3-Clause)",
            "cutlass_repository": CUTLASS_REPOSITORY,
            "cutlass_revision": CUTLASS_REVISION,
            "cutlass": f"resources/{CUTLASS_DIR}",
            "cutlass_files": {CUTLASS_EXAMPLE: CUTLASS_EXAMPLE_SHA256},
            "cutlass_include_tree_sha256": CUTLASS_INCLUDE_TREE_SHA256,
            "scope": "CUTLASS grouped (ptr-array) NVFP4 GEMM "
            "(KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100, 128x128x256, 1x1x1 "
            "cluster, E2M1 x E2M1 with UE4M3/16 scales -> FP16, alpha=1, beta=0, "
            "no C) in FlashInfer's fp4_gemm_template_sm100.h configuration; one "
            "kernel per cubin, one launch for all groups; Params and per-group "
            "device arrays built in Python from the probe's layout",
            "adaptations": [
                (
                    "The previous package launched pack_scales twice plus "
                    "FlashInfer's single NVFP4 GEMM once per group (3 G launches). "
                    "It is replaced by CUTLASS's grouped NVFP4 kernel (the "
                    "data type's only SM100 grouped kernel in the pinned FlashInfer/"
                    "CUTLASS sources) with FlashInfer's tile/cluster/epilogue "
                    "configuration: one launch per run."
                ),
                (
                    "Scale factors are inputs in the kernel's blocked layout "
                    "(as the NVIDIA task hands reordered scales to submissions); "
                    "pack_scales is not launched (see excluded_kernels)."
                ),
                (
                    "Groups have at least one row: experts without routed tokens "
                    "are left out of the launch (CUTLASS example 75 can draw empty "
                    "problems; the harness never passes them)."
                ),
                (
                    "Kernel type defined from FlashInfer's single-GEMM type aliases "
                    "with CUTLASS example 75's grouped changes (see the header); "
                    "host setup (Arguments -> Params, per-group arrays, workspace, "
                    "grid, smem) reimplemented in Python and checked byte-for-byte "
                    "against CUTLASS's to_underlying_arguments at compile time."
                ),
                (
                    "alpha = 1 as a scalar (the previous package passed a device "
                    "pointer to 1.0); static 1x1x1 cluster instead of FlashInfer's "
                    "dynamic cluster set to 1x1x1 at run time; no programmatic "
                    "dependent launch attribute."
                ),
                (
                    "Workspace: per-SM tensormap buffers for device-side TMA "
                    "descriptor updates, sized by CUTLASS's get_workspace_size for "
                    "the device's SM count and allocated with torch per launch "
                    "configuration."
                ),
            ],
            "builds": dict(sorted(builds.items())),
            "upstream_coverage": {
                "cases": "every resources/nvfp4_group_gemm/task.yml test line "
                "(smoke cases task_*, task distribution and seed) and benchmark "
                "line (throughput); CUTLASS example 75's randomize_problems regime "
                "(10 groups, m, n, k multiples of 32, smoke case cutlass_example75; "
                "alpha = 1, beta = 0 as configured here)",
                "not_served": "FlashInfer tests/gemm/test_group_gemm_fp4.py "
                "(SM120 kernel group_gemm_nvfp4_groupwise_sm120); empty groups",
                "check": "tests/test_nvfp4_group_gemm.py::"
                "test_upstream_parametrizations_are_cases",
            },
            "excluded_kernels": {
                "DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM": "previous "
                "per-group GEMM (launched G times); replaced by the grouped kernel",
                "pack_scales": PACK_SCALES,
                "DeviceGemmFp4GemmSm100_float_128_128_256_1_1_1_1SM": "nvfp4_dual_gemm "
                "FP32 intermediates only",
                "silu_product": "nvfp4_dual_gemm epilogue only",
                "CUBIN/*.cubin": "stale exports of the previous package (all four "
                "kernels above)",
            },
        }
        self._write_json(PROVENANCE, provenance)

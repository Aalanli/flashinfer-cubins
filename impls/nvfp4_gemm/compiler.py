"""Compile the nvfp4_gemm workload's live CUTLASS kernel into a single-kernel cubin.

The workload is FlashInfer's NVFP4 block-scaled GEMM on Blackwell
(``flashinfer::gemm::genericFp4GemmKernelLauncher<half, 128, 128, 256, 1, 1,
1, _1SM>`` from ``fp4_gemm_template_sm100.h``, pinned revision below):
``D[m, n, l] = (A[m, k, l] * SFA) @ (B[n, k, l] * SFB)^T`` with packed E2M1
A/B (K-major), UE4M3 scales per 16 K elements, FP32 accumulation and FP16 D.

Kernel inventory and the live/dead decision
-------------------------------------------

The previous package (``implementation_sm_100a.cpp`` = amalgamated
``fp4_gemm_template_sm100.h`` + ``impls/templates/nvfp4.cu`` with
``WORKLOAD_KIND 0``) is shared by the three NVFP4 workloads; its cubins
(``CUBIN/*.cubin``) each hold the same four kernels:

===============================================  ======  =====================================
kernel                                           status  reason
===============================================  ======  =====================================
``device_kernel<DeviceGemmFp4GemmSm100_half_     live    The GEMM itself: ``gemm<half>`` calls
128_128_256_1_1_1_1SM::Sm10x11xOnly<              (one    ``genericFp4GemmKernelLauncher<half,
GemmUniversal<..., PersistentScheduler>>>``      launch) ...>`` once per run for every m/n/k/l
                                                         (128x128x256 tile, dynamic cluster
                                                         launched 1x1x1, CLC scheduler, 5
                                                         stages). Workload ``nvfp4_gemm``.
``device_kernel<DeviceGemmFp4GemmSm100_float_    dead    FP32-output instantiation, called only
128_128_256_1_1_1_1SM::...>``                            by ``WORKLOAD_KIND == 1``
                                                         (nvfp4_dual_gemm's intermediates).
``silu_product``                                 dead    Dual-GEMM epilogue
                                                         (``WORKLOAD_KIND == 1`` only).
``pack_scales``                                  folded  Harness glue, not FlashInfer code: it
                                                 into    re-laid the logical (rows, k/16, l)
                                                 inputs  scales into CUTLASS's blocked
                                                         scale-factor layout before each GEMM
                                                         (two launches per run). The NVIDIA
                                                         task already supplies scales in that
                                                         layout (``sfa_permuted``/
                                                         ``sfb_permuted``, shape (32, 4,
                                                         ceil(mn/128), 4, ceil(k/64), l)), so
                                                         ``get_inputs`` produces it directly
                                                         (``quantization.to_blocked_scales``)
                                                         and no cubin is emitted here; the
                                                         kernel's one cubin is registered as
                                                         ``nvfp4_dual_gemm_pack_scales`` (a
                                                         stage of the previous dual GEMM
                                                         pipeline). Benchmark implication: the
                                                         timed launch is the GEMM alone; the
                                                         previous package's time included two
                                                         relayout kernels (for n=7168,
                                                         k=16384 about 7 MiB of scales read
                                                         and written).
``CUBIN/*.cubin`` (080965c8, 588bee38,           dead    Stale exports of successive builds of
61f3a321, c5251ad5)                                      the previous package: not distinct
                                                         kernels (all four hold the symbols
                                                         above).
===============================================  ======  =====================================

One kernel per cubin
--------------------

``kernels/nvfp4_gemm_sm100.cuh`` reproduces the definitions the FlashInfer
macro ``INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(half, 128, 128, 256, 1, 1, 1,
_1SM)`` makes (verbatim, same namespace and struct name, so the kernel symbol
is the previous package's) without the host launcher; it includes the pinned
``flashinfer/arch_condition.h`` for the ``Sm10x11xOnly`` guard.
``kernels/nvfp4_gemm_sm100.cu`` contains exactly one explicit instantiation
of ``cutlass::device_kernel<GemmKernel>`` and no host code, so ``nvcc -cubin
-lineinfo`` for a single ``-gencode`` emits a genuine single-kernel cubin;
``compile()`` checks its architecture, kernel count, symbol and parameter
size. Its SASS matches the half kernel of the previous package's cubins apart
from assert line numbers/string addresses (and the register choice of their
loads). Rebuilt cubins are not bit-identical (their SASS is): nvcc records
absolute directories in the line tables and a per-run hash in
internal-linkage symbol names. Commands use repo-relative paths (run from the
repository root) so the recorded provenance is portable.

Host-side setup without native code
-----------------------------------

The kernel takes one by-value ``GemmKernel::Params`` (3072 bytes: eight
mainloop TMA descriptors (A, B, SFA, SFB and their fallback-cluster copies),
an unused zero C descriptor (``ElementC = void``), the D store descriptor,
problem shape, scale-factor layouts, fusion params (``alpha_ptr``), tile
scheduler params and hardware info). ``kernels/nvfp4_gemm_probe.cu`` is a
host-only program built and run here, at compile time only. Its ``layout``
mode records offsets and sizes of every field (including every dynamic leaf
of the cute shapes/strides) plus the launch constants in the sidecar
``cubins/<arch>/nvfp4_gemm.json``; its ``params`` mode runs CUTLASS's own
``to_underlying_arguments``/``get_grid_shape`` on Arguments built exactly like
FlashInfer's ``prepareGemmArgs`` for several problems (fake, aligned device
addresses; ``cuTensorMapEncodeTiled`` and ``cudaDriverGetVersion`` interposed,
see the probe's header) and records the Params bytes, launch shape and every
descriptor's encode arguments in ``cubins/<arch>/nvfp4_gemm.fixtures.json``.
The static TMA parameters (data type, box, swizzle, L2 promotion, ...) in the
sidecar are taken from those recorded calls. ``harness/workloads/nvfp4_gemm.py``
rebuilds Params in Python from the sidecar and is checked byte-for-byte
against the fixtures by ``tests/test_nvfp4_gemm.py``.

Fixtures cover regime edges (``FIXTURE_PROBLEMS``, both driver versions) and
every case shape of the workload (``case_problems``, read from the registered
workload at compile time). The Params builder (``CutlassFp4Gemm``) is shared
with ``nvfp4_dual_gemm_gemm``, the FP32-output instantiation of the same
template, whose probe records the same field offsets.

Upstream tests and harness cases
--------------------------------

Every NVIDIA task test line (smoke cases ``task_*``, with the task
generator's distributions: all E2M1 bytes, integer scales 0..3, seed 1111)
and benchmark line (throughput), and FlashInfer's
``tests/gemm/test_mm_fp4.py::test_mm_fp4`` cutlass/nvfp4 cases (1, 128, 512)
and (31, 256, 256) with alpha != 1 (smoke cases ``mm_fp4_*``; the first asks
for BF16 output, whose instantiation is not built, and runs with FP16
output) are cases (``tests/test_nvfp4_gemm.py::
test_upstream_parametrizations_are_cases``). Not served: other ``mm_fp4``
backends, the 8x4 scale layout, and the randomized shapes of
``test_unified_gemm_fuzz.py`` (its regime -- odd m, partial tiles -- is
covered by the smoke cases).

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
WORKLOAD = "nvfp4_gemm"
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
# The header whose definitions kernels/nvfp4_gemm_sm100.cuh reproduces, and
# the header it includes from the pinned tree.
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

GENCODE = {"sm_100a": "-gencode=arch=compute_100a,code=sm_100a"}
KERNEL_SOURCE = "kernels/nvfp4_gemm_sm100.cu"
KERNEL_HEADER = "kernels/nvfp4_gemm_sm100.cuh"
PROBE_SOURCE = "kernels/nvfp4_gemm_probe.cu"
SYMBOL_PREFIX = (
    "_ZN7cutlass13device_kernelIN10flashinfer4gemm49"
    "DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM12Sm10x11xOnlyIN"
    "S_4gemm6kernel13GemmUniversalIN4cute5tupleIJiiiiEEE"
)

# Problems (m, n, k, l) whose Params CUTLASS builds at compile time (test
# fixtures) for both driver versions: partial M/N/K tiles and partial
# scale-factor blocks, batches (l > 1), both rasterization orders and CUTLASS's
# driver-version-dependent descriptor fix-up on both sides of its 128 KiB
# threshold (incl. one where only the batched size reaches it). Every case
# shape of the workload is added for CASE_DRIVER_VERSION (case_problems).
FIXTURE_PROBLEMS = (
    (128, 128, 128, 1),
    (128, 256, 256, 2),
    (256, 128, 512, 1),
    (128, 7168, 16384, 1),
    (128, 4096, 7168, 1),
    (128, 7168, 2048, 1),
    (200, 264, 512, 1),  # partial M and N tiles, AlongM
    (300, 136, 384, 3),  # partial M, N and K tiles, AlongN, batched
    (64, 128, 96, 1),  # k/16 = 6 scale columns: partial scale-factor block
    (1, 8, 32, 1),
    (129, 1032, 256, 2),
    (128, 256, 1024, 2),  # A is 64 KiB per batch, 128 KiB in total
)
FIXTURE_DRIVER_VERSIONS = (13020, 13010)  # without / with the bit-21 fix-up
CASE_DRIVER_VERSION = 13010
# Fake device addresses (16-byte aligned, bit 21 set so that the fix-up is
# visible in the descriptor stand-in, never dereferenced).
FIXTURE_POINTERS = {
    "a": 0x7F0000200000,
    "b": 0x7F1000200000,
    "sfa": 0x7F2000200000,
    "sfb": 0x7F3000200000,
    "alpha": 0x7F5000000100,
    "d": 0x7F4000200000,
}
PROBE_POINTER_ORDER = ("a", "b", "sfa", "sfb", "alpha", "d")
# Params slot -> operand whose descriptor it holds (encoded descriptors only;
# epilogue.tma_load_c stays value-initialized because ElementC is void).
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
UNUSED_DESCRIPTOR_SLOTS = ("epilogue.tma_load_c",)
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


def case_problems(workload: str) -> list[tuple[int, int, int, int]]:
    """Every (m, n, k, l) of the registered workload's cases (smoke, upstream,
    throughput), so that each case shape has a CUTLASS-built fixture."""
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


class ImplCompiler:
    """Builds ``cubins/<arch>/nvfp4_gemm.cubin`` (+ sidecars), merges ``kernels.json``."""

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
            raise ValueError(f"nvfp4_gemm supports {self.supported_arches}, not {arch}")
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
            "-I" + _relative(self.flashinfer / "include"),
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
        with tempfile.TemporaryDirectory(prefix="nvfp4_gemm-") as tmp:
            staged = Path(tmp) / f"{WORKLOAD}.cubin"
            subprocess.run(self.kernel_command(staged), check=True, cwd=ROOT)
            image = staged.read_bytes()
            arch, names, params = _cubin_info(image)
            if arch != self.arch:
                raise ValueError(f"{KERNEL_SOURCE}: cubin declares {arch}")
            if len(names) != 1 or not names[0].startswith(SYMBOL_PREFIX):
                raise ValueError(
                    f"{KERNEL_SOURCE}: expected one GemmUniversal, {names}"
                )
            probe = Path(tmp) / "nvfp4_gemm_probe"
            subprocess.run(self.probe_command(probe), check=True, cwd=ROOT)
            layout = json.loads(
                subprocess.check_output([str(probe), "layout"], text=True)
            )
            if [(p.offset, p.size) for p in params] != [(0, layout["params_size"])]:
                raise ValueError(
                    f"kernel parameters {params} do not match sizeof(Params) "
                    f"{layout['params_size']}"
                )
            runs = [(p, d) for p in FIXTURE_PROBLEMS for d in FIXTURE_DRIVER_VERSIONS]
            runs += [
                (p, CASE_DRIVER_VERSION)
                for p in case_problems(WORKLOAD)
                if p not in FIXTURE_PROBLEMS
            ]
            fixtures = [
                self._run_probe(probe, layout, problem, driver)
                for problem, driver in runs
            ]
            tensor_maps = self._static_tensor_maps(layout, fixtures)
            self._check_padding(layout, fixtures)

            # Only now replace the checked-in artifacts.
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
            "constants": layout["constants"],
            "tensor_maps": tensor_maps,
            "probe": self._portable(self.probe_command(Path("nvfp4_gemm_probe"))),
        }
        self._write_json(out_dir / f"{WORKLOAD}.json", sidecar)
        write_fixtures(
            out_dir / f"{WORKLOAD}.fixtures.json",
            {
                "workload": WORKLOAD,
                "arch": self.arch,
                "tensor_map_encoder": "fake (nvfp4_gemm_probe.cu fake_tensor_map)",
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
        problem: tuple[int, int, int, int],
        driver: int,
    ) -> dict[str, Any]:
        argv = [str(probe), "params", *map(str, problem)]
        argv += [hex(FIXTURE_POINTERS[key]) for key in PROBE_POINTER_ORDER]
        argv += [str(driver)]
        result = json.loads(subprocess.check_output(argv, text=True))
        if (
            not result["can_implement"]
            or result["workspace_size"] != 0
            or not result["workspace_independent"]
        ):
            raise ValueError(f"unexpected CUTLASS arguments for {problem}: {result}")
        if result["cluster"] != [1, 1, 1] or result["cluster_fallback"] != [1, 1, 1]:
            raise ValueError(f"unexpected cluster shape for {problem}")
        if len(result["tensor_maps"]) != len(DESCRIPTOR_SLOTS):
            raise ValueError("unexpected number of cuTensorMapEncodeTiled calls")
        params = bytes.fromhex(result["params_hex"])
        fields = layout["fields"]
        for slot in UNUSED_DESCRIPTOR_SLOTS:
            desc = fields[slot + ".desc"]
            if any(params[desc["offset"] : desc["offset"] + desc["size"]]):
                raise ValueError(f"{slot}: expected a zero (unused) descriptor")
        # Bytes the kernel never reads: those the probe found indeterminate (they
        # differ between 0x00/0xff stack poisoning), the fusion params' padding
        # and each TMA atom's bytes outside its descriptor (its aux strides are
        # static, i.e. empty, and the atom is padded to 128-byte alignment).
        holes = set(result.pop("indeterminate_bytes"))
        # Likewise the fusion params outside the six data members the probe located.
        thread = fields["epilogue.thread"]
        holes |= set(range(thread["offset"], thread["offset"] + thread["size"]))
        for name in THREAD_DATA:
            member = fields["epilogue.thread." + name]
            holes -= set(range(member["offset"], member["offset"] + member["size"]))
        for slot in (*DESCRIPTOR_SLOTS, *UNUSED_DESCRIPTOR_SLOTS):
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
        # Padding bytes may hold leftovers of the probe's memory; record them as
        # zero so that rebuilt fixtures are deterministic.
        zeroed = bytearray(params)
        for start, end in ranges:
            zeroed[start:end] = bytes(end - start)
        result["params_hex"] = zeroed.hex()
        return result

    @staticmethod
    def _static_tensor_maps(
        layout: dict[str, Any], fixtures: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Static encode parameters of each operand's descriptor, cross-checked
        against every recorded encode call (matched to its Params slot)."""
        static: dict[str, dict[str, Any]] = {}
        for fixture in fixtures:
            params = bytes.fromhex(fixture["params_hex"])
            stand_ins = {fake_tensor_map(call): call for call in fixture["tensor_maps"]}
            for slot, operand in DESCRIPTOR_SLOTS.items():
                offset = layout["fields"][slot + ".desc"]["offset"]
                descriptor = params[offset : offset + 128]
                # The bit-21 fix-up may have changed the stored descriptor.
                word = int.from_bytes(descriptor[8:16], "little") | (1 << 21)
                patched = descriptor[:8] + word.to_bytes(8, "little") + descriptor[16:]
                call = stand_ins.get(descriptor) or stand_ins.get(patched)
                if call is None:
                    raise ValueError(f"{slot}: descriptor matches no encode call")
                if call["address"] != FIXTURE_POINTERS[operand]:
                    raise ValueError(f"{slot}: descriptor of another operand")
                values = {key: call[key] for key in STATIC_TMA_KEYS}
                if static.setdefault(operand, values) != values:
                    raise ValueError(f"{operand}: static TMA parameters vary")
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
            "files": dict(FLASHINFER_FILES),
            "license": "LICENSE (FlashInfer, Apache-2.0); CUTLASS_LICENSE (BSD-3-Clause)",
            "cutlass_repository": CUTLASS_REPOSITORY,
            "cutlass_revision": CUTLASS_REVISION,
            "cutlass": f"resources/{CUTLASS_DIR}",
            "cutlass_include_tree_sha256": CUTLASS_INCLUDE_TREE_SHA256,
            "scope": "genericFp4GemmKernelLauncher<half,128,128,256,1,1,1,_1SM> "
            "(NVFP4 E2M1 x UE4M3/16 -> FP16, batched), one kernel per cubin; "
            "Params built in Python from the probe's layout (alpha_ptr -> the "
            "alpha input, beta=0, void C, as upstream); the Params builder is "
            "shared with nvfp4_dual_gemm_gemm (the FP32-output instantiation)",
            "adaptations": [
                (
                    "Kernel type copied from the upstream instantiation macro; host "
                    "setup (prepareGemmArgs -> to_underlying_arguments -> Params, "
                    "grid, smem, cluster 1x1x1 with fallback 1x1x1, PDL launch) "
                    "reimplemented in Python and checked byte-for-byte against "
                    "CUTLASS at compile time."
                ),
                (
                    "Scales are inputs in CUTLASS's blocked scale-factor layout (the "
                    "NVIDIA task's sfa_permuted/sfb_permuted), so the previous "
                    "package's pack_scales relayout kernels are not launched or timed."
                ),
                (
                    "No workspace: CUTLASS reports 0 bytes for this kernel (the "
                    "previous package reserved 32 MiB that was never used)."
                ),
            ],
            "builds": dict(sorted(builds.items())),
            "upstream_coverage": {
                "cases": "every resources/nvfp4_gemm/task.yml test line (smoke "
                "cases task_*, task distribution and seed) and benchmark line "
                "(throughput); FlashInfer tests/gemm/test_mm_fp4.py::test_mm_fp4 "
                "cutlass/nvfp4 cases (1, 128, 512) and (31, 256, 256) with alpha "
                "!= 1 (smoke cases mm_fp4_*; the m = 1 case asks for BF16 output, "
                "run with this FP16 kernel)",
                "not_served": "test_mm_fp4's other backends and the 8x4 scale "
                "layout; test_unified_gemm_fuzz.py's randomized mm_nvfp4 shapes "
                "(regime covered: odd m, partial tiles, m = 1 .. 16384)",
                "check": "tests/test_nvfp4_gemm.py::"
                "test_upstream_parametrizations_are_cases",
            },
            "kernels": {
                WORKLOAD: "live: DeviceGemmFp4GemmSm100_half_128_128_256_1_1_1_1SM "
                "GemmUniversal, one launch per run",
            },
            "excluded_kernels": {
                "DeviceGemmFp4GemmSm100_float_128_128_256_1_1_1_1SM": "dead: FP32 "
                "output, used only by nvfp4_dual_gemm (WORKLOAD_KIND 1)",
                "silu_product": "dead: dual-GEMM epilogue (WORKLOAD_KIND 1 only)",
                "pack_scales": PACK_SCALES,
                "CUBIN/*.cubin": "stale exports of the previous package; each holds "
                "the same four kernels",
            },
        }
        self._write_json(PROVENANCE, provenance)

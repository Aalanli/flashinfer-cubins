"""Batched NVFP4 block-scaled GEMM (FlashInfer's CUTLASS SM100 kernel), one kernel.

``D[:, :, l] = alpha * (A[:, :, l] * SFA) @ (B[:, :, l] * SFB)^T`` for packed
E2M1 A [m, k/2, l] and B [n, k/2, l] (uint8 bytes, two values per byte, low
nibble first, K-major, batch outermost in memory), UE4M3 scales per 16 K
elements, an FP32 device scalar ``alpha`` and FP16 D [m, n, l] (batch
outermost), as NVIDIA's task (``resources/nvfp4_gemm/{task.yml,reference.py}``)
defines it, plus FlashInfer's ``alpha`` (``mm_fp4``'s global-scale product,
1 in the task).

Inputs ``(a, b, sfa, sfb, alpha)``: the scales are in CUTLASS's blocked
scale-factor layout, the layout the NVIDIA task hands customized kernels as
``sfa_permuted``/``sfb_permuted``: a float8_e4m3fn tensor of shape (32, 4,
ceil(mn/128), 4, ceil(k/64), l) whose memory is a contiguous (l,
ceil(mn/128), ceil(k/64), 32, 4, 4) array (``quantization.to_blocked_scales``;
``to_task_scales``/``from_task_scales`` take the task's view). The previous
package converted logical scales with a ``pack_scales`` kernel on every run;
here that relayout is part of input generation and is not timed (see
``impls/nvfp4_gemm/compiler.py``).

The kernel is ``cutlass::device_kernel<...DeviceGemmFp4GemmSm100_half_128_128_
256_1_1_1_1SM...>`` built by ``impls/nvfp4_gemm/compiler.py``. Its single
by-value ``Params`` argument is rebuilt here (``CutlassFp4Gemm``, shared with
``nvfp4_dual_gemm_gemm``, the FP32-output instantiation of the same template),
mirroring ``GemmKernel::to_underlying_arguments`` for the Arguments
FlashInfer's ``prepareGemmArgs`` builds (``alpha_ptr`` -> the alpha input,
beta 0, void C, max_swizzle_size 1, heuristic rasterization, cluster and
fallback cluster 1x1x1). Field offsets, launch constants and static TMA
parameters come from the compile-time probe's sidecar
``cubins/<arch>/<workload>.json``; the builder is checked byte-for-byte
against CUTLASS by the package tests. Like ``GemmUniversalAdapter::run`` as
FlashInfer calls it (``enablePDL=true``), the launch carries a 1x1x1 cluster
and programmatic stream serialization. The kernel needs no workspace.

``alpha`` is read through ``alpha_ptr`` only by the epilogue warps, which run
after the producer warp's ``griddepcontrol.wait`` (``wait_on_dependent_grids``
before the first TMA load), so a value written by an earlier stream-ordered
operation (``get_inputs``) is visible despite the PDL launch: no host
synchronization is needed.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .. import cuda_driver
from ..cutlass_host import ceil_div, cutlass_small_tensor, driver_encode, fast_divmod
from ..registry import register
from ..throughput import gemm_case, model_case, upstream_case
from ..workload import ROOT, CaseSpec, Workload
from .quantization import (
    SF_COLS,
    SF_ROWS,
    SF_VEC,
    empty_fp4_output,
    fp4_matmul,
    from_blocked_scales,
    make_fp4,
    to_blocked_scales,
)

# Encodes one CUtensorMap: (data_type, address, dims, strides_bytes, box,
# element_strides, interleave, swizzle, l2_promotion, oob_fill) -> 128 bytes.
TensorMapEncoder = Callable[..., bytes]

INT32_MAX = (1 << 31) - 1
TASK = ROOT / "resources/nvfp4_gemm/task.yml"
# The task generator's distributions (reference.py): every E2M1 byte, integer
# scales 0..3 (quantization.make_fp4).
TASK_DIST = {"values": "full", "scales": "int0_3"}


def fake_encode(
    data_type: int,
    address: int,
    dims: Sequence[int],
    strides: Sequence[int],
    box: Sequence[int],
    element_strides: Sequence[int],
    interleave: int,
    swizzle: int,
    l2_promotion: int,
    oob_fill: int,
) -> bytes:
    """The compile-time probes' deterministic CUtensorMap stand-in
    (``fake_tensor_map`` in ``impls/nvfp4_gemm/kernels/nvfp4_gemm_probe.cu`` and
    ``impls/nvfp4_dual_gemm/kernels/nvfp4_dual_gemm_probe.cu``); for tests only."""
    out = bytearray(128)
    out[0:6] = bytes(
        (data_type, len(dims), interleave, swizzle, l2_promotion, oob_fill)
    )
    struct.pack_into("<Q", out, 8, address)
    struct.pack_into(f"<{len(dims)}Q", out, 16, *dims)
    struct.pack_into(f"<{len(strides)}Q", out, 56, *strides)
    struct.pack_into(f"<{len(box)}I", out, 88, *box)
    struct.pack_into(f"<{len(element_strides)}I", out, 108, *element_strides)
    return bytes(out)


def blocked_scale_shape(rows: int, k: int, batches: int) -> tuple[int, ...]:
    """Shape of the task's permuted view of the blocked scales for rows x k."""
    return (
        32,
        4,
        ceil_div(rows, SF_ROWS),
        SF_COLS,
        ceil_div(ceil_div(k, SF_VEC), SF_COLS),
        batches,
    )


def to_task_scales(scales: torch.Tensor) -> torch.Tensor:
    """Logical (rows, cols, l) scales -> the task's (32, 4, rb, 4, cb, l) view
    of the blocked layout."""
    return to_blocked_scales(scales).permute(3, 4, 1, 5, 2, 0)


def from_task_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``to_task_scales``: the logical (rows, cols, l) scales."""
    return from_blocked_scales(blocked.permute(5, 2, 4, 0, 1, 3), rows, cols)


class LaunchConfig:
    """Everything one launch needs: the Params bytes and the launch shape."""

    def __init__(
        self,
        params: bytes,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem: int,
        cluster: tuple[int, int, int],
    ):
        self.params = params
        self.grid = grid
        self.block = block
        self.shared_mem = shared_mem
        # GemmUniversalAdapter::run -> ClusterLauncher::launch_with_fallback_cluster
        # with hw_info.cluster_shape(_fallback) = 1x1x1 (the preferred-cluster
        # attribute it may add equals the cluster) and enablePDL=true.
        self.cluster = cluster
        self.programmatic_serialization = True


class CutlassFp4Gemm(Workload):
    """Host side of FlashInfer's ``genericFp4GemmKernelLauncher<T, 128, 128,
    256, 1, 1, 1, _1SM>`` kernels (``T`` = half for nvfp4_gemm, float for
    nvfp4_dual_gemm_gemm): sidecar, Params builder and PDL launch.

    Subclasses set ``OUT_DTYPE`` and implement ``operands`` (tensor checks and
    the problem) plus the workload contract's data side.
    """

    OUT_DTYPE: torch.dtype = torch.float16

    def __init__(self, cubin: bytes | None, *, device=None):
        super().__init__(cubin, device=device)
        self._layout: dict[str, Any] | None = None

    # -- compile-time layout -------------------------------------------------

    @property
    def layout(self) -> dict[str, Any]:
        """The probe sidecar next to this arch's cubin (``<workload>.json``)."""
        if self._layout is None:
            if self.arch is None:
                raise RuntimeError("reference-only workload has no kernel layout")
            path = self.cubin_path(self.arch).with_suffix(".json")
            layout = json.loads(path.read_text())
            if layout["symbol"] != self.image_name():
                raise ValueError(f"{path} describes another kernel")
            if [(p.offset, p.size) for p in self.kernel_params] != [
                (0, layout["params_size"])
            ]:
                raise ValueError("kernel parameters do not match sizeof(Params)")
            self._layout = layout
        return self._layout

    def configure(self, function: cuda_driver.Function) -> None:
        # GemmUniversalAdapter::initialize: smem >= 48 KiB needs the opt-in;
        # ClusterLauncher::init allows non-portable cluster sizes.
        smem = self.layout["constants"]["shared_storage_size"]
        if smem >= 48 << 10:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )
        function.set_attribute(
            cuda_driver.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1
        )

    # -- Params ------------------------------------------------------------------

    def build_params(
        self,
        m: int,
        n: int,
        k: int,
        l: int,  # noqa: E741  (the task's batch axis name)
        a: int,
        b: int,
        sfa: int,
        sfb: int,
        alpha: int,
        d: int,
        *,
        encode: TensorMapEncoder = driver_encode,
        driver: int | None = None,
    ) -> LaunchConfig:
        """Mirror prepareGemmArgs + CUTLASS's Arguments -> Params for device
        addresses ``a``..``d``; ``alpha`` is the address of the FP32 scalar
        read through the fusion's ``alpha_ptr``. ``encode``/``driver`` select
        the CUtensorMap encoder and the driver version CUTLASS's descriptor
        fix-up depends on (default: libcuda's)."""
        layout = self.layout
        fields, const, maps = (
            layout["fields"],
            layout["constants"],
            layout["tensor_maps"],
        )
        out_bytes = torch.empty((), dtype=self.OUT_DTYPE).element_size()
        if min(m, n, k, l) < 1 or k % 32 or (n * out_bytes) % 16:
            raise ValueError(f"unsupported problem m={m} n={n} k={k} l={l}")
        if l > 1 and max(m * k, n * k, m * n) > INT32_MAX:
            raise ValueError("FlashInfer's int batch strides would overflow")
        if any(p % 16 for p in (a, b, sfa, sfb, d)) or alpha % 4:
            raise ValueError("misaligned operand address")
        if driver is None:
            driver = cuda_driver.driver_version()
        if (
            const["sf_vec_size"] != SF_VEC
            or const["atom_thr_shape_mnk"] != [1, 1, 1]
            or not const["is_dynamic_cluster"]
            or const.get("cluster_shape_k", 1) != 1
            or const.get("cluster_shape", [1, 1, 1]) != [1, 1, 1]
            or const.get("cluster_shape_fallback", [1, 1, 1]) != [1, 1, 1]
        ):
            raise AssertionError("builder assumes a 1-SM MMA and a 1x1x1 cluster")
        # Every byte not written below is zero padding (see the fixtures'
        # "padding"): TMA atoms hold only their descriptor (static, i.e. empty,
        # aux strides), and epilogue.tma_load_c stays value-initialized (void C).
        if any(f["size"] for n_, f in fields.items() if n_.endswith(".aux_g_stride")):
            raise AssertionError("builder assumes static TMA aux strides")
        buf = bytearray(layout["params_size"])

        def put(name: str, fmt: str, *values: Any) -> None:
            field = fields[name]
            if struct.calcsize("<" + fmt) != field["size"]:
                raise AssertionError(f"{name}: {fmt} is not {field['size']} bytes")
            struct.pack_into("<" + fmt, buf, field["offset"], *values)

        def descriptor(operand: str, address: int, dims, strides, nbytes) -> bytes:
            static = maps[operand]
            if static["rank"] != len(dims):
                raise AssertionError(f"{operand}: unexpected TMA rank")
            desc = bytearray(
                encode(
                    static["data_type"],
                    address,
                    dims,
                    strides,
                    static["box"],
                    static["element_strides"],
                    static["interleave"],
                    static["swizzle"],
                    static["l2_promotion"],
                    static["oob_fill"],
                )
            )
            # cute::detail::make_tma_copy_desc: drivers <= 13.1 need bit 21 of
            # descriptor word 1 cleared for tensors (cosize) below 128 KiB (as
            # CUTLASS computes the size, see cutlass_small_tensor).
            if driver <= 13010 and cutlass_small_tensor(nbytes):
                (word,) = struct.unpack_from("<Q", desc, 8)
                struct.pack_into("<Q", desc, 8, word & ~(1 << 21))
            return bytes(desc)

        # GemmKernel::Params{mode, problem_shape (m, n, k, l), ...}.
        put("mode", "i", const["gemm_mode_kGemm"])
        put("problem_shape", "4i", m, n, k, l)

        # CollectiveMma::to_underlying_arguments. A (m, k, l) and B (n, k, l)
        # E2M1, K-major; FlashInfer's batch strides are m*k / n*k elements, or 0
        # for l == 1. The fallback descriptors equal the primary ones (cluster
        # and fallback cluster are both 1x1x1).
        batch_a, batch_b = (m * k // 2, n * k // 2) if l > 1 else (0, 0)
        desc_a = descriptor("a", a, [k, m, l], [k // 2, batch_a], l * m * k // 2)
        desc_b = descriptor("b", b, [k, n, l], [k // 2, batch_b], l * n * k // 2)
        # Sm1xxBlkScaledConfig::tile_atom_to_shape_SF{A,B}: 512-byte atoms of
        # 128 rows x 4 scale columns, K blocks inner, then MN blocks, then L:
        # (((32,4),mnb),((16,4),kb),(1,l)) : (((16,4),512kb),((0,1),512),(0,512kb*mnb)).
        # TMA views them as uint16 (256, kb, mnb, l) with byte strides
        # (512, 512kb, 512kb*mnb).
        kb = ceil_div(ceil_div(k, SF_VEC), SF_COLS)
        mb, nb = ceil_div(m, SF_ROWS), ceil_div(n, SF_ROWS)

        def scale_descriptor(operand: str, address: int, blocks: int) -> bytes:
            size = 512 * kb * blocks
            return descriptor(
                operand, address, [256, kb, blocks, l], [512, 512 * kb, size], size * l
            )

        desc_sfa = scale_descriptor("sfa", sfa, mb)
        desc_sfb = scale_descriptor("sfb", sfb, nb)
        for slot, desc in (
            ("mainloop.tma_load_a", desc_a),
            ("mainloop.tma_load_b", desc_b),
            ("mainloop.tma_load_sfa", desc_sfa),
            ("mainloop.tma_load_sfb", desc_sfb),
            ("mainloop.tma_load_a_fallback", desc_a),
            ("mainloop.tma_load_b_fallback", desc_b),
            ("mainloop.tma_load_sfa_fallback", desc_sfa),
            ("mainloop.tma_load_sfb_fallback", desc_sfb),
        ):
            put(slot + ".desc", "128s", desc)
        # The layouts' five dynamic leaves, in member order (the two probes
        # name them differently): MN blocks, K blocks, L, MN-block stride, L
        # stride.
        for operand, blocks in (("A", mb), ("B", nb)):
            name = f"mainloop.layout_SF{operand}"
            leaves = sorted(
                (f["offset"], f["size"])
                for key, f in fields.items()
                if key.startswith(name + ".")
            )
            start = fields[name]["offset"]
            if leaves != [(start + 4 * i, 4) for i in range(5)]:
                raise AssertionError(f"{name}: unexpected dynamic leaves")
            put(name, "5i", blocks, kb, l, 512 * kb, 512 * kb * blocks)
        put("mainloop.cluster_shape_fallback", "3I", 1, 1, 1)
        put("mainloop.runtime_data_type_a", "Q", 0)  # unused for static types
        put("mainloop.runtime_data_type_b", "Q", 0)

        # CollectiveEpilogue::to_underlying_arguments: LinearCombination with the
        # default alpha=1, beta=0 and alpha_ptr (global_sf) set; no C (void);
        # D row-major (m, n, l), batch stride m*n (0 for l == 1).
        default_alpha = const.get("fusion_default_alpha", const.get("default_alpha"))
        default_beta = const.get("fusion_default_beta", const.get("default_beta"))
        put("epilogue.thread.alpha", "f", default_alpha)
        put("epilogue.thread.beta", "f", default_beta)
        put("epilogue.thread.alpha_ptr", "Q", alpha)
        put("epilogue.thread.beta_ptr", "Q", 0)
        put("epilogue.thread.alpha_stride_l", "q", 0)
        put("epilogue.thread.beta_stride_l", "q", 0)
        batch_d = out_bytes * m * n if l > 1 else 0
        put(
            "epilogue.tma_store_d.desc",
            "128s",
            descriptor(
                "d", d, [n, m, l], [out_bytes * n, batch_d], out_bytes * l * m * n
            ),
        )

        # PersistentTileSchedulerSm100::to_underlying_arguments with
        # max_swizzle_size=1 and RasterOrderOptions::Heuristic, cluster 1x1x1.
        tile_m, tile_n, _ = const["cta_shape_mnk"]
        tiles_m, tiles_n = ceil_div(m, tile_m), ceil_div(n, tile_n)
        along_n = tiles_n <= tiles_m and tiles_m <= (1 << 16) - 1
        raster = const["raster_order_AlongN" if along_n else "raster_order_AlongM"]
        put("scheduler.problem_tiles_m", "I", tiles_m)
        put("scheduler.problem_tiles_n", "I", tiles_n)
        put("scheduler.problem_tiles_l", "I", l)
        put("scheduler.divmod_cluster_shape_m", "iII", *fast_divmod(1))
        put("scheduler.divmod_cluster_shape_n", "iII", *fast_divmod(1))
        put("scheduler.divmod_swizzle_size", "iII", 0, 0, 0)  # unused (divisor 0)
        put("scheduler.raster_order", "i", raster)
        put("scheduler.log_swizzle_size", "i", 0)
        # KernelHardwareInfo as FlashInfer sets it (cluster shapes only).
        put("hw_info.device_id", "i", 0)
        put("hw_info.sm_count", "i", 0)
        put("hw_info.max_active_clusters", "i", 0)
        put("hw_info.cluster_shape", "3I", 1, 1, 1)
        put("hw_info.cluster_shape_fallback", "3I", 1, 1, 1)

        # GemmKernel::get_grid_shape -> PersistentTileSchedulerSm100::get_grid_shape:
        # one CTA per output tile and batch, transposed for AlongN rasterization.
        grid = (tiles_n, tiles_m, l) if along_n else (tiles_m, tiles_n, l)
        if grid[1] > (1 << 16) - 1 or grid[2] > (1 << 16) - 1:
            raise ValueError("problem exceeds the grid's y/z limit")
        block = (const["max_threads_per_block"], 1, 1)
        return LaunchConfig(
            bytes(buf), grid, block, const["shared_storage_size"], (1, 1, 1)
        )

    # -- launch ----------------------------------------------------------------

    def operands(self, inputs: tuple, out: torch.Tensor) -> tuple[int, int, int, int]:
        """Check ``inputs`` (a, b, sfa, sfb, alpha) and ``out``; (m, n, k, l)."""
        raise NotImplementedError

    def launch_config(
        self,
        inputs: tuple,
        out: torch.Tensor,
        *,
        encode: TensorMapEncoder = driver_encode,
        driver: int | None = None,
    ) -> LaunchConfig:
        """Check the tensors and build the launch for ``inputs`` into ``out``."""
        m, n, k, l = self.operands(inputs, out)  # noqa: E741
        a, b, sfa, sfb, alpha = inputs
        if tuple(alpha.shape) != (1,) or alpha.dtype != torch.float32:
            raise ValueError("alpha must be one FP32 element")
        if any(t.device != out.device for t in inputs):
            raise ValueError("operands must be on one device")
        if encode is driver_encode:
            cuda_driver.ensure_context(out.device)
        return self.build_params(
            m,
            n,
            k,
            l,
            a.data_ptr(),
            b.data_ptr(),
            sfa.data_ptr(),
            sfb.data_ptr(),
            alpha.data_ptr(),
            out.data_ptr(),
            encode=encode,
            driver=driver,
        )

    def _launch(self, config: LaunchConfig) -> None:
        self.launch(
            config.grid,
            config.block,
            [config.params],
            shared_mem=config.shared_mem,
            cluster=config.cluster,
            programmatic_serialization=config.programmatic_serialization,
        )

    def allocate_outputs(self, inputs):
        return (empty_fp4_output(inputs[0], inputs[1], self.OUT_DTYPE),)

    def _check_supported(self) -> None:
        # Fail before any driver work (descriptor encoding needs sm_90+).
        if not self.is_supported():
            raise RuntimeError(f"{self.arch} cubin cannot run on {self.device}")

    def run(self, inputs):
        self._check_supported()
        (out,) = self.allocate_outputs(inputs)
        self._launch(self.launch_config(inputs, out))
        return (out,)

    def prepare(self, inputs):
        self._check_supported()
        outputs = self.allocate_outputs(inputs)
        config = self.launch_config(inputs, outputs[0])

        def launch() -> tuple:
            self._launch(config)
            return outputs

        return launch, outputs


def check_layout(tensor, shape, dtype, order) -> None:
    """``tensor`` has ``shape``/``dtype`` and ``permute(*order)`` is contiguous."""
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(
            f"expected {dtype} {tuple(shape)}, got {tensor.dtype} {tuple(tensor.shape)}"
        )
    if not tensor.permute(*order).is_contiguous():
        raise ValueError(f"{dtype} {tuple(shape)} operand has an unsupported layout")


def fp4_case(m, n, k, l=1, *, seed, suite="smoke", **extra) -> CaseSpec:  # noqa: E741
    """A plain (m, n, k, l) case; ``extra`` holds non-default distributions
    (``values``/``scales``) and ``alpha``, also named in the case."""
    params = dict(m=m, n=n, k=k, l=l, **extra)
    label = "_".join(f"{key}{value}" for key, value in params.items())
    return CaseSpec(label, params, seed, suite)


def task_shapes(path, section: str) -> list[dict[str, Any]]:
    """The ``tests`` or ``benchmarks`` entries of a gpu-mode task.yml."""
    lines = path.read_text().split(f"\n{section}:", 1)[1].splitlines()[1:]
    entries = []
    for line in lines:
        item = line.strip()
        if item.startswith("- {"):
            entries.append(json.loads(item[2:]))
        elif item:
            break
    return entries


def task_cases(path, name: str, **extra) -> list[CaseSpec]:
    """Every distinct test line of the task (``upstream_case``, the task's seed)."""
    cases: dict[str, CaseSpec] = {}
    for index, s in enumerate(task_shapes(path, "tests")):
        label = "task_" + "_".join(f"{key}{s[key]}" for key in ("m", "n", "k", "l"))
        cases.setdefault(
            label,
            upstream_case(
                label,
                dict(m=s["m"], n=s["n"], k=s["k"], l=s["l"], **extra),
                f"{name}/task.yml::tests[{index}]",
                seed=s["seed"],
            ),
        )
    return list(cases.values())


@register(name="nvfp4_gemm", supported_arches=("sm_100a",))
class NVFP4GEMM(CutlassFp4Gemm):
    """Batched NT NVFP4 GEMM with UE4M3/16 block scales, alpha and FP16 output.

    Wraps ``genericFp4GemmKernelLauncher<half, 128, 128, 256, 1, 1, 1, _1SM>``
    (128x128x256 tile, dynamic cluster launched 1x1x1, CLC persistent
    scheduler, 5 stages). Requires k % 32 == 0 (16-byte TMA rows of packed
    A/B) and n % 8 == 0 (16-byte D rows).

    Smoke cases (default distribution: the task's, ``TASK_DIST``): partial M,
    N and K tiles (k = 96: K below one 256-wide MMA tile and 6 scale columns,
    a partial scale atom), batches l = 2, 3, a K loop of 9 tiles (longer than
    the 5-stage pipeline) ending in a partial tile, 195 output tiles (more than
    a B200's 148 SMs, so CLC hands CTAs further tiles), both rasterizations,
    zero (``int0_3``), subnormal (``unit``) and three-binade (``uniform``)
    scales, alpha != 1, every NVIDIA task test line and FlashInfer's
    ``mm_fp4`` cutlass cases (``UPSTREAM``).
    """

    OUT_DTYPE = torch.float16

    SMOKE = (
        fp4_case(128, 128, 128, seed=1),
        fp4_case(128, 256, 256, 2, seed=2),
        fp4_case(256, 128, 512, seed=3),
        fp4_case(200, 136, 96, seed=4),
        fp4_case(200, 136, 96, seed=5, scales="uniform"),
        fp4_case(300, 264, 384, 3, seed=6, alpha=0.0123),
        fp4_case(129, 1032, 2080, 2, seed=7),
        fp4_case(640, 1544, 768, 3, seed=8),
        fp4_case(192, 520, 1024, 2, seed=9, scales="unit"),
        fp4_case(17, 4104, 4096, seed=10, scales="uniform", alpha=0.37),
        fp4_case(1000, 264, 512, seed=11, values="restricted", scales="int1_2"),
    )

    def operands(self, inputs, out):
        a, b, sfa, sfb, _ = inputs
        if a.dim() != 3 or b.dim() != 3:
            raise ValueError("A and B must be (rows, k/2, l)")
        m, half_k, l = a.shape  # noqa: E741
        n, k = b.shape[0], 2 * half_k
        blocked = (5, 2, 4, 0, 1, 3)
        check_layout(a, (m, half_k, l), torch.uint8, (2, 0, 1))
        check_layout(b, (n, half_k, l), torch.uint8, (2, 0, 1))
        check_layout(sfa, blocked_scale_shape(m, k, l), torch.float8_e4m3fn, blocked)
        check_layout(sfb, blocked_scale_shape(n, k, l), torch.float8_e4m3fn, blocked)
        check_layout(out, (m, n, l), torch.float16, (2, 0, 1))
        return m, n, k, l

    # -- workload contract ---------------------------------------------------------

    def get_cases(self):
        cases = [*self.SMOKE, *task_cases(TASK, "nvfp4_gemm")]
        cases += [
            upstream_case(
                f"mm_fp4_m{m}_n{n}_k{k}",
                dict(m=m, n=n, k=k, l=1, alpha=alpha),
                test,
                seed=index,
            )
            for index, (test, (m, n, k), alpha) in enumerate(UPSTREAM)
        ]
        return cases + throughput_cases()

    def get_inputs(self, case):
        p, g = case.params, self.generator(case)
        dist = {key: p.get(key, TASK_DIST[key]) for key in TASK_DIST}
        a, sa = make_fp4(self, g, p["m"], p["k"], p["l"], **dist)
        b, sb = make_fp4(self, g, p["n"], p["k"], p["l"], **dist)
        alpha = torch.full((1,), p.get("alpha", 1.0), device=self.device)
        return a, b, to_task_scales(sa), to_task_scales(sb), alpha

    def get_reference(self, inputs):
        a, b, sfa, sfb, alpha = inputs
        cols = 2 * a.shape[1] // SF_VEC
        sa = from_task_scales(sfa, a.shape[0], cols)
        sb = from_task_scales(sfb, b.shape[0], cols)
        out = fp4_matmul(
            a, b, sa, sb, alpha=float(alpha.item()), out_dtype=torch.float16
        )
        return (out,)

    def validate(self, ref, impl):
        # NVIDIA's tolerance (reference.py: rtol=atol=1e-3). Both accumulate
        # exact FP4 x UE4M3 products in FP32 and round once to FP16 (after the
        # same FP32 alpha multiplication). With integer scales the FP32 sums
        # are exact in any order (quantization.py), so outputs are identical;
        # with unit/uniform scales the orders differ by far less than an FP16
        # ulp (< 2**-10 relative), so they differ by at most one ulp.
        self.assert_close(ref, impl, rtol=1e-3, atol=1e-3)


# FlashInfer tests (pinned revision, tests/gemm/test_mm_fp4.py::test_mm_fp4)
# that run this template (cutlass backend, nvfp4, 128x4 scale layout) and the
# alpha they pass (a global-scale product; any value != 1 exercises alpha_ptr).
# The m = 1 case asks for BF16 output, whose instantiation is not built: it
# runs here with FP16 output (same mainloop, tile and scheduler).
UPSTREAM = (
    ("tests/gemm/test_mm_fp4.py::test_mm_fp4[1-128-512-bfloat16-cutlass]", (1, 128, 512), 0.0123),
    ("tests/gemm/test_mm_fp4.py::test_mm_fp4[31-256-256-float16-cutlass]", (31, 256, 256), 0.0123),
)  # fmt: skip

THROUGHPUT = "nvfp4_gemm"

# (model, decode/prefill m values): every linear layer of each model
# (harness.models linear_shapes; all satisfy k % 32 == 0, n % 8 == 0).
MODEL_PLAN = (
    ("deepseek_v3", (1, 16, 4096, 16384)),
    ("llama3_70b", (1, 16, 4096, 16384)),
    ("llama3_405b", (8, 8192)),
    ("qwen3_235b_a22b", (8, 8192)),
    ("gpt_oss_120b", (8, 8192)),
    ("llama4_maverick", (8, 8192)),
)


def model_gemm_cases(plan, prefix=""):
    """``model_case`` (m, n, k, l=1) for every linear layer of each planned model."""
    from ..models import MODELS

    cases = []
    for model, ms in plan:
        for layer, (n, k) in MODELS[model].linear_shapes().items():
            for m in ms:
                cases.append(
                    model_case(
                        f"{prefix}{model}_{layer}_m{m}",
                        dict(m=m, n=n, k=k, l=1),
                        model,
                        layer,
                    )
                )
    return cases


def throughput_cases():
    """Throughput (benchmark) cases: every task benchmark line and the linear
    layers of DeepSeek-V3, Llama-3.1 70B/405B, Qwen3-235B, gpt-oss-120b and
    Llama-4 Maverick from decode (m = 1..16) to 16k-token prefill."""
    name = THROUGHPUT
    source = {
        "kind": "official_shape",
        "snapshot": f"resources/{name}/task.yml",
        "url": f"https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/{name}/task.yml",
        "selection": "All benchmark (not just test) shapes; generated using the harness quantization contract.",
    }
    cases = [
        gemm_case(name, (s["m"], s["n"], s["k"], s["l"]), source)
        for s in task_shapes(TASK, "benchmarks")
    ]
    seen = {tuple(c.params.values()) for c in cases}
    for case in model_gemm_cases(MODEL_PLAN):
        if tuple(case.params.values()) not in seen:
            seen.add(tuple(case.params.values()))
            cases.append(case)
    return cases

"""Groupwise-scaled FP8 GEMM (FlashInfer's CUTLASS SM100 kernels), one kernel each.

``D[m, n] = (A * SFA) @ (B * SFB)^T`` with A [m, k] and B [n, k] FP8 E4M3
(K-major), SFA [m, k/128] (one scale per row and 128-wide K block), SFB
[ceil(n/128), k/128] (one scale per 128x128 block), both FP32 K-major
contiguous, and BF16 D [m, n] row-major. These are exactly the layouts both
kernels consume, so ``run`` does no relayout.

FlashInfer's dispatcher (``csrc/gemm_groupwise_sm100.cu``) runs one of two
kernels per call, and so does this module, one workload per kernel:

* ``fp8_gemm_small_m`` (m <= 32): ``CutlassGroupwiseScaledGEMMSM100LowLatency``,
  a swap-AB kernel computing ``D^T[n, m] = B A^T`` with a 128x16 tile;
* ``fp8_gemm`` (m > 32): ``CutlassGroupwiseScaledGEMMSM100<..., MmaSM=1>``,
  128x128 tile.

Both kernels are ``cutlass::device_kernel<GemmUniversal<...>>`` built by
``impls/fp8_gemm/compiler.py`` (see its live/dead table). Their single
by-value ``Params`` argument is rebuilt here, mirroring ``GemmKernel::
to_underlying_arguments`` for the Arguments FlashInfer's host functions build
(alpha=1, beta=0; C=D for ``fp8_gemm``, no C for ``fp8_gemm_small_m``). Field
offsets, launch constants, static TMA parameters, the launch cluster and the
hardware query come from the compile-time probe's sidecar
``cubins/<arch>/<workload>.json``; the builder is checked byte-for-byte against
CUTLASS by ``tests/test_fp8_gemm.py``. Neither kernel needs a workspace
(CUTLASS reports 0 bytes).
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
from ..throughput import model_case, upstream_case
from ..workload import CaseSpec, Workload

# Encodes one CUtensorMap: (data_type, address, dims, strides_bytes, box,
# element_strides, interleave, swizzle, l2_promotion, oob_fill) -> 128 bytes.
TensorMapEncoder = Callable[..., bytes]

# csrc/gemm_groupwise_sm100.cu: scale granularity M == 1 and m <= 32 run the
# low-latency kernel (fp8_gemm_small_m), everything else fp8_gemm.
SMALL_M_MAX_M = 32

# Elements of the FP32 reference temporaries materialized at a time.
CHUNK = 1 << 26


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
    """The compile-time probe's deterministic CUtensorMap stand-in (``fake_tensor_map``
    in ``impls/fp8_gemm/kernels/fp8_gemm_probe.cu``); for tests only."""
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


driver_version = cuda_driver.driver_version


class LaunchConfig:
    """Everything one launch needs: the Params bytes and the launch shape."""

    def __init__(
        self,
        params: bytes,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem: int,
        cluster: tuple[int, int, int] | None = None,
    ):
        self.params = params
        self.grid = grid
        self.block = block
        self.shared_mem = shared_mem
        # None: launched without a cluster attribute (GemmUniversalAdapter::run
        # for a static 1x1x1 cluster kernel).
        self.cluster = cluster


class _GroupwiseFP8GEMM(Workload):
    """Shared host setup of the package's two CUTLASS blockwise-scaled kernels."""

    def __init__(self, cubin: bytes | None, *, device=None):
        super().__init__(cubin, device=device)
        self._layout: dict[str, Any] | None = None
        self._hardware: tuple[int, int] | None = None

    @classmethod
    def serves(cls, m: int) -> bool:
        """Whether upstream's dispatcher sends a problem with ``m`` rows here."""
        raise NotImplementedError

    @classmethod
    def validated(cls, m: int, k: int) -> bool:
        """Whether this kernel is validated for the shape (see VALIDATION.md)."""
        return True

    # -- compile-time layout -------------------------------------------------

    @property
    def layout(self) -> dict[str, Any]:
        """The probe sidecar next to this arch's cubin (``<workload>.json``)."""
        if self._layout is None:
            if self.arch is None:
                raise RuntimeError("reference-only workload has no kernel layout")
            path = self.cubin_path(self.arch).with_suffix(".json")
            layout = json.loads(path.read_text())
            if layout["workload"] != self.name or layout["symbol"] != self.image_name():
                raise ValueError(f"{path} describes another kernel")
            if [(p.offset, p.size) for p in self.kernel_params] != [
                (0, layout["params_size"])
            ]:
                raise ValueError("kernel parameters do not match sizeof(Params)")
            self._layout = layout
        return self._layout

    def configure(self, function: cuda_driver.Function) -> None:
        # GemmUniversalAdapter::initialize: smem >= 48 KiB needs the opt-in.
        smem = self.layout["constants"]["shared_storage_size"]
        if smem >= 48 << 10:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )

    def hardware_info(self) -> tuple[int, int]:
        """(sm_count, max_active_clusters) as ``KernelHardwareInfo::
        make_kernel_hardware_info<GemmKernel>()`` queries them on this device.

        Only kernels whose upstream host function passes a KernelHardwareInfo
        (the sidecar records its occupancy query) store them in Params; the
        device code never reads them. A failing occupancy query yields 0, as
        in CUTLASS.
        """
        query = self.layout["occupancy_query"]
        if query is None:
            return 0, 0
        if self._hardware is None:
            sm_count = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count
            # cudaOccupancyMaxActiveClusters(kernel, config) through its driver
            # equivalent, with the recorded launch configuration.
            clusters = None
            for attr in query["attrs"]:
                if attr["id"] != cuda_driver.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION:
                    raise ValueError(f"unexpected occupancy attribute {attr}")
                clusters = attr["cluster_dim"]
            active = cuda_driver.max_active_clusters(
                self.function,
                query["grid"],
                query["block"],
                query["shared_mem"],
                clusters,
            )
            self._hardware = (sm_count, active)
        return self._hardware

    # -- Params ------------------------------------------------------------------

    def build_params(
        self,
        m: int,
        n: int,
        k: int,
        a: int,
        b: int,
        sfa: int,
        sfb: int,
        d: int,
        *,
        encode: TensorMapEncoder = driver_encode,
        driver: int | None = None,
        hardware: tuple[int, int] = (0, 0),
    ) -> LaunchConfig:
        """Mirror the upstream host function's Arguments -> Params for the
        problem ``D[m, n] = A[m, k] B[n, k]^T`` at device addresses ``a``..``d``.

        ``encode``/``driver`` select the CUtensorMap encoder and the driver
        version CUTLASS's descriptor fix-up depends on (default: libcuda's);
        ``hardware`` is (sm_count, max_active_clusters) of ``hardware_info``.
        """
        layout = self.layout
        fields, const, maps = (
            layout["fields"],
            layout["constants"],
            layout["tensor_maps"],
        )
        if min(m, n, k) < 1 or k % 16 or n % 8:
            raise ValueError(f"unsupported problem m={m} n={n} k={k}")
        if any(p % 16 for p in (a, b, d)) or any(p % 4 for p in (sfa, sfb)):
            raise ValueError("misaligned operand address")
        if driver is None:
            driver = driver_version()
        # Every byte not written below is padding (see the fixtures' "padding"):
        # TMA atoms hold only their descriptor (static, i.e. empty, aux strides).
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
            # descriptor word 1 cleared for tensors smaller than 128 KiB (as
            # CUTLASS computes the size, see cutlass_small_tensor).
            if driver <= 13010 and cutlass_small_tensor(nbytes):
                (word,) = struct.unpack_from("<Q", desc, 8)
                struct.pack_into("<Q", desc, 8, word & ~(1 << 21))
            return bytes(desc)

        # The kernel's own problem. The low-latency host function swaps (m, n),
        # (A, B) and (SFA, SFB): the kernel computes D^T[n, m] = B A^T with
        # B (n, k) as its K-major "A", A (m, k) as its K-major "B", and stores
        # its (n, m) result column-major, i.e. into row-major D[m, n].
        swap = const["swap_ab"]
        km, kn = (n, m) if swap else (m, n)
        ka, kb, ksfa, ksfb = (b, a, sfb, sfa) if swap else (a, b, sfa, sfb)

        # GemmKernel::Params{mode, problem_shape, ...}; L = 1.
        put("mode", "i", const["gemm_mode_kGemm"])
        put("problem_shape", "4i", km, kn, k, 1)

        # KernelHardwareInfo: default (all zero) for fp8_gemm; for the
        # low-latency kernel make_kernel_hardware_info() (device_id 0) with
        # cluster_shape = cluster_shape_fallback = 1x1x1.
        if const["is_dynamic_cluster"]:
            sm_count, max_active_clusters = hardware
            cluster_shape = tuple(layout["preferred_cluster"])
            cluster_fallback = tuple(layout["cluster"])
        else:
            sm_count = max_active_clusters = 0
            cluster_shape = cluster_fallback = (0, 0, 0)

        # CollectiveMma::to_underlying_arguments: K-major A (km,k,1) and B (kn,k,1)
        # with packed strides (k, 1, 0) as make_cute_packed_stride gives for L=1;
        # the fallback descriptors equal the primary ones (1x1x1 clusters: no
        # multicast).
        desc_a = descriptor("a", ka, [k, km, 1], [k, 0], km * k)
        desc_b = descriptor("b", kb, [k, kn, 1], [k, 0], kn * k)
        put("mainloop.runtime_data_type_a", "Q", 0)  # unused for static types
        put("mainloop.runtime_data_type_b", "Q", 0)
        for slot, desc in (
            ("mainloop.tma_load_a", desc_a),
            ("mainloop.tma_load_b", desc_b),
            ("mainloop.tma_load_a_fallback", desc_a),
            ("mainloop.tma_load_b_fallback", desc_b),
        ):
            put(slot + ".desc", "128s", desc)
        put("mainloop.cluster_shape_fallback", "3I", *cluster_fallback)
        put("mainloop.ptr_SFA", "Q", ksfa)
        put("mainloop.ptr_SFB", "Q", ksfb)
        # ScaleConfig::tile_atom_to_shape_SF{A,B} (K-major) of the kernel's
        # granularities (swapped M/N for the low-latency kernel):
        # ((gm|gn, ceil(km|kn/g)), (128, ceil(k/128)), L):((0, ceil(k/128)), (0, 1), size)
        gm, gn, gk = const["scale_granularity_mnk"]
        k_blocks, m_blocks, n_blocks = (
            ceil_div(k, gk),
            ceil_div(km, gm),
            ceil_div(kn, gn),
        )
        put("mainloop.layout_SFA.shape_m_blocks", "i", m_blocks)
        put("mainloop.layout_SFA.shape_k_blocks", "i", k_blocks)
        put("mainloop.layout_SFA.shape_l", "i", 1)
        put("mainloop.layout_SFA.stride_m", "i", k_blocks)
        put("mainloop.layout_SFA.stride_l", "i", m_blocks * k_blocks)
        put("mainloop.layout_SFB.shape_n_blocks", "i", n_blocks)
        put("mainloop.layout_SFB.shape_k_blocks", "i", k_blocks)
        put("mainloop.layout_SFB.shape_l", "i", 1)
        put("mainloop.layout_SFB.stride_n", "i", k_blocks)
        put("mainloop.layout_SFB.stride_l", "i", n_blocks * k_blocks)

        # CollectiveEpilogue::to_underlying_arguments: LinearCombination with
        # alpha=1, beta=0 (null scalar pointers, zero strides) and BF16 D. The
        # kernel's D is row-major (km, kn) (strides (kn, 1, 0)) or, for swap-AB,
        # column-major (km, kn) (strides (1, km, 0)); both are row-major D[m, n]
        # in memory. fp8_gemm also loads C = D; the low-latency kernel has no
        # source, its tma_load_c stays value-initialized (zero).
        put("epilogue.thread.alpha", "f", 1.0)
        put("epilogue.thread.beta", "f", 0.0)
        put("epilogue.thread.alpha_ptr", "Q", 0)
        put("epilogue.thread.beta_ptr", "Q", 0)
        put("epilogue.thread.alpha_stride_l", "q", 0)
        put("epilogue.thread.beta_stride_l", "q", 0)
        d_dims, d_strides = (
            ([km, kn, 1], [2 * km, 0]) if swap else ([kn, km, 1], [2 * kn, 0])
        )
        if const["epilogue_source_supported"]:
            put(
                "epilogue.tma_load_c.desc",
                "128s",
                descriptor("c", d, d_dims, d_strides, 2 * m * n),
            )
        elif "c" in maps:
            raise AssertionError("C descriptor for a kernel without source")
        put(
            "epilogue.tma_store_d.desc",
            "128s",
            descriptor("d", d, d_dims, d_strides, 2 * m * n),
        )

        # PersistentTileSchedulerSm100::to_underlying_arguments with default
        # Arguments (max_swizzle_size=0, RasterOrderOptions::Heuristic) and the
        # static or hw_info cluster shape.
        tile_m, tile_n, _ = const["cta_shape_mnk"]
        if const["is_dynamic_cluster"]:
            cluster_m, cluster_n, _ = cluster_shape
        else:
            cluster_m, cluster_n, _ = const["cluster_shape_mnk"]
        if const["atom_thr_shape_mnk"] != [1, 1, 1] or (cluster_m, cluster_n) != (1, 1):
            raise AssertionError("builder assumes a 1-SM MMA and a 1x1 cluster")
        blocks_m = ceil_div(ceil_div(km, tile_m), cluster_m) * cluster_m
        blocks_n = ceil_div(ceil_div(kn, tile_n), cluster_n) * cluster_n
        tiles_m, tiles_n = blocks_m // cluster_m, blocks_n // cluster_n
        along_n = tiles_n <= tiles_m and tiles_m * cluster_n <= (1 << 16) - 1
        raster = const["raster_order_AlongN" if along_n else "raster_order_AlongM"]
        put("scheduler.problem_tiles_m", "I", tiles_m)
        put("scheduler.problem_tiles_n", "I", tiles_n)
        put("scheduler.problem_tiles_l", "I", 1)
        put("scheduler.divmod_cluster_shape_m", "iII", *fast_divmod(cluster_m))
        put("scheduler.divmod_cluster_shape_n", "iII", *fast_divmod(cluster_n))
        put("scheduler.divmod_swizzle_size", "iII", 0, 0, 0)  # unused (divisor 0)
        put("scheduler.raster_order", "i", raster)
        put("scheduler.log_swizzle_size", "i", 0)
        put("hw_info.device_id", "i", 0)
        put("hw_info.sm_count", "i", sm_count)
        put("hw_info.max_active_clusters", "i", max_active_clusters)
        put("hw_info.cluster_shape", "3I", *cluster_shape)
        put("hw_info.cluster_shape_fallback", "3I", *cluster_fallback)

        # GemmKernel::get_grid_shape -> PersistentTileSchedulerSm100::get_grid_shape:
        # one CTA per output tile, transposed for AlongN rasterization.
        grid = (blocks_m, blocks_n, 1)
        if along_n:
            grid = (
                (blocks_n // cluster_n) * cluster_m,
                (blocks_m // cluster_m) * cluster_n,
                1,
            )
        if grid[1] > (1 << 16) - 1:
            raise ValueError("problem exceeds the grid's y limit")
        block = (const["max_threads_per_block"], 1, 1)
        # GemmUniversalAdapter::run: a dynamic-cluster kernel goes through
        # ClusterLauncher::launch_with_fallback_cluster, whose cluster dimension
        # attribute is hw_info.cluster_shape_fallback (its preferred cluster
        # dimension, hw_info.cluster_shape = the same 1x1x1, is redundant).
        cluster: tuple[int, int, int] | None = None
        if layout["cluster"] is not None:
            x, y, z = layout["cluster"]
            cluster = (x, y, z)
        return LaunchConfig(
            bytes(buf),
            grid,
            block,
            const["shared_storage_size"],
            cluster,
        )

    def launch_config(
        self,
        inputs: tuple,
        out: torch.Tensor,
        *,
        encode: TensorMapEncoder = driver_encode,
        driver: int | None = None,
        hardware: tuple[int, int] | None = None,
    ) -> LaunchConfig:
        """Check the tensors and build the launch for ``inputs`` into ``out``.

        ``hardware`` defaults to this device's ``hardware_info()`` when the real
        encoder is used, else (0, 0).
        """
        a, b, sfa, sfb = inputs
        m, k = a.shape
        n = b.shape[0]
        k_blocks = ceil_div(k, 128)
        expected = (
            (a, (m, k), torch.float8_e4m3fn),
            (b, (n, k), torch.float8_e4m3fn),
            (sfa, (m, k_blocks), torch.float32),
            (sfb, (ceil_div(n, 128), k_blocks), torch.float32),
            (out, (m, n), torch.bfloat16),
        )
        for tensor, shape, dtype in expected:
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(
                    f"expected {dtype} {shape}, got {tensor.dtype} {tuple(tensor.shape)}"
                )
            if not tensor.is_contiguous() or tensor.device != out.device:
                raise ValueError("operands must be contiguous and on one device")
        if encode is driver_encode:
            cuda_driver.ensure_context(out.device)
            if hardware is None:
                hardware = self.hardware_info()
        return self.build_params(
            m,
            n,
            k,
            a.data_ptr(),
            b.data_ptr(),
            sfa.data_ptr(),
            sfb.data_ptr(),
            out.data_ptr(),
            encode=encode,
            driver=driver,
            hardware=hardware or (0, 0),
        )

    def _launch(self, config: LaunchConfig) -> None:
        self.launch(
            config.grid,
            config.block,
            [config.params],
            shared_mem=config.shared_mem,
            cluster=config.cluster,
        )

    # -- workload contract ---------------------------------------------------------

    def get_cases(self):
        # Every shape of the package's case lists runs on the kernel upstream
        # dispatches it to (and is validated there).
        cases = [
            case
            for case in (*smoke_cases(), *upstream_cases(), *throughput_cases())
            if self.serves(case.params["m"])
            and self.validated(case.params["m"], case.params["k"])
        ]
        names = [case.name for case in cases]
        if len(names) != len(set(names)):
            raise AssertionError(f"{self.name}: duplicate case names")
        return cases

    def get_inputs(self, case):
        p, g = case.params, self.generator(case)
        m, n, k = p["m"], p["n"], p["k"]

        def scales(shape):
            u = torch.rand(shape, device=self.device, generator=g)
            if p.get("scales") == "wide":  # 2**-6 .. 2**6 times 0.01: 12 binades
                return torch.exp2(u * 12 - 6) * 0.01
            return u * 0.1 + 0.01

        return (
            self.randn((m, k), g, torch.float8_e4m3fn),
            self.randn((n, k), g, torch.float8_e4m3fn),
            scales((m, k // 128)),
            scales((ceil_div(n, 128), k // 128)),
        )

    def get_reference(self, inputs):
        # FP32 (A * SFA) @ (B * SFB)^T, B dequantized in 128-row-aligned chunks
        # so that FP32 copies of the largest weights are never materialized
        # whole (the per-element result does not depend on the chunking).
        a, b, sa, sb = inputs
        m, k = a.shape
        n = b.shape[0]
        lhs = a.float() * sa.repeat_interleave(128, -1)
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        step = max(128, CHUNK // max(k, m, 1) // 128 * 128)
        for start in range(0, n, step):
            rhs = b[start : start + step].float()
            scale = sb[start // 128 : ceil_div(start + step, 128)]
            rhs *= scale.repeat_interleave(128, 0)[: rhs.shape[0]].repeat_interleave(
                128, 1
            )
            out[:, start : start + step] = lhs @ rhs.T
        return (out,)

    def allocate_outputs(self, inputs):
        return (
            torch.empty(
                (inputs[0].shape[0], inputs[1].shape[0]),
                device=inputs[0].device,
                dtype=torch.bfloat16,
            ),
        )

    def _check_supported(self, inputs: tuple) -> None:
        # Fail before any driver work (descriptor encoding needs sm_90+).
        m, k = inputs[0].shape
        if not self.validated(m, k):
            raise ValueError(
                f"{self.name}: m={m}, k={k} is outside the validated range"
            )
        if not self.is_supported():
            raise RuntimeError(f"{self.arch} cubin cannot run on {self.device}")

    def run(self, inputs):
        self._check_supported(inputs)
        (out,) = self.allocate_outputs(inputs)
        self._launch(self.launch_config(inputs, out))
        return (out,)

    def prepare(self, inputs):
        self._check_supported(inputs)
        outputs = self.allocate_outputs(inputs)
        config = self.launch_config(inputs, outputs[0])
        return (lambda: (self._launch(config), outputs)[1]), outputs

    def validate(self, ref, impl):
        # FP32 accumulation in both; outputs differ by at most BF16 rounding.
        self.assert_close(ref, impl, rtol=1e-2, atol=1e-2)


@register(name="fp8_gemm", supported_arches=("sm_100a",))
class FP8GEMM(_GroupwiseFP8GEMM):
    """NT GEMM, A scales per (1,128), B scales per (128,128), BF16 output, m > 32.

    Wraps ``CutlassGroupwiseScaledGEMMSM100<1, 128, 128, ScaleMajorK=true,
    MmaSM=1>`` (128x128x128 tile, static 1x1x1 cluster, CLC persistent
    scheduler, 5 stages, C = D), the kernel FlashInfer dispatches m > 32 to.
    Requires k % 16 == 0 and n % 8 == 0 (TMA stride alignment); upstream's
    scale shapes (``k // 128`` blocks) require k % 128 == 0, so a K tile is
    never partial.

    Cases (``smoke_cases``, ``upstream_cases``, ``throughput_cases`` with
    m > 32): partial M and N tiles in both rasterization orders, the dispatch
    boundary m = 33, a 32-tile K loop (beyond the 5 stages), 288 tiles (more
    than a B200's 148 SMs: CLC hands CTAs further tiles), 12-binade scales,
    all 125 K-major shapes of FlashInfer's ``test_fp8_groupwise_gemm``
    (cutlass backend), DeepGEMM's FP8 ``enumerate_normal`` forward shapes with
    m = 128, 4096 and model linear layers up to m = 16384.
    """

    @classmethod
    def serves(cls, m: int) -> bool:
        return m > SMALL_M_MAX_M


@register(name="fp8_gemm_small_m", supported_arches=("sm_100a",))
class FP8GEMMSmallM(_GroupwiseFP8GEMM):
    """The same GEMM for m <= 32 on FlashInfer's low-latency swap-AB kernel.

    Wraps ``CutlassGroupwiseScaledGEMMSM100LowLatency<1, 128, 128,
    ScaleMajorK=true, MmaSM=1>``, the kernel FlashInfer's dispatcher uses for
    m <= 32: it computes ``D^T[n, m] = B[n, k] A[m, k]^T`` (B is the kernel's
    "A" with (128, 128) scales, A its "B" with (1, 128) scales) on a 128x16x128
    tile with a dynamic cluster launched as 1x1x1 and no C source, writing the
    column-major (n, m) result, i.e. row-major ``D[m, n]``. Inputs, reference
    and validation are those of ``fp8_gemm``; same alignment requirements.

    Restricted to ``m <= 16`` (``validated``): with two CTAs along m the
    kernel races on a B200 (see VALIDATION.md); ``run`` rejects such shapes,
    so 16 < m <= 32 is currently neither benchmarked nor tested, including
    upstream's ``test_fp8_groupwise_gemm_small_batch_size`` m = 32 cases
    (``UNSERVED_UPSTREAM``). Cases: partial tiles of the transposed problem in
    both dimensions (n % 128 != 0, m % 16 != 0), 157 tiles along n (more than
    148 SMs), 12-binade scales, upstream's small-batch m = 1, 4, 16 cases,
    DeepGEMM's m = 1 shapes and decode GEMMs (m = 1, 4, 8, 16) of DeepSeek-V3
    (all linear layers, incl. n = 24576: 192 tiles), Llama-3.1 70B and
    Qwen3-235B. Every case is a probe fixture.
    """

    @classmethod
    def serves(cls, m: int) -> bool:
        return m <= SMALL_M_MAX_M

    @classmethod
    def validated(cls, m: int, k: int) -> bool:
        # On a B200 the kernel returns nondeterministically wrong values when
        # the swapped problem has two CTAs along x (m > 16), already at
        # K = 256 for large n; racecheck reports shared-memory hazards on its
        # cp.async scale loads. Restricted to one CTA along x until resolved.
        return m <= 16


def fp8_case(m, n, k, seed, **extra) -> CaseSpec:
    params = dict(m=m, n=n, k=k, **extra)
    return CaseSpec(
        "_".join(f"{key}{value}" for key, value in params.items()), params, seed
    )


def smoke_cases() -> list[CaseSpec]:
    """Harness smoke cases of both kernels (each serves its m range)."""
    return [
        # fp8_gemm (m > 32)
        fp8_case(128, 512, 4096, 128),
        fp8_case(200, 264, 512, 200),  # partial M and N tiles, AlongM
        fp8_case(300, 136, 384, 300),  # partial M and N tiles, AlongN
        fp8_case(33, 1032, 256, 33),  # smallest m upstream sends here
        fp8_case(1536, 3000, 768, 1),  # 288 tiles > 148 SMs, partial N
        fp8_case(200, 264, 512, 2, scales="wide"),
        # fp8_gemm_small_m (m <= 16)
        fp8_case(4, 128, 512, 4),
        fp8_case(1, 136, 256, 1),
        fp8_case(7, 264, 512, 7),
        fp8_case(16, 7168, 2048, 16),
        fp8_case(13, 19976, 512, 3),  # 157 tiles of the swapped problem
        fp8_case(5, 264, 512, 5, scales="wide"),
    ]  # fmt: skip


FLASHINFER_TESTS = "tests/gemm/test_groupwise_scaled_gemm_fp8.py"
# test_fp8_groupwise_gemm (cutlass backend, scale_major_mode "K"; "MN" needs
# the ScaleMajorK=false template, not built) and
# test_fp8_groupwise_gemm_small_batch_size ("K").
GROUPWISE_DIMS = (128, 256, 512, 4096, 8192)
SMALL_BATCH = [(m, n, 256) for m in (1, 4, 16, 32) for n in (128, 256)]
# Upstream parametrizations no kernel here serves: m = 32 runs the
# low-latency kernel upstream, which this harness restricts to m <= 16.
UNSERVED_UPSTREAM = [shape for shape in SMALL_BATCH if shape[0] > 16]


def upstream_cases() -> list[CaseSpec]:
    """FlashInfer's tests of both kernels at the pinned revision (upstream_case)."""
    cases = [
        upstream_case(
            f"groupwise_m{m}_n{n}_k{k}",
            dict(m=m, n=n, k=k),
            f"{FLASHINFER_TESTS}::test_fp8_groupwise_gemm[cutlass-K-{k}-{n}-{m}]",
        )
        for m in GROUPWISE_DIMS
        for n in GROUPWISE_DIMS
        for k in GROUPWISE_DIMS
    ]
    cases += [
        upstream_case(
            f"small_batch_m{m}_n{n}_k{k}",
            dict(m=m, n=n, k=k),
            f"{FLASHINFER_TESTS}::test_fp8_groupwise_gemm_small_batch_size"
            f"[K-{k}-{n}-{m}]",
        )
        for m, n, k in SMALL_BATCH
    ]
    return cases


THROUGHPUT = "fp8_gemm"

DEEPGEMM = "https://github.com/deepseek-ai/DeepGEMM/blob/78b69000794d0937b47ae3387eff7663410264d1/tests/generators.py"
# DeepGEMM tests/generators.py enumerate_normal(float8_e4m3fn), legacy (1,128)/
# (128,128) quantization, forward (K-major A/B, BF16 output): m x (n, k).
# (Its BF16-accumulation variants, the FP4/UE8M0 configs and the backward
# MN-major shapes are other kernels.)
DEEPGEMM_M = (1, 128, 4096)
DEEPGEMM_NK = (
    (2112, 7168),
    (576, 7168),
    (24576, 1536),
    (32768, 512),
    (7168, 16384),
    (4096, 7168),
    (7168, 2048),
)
# (model, m values): every linear layer of each model (harness.models;
# all satisfy k % 128 == 0, n % 8 == 0); m <= 16 runs on fp8_gemm_small_m.
MODEL_PLAN = (
    ("deepseek_v3", (1, 4, 8, 16, 2048, 16384)),
    ("llama3_70b", (1, 16, 3000, 8192)),  # 3000: partial M tile
    ("llama3_405b", (2048, 16384)),
    ("qwen3_235b_a22b", (8, 1000, 8192)),
)


def throughput_cases():
    """Throughput (benchmark) cases of this package: DeepGEMM's FP8 GEMM test
    shapes and model linear layers from decode (m = 1..16) to 16k-token
    prefill."""
    from ..models import MODELS

    cases: dict[tuple[int, int, int], CaseSpec] = {}
    for m in DEEPGEMM_M:
        for n, k in DEEPGEMM_NK:
            cases[m, n, k] = upstream_case(
                f"deepgemm_m{m}_n{n}_k{k}",
                dict(m=m, n=n, k=k),
                f"{DEEPGEMM}::enumerate_normal[float8_e4m3fn-{m}-{n}-{k}]",
                suite="throughput",
                seed=1111,
                revision="78b69000794d0937b47ae3387eff7663410264d1",
            )
    for model, ms in MODEL_PLAN:
        for layer, (n, k) in MODELS[model].linear_shapes().items():
            for m in ms:
                cases.setdefault(
                    (m, n, k),
                    model_case(
                        f"{model}_{layer}_m{m}", dict(m=m, n=n, k=k), model, layer
                    ),
                )
    return list(cases.values())

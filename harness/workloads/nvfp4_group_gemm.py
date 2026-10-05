"""Grouped NVFP4 block-scaled GEMM: all groups in one CUTLASS SM100 launch.

For every group g, ``D_g[m_g, n_g] = (A_g * SFA_g) @ (B_g * SFB_g)^T`` with
packed FP4 E2M1 A_g [m_g, k_g/2] and B_g [n_g, k_g/2] (two values per uint8,
low nibble first, K-major), UE4M3 scale factors per 16 K elements and FP16
D_g [m_g, n_g] row-major (FP32 accumulation). One output tensor per group.

Inputs, in order, four tensors per group: ``a_g, b_g, sfa_g, sfb_g``. The
scale factors are given in the kernel's blocked layout ``(ceil(rows/128),
ceil(k/64), 32, 4, 4)`` (float8_e4m3fn bytes, zero padded): element (r, c) of
the logical (rows, k/16) scale matrix lives at ``[r // 128, c // 4, r % 32,
(r // 32) % 4, c % 4]``. That is cuBLAS's / tcgen05's block scaling-factor
layout, the one ``to_blocked`` and ``create_reordered_scale_factor_tensor``
of the NVIDIA task produce and hand to submissions; see ``to_blocked`` and
``from_blocked``. The previous package converted logical scales with a
``pack_scales`` kernel per operand and group at run time; that relayout is
input preparation here (``get_inputs``), not a timed launch.

The kernel is CUTLASS's grouped (ptr-array) NVFP4 GEMM
(``KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100``) in FlashInfer's
single-GEMM configuration, built by ``impls/nvfp4_group_gemm/compiler.py``
(see its docstring for why and for the live/dead table: the previous package
launched pack_scales x2 + one GEMM per group). Its by-value ``Params`` and
the per-group device arrays it reads (problem shapes, pointers, strides,
scale-factor layouts) are rebuilt here, mirroring CUTLASS's
``GemmKernel::to_underlying_arguments`` for the Arguments CUTLASS's example 75
builds (alpha = 1, beta = 0, no C, host problem shapes available, default
group-scheduler arguments, ``hw_info.sm_count`` = the device's SM count).
Field offsets, launch/workspace constants and the placeholder TMA encode
calls come from the compile-time probe's sidecar
``cubins/<arch>/nvfp4_group_gemm.json``; ``tests/test_nvfp4_group_gemm.py``
checks Params and arrays byte-for-byte against CUTLASS. The kernel rewrites
its TMA descriptors per group on the device inside a tensormap workspace
(allocated with torch, uninitialized, size per CUTLASS's
``get_workspace_size``).
"""

from __future__ import annotations

import ctypes
import json
import random
import struct
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .. import cuda_driver
from ..cutlass_host import ceil_div, driver_encode
from ..registry import register
from ..throughput import model_case, skewed_lengths, upstream_case
from ..workload import ROOT, CaseSpec, Workload
from .quantization import (
    fp4_matmul,
    from_blocked_scales,
    make_fp4,
    to_blocked_scales,
)

# Encodes one CUtensorMap: (data_type, address, dims, strides_bytes, box,
# element_strides, interleave, swizzle, l2_promotion, oob_fill) -> 128 bytes.
TensorMapEncoder = Callable[..., bytes]

SF_VEC = 16  # K elements per scale factor
SF_BLOCK_ROWS, SF_BLOCK_COLS = 128, 4  # rows x scale columns per 512-byte block
B200_SM_COUNT = 148  # default SM count for argument building off-device
FIXUP_BIT = 1 << 21
ARRAYS = (
    "problem_shapes",
    "ptr_a",
    "ptr_b",
    "ptr_sfa",
    "ptr_sfb",
    "ptr_d",
    "stride_a",
    "stride_b",
    "stride_d",
    "layout_sfa",
    "layout_sfb",
)
ARRAY_ALIGN = 64


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
    """The compile-time probe's deterministic CUtensorMap stand-in
    (``fake_tensor_map`` in ``impls/nvfp4_group_gemm/kernels/
    nvfp4_group_gemm_probe.cu``, which sets bit 21 of word 1 so CUTLASS's
    descriptor fix-up is observable); for tests only."""
    out = bytearray(128)
    out[0:6] = bytes(
        (data_type, len(dims), interleave, swizzle, l2_promotion, oob_fill)
    )
    struct.pack_into("<Q", out, 8, address | FIXUP_BIT)
    struct.pack_into(f"<{len(dims)}Q", out, 16, *dims)
    struct.pack_into(f"<{len(strides)}Q", out, 56, *strides)
    struct.pack_into(f"<{len(box)}I", out, 88, *box)
    struct.pack_into(f"<{len(element_strides)}I", out, 108, *element_strides)
    return bytes(out)


driver_version = cuda_driver.driver_version


def round_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def fast_divmod_u64(divisor: int) -> tuple[int, int, int, int]:
    """cutlass::FastDivmodU64(divisor) as (divisor, multiplier, shift_right, round_up)."""
    if divisor == 0:
        return 0, 1, 0, 0
    shift = divisor.bit_length() - 1  # integer_log2: floor(log2(divisor))
    if divisor & (divisor - 1) == 0:
        return divisor, 0, shift, 0
    power = 1 << shift
    multiplier_lo = (power << 64) // divisor  # uint128_t(0, power) / divisor
    multiplier = ((power << 64) + power) // divisor  # uint128_t(power, power) / divisor
    if multiplier >> 64:
        raise AssertionError("FastDivmodU64 multiplier overflow")
    return divisor, multiplier, shift, int(multiplier_lo == multiplier)


def fast_divmod_u64_pow2(divisor: int) -> tuple[int, int]:
    """cutlass::FastDivmodU64Pow2(divisor) as (divisor, shift_right)."""
    return divisor, max(divisor.bit_length() - 1, 0)


def blocked_shape(rows: int, k: int) -> tuple[int, int, int, int, int]:
    """Shape of one group's blocked scale-factor tensor for (rows, k/16) scales."""
    return (
        ceil_div(rows, SF_BLOCK_ROWS),
        ceil_div(k // SF_VEC, SF_BLOCK_COLS),
        32,
        4,
        SF_BLOCK_COLS,
    )


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """Logical (rows, cols) scales -> one group's zero-padded blocked layout
    ``(rb, cb, 32, 4, 4)``: ``quantization.to_blocked_scales`` with l = 1 (the
    element order of the NVIDIA task's ``to_blocked``)."""
    return to_blocked_scales(scales[..., None])[0]


def from_blocked(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``to_blocked`` (drops the padding)."""
    return from_blocked_scales(blocked[None], rows, cols)[..., 0]


class LaunchConfig:
    """Everything one launch needs: Params, launch shape and the device memory
    (argument arrays, workspace) and host problem shapes Params points to."""

    def __init__(
        self,
        params: bytes,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem: int,
        workspace_size: int,
    ):
        self.params = params
        self.grid = grid
        self.block = block
        self.shared_mem = shared_mem
        self.workspace_size = workspace_size
        # Static 1x1x1 cluster: GemmUniversalAdapter launches without a cluster
        # launch attribute.
        self.cluster: tuple[int, int, int] | None = None
        # Buffers Params refers to; kept alive with the configuration.
        self.keepalive: list[Any] = []


@register(name="nvfp4_group_gemm", supported_arches=("sm_100a",))
class NVFP4GroupGEMM(Workload):
    """Ragged grouped NVFP4 GEMM, one FP16 output per group, one launch.

    Wraps CUTLASS's ``KernelPtrArrayTmaWarpSpecialized1SmNvf4Sm100`` grouped
    GEMM (128x128x256 tile, 1x1x1 cluster, persistent group tile scheduler).
    Requires k % 32 == 0 and n % 8 == 0 (TMA alignment of A/B and D) and at
    least one row per group.

    Cases: ``SMOKE`` (see its comment), every NVIDIA task test line (task
    distribution and seed) and benchmark line, and MoE expert GEMMs with
    skewed per-expert token counts (``MOE_PLAN``).
    """

    rtol, atol = 1e-3, 1e-3

    def __init__(self, cubin: bytes | None, *, device=None):
        super().__init__(cubin, device=device)
        self._layout: dict[str, Any] | None = None

    # -- compile-time layout -------------------------------------------------

    @property
    def layout(self) -> dict[str, Any]:
        """The probe sidecar next to this arch's cubin (``nvfp4_group_gemm.json``)."""
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
        # GemmUniversalAdapter::initialize: smem >= 48 KiB needs the opt-in.
        smem = self.layout["constants"]["shared_storage_size"]
        if smem >= 48 << 10:
            function.set_attribute(
                cuda_driver.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem
            )

    # -- per-group device arrays -------------------------------------------------

    def workspace_size(self, sm_count: int) -> int:
        """GemmKernel::get_workspace_size: epilogue then mainloop tensormaps per
        SM, each rounded to the tensormap alignment; the group scheduler needs none."""
        const = self.layout["constants"]
        align = const["min_tensormap_workspace_alignment"]
        size = round_up(
            const["epilogue_workspace_base"]
            + const["epilogue_workspace_per_sm"] * sm_count,
            align,
        )
        size += const["mainloop_workspace_base"]
        size += const["mainloop_workspace_per_sm"] * sm_count
        return round_up(size, align)

    def array_bytes(
        self,
        problems: Sequence[tuple[int, int, int]],
        pointers: dict[str, Sequence[int]],
    ) -> dict[str, bytes]:
        """Contents of the per-group device arrays, as CUTLASS's example 75
        fills them: problem shapes, operand pointers (``pointers[a|b|sfa|sfb|d]``),
        packed strides and ``tile_atom_to_shape_SF{A,B}`` layouts."""
        elements = self.layout["elements"]

        def element(prefix: str, values: dict[str, int], fmt: str) -> bytes:
            out = bytearray(elements[prefix]["size"])
            for name, value in values.items():
                field = elements[f"{prefix}.{name}"]
                if struct.calcsize("<" + fmt) != field["size"]:
                    raise AssertionError(f"{prefix}.{name}: unexpected size")
                struct.pack_into("<" + fmt, out, field["offset"], value)
            # Every other member is static (empty): must hold no data.
            for name, field in elements.items():
                if name.startswith(prefix + ".") and field["size"]:
                    if name.removeprefix(prefix + ".") not in values:
                        raise AssertionError(f"{name} is not written")
            return bytes(out)

        def layout_sf(mn: int, k: int) -> bytes:
            # tile_atom_to_shape_SF*: (((32,4), mn_blocks), ((16,4), k_blocks), (1, L))
            # : (((16,4), 512 * k_blocks), ((0,1), 512), (_, 512 * mn_blocks * k_blocks))
            mn_blocks = ceil_div(mn, SF_BLOCK_ROWS)
            k_blocks = ceil_div(ceil_div(k, SF_VEC), SF_BLOCK_COLS)
            block = SF_BLOCK_ROWS * SF_BLOCK_COLS
            return element(
                "layout_sf",
                {
                    "shape_mn_blocks": mn_blocks,
                    "shape_k_blocks": k_blocks,
                    "shape_l": 1,
                    "stride_mn_blocks": block * k_blocks,
                    "stride_l": block * mn_blocks * k_blocks,
                },
                "i",
            )

        out = {
            "problem_shapes": b"".join(
                element("problem_shape", {"m": m, "n": n, "k": k}, "i")
                for m, n, k in problems
            ),
            # make_cute_packed_stride(Stride<int64_t, _1, _0>, (rows, cols, 1)).
            "stride_a": b"".join(
                element("stride_a", {"0": k}, "q") for _, _, k in problems
            ),
            "stride_b": b"".join(
                element("stride_b", {"0": k}, "q") for _, _, k in problems
            ),
            "stride_d": b"".join(
                element("stride_d", {"0": n}, "q") for _, n, _ in problems
            ),
            "layout_sfa": b"".join(layout_sf(m, k) for m, _, k in problems),
            "layout_sfb": b"".join(layout_sf(n, k) for _, n, k in problems),
        }
        for operand in ("a", "b", "sfa", "sfb", "d"):
            values = list(pointers[operand])
            if len(values) != len(problems):
                raise ValueError(f"expected one {operand} pointer per group")
            out["ptr_" + operand] = struct.pack(f"<{len(values)}Q", *values)
        return out

    # -- Params ------------------------------------------------------------------

    def build_params(
        self,
        problems: Sequence[tuple[int, int, int]],
        arrays: dict[str, int],
        workspace: int,
        host_problem_shapes: int,
        sm_count: int,
        *,
        encode: TensorMapEncoder = driver_encode,
        driver: int | None = None,
    ) -> LaunchConfig:
        """Mirror CUTLASS's grouped Arguments -> Params.

        ``arrays`` maps each name of ``ARRAYS`` to the device address of that
        per-group array, ``workspace`` is the tensormap workspace (at least
        ``workspace_size(sm_count)`` bytes) and ``host_problem_shapes`` the
        host address of the (m, n, k) int32 triples CUTLASS's host code reads
        (stored in Params, never dereferenced by the kernel). ``encode``/
        ``driver`` select the CUtensorMap encoder and the driver version
        CUTLASS's descriptor fix-up depends on (default: libcuda's).
        """
        layout = self.layout
        fields, const, maps = (
            layout["fields"],
            layout["constants"],
            layout["tensor_maps"],
        )
        groups = len(problems)
        if groups < 1:
            raise ValueError("at least one group is required")
        for m, n, k in problems:
            if min(m, n, k) < 1 or k % 32 or n % 8:
                raise ValueError(f"unsupported problem m={m} n={n} k={k}")
        if set(arrays) != set(ARRAYS) or any(arrays[a] % 16 for a in ARRAYS):
            raise ValueError("expected 16-byte aligned addresses of every array")
        if workspace % ARRAY_ALIGN or sm_count < 1:
            raise ValueError("misaligned workspace or invalid SM count")
        if driver is None:
            driver = driver_version()
        if const["atom_thr_shape_mnk"] != [1, 1, 1] or const["cluster_shape_mnk"] != [
            1,
            1,
            1,
        ]:
            raise AssertionError("builder assumes a 1-SM MMA and a 1x1x1 cluster")
        if const["is_dynamic_cluster"] or const["is_sched_dynamic_persistent"]:
            raise AssertionError("builder assumes the static group tile scheduler")
        if any(f["size"] for n_, f in fields.items() if n_.endswith(".aux_g_stride")):
            raise AssertionError("builder assumes static TMA aux strides")
        # Every byte not written below is padding or a value-initialized zero
        # (see the fixtures' "padding"; tma_load_c is never encoded: no C).
        buf = bytearray(layout["params_size"])

        def put(name: str, fmt: str, *values: Any) -> None:
            field = fields[name]
            if struct.calcsize("<" + fmt) != field["size"]:
                raise AssertionError(f"{name}: {fmt} is not {field['size']} bytes")
            struct.pack_into("<" + fmt, buf, field["offset"], *values)

        def descriptor(operand: str) -> bytes:
            call = maps[operand]
            desc = bytearray(
                encode(
                    call["data_type"],
                    call["address"],
                    call["dims"],
                    call["strides"],
                    call["box"],
                    call["element_strides"],
                    call["interleave"],
                    call["swizzle"],
                    call["l2_promotion"],
                    call["oob_fill"],
                )
            )
            # cute::detail::make_tma_copy_desc: drivers <= 13.1 need bit 21 of
            # descriptor word 1 cleared for tensors smaller than 128 KiB (all
            # of the grouped kernel's tile-sized placeholders).
            if driver <= 13010 and call["small_tensor_fixup"]:
                (word,) = struct.unpack_from("<Q", desc, 8)
                struct.pack_into("<Q", desc, 8, word & ~FIXUP_BIT)
            return bytes(desc)

        def group_problem_shape(prefix: str) -> None:
            put(prefix + ".num_groups", "i", groups)
            put(prefix + ".problem_shapes", "Q", arrays["problem_shapes"])
            put(prefix + ".host_problem_shapes", "Q", host_problem_shapes)

        put("mode", "i", const["gemm_mode_kGrouped"])
        group_problem_shape("problem_shape")

        # CollectiveMma::to_underlying_arguments (grouped): placeholder
        # descriptors for a tile-sized problem at a null address (the kernel
        # replaces address/shape/stride per group before the first load);
        # fallbacks equal the primary ones (static cluster).
        descs = {operand: descriptor(operand) for operand in maps}
        for operand in ("a", "b", "sfa", "sfb"):
            put(f"mainloop.tma_load_{operand}.desc", "128s", descs[operand])
            put(f"mainloop.tma_load_{operand}_fallback.desc", "128s", descs[operand])
        put("mainloop.cluster_shape_fallback", "3I", 0, 0, 0)  # hw_info default
        epilogue_size = round_up(
            const["epilogue_workspace_base"]
            + const["epilogue_workspace_per_sm"] * sm_count,
            const["min_tensormap_workspace_alignment"],
        )
        put("mainloop.tensormaps", "Q", workspace + epilogue_size)
        put("mainloop.ptr_A", "Q", arrays["ptr_a"])
        put("mainloop.dA", "Q", arrays["stride_a"])
        put("mainloop.ptr_B", "Q", arrays["ptr_b"])
        put("mainloop.dB", "Q", arrays["stride_b"])
        put("mainloop.ptr_SFA", "Q", arrays["ptr_sfa"])
        put("mainloop.layout_SFA", "Q", arrays["layout_sfa"])
        put("mainloop.ptr_SFB", "Q", arrays["ptr_sfb"])
        put("mainloop.layout_SFB", "Q", arrays["layout_sfb"])

        # CollectiveEpilogue::to_underlying_arguments: LinearCombination with
        # alpha = 1, beta = 0 (no scalar pointers/arrays, zero batch strides),
        # no C, D through the per-group arrays; tensormaps after the (empty)
        # fusion workspace.
        put("epilogue.thread.alpha", "f", 1.0)
        put("epilogue.thread.beta", "f", 0.0)
        for name in ("alpha_ptr", "beta_ptr", "alpha_ptr_array", "beta_ptr_array"):
            put(f"epilogue.thread.{name}", "Q", 0)
        put("epilogue.thread.alpha_stride_l", "q", 0)
        put("epilogue.thread.beta_stride_l", "q", 0)
        put("epilogue.tma_store_d.desc", "128s", descs["d"])
        fusion = round_up(
            const["fusion_workspace_size"], const["min_tensormap_workspace_alignment"]
        )
        put("epilogue.tensormaps", "Q", workspace + fusion)
        put("epilogue.ptr_C", "Q", 0)
        put("epilogue.dC", "Q", 0)
        put("epilogue.ptr_D", "Q", arrays["ptr_d"])
        put("epilogue.dD", "Q", arrays["stride_d"])

        # PersistentTileSchedulerSm100Group::to_underlying_arguments with the
        # default (SM90 group) scheduler Arguments: max_swizzle_size = 1,
        # RasterOrderOptions::AlongM. Host problem shapes are available, so the
        # problem is linearized into (total CTAs, 1, 1).
        tile_m, tile_n, tile_k = const["cta_shape_mnk"]
        total = sum(
            max(ceil_div(m, tile_m), 1) * max(ceil_div(n, tile_n), 1)
            for m, n, _ in problems
        )
        raster = const["raster_order_AlongM"]
        # AlongM: major = cluster M, minor = cluster N (both 1).
        put("scheduler.divmod_cluster_shape_major.divisor", "Q", 1)
        put("scheduler.divmod_cluster_shape_major.shift_right", "I", 0)
        put("scheduler.divmod_cluster_shape_minor.divisor", "Q", 1)
        put("scheduler.divmod_cluster_shape_minor.shift_right", "I", 0)
        for name, size in (("m", tile_m), ("n", tile_n)):
            divisor, multiplier, shift, up = fast_divmod_u64(size)
            put(f"scheduler.divmod_cta_shape_{name}.divisor", "Q", divisor)
            put(f"scheduler.divmod_cta_shape_{name}.multiplier", "Q", multiplier)
            put(f"scheduler.divmod_cta_shape_{name}.shift_right", "I", shift)
            put(f"scheduler.divmod_cta_shape_{name}.round_up", "I", up)
        put("scheduler.blocks_across_problem", "Q", total)
        put("scheduler.pre_processed_problem_shapes", "?", True)
        put("scheduler.max_swizzle_size", "i", 1)
        put("scheduler.raster_order", "i", raster)
        group_problem_shape("scheduler.problem_shapes")
        put("scheduler.cta_shape", "3i", tile_m, tile_n, tile_k)
        put("scheduler.cluster_shape", "3i", 1, 1, 1)

        # KernelHardwareInfo as example 75 sets it (device 0, SM count).
        put("hw_info.device_id", "i", 0)
        put("hw_info.sm_count", "i", sm_count)
        put("hw_info.max_active_clusters", "i", 0)
        put("hw_info.cluster_shape", "3I", 0, 0, 0)
        put("hw_info.cluster_shape_fallback", "3I", 0, 0, 0)

        # GemmKernel::get_grid_shape -> PersistentTileSchedulerSm100GroupParams::
        # get_grid_shape (AlongM, static 1x1 cluster): one persistent CTA per SM,
        # truncated to the number of output tiles.
        grid = (min(sm_count, total), 1, 1)
        block = (const["max_threads_per_block"], 1, 1)
        return LaunchConfig(
            bytes(buf),
            grid,
            block,
            const["shared_storage_size"],
            self.workspace_size(sm_count),
        )

    @staticmethod
    def groups_of(inputs: tuple) -> list[tuple[torch.Tensor, ...]]:
        if not inputs or len(inputs) % 4:
            raise ValueError("expected four tensors (a, b, sfa, sfb) per group")
        return [tuple(inputs[i : i + 4]) for i in range(0, len(inputs), 4)]

    @staticmethod
    def problem_of(group: tuple[torch.Tensor, ...]) -> tuple[int, int, int]:
        a, b = group[:2]
        return a.shape[0], b.shape[0], a.shape[1] * 2

    def launch_config(
        self,
        inputs: tuple,
        outputs: tuple,
        *,
        encode: TensorMapEncoder = driver_encode,
        driver: int | None = None,
        sm_count: int | None = None,
    ) -> LaunchConfig:
        """Check the tensors, fill the device arrays and build the launch for
        ``inputs`` into ``outputs`` (one FP16 tensor per group)."""
        groups = self.groups_of(inputs)
        if len(outputs) != len(groups):
            raise ValueError("expected one output per group")
        device = outputs[0].device
        problems = []
        for (a, b, sfa, sfb), out in zip(groups, outputs):
            m, n, k = self.problem_of((a, b))
            expected = (
                (a, (m, k // 2), torch.uint8),
                (b, (n, k // 2), torch.uint8),
                (sfa, blocked_shape(m, k), torch.float8_e4m3fn),
                (sfb, blocked_shape(n, k), torch.float8_e4m3fn),
                (out, (m, n), torch.float16),
            )
            for tensor, shape, dtype in expected:
                if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                    raise ValueError(
                        f"expected {dtype} {shape}, got {tensor.dtype} "
                        f"{tuple(tensor.shape)}"
                    )
                if not tensor.is_contiguous() or tensor.device != device:
                    raise ValueError("operands must be contiguous and on one device")
                if tensor.data_ptr() % 16:
                    raise ValueError("operands must be 16-byte aligned")
            problems.append((m, n, k))
        if sm_count is None:
            sm_count = (
                torch.cuda.get_device_properties(device).multi_processor_count
                if device.type == "cuda"
                else B200_SM_COUNT
            )
        contents = self.array_bytes(
            problems,
            {
                operand: [group[i].data_ptr() for group in groups]
                for i, operand in enumerate(("a", "b", "sfa", "sfb"))
            }
            | {"d": [out.data_ptr() for out in outputs]},
        )
        offsets, size = {}, 0
        for name in ARRAYS:
            offsets[name] = size
            size = round_up(size + len(contents[name]), ARRAY_ALIGN)
        host = bytearray(size)
        for name in ARRAYS:
            host[offsets[name] : offsets[name] + len(contents[name])] = contents[name]
        # One host -> device copy, ordered on the current stream before the launch.
        args = torch.frombuffer(host, dtype=torch.uint8).to(device)
        workspace = torch.empty(
            self.workspace_size(sm_count), dtype=torch.uint8, device=device
        )
        host_shapes = (ctypes.c_int32 * (3 * len(problems)))(
            *(v for problem in problems for v in problem)
        )
        if encode is driver_encode:
            cuda_driver.ensure_context(device)
        config = self.build_params(
            problems,
            {name: args.data_ptr() + offsets[name] for name in ARRAYS},
            workspace.data_ptr(),
            ctypes.addressof(host_shapes),
            sm_count,
            encode=encode,
            driver=driver,
        )
        config.keepalive = [args, workspace, host_shapes, inputs, outputs]
        return config

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
        return [*SMOKE, *task_group_cases(), *throughput_cases()]

    def get_inputs(self, case):
        p, g = case.params, self.generator(case)
        dist = {key: p.get(key, TASK_DIST[key]) for key in TASK_DIST}
        tensors = []
        for m, n, k in p["groups"]:
            a, sa = make_fp4(self, g, m, k, 1, **dist)
            b, sb = make_fp4(self, g, n, k, 1, **dist)
            tensors += [
                a[..., 0],
                b[..., 0],
                to_blocked(sa[..., 0]),
                to_blocked(sb[..., 0]),
            ]
        return tuple(tensors)

    def get_reference(self, inputs):
        out = []
        for a, b, sfa, sfb in self.groups_of(inputs):
            m, n, k = self.problem_of((a, b))
            sa = from_blocked(sfa, m, k // SF_VEC)
            sb = from_blocked(sfb, n, k // SF_VEC)
            d = fp4_matmul(
                a[..., None],
                b[..., None],
                sa[..., None],
                sb[..., None],
                out_dtype=torch.float16,
            )
            out.append(d[..., 0])
        return tuple(out)

    def allocate_outputs(self, inputs):
        return tuple(
            torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float16)
            for a, b, _, _ in self.groups_of(inputs)
        )

    def _check_supported(self) -> None:
        # Fail before any driver work (descriptor encoding needs sm_90+).
        if not self.is_supported():
            raise RuntimeError(f"{self.arch} cubin cannot run on {self.device}")

    def run(self, inputs):
        self._check_supported()
        outputs = self.allocate_outputs(inputs)
        self._launch(self.launch_config(inputs, outputs))
        return outputs

    def prepare(self, inputs):
        self._check_supported()
        outputs = self.allocate_outputs(inputs)
        config = self.launch_config(inputs, outputs)
        return (lambda: (self._launch(config), outputs)[1]), outputs

    def validate(self, ref, impl):
        # FP32 accumulation of exactly representable FP4 x UE4M3 products in
        # both; outputs differ by at most one FP16 rounding step (the NVIDIA
        # task's tolerance).
        self.assert_close(ref, impl, rtol=self.rtol, atol=self.atol)


TASK = ROOT / "resources/nvfp4_group_gemm/task.yml"
# The task generator's distributions (reference.py): 0xBB-masked E2M1 bytes,
# integer scales 1..2 (quantization.make_fp4).
TASK_DIST = {"values": "restricted", "scales": "int1_2"}


def group_case(label, groups, seed, suite="smoke", **extra) -> CaseSpec:
    return CaseSpec(label, {"groups": [tuple(g) for g in groups], **extra}, seed, suite)


def example75_groups(seed: int = 75, count: int = 10) -> list[tuple[int, int, int]]:
    """CUTLASS example 75's ``randomize_problems``: ``count`` groups with m, n,
    k = 32 * (rand() % 64) (FP4 TMA alignment), seeded; 0 (an empty problem,
    which the harness does not pass, see ``build_params``) becomes 32."""
    rng = random.Random(seed)
    return [
        tuple(32 * max(rng.randrange(64), 1) for _ in range(3)) for _ in range(count)
    ]


# Smoke cases: one tile; three ragged groups (partial M); distinct (m, n, k)
# per group down to a 1-row group with k = 32 (one quarter of a K tile), 6
# scale columns (a partial scale atom), partial N and a 9-tile K loop
# (beyond the 5-stage pipeline); MoE-like skewed groups (seeded log-normal,
# 1-row groups included) over a shared partial-N/long-K weight shape with
# about 200 tiles (> 148 SMs: persistent CTAs take several tiles, across
# groups); CUTLASS example 75's random problems; the task's distribution
# (0xBB-masked E2M1, scales {1, 2}) by default and unrestricted E2M1 with
# zero scales, subnormal/zero scales and three-binade scales.
SMOKE = (
    group_case("single", [(128, 128, 128)], 1),
    group_case("ragged", [(17, 128, 128), (128, 256, 256), (65, 128, 512)], 2),
    group_case("distinct", [(1, 8, 32), (200, 136, 96), (129, 264, 384), (300, 1032, 2080)], 3),
    group_case(
        "skewed",
        [(m, 2056, 1024) for m in skewed_lengths(640, 12, seed=4, sigma=1.5)],
        4,
    ),
    group_case(
        "skewed_full",
        [(m, 520, 2080) for m in skewed_lengths(300, 6, seed=5, sigma=1.5)],
        5,
        values="full",
        scales="int0_3",
    ),
    group_case("distinct_unit", [(64, 72, 160), (257, 136, 512)], 6, scales="unit"),
    group_case("ragged_uniform", [(33, 264, 256), (190, 128, 768)], 7, scales="uniform"),
    CaseSpec(
        "cutlass_example75",
        {"groups": example75_groups()},
        75,
        "smoke",
        {
            "kind": "upstream_test",
            "test": "cutlass examples/75_blackwell_grouped_gemm/"
            "75_blackwell_grouped_gemm_block_scaled.cu::randomize_problems "
            "(groups=10, alpha=1, beta=0 as configured here)",
        },
    ),
)  # fmt: skip


def task_group_cases() -> list[CaseSpec]:
    """Every NVIDIA task test line (``upstream_case``, the task's seed)."""
    from .nvfp4_gemm import task_shapes

    cases = []
    for index, s in enumerate(task_shapes(TASK, "tests")):
        groups = list(zip(s["m"], s["n"], s["k"], strict=True))
        label = "task_" + "_".join(f"{m}x{n}x{k}" for m, n, k in groups)
        cases.append(
            upstream_case(
                label,
                {"groups": groups},
                f"nvfp4_group_gemm/task.yml::tests[{index}]",
                seed=s["seed"],
            )
        )
    return cases


THROUGHPUT = "nvfp4_group_gemm"

# (model, local experts per GPU (expert parallelism), token counts): MoE
# expert GEMMs (gate_up and down) where each local expert is one group with
# its routed tokens; per-expert counts are seeded log-normal (skewed, as
# routers produce), experts without tokens are left out of the launch.
MOE_PLAN = (
    ("deepseek_v3", 32, (16, 1024, 16384)),  # EP8 of 256 experts, top-8
    ("qwen3_235b_a22b", 32, (16, 1024, 16384)),  # EP4 of 128 experts, top-8
    ("gpt_oss_120b", 128, (16, 1024, 16384)),  # all 128 experts, top-4
    ("llama4_maverick", 16, (16, 1024, 16384)),  # EP8 of 128 experts, top-1
)


def moe_cases() -> list[CaseSpec]:
    from ..models import MODELS

    cases = []
    for model, local, token_counts in MOE_PLAN:
        spec = MODELS[model]
        shapes = spec.linear_shapes()
        for tokens in token_counts:
            routed = tokens * spec.top_k * local // spec.experts
            counts = [
                m for m in skewed_lengths(routed, local, seed=tokens, minimum=0) if m
            ]
            for layer in ("expert_gate_up", "expert_down"):
                n, k = shapes[layer]
                case = model_case(
                    f"{model}_{layer}_t{tokens}",
                    {"groups": [(m, n, k) for m in counts]},
                    model,
                    layer,
                )
                case.source.update(tokens=tokens, local_experts=local)
                cases.append(case)
    return cases


def throughput_cases():
    """Throughput (benchmark) cases: every task benchmark line and MoE expert
    GEMMs of DeepSeek-V3, Qwen3-235B, gpt-oss-120b and Llama-4 Maverick for
    16 to 16k tokens."""
    name = THROUGHPUT
    source = {
        "kind": "official_shape",
        "snapshot": f"resources/{name}/task.yml",
        "url": f"https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/{name}/task.yml",
        "selection": "All benchmark (not just test) shapes; generated using the harness quantization contract.",
    }
    from .nvfp4_gemm import task_shapes

    cases = [
        CaseSpec(
            f"throughput_groups{s['g']}_n{s['n'][0]}_k{s['k'][0]}",
            {"groups": list(zip(s["m"], s["n"], s["k"], strict=True))},
            1111,
            "throughput",
            source,
        )
        for s in task_shapes(TASK, "benchmarks")
    ]
    return cases + moe_cases()

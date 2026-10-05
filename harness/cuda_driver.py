"""Minimal ctypes bindings to the CUDA Driver API (``libcuda.so``).

Workloads load a single-kernel cubin and launch it from Python; nothing here
compiles code. Modules are loaded into the device's primary context, which is
the context PyTorch uses, so torch tensors and streams are directly usable.
"""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Sequence
from typing import Any

import torch

CUresult = ctypes.c_int
CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0
CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1
CU_FUNC_ATTRIBUTE_NUM_REGS = 4
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14
CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4
CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5
CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6
CUDA_ERROR_NOT_SUPPORTED = 801

_lock = threading.Lock()
_library: ctypes.CDLL | None = None


class CudaDriverError(RuntimeError):
    def __init__(self, status: int, what: str):
        self.status = status
        super().__init__(f"{what} failed: {error_string(status)} ({status})")


def lib() -> ctypes.CDLL:
    """dlopen libcuda once and initialize the driver."""
    global _library
    with _lock:
        if _library is None:
            library = ctypes.CDLL("libcuda.so.1")
            status = library.cuInit(0)
            if status:
                raise RuntimeError(f"cuInit failed with CUDA status {status}")
            _library = library
        return _library


def error_string(status: int) -> str:
    name, text = ctypes.c_char_p(), ctypes.c_char_p()
    library = lib()
    library.cuGetErrorName(status, ctypes.byref(name))
    library.cuGetErrorString(status, ctypes.byref(text))
    return ": ".join(
        value.decode() for value in (name.value, text.value) if value is not None
    )


def check(status: int, what: str) -> None:
    if status:
        raise CudaDriverError(status, what)


def driver_version() -> int:
    """cuDriverGetVersion, e.g. 13020 for CUDA 13.2."""
    version = ctypes.c_int()
    check(lib().cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
    return version.value


def ensure_context(device: torch.device) -> None:
    """Make the device's primary context (shared with PyTorch) current."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    torch.cuda.init()
    torch.empty(0, device=torch.device("cuda", index))  # creates the primary ctx
    library = lib()
    current = ctypes.c_void_p()
    check(library.cuCtxGetCurrent(ctypes.byref(current)), "cuCtxGetCurrent")
    dev = ctypes.c_int()
    check(library.cuDeviceGet(ctypes.byref(dev), index), "cuDeviceGet")
    primary = ctypes.c_void_p()
    check(
        library.cuDevicePrimaryCtxRetain(ctypes.byref(primary), dev),
        "cuDevicePrimaryCtxRetain",
    )
    # Retained once per call; the matching release keeps the refcount balanced
    # because PyTorch already holds the primary context for the process.
    check(library.cuDevicePrimaryCtxRelease(dev), "cuDevicePrimaryCtxRelease")
    if current.value != primary.value:
        check(library.cuCtxSetCurrent(primary), "cuCtxSetCurrent")


class _LaunchAttribute(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("pad", ctypes.c_char * 4),
        ("value", ctypes.c_char * 64),
    ]


def _launch_attribute(kind: int, payload: ctypes.Array) -> _LaunchAttribute:
    """A CUlaunchAttribute whose value union starts with ``payload``.

    Writes at the field's address: reading a ``c_char`` array field returns a
    ``bytes`` copy, so ``memmove(attr.value, ...)`` would leave the value zero.
    """
    attr = _LaunchAttribute(id=kind)
    size = ctypes.sizeof(payload)
    if size > ctypes.sizeof(attr) - _LaunchAttribute.value.offset:
        raise ValueError("launch attribute payload too large")
    ctypes.memmove(
        ctypes.addressof(attr) + _LaunchAttribute.value.offset, payload, size
    )
    return attr


class _LaunchConfig(ctypes.Structure):
    _fields_ = [
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_LaunchAttribute)),
        ("numAttrs", ctypes.c_uint),
    ]


def _dim3(value: int | Sequence[int]) -> tuple[int, int, int]:
    dims = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(dims) <= 3 or any(int(d) < 1 for d in dims):
        raise ValueError(f"invalid launch dimensions: {value!r}")
    return tuple(int(d) for d in (*dims, 1, 1)[:3])  # type: ignore[return-value]


class Function:
    """A kernel handle owned by a loaded Module."""

    def __init__(self, handle: ctypes.c_void_p, name: str):
        self.handle = handle
        self.name = name

    def get_attribute(self, attribute: int) -> int:
        value = ctypes.c_int()
        check(
            lib().cuFuncGetAttribute(ctypes.byref(value), attribute, self.handle),
            "cuFuncGetAttribute",
        )
        return value.value

    def set_attribute(self, attribute: int, value: int) -> None:
        check(
            lib().cuFuncSetAttribute(self.handle, attribute, int(value)),
            "cuFuncSetAttribute",
        )

    def launch(
        self,
        grid: int | Sequence[int],
        block: int | Sequence[int],
        params: Sequence[Any],
        *,
        shared_mem: int = 0,
        stream: int | None = None,
        cluster: int | Sequence[int] | None = None,
        programmatic_serialization: bool = False,
        cooperative: bool = False,
        cluster_scheduling_policy: int | None = None,
    ) -> None:
        """Launch with ``params`` given as ctypes instances, one per kernel parameter.

        Each instance holds the parameter's *value* (a pointer is ``c_void_p``,
        a by-value struct is a ``Structure`` or ``c_char`` array of its bytes).
        ``cluster_scheduling_policy`` adds the
        CLUSTER_SCHEDULING_POLICY_PREFERENCE attribute (CUclusterSchedulingPolicy).
        """
        g, b = _dim3(grid), _dim3(block)
        argv = (ctypes.c_void_p * max(1, len(params)))(
            *(ctypes.addressof(p) for p in params)
        )
        attrs = []
        if cluster is not None:
            dims = (ctypes.c_uint * 3)(*_dim3(cluster))
            attrs.append(_launch_attribute(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, dims))
        if cluster_scheduling_policy is not None:
            attrs.append(
                _launch_attribute(
                    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                    (ctypes.c_int * 1)(int(cluster_scheduling_policy)),
                )
            )
        if programmatic_serialization:
            attrs.append(
                _launch_attribute(
                    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
                    (ctypes.c_int * 1)(1),
                )
            )
        if cooperative:  # grid-wide sync (cooperative_groups::this_grid())
            attrs.append(
                _launch_attribute(
                    CU_LAUNCH_ATTRIBUTE_COOPERATIVE, (ctypes.c_int * 1)(1)
                )
            )
        attr_array = (_LaunchAttribute * max(1, len(attrs)))(*attrs)
        config = _LaunchConfig(
            *g,
            *b,
            int(shared_mem),
            ctypes.c_void_p(stream),
            attr_array,
            len(attrs),
        )
        check(
            lib().cuLaunchKernelEx(ctypes.byref(config), self.handle, argv, None),
            f"cuLaunchKernelEx({self.name})",
        )


def max_active_clusters(
    function: Function,
    grid: int | Sequence[int],
    block: int | Sequence[int],
    shared_mem: int,
    cluster: int | Sequence[int] | None = None,
) -> int:
    """cuOccupancyMaxActiveClusters for a launch configuration; 0 on failure
    (as CUTLASS's KernelHardwareInfo treats a failed query).

    Dimensions are forwarded verbatim: CUTLASS queries kernels with runtime
    cluster shapes using a grid and cluster of 0, which the driver rejects.
    """

    def raw3(value: int | Sequence[int]) -> tuple[int, int, int]:
        values = (value,) if isinstance(value, int) else tuple(value)
        return tuple(int(d) for d in (*values, 1, 1)[:3])  # type: ignore[return-value]

    if 0 in (*raw3(grid), *raw3(block), *(raw3(cluster) if cluster else ())):
        return 0  # the driver rejects zero dimensions; skip the failing call
    attrs = []
    if cluster is not None:
        dims = (ctypes.c_uint * 3)(*raw3(cluster))
        attrs.append(_launch_attribute(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION, dims))
    attr_array = (_LaunchAttribute * max(1, len(attrs)))(*attrs)
    config = _LaunchConfig(
        *raw3(grid),
        *raw3(block),
        int(shared_mem),
        ctypes.c_void_p(None),
        attr_array,
        len(attrs),
    )
    clusters = ctypes.c_int(0)
    status = lib().cuOccupancyMaxActiveClusters(
        ctypes.byref(clusters), function.handle, ctypes.byref(config)
    )
    return clusters.value if status == 0 else 0


def max_active_blocks_per_sm(
    function: Function, block_size: int, shared_mem: int
) -> int:
    """cuOccupancyMaxActiveBlocksPerMultiprocessor (as
    ``cudaOccupancyMaxActiveBlocksPerMultiprocessor``, which upstream planners
    and persistent launchers call)."""
    blocks = ctypes.c_int(0)
    check(
        lib().cuOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks),
            function.handle,
            int(block_size),
            ctypes.c_size_t(shared_mem),
        ),
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
    )
    return blocks.value


class Module:
    """A CUmodule loaded from in-memory image bytes into the current context."""

    def __init__(self, image: bytes):
        self._buffer = ctypes.create_string_buffer(image, len(image))
        self.handle = ctypes.c_void_p()
        check(
            lib().cuModuleLoadData(ctypes.byref(self.handle), self._buffer),
            "cuModuleLoadData",
        )

    def function(self, name: str) -> Function:
        handle = ctypes.c_void_p()
        check(
            lib().cuModuleGetFunction(ctypes.byref(handle), self.handle, name.encode()),
            f"cuModuleGetFunction({name})",
        )
        return Function(handle, name)

    def unload(self) -> None:
        if self.handle.value:
            check(lib().cuModuleUnload(self.handle), "cuModuleUnload")
            self.handle = ctypes.c_void_p()


# --- Tensor Memory Accelerator descriptors (sm_90+) -------------------------

CU_TENSOR_MAP_DATA_TYPE = {
    torch.uint8: 0,
    torch.float8_e4m3fn: 0,
    torch.int8: 0,
    torch.uint16: 1,
    torch.int32: 3,
    torch.int64: 5,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}


class TensorMap(ctypes.Structure):
    """CUtensorMap: 128 opaque bytes, 64-byte aligned."""

    _fields_ = [("opaque", ctypes.c_uint64 * 16)]


def encode_tensor_map_tiled(
    data_type: int,
    global_address: int,
    global_dims: Sequence[int],
    global_strides_bytes: Sequence[int],
    box_dims: Sequence[int],
    element_strides: Sequence[int] | None = None,
    *,
    interleave: int = 0,
    swizzle: int = 0,
    l2_promotion: int = 0,
    oob_fill: int = 0,
) -> TensorMap:
    """Wrap cuTensorMapEncodeTiled. ``global_strides_bytes`` excludes dim 0.

    Enum arguments use the raw CUtensorMap* values from cuda.h. The driver only
    encodes descriptors on a TMA-capable (sm_90+) device; elsewhere, e.g. on
    sm_86, it returns CUDA_ERROR_NOT_SUPPORTED even without any launch.
    """
    rank = len(global_dims)
    if len(global_strides_bytes) != rank - 1 or len(box_dims) != rank:
        raise ValueError("tensor map rank mismatch")
    element_strides = element_strides or [1] * rank
    tensor_map = TensorMap()
    status = lib().cuTensorMapEncodeTiled(
        ctypes.byref(tensor_map),
        data_type,
        ctypes.c_uint(rank),
        ctypes.c_void_p(global_address),
        (ctypes.c_uint64 * rank)(*global_dims),
        (ctypes.c_uint64 * max(1, rank - 1))(*global_strides_bytes),
        (ctypes.c_uint32 * rank)(*box_dims),
        (ctypes.c_uint32 * rank)(*element_strides),
        interleave,
        swizzle,
        l2_promotion,
        oob_fill,
    )
    if status == CUDA_ERROR_NOT_SUPPORTED:
        raise CudaDriverError(
            status, "cuTensorMapEncodeTiled (requires a TMA-capable sm_90+ GPU)"
        )
    check(status, "cuTensorMapEncodeTiled")
    return tensor_map

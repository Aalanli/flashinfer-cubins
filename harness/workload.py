"""Single-kernel workloads: one cubin, one kernel, one Python-side launch.

A workload wraps exactly one CUDA kernel compiled ahead of time by the
``ImplCompiler`` of its ``impls/<package>/compiler.py``. At run time nothing is
compiled: the cubin is loaded with ``cuModuleLoadData`` through ``libcuda.so``
(see ``harness.cuda_driver``) and launched from Python on the caller's current
PyTorch stream.

Each package writes ``impls/<package>/kernels.json``::

    {"sm_100a": {"<workload name>": "<cubin path relative to kernels.json>"}}

Only trusted cubins should be loaded: they are executable code.
"""

from __future__ import annotations

import ctypes
import json
import re
import struct
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch

from . import cuda_driver

ROOT = Path(__file__).resolve().parents[1]
IMPLS = ROOT / "impls"
MANIFEST = "kernels.json"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    suite: str = "smoke"
    source: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelParam:
    """One kernel parameter as recorded in the cubin's EIATTR_KPARAM_INFO."""

    ordinal: int
    offset: int
    size: int


def normalize_arch(arch: str | int) -> str:
    value = (
        str(arch).removeprefix("-arch=").removeprefix("sm_").removeprefix("compute_")
    )
    if not re.fullmatch(r"[0-9]{2,3}[af]?", value):
        raise ValueError(f"Invalid CUDA architecture: {arch!r}")
    return "sm_" + value


def arch_capability(arch: str) -> tuple[int, int]:
    number = int(normalize_arch(arch)[3:].rstrip("af"))
    return number // 10, number % 10


def arch_compatible(cubin_arch: str, arch: str) -> bool:
    """Whether SASS built for ``cubin_arch`` runs on the GPUs of ``arch``.

    CUDA binary compatibility: a plain ``sm_XY`` cubin runs on every ``sm_XZ``
    with ``Z >= Y`` (sm_80 SASS runs on sm_86); a family ``sm_XYf`` cubin
    likewise (sm_100f runs on sm_100a/sm_103a); an arch-specific ``sm_XYa``
    cubin only on ``sm_XY``/``sm_XYa``.
    """
    cubin_arch, arch = normalize_arch(cubin_arch), normalize_arch(arch)
    (major, minor), (target_major, target_minor) = map(
        arch_capability, (cubin_arch, arch)
    )
    if major != target_major:
        return False
    if cubin_arch.endswith("a"):
        return minor == target_minor and not arch.endswith("f")
    return minor <= target_minor


def device_matches(arch: str, device: torch.device | str | None = None) -> bool:
    """Exact compute-capability match; an sm_100a cubin never runs on sm_86."""
    capability = device_capability(device)
    return capability is not None and capability == arch_capability(arch)


def device_capability(
    device: torch.device | str | None = None,
) -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    device = torch.device(device or "cuda")
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_capability(device)


def device_runs(cubin_arch: str, device: torch.device | str | None = None) -> bool:
    """Whether ``device`` can execute SASS built for ``cubin_arch``."""
    capability = device_capability(device)
    if capability is None:
        return False
    major, minor = capability
    # Every sm_9x/sm_10x/sm_12x device supports its arch-specific features.
    suffix = "a" if major >= 9 else ""
    return arch_compatible(cubin_arch, f"sm_{major}{minor}{suffix}")


EF_CUDA_SM_SHIFT = 8
EF_CUDA_VARIANT_A = 0x8
EIFMT_SVAL = 4
EIATTR_KPARAM_INFO = 0x17
EIATTR_KPARAM_INFO_V2 = 0x45  # TRT-LLM/cubin-gen kernels; size stored unpacked


def validate_image(image: bytes) -> None:
    """Check CUDA ELF headers and section/segment bounds before parsing."""
    if not isinstance(image, bytes):
        raise TypeError("image must be bytes")
    if len(image) < 64 or image[:4] != b"\x7fELF":
        raise ValueError("expected a cubin (CUDA ELF) image")
    h = struct.unpack_from("<16sHHIQQQIHHHHHH", image)
    if (
        h[0][4:6] != b"\x02\x01"
        or h[2] != 190
        or h[11] != 64
        or not h[12]
        or (h[10] and h[9] != 56)
    ):
        raise ValueError("invalid CUDA ELF header")

    def bounds(offset: int, size: int) -> None:
        if offset > len(image) or size > len(image) - offset:
            raise ValueError("truncated CUDA ELF image")

    bounds(h[6], h[11] * h[12])
    bounds(h[5], h[9] * h[10])
    for i in range(h[12]):
        section = struct.unpack_from("<IIQQQQIIQQ", image, h[6] + i * h[11])
        if section[1] != 8:  # SHT_NOBITS
            bounds(section[4], section[5])
    for i in range(h[10]):
        segment = struct.unpack_from("<IIQQQQQQ", image, h[5] + i * h[9])
        bounds(segment[2], segment[5])


def _sections(image: bytes) -> dict[str, bytes]:
    """Section name -> contents of a (validated) little-endian ELF64 image."""
    shoff = struct.unpack_from("<Q", image, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 0x3A)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", image, shoff + i * shentsize)
        for i in range(shnum)
    ]
    strtab = headers[shstrndx]
    names = image[strtab[4] : strtab[4] + strtab[5]]
    result = {}
    for name, kind, _, _, offset, size, *_ in headers:
        label = names[name : names.index(b"\0", name)].decode()
        result[label] = b"" if kind == 8 else image[offset : offset + size]  # NOBITS
    return result


def cubin_info(image: bytes) -> tuple[str, list[str], list[KernelParam]]:
    """(arch, kernel names, parameters of the first kernel) of a cubin ELF.

    Kernels are the ``.text.<name>`` sections; parameters come from the
    EIATTR_KPARAM_INFO (or _V2) entries of ``.nv.info.<name>``.
    """
    if image[:4] != b"\x7fELF":
        raise ValueError("expected a cubin (CUDA ELF) image")
    validate_image(image)
    sections = _sections(image)
    flags = struct.unpack_from("<I", image, 48)[0]
    sm = (flags >> EF_CUDA_SM_SHIFT) & 0xFF
    # The toolkit note records the -arch suffix: "a" (arch-specific) or "f"
    # (family-portable, e.g. the trtllm-gen sm_100f cubins).
    note = re.search(
        rb"-arch sm_%d([af]?)\b" % sm, sections.get(".note.nv.tkinfo", b"")
    )
    suffix = note.group(1).decode() if note else ""
    if flags & EF_CUDA_VARIANT_A:
        suffix = "a"
    arch = f"sm_{sm}" + suffix
    names = [n.removeprefix(".text.") for n in sections if n.startswith(".text.")]
    params: list[KernelParam] = []
    if names:
        info, pos = sections.get(f".nv.info.{names[0]}", b""), 0
        while pos + 4 <= len(info):
            fmt, attr = info[pos], info[pos + 1]
            if fmt != EIFMT_SVAL:
                pos += 4
                continue
            length = struct.unpack_from("<H", info, pos + 2)[0]
            if attr in (EIATTR_KPARAM_INFO, EIATTR_KPARAM_INFO_V2) and length >= 12:
                _, location, layout = struct.unpack_from("<III", info, pos + 4)
                size = (
                    (layout >> 18) & 0x3FFF
                    if attr == EIATTR_KPARAM_INFO
                    else layout & 0xFFFF
                )
                params.append(KernelParam(location & 0xFFFF, location >> 16, size))
            pos += 4 + length
        params.sort(key=lambda p: p.ordinal)
    return arch, names, params


def ctypes_value(value: Any) -> Any:
    """Coerce a launch argument to a ctypes instance holding its value.

    Tensors become device pointers; Python ints/floats are rejected because
    their width is ambiguous: wrap them in an explicit ctypes type.
    """
    if isinstance(value, torch.Tensor):
        return ctypes.c_void_p(value.data_ptr())
    if isinstance(value, (bytes, bytearray)):
        return (ctypes.c_char * len(value)).from_buffer_copy(value)
    if isinstance(value, (ctypes._SimpleCData, ctypes.Structure, ctypes.Array)):
        return value
    raise TypeError(
        f"kernel argument {value!r} must be a tensor, bytes or ctypes instance"
    )


class Workload(ABC):
    """One kernel in one cubin. Subclasses register with ``@register``.

    ``name`` and ``supported_arches`` are set by the registry decorator.
    ``package`` names the ``impls/`` directory whose ``kernels.json`` holds the
    cubin; it defaults to the defining module's basename.
    """

    name: ClassVar[str]
    supported_arches: ClassVar[tuple[str, ...]] = ()
    package: ClassVar[str | None] = None

    def __init__(
        self,
        cubin: bytes | None,
        *,
        device: str | torch.device | None = None,
    ):
        """``cubin=None`` selects reference-only use (no native launches)."""
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # ``arch`` is the registered target (``from_arch``), else the cubin's
        # own; ``cubin_arch`` is what the cubin declares (e.g. sm_80 SASS
        # registered for sm_86, sm_100f for sm_100a).
        self.arch: str | None = None
        self.cubin_arch: str | None = None
        self._image: bytes | None = None
        self._kernel_name: str | None = None
        self._params: list[KernelParam] = []
        self._module: cuda_driver.Module | None = None
        self._function: cuda_driver.Function | None = None
        self._closed = False
        self._lock = threading.RLock()
        if cubin is not None:
            self._accept_image(cubin)

    # -- construction from compiled artifacts --------------------------------

    @classmethod
    def manifest_path(cls) -> Path:
        package = cls.package or cls.__module__.rsplit(".", 1)[-1]
        return IMPLS / package / MANIFEST

    @classmethod
    def cubin_path(cls, arch: str) -> Path:
        arch = normalize_arch(arch)
        manifest = cls.manifest_path()
        if not manifest.is_file():
            raise FileNotFoundError(
                f"{manifest} is missing; run ./compile_kernels.py first"
            )
        entries = json.loads(manifest.read_text()).get(arch, {})
        if cls.name not in entries:
            raise KeyError(f"{manifest} has no {cls.name!r} cubin for {arch}")
        return manifest.parent / entries[cls.name]

    @classmethod
    def default_arch(cls, device: str | torch.device | None = None) -> str | None:
        """The supported arch matching ``device``, if any."""
        for arch in cls.supported_arches:
            if device_matches(arch, device):
                return arch
        return None

    @classmethod
    def from_arch(
        cls,
        arch: str | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> Workload:
        """Load this workload's cubin for ``arch`` (default: the device's arch)."""
        arch = arch or cls.default_arch(device)
        if arch is None:
            raise RuntimeError(
                f"{cls.name} supports {cls.supported_arches}, not the current device"
            )
        arch = normalize_arch(arch)
        if arch not in cls.supported_arches:
            raise ValueError(f"{cls.name} does not support {arch}")
        workload = cls(cls.cubin_path(arch).read_bytes(), device=device)
        assert workload.cubin_arch is not None
        if not arch_compatible(workload.cubin_arch, arch):
            raise ValueError(f"cubin for {arch} declares {workload.cubin_arch}")
        workload.arch = arch
        return workload

    # -- the single image ----------------------------------------------------

    def _accept_image(self, image: bytes) -> None:
        if not isinstance(image, bytes):
            raise TypeError("image must be bytes")
        arch, names, params = cubin_info(image)
        if len(names) != 1:
            raise ValueError(f"cubin must contain exactly one kernel, found {names}")
        if self._image is not None and [(p.offset, p.size) for p in params] != [
            (p.offset, p.size) for p in self._params
        ]:
            raise ValueError("patched kernel must keep the parameter layout")
        if self.arch is not None and not arch_compatible(arch, self.arch):
            raise ValueError(f"patched kernel is {arch}; it cannot run as {self.arch}")
        self.cubin_arch, self._kernel_name, self._params = arch, names[0], params
        self.arch = self.arch or arch
        self._image = image

    def image_name(self) -> str:
        """The (mangled) name of the cubin's single kernel."""
        if self._kernel_name is None:
            raise RuntimeError("reference-only workload has no image")
        return self._kernel_name

    def get_image(self) -> bytes:
        """The current cubin bytes; requires no GPU."""
        with self._lock:
            self._check_open()
            if self._image is None:
                raise RuntimeError("reference-only workload has no image")
            return self._image

    def patch_image(self, image: bytes) -> None:
        """Replace the cubin. It must hold one kernel with the same parameters."""
        with self._lock:
            self._check_open()
            if self._image is None:
                raise RuntimeError("reference-only workload has no image")
            self._accept_image(image)
            if self._module is not None:
                torch.cuda.synchronize(self.device)
                old, self._module, self._function = self._module, None, None
                old.unload()

    @property
    def kernel_params(self) -> list[KernelParam]:
        return list(self._params)

    # -- native launch --------------------------------------------------------

    def is_supported(self) -> bool:
        return (
            self.cubin_arch is not None
            and not self._closed
            and device_runs(self.cubin_arch, self.device)
            and (self.arch is None or device_matches(self.arch, self.device))
        )

    def configure(self, function: cuda_driver.Function) -> None:
        """Hook run once after the module loads, e.g. to raise the smem limit."""

    @property
    def function(self) -> cuda_driver.Function:
        """The loaded kernel; loads the module lazily under the device context."""
        with self._lock:
            self._check_open()
            if self._function is None:
                if not self.is_supported():
                    raise RuntimeError(f"{self.arch} cubin cannot run on {self.device}")
                cuda_driver.ensure_context(self.device)
                assert self._image is not None and self._kernel_name is not None
                module = cuda_driver.Module(self._image)
                try:
                    function = module.function(self._kernel_name)
                    self.configure(function)
                except BaseException:
                    module.unload()
                    raise
                self._module, self._function = module, function
            return self._function

    def launch(
        self,
        grid: int | Sequence[int],
        block: int | Sequence[int],
        args: Sequence[Any],
        *,
        shared_mem: int = 0,
        cluster: int | Sequence[int] | None = None,
        stream: torch.cuda.Stream | None = None,
        programmatic_serialization: bool = False,
        cooperative: bool = False,
        cluster_scheduling_policy: int | None = None,
    ) -> None:
        """Launch the kernel on ``stream`` (default: current PyTorch stream).

        ``args`` is checked against the cubin's parameter sizes.
        ``programmatic_serialization`` enables PDL (sm_90+), as upstream
        launchers do with ``enable_pdl``; ``cooperative`` is required by
        kernels that synchronize the whole grid; ``cluster_scheduling_policy``
        sets the cluster scheduling preference (e.g. trtllm-gen's SPREAD).
        """
        values = [ctypes_value(a) for a in args]
        sizes = [ctypes.sizeof(v) for v in values]
        expected = [p.size for p in self._params]
        if sizes != expected:
            raise ValueError(
                f"{self.name}: argument sizes {sizes} do not match kernel {expected}"
            )
        with self._lock, torch.cuda.device(self.device):
            function = self.function
            stream = stream or torch.cuda.current_stream(self.device)
            function.launch(
                grid,
                block,
                values,
                shared_mem=shared_mem,
                stream=stream.cuda_stream,
                cluster=cluster,
                programmatic_serialization=programmatic_serialization,
                cooperative=cooperative,
                cluster_scheduling_policy=cluster_scheduling_policy,
            )

    def prepare(self, inputs: tuple) -> tuple[Callable[[], tuple], tuple | None]:
        """(launch callable, outputs) for timing. Override to keep allocation
        and argument marshalling out of the timed callable."""
        return (lambda: self.run(inputs)), None

    # -- workload contract ------------------------------------------------------

    @abstractmethod
    def get_cases(self) -> list[CaseSpec]: ...

    @abstractmethod
    def get_inputs(self, case: CaseSpec) -> tuple: ...

    @abstractmethod
    def get_reference(self, inputs: tuple) -> tuple:
        """PyTorch reference on CPU or CUDA; must not mutate inputs."""

    @abstractmethod
    def run(self, inputs: tuple) -> tuple:
        """Allocate outputs and launch the kernel once (asynchronously)."""

    @abstractmethod
    def validate(self, ref: tuple, impl: tuple) -> None:
        """Raise AssertionError unless ``impl`` matches ``ref``."""

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def assert_close(
        ref: tuple, impl: tuple, rtol: float = 1e-2, atol: float = 1e-2
    ) -> None:
        if (
            not isinstance(ref, tuple)
            or not isinstance(impl, tuple)
            or len(ref) != len(impl)
        ):
            raise AssertionError("outputs must be tuples of equal length")
        for expected, actual in zip(ref, impl):
            if not isinstance(expected, torch.Tensor) or not isinstance(
                actual, torch.Tensor
            ):
                raise AssertionError("every output must be a torch tensor")
            torch.testing.assert_close(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=False,
                check_device=True,
                check_dtype=True,
            )

    def generator(self, case: CaseSpec) -> torch.Generator:
        return torch.Generator(device=self.device).manual_seed(case.seed)

    def randn(self, shape, generator, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(shape, device=self.device, generator=generator) * scale).to(
            dtype
        )

    def scalar(self, value, dtype=torch.float32):
        return torch.tensor(value, dtype=dtype)

    # -- lifecycle -----------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("workload is closed")

    def close(self) -> None:
        with self._lock:
            if self._module is not None:
                torch.cuda.synchronize(self.device)
                self._module.unload()
                self._module, self._function = None, None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

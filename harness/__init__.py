"""Uniform, standalone CUDA workload benchmarking."""

from .registry import REGISTRY, get_workload, register, register_variant
from .workload import CaseSpec, Workload, normalize_arch


def enumerate_workloads(arch: str | None = None) -> list[type[Workload]]:
    """Registered workload classes, by name; only those supporting ``arch`` if given."""
    from . import workloads  # noqa: F401  (registers the built-in workloads)

    target = normalize_arch(arch) if arch is not None else None
    return [
        cls
        for _, cls in sorted(REGISTRY.items())
        if target is None or target in cls.supported_arches
    ]


__all__ = [
    "CaseSpec",
    "REGISTRY",
    "Workload",
    "enumerate_workloads",
    "get_workload",
    "normalize_arch",
    "register",
    "register_variant",
]

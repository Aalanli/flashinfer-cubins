"""Workload registry: ``@register(name=..., supported_arches=...)`` and
``register_variant`` for programmatically generated variants."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from .workload import Workload, normalize_arch

W = TypeVar("W", bound=type[Workload])

REGISTRY: dict[str, type[Workload]] = {}


def register(*, name: str, supported_arches: Iterable[str]) -> Callable[[W], W]:
    """Register a single-kernel Workload subclass under a unique name."""
    arches = tuple(dict.fromkeys(normalize_arch(a) for a in supported_arches))
    if not arches:
        raise ValueError(f"{name}: supported_arches must be nonempty")

    def decorate(cls: W) -> W:
        if not (isinstance(cls, type) and issubclass(cls, Workload)):
            raise TypeError("@register applies to Workload subclasses")
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise ValueError(f"workload {name!r} is already registered")
        cls.name = name
        cls.supported_arches = arches
        REGISTRY[name] = cls
        return cls

    return decorate


def register_variant(
    base: type[Workload],
    *,
    name: str,
    supported_arches: Iterable[str],
    **attributes: Any,
) -> type[Workload]:
    """Register a generated subclass of ``base`` holding ``attributes``.

    For packages whose kernels are variants of one workload (dtypes, head
    dims, tile shapes): one ``base`` class implements the contract from
    class attributes, and each kernel registers as a variant instead of a
    hand-written subclass. The class lives in ``base``'s module, so
    ``package`` and ``sys.modules[cls.__module__]`` resolve as for ordinary
    workloads.
    """
    qualname = "".join(part.title() for part in re.split(r"[^0-9A-Za-z]+", name))
    namespace = {"__module__": base.__module__, "__qualname__": qualname}
    cls = type(qualname, (base,), {**namespace, **attributes})
    return register(name=name, supported_arches=supported_arches)(cls)


def get_workload(name: str) -> type[Workload]:
    from . import workloads  # noqa: F401  (registers the built-in workloads)

    return REGISTRY[name]

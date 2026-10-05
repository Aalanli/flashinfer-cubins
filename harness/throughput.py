"""Shape-based throughput coverage; inputs are regenerated, not trace tensor replay.

Every throughput case records where its shape comes from (``CaseSpec.source``):

* ``official_shape`` (``trace``): a row of the downloaded official inventories;
* ``model_shape`` (``model_case``): a layer of a model in ``harness.models``;
* ``upstream_test`` (``upstream_case``): a parametrization of an upstream
  test suite, so upstream's tested cases stay a subset of the harness's;
* ``synthetic_stress`` (``synthetic``): a stress shape with its reason.

Ragged lengths use ``skewed_lengths`` (seeded long-tailed) where the recorded
lengths are not replayed; ``split_lengths`` (balanced) hides scheduler tail
imbalance and is kept only for cases that need equal lengths.
"""

import json
import random
from functools import lru_cache

from .models import MODELS
from .workload import ROOT, CaseSpec

DATA = ROOT / "resources/benchmark_workloads"


def split_lengths(total, batch):
    """Balanced synthetic ragged lengths, preserving recorded batch/total axes."""
    return [total // batch + (i < total % batch) for i in range(batch)]


def skewed_lengths(total, batch, seed=0, sigma=1.0, minimum=1):
    """Seeded long-tailed (log-normal) lengths of ``batch`` requests summing
    to ``total``, each at least ``minimum``: a few long requests among many
    short ones, as in serving traces."""
    if batch * minimum > total:
        raise ValueError(f"{batch} lengths of at least {minimum} exceed {total}")
    rng = random.Random(seed)
    weights = [rng.lognormvariate(0.0, sigma) for _ in range(batch)]
    spare = total - batch * minimum
    scale = spare / sum(weights)
    lengths = [minimum + int(w * scale) for w in weights]
    for i in sorted(range(batch), key=lambda i: -weights[i])[: total - sum(lengths)]:
        lengths[i] += 1
    return lengths


@lru_cache(None)
def inventory(name):
    return (
        [
            json.loads(line)["workload"]
            for line in (DATA / f"{name}.jsonl").read_text().splitlines()
        ],
        json.loads((DATA / "manifest.json").read_text())[name],
    )


def trace(name, label, params, axes):
    rows, manifest = inventory(name)
    row = next(r for r in rows if r["axes"] == axes)
    return CaseSpec(
        "throughput_" + label,
        params,
        1111,
        "throughput",
        {
            **manifest,
            "kind": "official_shape",
            "inventory": name,
            "uuid": row["uuid"],
            "axes": axes,
            "input_policy": "Seeded synthetic values, lengths and page mappings; not tensor-blob replay.",
        },
    )


def synthetic(label, params, reason):
    return CaseSpec(
        "throughput_" + label,
        params,
        1111,
        "throughput",
        {"kind": "synthetic_stress", "reason": reason},
    )


def model_case(label, params, model, layer=None, *, suite="throughput", seed=1111):
    """A case whose shape is ``layer`` of ``model`` (a ``harness.models`` key)."""
    spec = MODELS[model]
    source = {"kind": "model_shape", "model": model, "hf": spec.hf}
    if layer is not None:
        source["layer"] = layer
    prefix = "throughput_" if suite == "throughput" else ""
    return CaseSpec(prefix + label, params, seed, suite, source)


def upstream_case(label, params, test, *, suite="smoke", seed=0, revision=None):
    """A case mirroring the upstream test parametrization ``test``
    (``path::function[params]`` inside the pinned upstream tree)."""
    source = {"kind": "upstream_test", "test": test}
    if revision is not None:
        source["revision"] = revision
    prefix = "throughput_" if suite == "throughput" else ""
    return CaseSpec(prefix + label, params, seed, suite, source)


def gemm_case(name, shape, source):
    m, n, k, *batch = shape
    params = dict(m=m, n=n, k=k)
    if batch:
        params["l"] = batch[0]
    label = "_".join(f"{key}{value}" for key, value in params.items())
    return CaseSpec("throughput_" + label, params, 1111, "throughput", source)


def throughput_cases(package):
    """The throughput cases a package's kernels must serve together, defined
    by ``throughput_cases()`` of ``harness.workloads.<package>``."""
    from importlib import import_module

    return import_module(f"harness.workloads.{package}").throughput_cases()

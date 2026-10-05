"""python -m harness.benchmark [--mode check|benchmark|reference].

Kernels are precompiled by ``compile_kernels.py``; this tool only loads cubins.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from . import enumerate_workloads


def benchmark(workload, inputs, warmup=5, iterations=20):
    """CUDA event timings, excluding generation, reference, allocation and setup.

    ``prepare`` retains outputs and arguments, so each timed call is the
    workload's single kernel launch. No hidden reference fallback.
    """
    launch, _ = workload.prepare(inputs)
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize(workload.device)
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "samples_us": samples,
        "timing": "cuda_events_preallocated",
    }


def open_workload(cls, mode, device):
    """(workload, cubin path or None) — reference-only when not runnable here."""
    arch = cls.default_arch(device)
    if mode == "reference" or arch is None:
        return cls(None, device=device), None
    return cls.from_arch(arch, device=device), str(cls.cubin_path(arch))


def main(argv=None):
    registered = {cls.name: cls for cls in enumerate_workloads()}
    packages = sorted({cls.manifest_path().parent.name for cls in registered.values()})
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--workload",
        action="append",
        choices=["all", *registered, *(n for n in packages if n not in registered)],
        help="Workload or package name (repeatable; default: all)",
    )
    p.add_argument(
        "--mode", choices=["check", "benchmark", "reference"], default="check"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--case", help="Exact case name")
    p.add_argument(
        "--suite",
        choices=("all", "smoke", "throughput"),
        default="all",
        help="Small correctness cases, larger throughput shapes, or both (default)",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if a.iterations < 1 or a.warmup < 0:
        p.error("iterations must be positive and warmup nonnegative")
    torch.set_grad_enabled(False)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    # Reference arithmetic must not silently use TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    wanted = set(a.workload or ["all"])
    selected = [
        cls
        for name, cls in registered.items()
        if "all" in wanted
        or name in wanted
        or cls.manifest_path().parent.name in wanted
    ]
    report = {
        "torch": torch.__version__,
        "device": a.device,
        "mode": a.mode,
        "suite": a.suite,
        "results": [],
    }
    if a.device.startswith("cuda") and torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(a.device)
        report["capability"] = torch.cuda.get_device_capability(a.device)
    failed = False
    for cls in selected:
        record: dict[str, Any] = {"workload": cls.name, "cubin": None, "cases": []}
        report["results"].append(record)
        try:
            workload, record["cubin"] = open_workload(cls, a.mode, a.device)
            with workload:
                cases = [
                    c
                    for c in workload.get_cases()
                    if (a.case is None or c.name == a.case)
                    and (a.suite == "all" or c.suite == a.suite)
                ]
                if not cases and a.case is not None and len(selected) == 1:
                    raise ValueError(f"no cases match {a.case!r}")
                for case in cases:
                    entry = {
                        "case": case.name,
                        "params": case.params,
                        "seed": case.seed,
                        "suite": case.suite,
                        "source": case.source,
                    }
                    record["cases"].append(entry)
                    before = time.perf_counter()
                    inputs = workload.get_inputs(case)
                    if not isinstance(inputs, tuple) or not all(
                        isinstance(t, torch.Tensor) for t in inputs
                    ):
                        raise AssertionError(
                            "get_inputs must return a tuple of torch tensors"
                        )
                    ref = workload.get_reference(inputs)
                    workload.validate(ref, ref)
                    entry["reference"] = "passed"
                    if a.mode != "reference" and workload.is_supported():
                        out = workload.run(inputs)
                        torch.cuda.synchronize(workload.device)
                        workload.validate(ref, out)
                        entry["implementation"] = "passed"
                        if a.mode == "benchmark":
                            entry.update(
                                benchmark(workload, inputs, a.warmup, a.iterations)
                            )
                        del out
                    else:
                        entry["implementation"] = (
                            "reference_only"
                            if a.mode == "reference"
                            else "skipped_architecture"
                        )
                    entry["check_seconds"] = time.perf_counter() - before
                    print(
                        cls.name,
                        case.name,
                        entry["reference"],
                        entry["implementation"],
                        f"{entry['median_us']:.2f} us" if "median_us" in entry else "",
                        flush=True,
                    )
                    del inputs, ref
                    gc.collect()
        except Exception as exc:
            failed = True
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(cls.name, "FAILED", record["error"], flush=True)
    if a.output:
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

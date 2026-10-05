"""Registered single-kernel workloads: artifacts, references and native runs.

``NativeWorkloads`` checks every workload whose architecture matches the
current GPU against its own cases, on the GPU only: inputs, reference, launch
and validation all run on CUDA (``python -m unittest
tests.test_workloads.NativeWorkloads``). One long-lived worker process
checks the workloads one at a time (only one, so a large case always has the
whole GPU); each workload synchronizes after every launch, so an error is
attributed to it. A worker whose CUDA context is lost (e.g. an illegal memory
access) reports that workload as failed and exits, and a fresh worker takes
over. Results, with the traceback of every failure, are written
to WORKLOAD_TEST_REPORT (default ``test_results_native.json``) as they arrive;
progress goes to stderr.

``RegisteredWorkloads`` checks registry, artifacts, references (every smoke
case, on CUDA) and parameter layouts.

Environment:

* WORKLOAD_TEST_SUITE: ``smoke`` (default) or ``all`` (adds throughput cases).
* WORKLOAD_TEST_FILTER=<regex>: only workloads whose name it matches.
* WORKLOAD_TEST_TIMEOUT: seconds one workload may take (default 900).

``python tests/test_workloads.py <name>[,<name>...] <arch> <suite>`` checks
the given workloads in this process (e.g. under compute-sanitizer).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import enumerate_workloads, get_workload  # noqa: E402
from harness.workload import (  # noqa: E402
    arch_compatible,
    cubin_info,
    device_matches,
)

SUITE = os.environ.get("WORKLOAD_TEST_SUITE", "smoke")
FILTER = re.compile(os.environ.get("WORKLOAD_TEST_FILTER", ""))
TIMEOUT = float(os.environ.get("WORKLOAD_TEST_TIMEOUT", "900"))
REPORT = Path(
    os.environ.get("WORKLOAD_TEST_REPORT", str(ROOT / "test_results_native.json"))
)
RESULT = "\x1eRESULT\t"  # worker -> parent protocol line prefix


def selected_workloads(arch: str | None = None):
    return [cls for cls in enumerate_workloads(arch) if FILTER.search(cls.name)]


def snapshot(inputs):
    """Host copies of the tensor inputs: a GPU clone of a multi-GiB input
    (deep_gemm's 256-group FP8 weights) would not fit beside it."""
    return [
        t.to("cpu", copy=True) if isinstance(t, torch.Tensor) else t for t in inputs
    ]


UNCHANGED_CHUNK = 1 << 24  # elements compared at a time


def unchanged(old, new) -> bool:
    """Exact equality of a host snapshot ``old`` and the tensor ``new``; NaN
    inputs (e.g. undefined rows) compare equal. Compared in chunks moved to
    the host, so no comparison temporaries are allocated on the GPU."""
    if old.shape != new.shape:
        return False
    old, new = old.reshape(-1), new.reshape(-1)
    try:
        for start in range(0, max(old.numel(), 1), UNCHANGED_CHUNK):
            stop = start + UNCHANGED_CHUNK
            torch.testing.assert_close(
                new[start:stop].cpu(), old[start:stop], rtol=0, atol=0, equal_nan=True
            )
    except AssertionError:
        return False
    return True


def check_native(names: str, arch: str, suite: str) -> None:
    """Check each comma-separated workload in this process."""
    for name in names.split(","):
        check_one(name, arch, suite)
        print(f"{name} {arch} passed", flush=True)


def check_one(name: str, arch: str, suite: str) -> int:
    """Every ``suite`` case of one workload on the GPU: the native output
    validates against the reference, and inputs are not mutated. The first
    case is run again after re-patching the same image (module reload)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    cls = get_workload(name)
    with cls.from_arch(arch, device="cuda") as workload:
        assert workload.is_supported(), f"{name} {arch} unsupported on this GPU"
        cases = [c for c in workload.get_cases() if suite == "all" or c.suite == suite]
        assert cases, f"{name} has no {suite} cases"
        for index, case in enumerate(cases):
            inputs = workload.get_inputs(case)
            before = snapshot(inputs)
            ref = workload.get_reference(inputs)
            out = workload.run(inputs)
            torch.cuda.synchronize()
            workload.validate(ref, out)
            for old, new in zip(before, inputs):
                if isinstance(old, torch.Tensor):
                    assert unchanged(old, new), f"{name}/{case.name} mutated inputs"
            if index == 0:
                workload.patch_image(workload.get_image())
                again = workload.run(inputs)
                torch.cuda.synchronize()
                workload.validate(ref, again)
                del again
            del out
            check_prepared(workload, case, inputs, before, ref)
            del inputs, before, ref
    torch.cuda.synchronize()
    return len(cases)


def check_prepared(workload, case, inputs, before, ref) -> None:
    """``prepare``'s callable launches without allocating and returns the
    prepared outputs, which validate after the first call. Later calls may
    change the values (e.g. accumulating epilogues) but not the work, so they
    are only checked for allocations."""
    tag = f"{workload.name}/{case.name}"
    launch, outputs = workload.prepare(inputs)
    torch.cuda.synchronize()
    for call in range(2):
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
        result = launch()
        new = torch.cuda.memory_stats()["allocation.all.allocated"] - allocated
        assert not new, f"{tag}: prepared launch {call} allocated {new} blocks"
        assert result is outputs, f"{tag}: prepared launch must return its outputs"
        torch.cuda.synchronize()
        if call == 0:
            workload.validate(ref, outputs)
    for old, new_input in zip(before, inputs):
        if isinstance(old, torch.Tensor):
            assert unchanged(old, new_input), f"{tag}: prepared launch mutated inputs"


def cuda_usable() -> bool:
    """False once the context is lost (sticky errors such as illegal address)."""
    try:
        torch.cuda.synchronize()
        return torch.ones(1, device="cuda").sum().item() == 1
    except Exception:
        return False


def serve(arch: str, suite: str) -> int:
    """Worker process: check the workload named on each stdin line and report
    one ``RESULT`` JSON line on the original stdout; anything else written to
    stdout goes to stderr. Exits after a failure that lost the CUDA context."""
    channel = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    torch.set_num_threads(1)  # nothing runs on the CPU
    for line in sys.stdin:
        name = line.strip()
        start = time.monotonic()
        cases = 0
        try:
            cases = check_one(name, arch, suite)
            # A fault can surface only at the next CUDA call (e.g. after the
            # module unload): it still belongs to this workload.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            status, error = "ok", ""
        except Exception:
            status, error = "fail", traceback.format_exc()
        fatal = status == "fail" and not cuda_usable()
        if status == "fail" and not fatal:
            try:
                torch.cuda.empty_cache()
            except Exception:
                fatal = True
        record = {
            "name": name,
            "status": status,
            "cases": cases,
            "seconds": round(time.monotonic() - start, 3),
            "error": error,
            "fatal": fatal,
        }
        channel.write(RESULT + json.dumps(record) + "\n")
        if fatal:
            return 1
    return 0


class Worker:
    """One ``serve`` child; ``check`` returns its result for one workload."""

    def __init__(self, arch: str, log_dir: str):
        self.log = tempfile.NamedTemporaryFile(
            "w+", dir=log_dir, prefix="worker-", suffix=".log", delete=False
        )
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--worker", arch, SUITE],
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if line.startswith(RESULT):
                self.lines.put(line[len(RESULT) :])
        self.lines.put(None)

    def log_tail(self, size: int = 4000) -> str:
        self.log.flush()
        with open(self.log.name, errors="replace") as f:
            return f.read()[-size:]

    def check(self, name: str) -> dict:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(name + "\n")
            self.proc.stdin.flush()
            line = self.lines.get(timeout=TIMEOUT)
            if line is None:  # the reader saw EOF: the worker is gone
                error = f"worker exited ({self.proc.wait()}) before reporting"
        except BrokenPipeError:
            line, error = None, f"worker exited ({self.proc.wait()})"
        except queue.Empty:
            line, error = None, f"timed out after {TIMEOUT:.0f} s"
        if line is None:
            self.kill()
            return {"name": name, "status": "fail", "error": error, "fatal": True}
        return json.loads(line)

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()

    def close(self) -> None:
        """Stop the worker (if still running) and release every pipe."""
        assert self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.close()
        except BrokenPipeError:  # the worker already exited
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.kill()
        self.reader.join(timeout=10)
        self.proc.stdout.close()
        self.log.close()
        os.unlink(self.log.name)


def run_native(names: list[str], arch: str) -> dict[str, dict]:
    """Check ``names`` with one worker at a time; results by workload name."""
    results: dict[str, dict] = {}
    began = last = time.monotonic()

    def report(final: bool = False) -> None:
        failed = sorted(n for n, r in results.items() if r["status"] != "ok")
        summary = {
            "arch": arch,
            "suite": SUITE,
            "total": len(names),
            "done": len(results),
            "cases": sum(r.get("cases", 0) for r in results.values()),
            "failed": failed,
            "seconds": round(time.monotonic() - began, 1),
            "complete": final,
            "results": results,
        }
        tmp = REPORT.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=1))
        os.replace(tmp, REPORT)
        print(
            f"native {arch}: {len(results)}/{len(names)} checked, "
            f"{summary['cases']} cases, {len(failed)} failed, {summary['seconds']} s",
            file=sys.stderr,
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="native-") as log_dir:
        worker: Worker | None = None
        for name in names:
            worker = worker or Worker(arch, log_dir)
            result = worker.check(name)
            if result["status"] != "ok":
                result["log"] = worker.log_tail()
            if result.get("fatal"):
                worker.close()
                worker = None
            results[name] = result
            if result["status"] != "ok" or time.monotonic() - last > 30:
                last = time.monotonic()
                report()
        if worker is not None:
            worker.close()
    report(final=True)
    return results


class LaunchAttributes(unittest.TestCase):
    def test_attribute_values_are_written(self):
        """The CUlaunchAttribute value union must hold the requested payload
        (cluster dims, PDL and cooperative flags), not stay zero."""
        import ctypes
        import struct

        from harness import cuda_driver

        attr = cuda_driver._launch_attribute(
            cuda_driver.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION,
            (ctypes.c_uint * 3)(2, 1, 1),
        )
        raw = bytes(ctypes.string_at(ctypes.addressof(attr), ctypes.sizeof(attr)))
        self.assertEqual(ctypes.sizeof(attr), 72)
        self.assertEqual(struct.unpack_from("<i", raw, 0)[0], 4)
        self.assertEqual(struct.unpack_from("<3I", raw, 8), (2, 1, 1))
        flag = cuda_driver._launch_attribute(
            cuda_driver.CU_LAUNCH_ATTRIBUTE_COOPERATIVE, (ctypes.c_int * 1)(1)
        )
        raw = bytes(ctypes.string_at(ctypes.addressof(flag), ctypes.sizeof(flag)))
        self.assertEqual(struct.unpack_from("<ii", raw, 0)[0], 2)
        self.assertEqual(struct.unpack_from("<i", raw, 8)[0], 1)


class RegisteredWorkloads(unittest.TestCase):
    def test_registry(self):
        workloads = selected_workloads()
        self.assertTrue(workloads, "no workloads registered")
        names = [cls.name for cls in workloads]
        self.assertEqual(len(names), len(set(names)))
        by_arch: dict[str, set] = {}
        for cls in workloads:
            self.assertTrue(cls.supported_arches, cls.name)
            for arch in cls.supported_arches:
                if arch not in by_arch:
                    by_arch[arch] = set(selected_workloads(arch))
                self.assertIn(cls, by_arch[arch])

    def test_artifacts(self):
        """Every supported arch has a one-kernel cubin; no GPU needed."""
        for cls in selected_workloads():
            for arch in cls.supported_arches:
                with self.subTest(workload=cls.name, arch=arch):
                    image = cls.cubin_path(arch).read_bytes()
                    declared, kernels, _ = cubin_info(image)
                    self.assertTrue(arch_compatible(declared, arch), declared)
                    self.assertEqual(len(kernels), 1)
                    with cls.from_arch(arch, device="cpu") as workload:
                        self.assertEqual(workload.image_name(), kernels[0])
                        self.assertEqual(workload.get_image(), image)
                        self.assertFalse(workload.is_supported())

    def check_reference(self, workload, inputs) -> None:
        self.assertIsInstance(inputs, tuple)
        before = snapshot(inputs)
        ref = workload.get_reference(inputs)
        workload.validate(ref, ref)
        for old, new in zip(before, inputs):
            if isinstance(old, torch.Tensor):
                self.assertTrue(unchanged(old, new))

    def test_references(self):
        """References validate against themselves and never mutate inputs,
        for every smoke case, on CUDA."""
        if not torch.cuda.is_available():
            self.skipTest("references are checked on CUDA")
        for cls in selected_workloads():
            with self.subTest(workload=cls.name):
                workload = cls(None, device="cuda")
                cases = [c for c in workload.get_cases() if c.suite == "smoke"]
                self.assertTrue(cases)
                for case in cases:
                    inputs = workload.get_inputs(case)
                    self.check_reference(workload, inputs)
                    del inputs

    def test_param_layouts(self):
        """Modules that rebuild by-value structs from probe sidecars expose
        ``check_param_layouts()``; it byte-compares every probe example."""
        modules = {sys.modules[cls.__module__] for cls in selected_workloads()}
        for module in sorted(modules, key=lambda m: m.__name__):
            check = getattr(module, "check_param_layouts", None)
            if check is not None:
                with self.subTest(module=module.__name__):
                    self.assertTrue(check())


class NativeWorkloads(unittest.TestCase):
    def test_native(self):
        """Every workload matching the GPU validates against its cases."""
        # The workers need the GPU: return what this process's allocator
        # cached for earlier tests (e.g. test_references) to the driver.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        groups: dict[str, list[str]] = {}
        for cls in selected_workloads():
            for arch in cls.supported_arches:
                if device_matches(arch):
                    groups.setdefault(arch, []).append(cls.name)
        if not groups:
            self.skipTest("no registered workload matches the current GPU")
        for arch, names in sorted(groups.items()):
            results = run_native(names, arch)
            for name in names:
                result = results[name]
                if result["status"] != "ok":
                    with self.subTest(workload=name, arch=arch):
                        self.fail(
                            result["error"][-6000:] + "\n" + result.get("log", "")
                        )


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(serve(*sys.argv[2:]))
    if len(sys.argv) == 4 and not sys.argv[1].startswith("-"):
        check_native(*sys.argv[1:])
    else:
        unittest.main()

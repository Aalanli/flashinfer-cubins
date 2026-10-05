"""sm_100a build of the fmha package: the trtllm-gen FMHA cubins of ``cubins/fmha``.

Source: the 7,634 cubins of ``cubins/fmha`` are the trtllm-gen FMHA artifact
``55bba55929d4093682e32d817bd11ffb0441c749/fmha/trtllm-gen`` that FlashInfer
v0.6.8/v0.6.9 pin (``flashinfer/artifacts.py``), shipped in the
``flashinfer_cubin`` wheel. Every file is checked against the artifact's
``checksums.txt`` and the SHA256 of its meta-info row
(``include/flashInferMetaInfo.h``, fetched by ``scripts/fetch_resources.py
--only trtllm-gen``). Each cubin holds the attention kernel and a
``<kernel>GetSmemSize`` helper that host code never launches; the helper is
stripped (``harness.cubin_strip.strip_cubin`` + ``check_strip``), giving
``cubins/sm_100a/<workload>.cubin``.

Kernel selection is FlashInfer v0.6.9's ``TllmGenFmhaKernel::run``
(``include/flashinfer/trtllm/fmha/fmhaKernels.cuh``) on a B200 (``kSM_100``,
148 SMs, ``harness.workloads.fmha.trtllm.NUM_SMS``;
``cuOccupancyMaxActiveClusters`` answers with the cluster counts measured
there, so CgaSmemReduction kernels get only shapes that fit one wave and the
rest fall back to GmemReduction as upstream does). The host-only probe
``kernels/trtllm_probe.cu`` runs that code unmodified with the driver API
interposed: for each ``TllmGenFmhaRunnerParams`` it reports the kernel
selected, the launch configuration and the ``KernelParams`` bytes
(deterministic stand-in TMA descriptors).

Cases, in two probe passes:

1. Selection sweep (``emit=0``): per group of rows sharing selection-relevant
   traits, runner parameters over head counts (GQA groups 1-128, 1-8 KV
   heads), query lengths, batch sizes (1-256), KV lengths (40-131072),
   sliding windows and runner modes (FlashInfer's launchers and the generic
   runner's), a seeded sample of that product plus the former grid; the
   FlashInfer test parametrizations (``trtllm_upstream``); model and stress
   shapes. Selection depends only on the maxima, batch and heads.
2. Per kernel: ``smoke`` cases picked greedily for coverage under a cost cap
   (batch >= 3 with skewed ragged lengths incl. a minimal request, several
   KV heads and GQA groups, >= 3 KV tiles, multi-CTA KV splits, > 148 CTAs,
   several query tokens/tiles, partial pages and tiles), with the runtime
   features cycled over them (attention sinks, device bmm1/bmm2 scales, bmm2
   scale != 1, NHD caches, separate K/V page indices, separate K/V tensors,
   non-contiguous queries, shared prefix pages, NVFP4 scale-factor offsets,
   skip-softmax tile patterns and FlashInfer's tiny threshold); one
   ``upstream_test`` case per distinct upstream shape that selects the kernel
   (flags chosen so every upstream flag value of the kernel appears; large
   ones in the throughput suite); ``throughput`` cases from model head
   configurations (``harness.models``, tensor-parallel shards) at serving
   scale or, where no model shape selects the kernel, the largest selecting
   stress shape; official DSA inventory rows for sparse MLA. Every final case
   is probed again with its full runner fields (``emit=1``): it must select
   its kernel and the Python port must reproduce the launch and
   ``KernelParams`` byte for byte (``check_param_layouts`` re-checks).

Upstream coverage: ``fixtures/sm_100a.json`` ``upstream`` maps each
upstream shape key (hash) to the kernel it selects (or the reason it is not
a workload); ``tests/test_fmha.py`` re-enumerates the parametrizations and
checks that each selected live kernel has a case with that shape and every
upstream flag value. ``provenance.json`` summarizes the mapping.

Live/dead (``provenance.json`` lists every excluded kernel with its reason):

=====================================  =========================================
kernels                                status / reason
=====================================  =========================================
main table, selected by some runner    live
parameters on sm_100
``kSM_100f`` twin of a ``kSM_100``     dead: ``loadKernels`` keeps the sm_100
kernel with the same hash                specific kernel for that hash
``mReuseSmemKForV``                    dead: ``TllmGenSelectKernelParams``
                                         initialises it false; nothing sets it
non-sparse MLA SwapsMmaAb, tileQ 32    dead: that selection caps tileQ at 16
custom mask                            excluded: the TRT-LLM packed custom
                                         (tree) mask is not modelled
``GmemReductionWithSeparateKernel``    excluded: the output needs the separate
                                         ``runFmhaReduction`` kernel (FlashInfer
                                         JIT source, not in this collection)
context kernels with tile sizes,       dead: context selection keeps tileQ =
head-dim split, 2-CTA or sparse traits   tileKv = 128, headDimPerCtaV = headDimV,
other than selection produces            no 2-CTA MMA, no sparse MLA
NVFP4 output with multi-CTA KV        excluded (defect, measured on B200):
reduction (GMEM and CGA)                 wrong (up to saturated) output scale
                                         factors, some only after an unrelated
                                         kernel ran; racecheck reports a smem
                                         hazard
FP8 output, head dim 64, SwapsMmaAb    excluded (defect): heads 8-15 of each
tileQ >= 16, one CTA per KV              tile stored pairwise swapped
NVFP4 KV, head dim 256, context        excluded (defect): V scale factors of
                                         blocks >= 8 read at overlapping offsets
NVFP4 KV, head dim 64, 16-token pages  excluded (defect): misaligned shared-
                                         memory address fault
PackedQkv kernel whose code equals a   removed (duplicate): identical .text,
SeparateQkv kernel                       .nv.info and constant sections; the
                                         SeparateQkv kernel keeps its cases and
                                         serves packed QKV (same kernel code,
                                         the PackedQkv runner layout)
other rows no candidate selects        dead on sm_100 for every probed shape
``sTllmGenFmhaKernelMetaInfosVx``      dead: Sage / int8-QK kernels of a table
(44 cubins)                              no FlashInfer v0.6.9 code loads
=====================================  =========================================

Case restrictions (shape-dependent defects, measured on B200 with upstream's
KernelParams and reproduced with FlashInfer v0.6.9's
``trtllm_batch_decode_with_kv_cache``; the kernels compute every other shape
correctly, so they stay live and ``case_restriction`` keeps their cases out
of these shapes; ``provenance.json`` ``case_restrictions`` counts the dropped
candidates):

=====================================  =========================================
kernels                                restriction / defect
=====================================  =========================================
generation, Q dtype != KV dtype        numHeadsQPerKv <= stepQ: above it only
                                         the first tile of query heads (grid
                                         y = 0) is right
SlidingOrChunkedCausal generation,     query token i takes its KV tile range
one query token per CTA                  from kv - 1 - 2 (q-1-i) instead of
(groupsTokensHeadsQ false)               kv - 1 - (q-1-i): it loses its last
                                         tile (or its whole row) or, at the
                                         window start, also reads the previous
                                         tile unmasked; cases keep both in the
                                         same tile at both ends
FP16/BF16 Q, E4M3 KV, head dim 64      NHD caches only with one KV head: NHD
                                         with more KV heads gives wrong output
=====================================  =========================================
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import itertools
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ARCH = "sm_100a"
FLASHINFER_V069 = "a1aa676196f798435248d9ea205c67674476f473"
ARTIFACT = "55bba55929d4093682e32d817bd11ffb0441c749/fmha/trtllm-gen"
CUTLASS = "cutlass-b46b16d003484063bca4ed365e44095c4c6ed633"
PROBE = "kernels/trtllm_probe.cu"
DTYPES = {
    "DATA_TYPE_FP16": "fp16",
    "DATA_TYPE_BF16": "bf16",
    "DATA_TYPE_E4M3": "e4m3",
    "DATA_TYPE_E2M1": "e2m1",
    "DATA_TYPE_FP32": "fp32",
}
SM = {"kSM_100": "100a", "kSM_100f": "100f", "kSM_103": "103a"}
# Examples (full KernelParams bytes) kept in the fixtures for this many kernels.
EXAMPLES = 64
# Smoke shape picks per kernel and their cost cap (reference FLOPs, input
# bytes): well under a second on a B200, and the RTX 3090 computes every
# smoke reference in tests.test_workloads.
SMOKE_PICKS = 3
SMOKE_FLOPS = 3e10
SMOKE_BYTES = 1 << 28
# Throughput cases per kernel and their caps (inputs; the reference adds at
# most a 512 MiB logit chunk and one request's float32 K/V).
THROUGHPUT_PICKS = 2
THROUGHPUT_FLOPS = 4e13
THROUGHPUT_BYTES = 12 << 30
# Seeded generation-kernel selection candidates per group (besides the grid).
POOL_SAMPLES = 4000
SKIP_SCALE = 1e-3  # skip-softmax threshold scale of the tile-pattern cases


def _log(message: str) -> None:
    print(f"[fmha sm_100a] {message}", file=sys.stderr, flush=True)


def _compiler_module(compiler: Any) -> Any:
    return sys.modules[type(compiler).__module__]


def load_upstream(package: Path) -> Any:
    """``trtllm_upstream.py`` next to this file (loaded by path)."""
    name = "impls_fmha_trtllm_upstream"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, package / "trtllm_upstream.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# -- meta-info -------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    body = line.strip().rstrip(",").strip()
    assert body.startswith("{") and body.endswith("}")
    return [p.strip() for p in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', body[1:-1])]


def parse_meta_info(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """(rows of ``sTllmGenFmhaKernelMetaInfos``, sm_100 kernel names of the
    Vx table)."""
    text = path.read_text().splitlines()
    rows: list[dict[str, Any]] = []
    vx: list[str] = []
    table = None
    for line in text:
        if "sTllmGenFmhaKernelMetaInfos[] = {" in line:
            table = "main"
            continue
        if "sTllmGenFmhaKernelMetaInfosVx[] = {" in line:
            table = "vx"
            continue
        if line.startswith("};"):
            table = None
            continue
        if table is None or not line.startswith("{ DATA_TYPE"):
            continue
        p = _split_row(line)
        name = p[13].strip('"')
        if table == "vx":
            if SM[p[10]] != "103a":  # kSM_103 kernels are not in cubins/fmha
                vx.append(name)
            continue
        if len(p) != 31:
            raise ValueError(f"{name}: {len(p)} fields, expected 31")

        def flag(s: str) -> bool:
            return {"true": True, "false": False}[s]

        rows.append(
            {
                "name": name,
                "sm": SM[p[10]],
                "dtq": DTYPES[p[0]],
                "dtkv": DTYPES[p[1]],
                "dto": DTYPES[p[2]],
                "tileQ": int(p[3]),
                "tileKv": int(p[4]),
                "stepQ": int(p[5]),
                "stepKv": int(p[6]),
                "hdPerCtaV": int(p[7]),
                "hdQk": int(p[8]),
                "hdV": int(p[9]),
                "smem": int(p[14]),
                "threads": int(p[15]),
                "layout": int(p[16]),
                "tpp": int(p[17]),
                "mask": int(p[18]),
                "ktype": int(p[19]),
                "sched": int(p[20]),
                "mcta": int(p[21]),
                "groupsHeadsQ": flag(p[22]),
                "groupsTokensHeadsQ": flag(p[23]),
                "reuseK": flag(p[24]),
                "twoCta": flag(p[25]),
                "sparse": flag(p[26]),
                "skips": flag(p[27]),
                "sha256": p[30].strip('"'),
            }
        )
    return rows, vx


def hash_id(m: dict[str, Any]) -> int:
    """TllmGenFmhaKernel::hashID(kernelMeta); ``log2(0)`` (non-paged) casts
    to 2^63, which the shift by 44 drops."""

    def log2(x: int) -> int:
        return int(math.log2(x)) if x > 0 else 1 << 63

    value = (
        m["layout"]
        | m["mask"] << 4
        | m["ktype"] << 8
        | m["sched"] << 12
        | m["mcta"] << 16
        | (m["hdPerCtaV"] >> 3) << 18
        | (m["hdQk"] >> 3) << 26
        | (m["hdV"] >> 3) << 34
        | (m["tileKv"] >> 6) << 42
        | log2(m["tpp"]) << 44
        | log2(m["tileQ"]) << 49
        | int(m["reuseK"]) << 53
        | int(m["twoCta"]) << 54
        | int(m["sparse"]) << 55
        | int(m["skips"]) << 56
    )
    return value & 0xFFFFFFFFFFFFFFFF


# -- cases ------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def is_mla(m: dict[str, Any]) -> bool:
    return m["hdQk"] in (576, 320) and m["hdQk"] > m["hdV"]


def finish_case(t: Any, m: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Fill the paging fields of a paged case from its lengths (pages per
    request, a shared prefix between requests 0 and 1, three spare pages)."""
    if m["layout"] != t.PAGED_KV:
        return case
    page = case.setdefault("page_size", m["tpp"] if not m["sparse"] else 64)
    counts = [_ceil_div(n, page) for n in case["kv_lens"]]
    shared = 0
    if case.get("prefix_pages") and len(counts) > 1:
        shared = min(case["prefix_pages"], counts[0], counts[1])
        if shared == 0:
            del case["prefix_pages"]
    case["num_pages"] = sum(counts) - shared + 3
    case["max_pages"] = case["topk"] if case.get("topk") else max(counts)
    case["shared_kv"] = is_mla(m)
    return case


def skeleton(
    t: Any,
    m: dict[str, Any],
    *,
    hq: int,
    hkv: int,
    batch: int,
    max_q: int,
    max_kv: int,
    runner: dict[str, int],
    window_left: int = -1,
    topk: int = 0,
    cum_q: bool = False,
) -> dict[str, Any]:
    """A selection-level case: every request at the maxima (selection reads
    only the maxima, batch, heads and runner fields)."""
    case: dict[str, Any] = {
        "hq": hq,
        "hkv": hkv,
        "q_lens": [max_q] * batch,
        "kv_lens": [max_kv] * batch,
        "window_left": window_left,
        "skip_thr": SKIP_SCALE if m["skips"] else 0.0,
        "bmm1_scale": 1.0 / math.sqrt(m["hdQk"]),
        "bmm2_scale": 1.0,
        "runner": dict(runner),
    }
    if topk:
        case["topk"] = topk
    if cum_q:
        case["cum_q"] = True
    return finish_case(t, m, case)


def _runner_masks(t: Any, m: dict[str, Any]) -> list[tuple[int, bool]]:
    """(runner mask, needs a sliding window) for the row's mask."""
    if m["mask"] == t.DENSE:
        return [(t.DENSE, False)]
    if m["mask"] == t.CAUSAL:
        return [(t.CAUSAL, False)]
    if m["mask"] == t.SLIDING:
        return [(t.CAUSAL, True)]
    return []


def legacy_generation_grid(t: Any, m: dict[str, Any]) -> list[dict[str, Any]]:
    """The former candidate grid (keeps every previously live kernel
    selected)."""
    if is_mla(m):
        heads = [(h, 1) for h in (8, 16, 32, 64, 128)]
    else:
        heads = [(hpk * hkv, hkv) for hpk in (1, 2, 4, 8, 16, 32, 64) for hkv in (1, 4)]
        heads = [h for h in heads if h[0] <= 128]
    kvs = (512, 4096) if m["sparse"] else (96, 1024, 4096)
    modes = [(t.STATIC, 1), (t.PERSISTENT, 0), (t.STATIC, 0)]
    out = []
    for (mask, sliding), (hq, hkv), max_q, batch in itertools.product(
        _runner_masks(t, m), heads, (1, 2, 4, 8), (1, 4, 32)
    ):
        extra = (8192, 16384) if batch == 1 and not m["sparse"] else ()
        windows = (63, 2047, 8191) if sliding else (-1,)
        for kv, window, (sched, mcta) in itertools.product(kvs + extra, windows, modes):
            if window >= kv - 1:
                continue
            out.append(
                skeleton(
                    t,
                    m,
                    hq=hq,
                    hkv=hkv,
                    batch=batch,
                    max_q=max_q,
                    max_kv=kv,
                    runner={
                        "mask": mask,
                        "ktype": t.GENERATION,
                        "sched": sched,
                        "mcta": mcta,
                    },
                    window_left=window,
                    topk=256 if m["sparse"] else 0,
                    cum_q=max_q > 1 and not is_mla(m),
                )
            )
    return out


HPK = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 128)
GEN_HKV = (1, 2, 4, 8)
GEN_Q = (1, 2, 3, 4, 5, 8, 16)
MLA_Q = (1, 2, 3, 4)
GEN_BATCH = (1, 2, 3, 4, 6, 8, 16, 32, 64, 128, 256)
GEN_KV = (40, 110, 300, 777, 1500, 3000, 5000, 9000, 17000, 33000, 66000, 131072)
WINDOWS = (127, 300, 1000, 4000)
TOPKS = (128, 256, 2048)


def generation_pool(t: Any, m: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    """A seeded sample of the generation selection space of ``m``'s group."""
    rng = random.Random(seed)
    mla = is_mla(m)
    if mla:
        heads = [(h, 1) for h in (8, 16, 32, 64, 128)]
    else:
        heads = [(p * k, k) for p in HPK for k in GEN_HKV if p * k <= 256]
    qs = MLA_Q if mla else GEN_Q
    modes = [(t.STATIC, 1), (t.PERSISTENT, 0), (t.STATIC, 0)]
    masks = _runner_masks(t, m)
    out = []
    for _ in range(POOL_SAMPLES):
        mask, sliding = rng.choice(masks)
        hq, hkv = rng.choice(heads)
        max_q = rng.choice(qs if not m["sparse"] else (1, 2))
        batch = rng.choice(GEN_BATCH)
        kv = rng.choice(GEN_KV)
        topk = rng.choice(TOPKS) if m["sparse"] else 0
        if topk:
            kv = max(kv, topk)
        window = -1
        if sliding:
            window = rng.choice(WINDOWS)
            if kv <= window + 1:
                kv = rng.choice([k for k in GEN_KV if k > window + 1])
        sched, mcta = rng.choice(modes)
        cum_q = max_q > 1 and not mla and rng.random() < 0.5
        out.append(
            skeleton(
                t,
                m,
                hq=hq,
                hkv=hkv,
                batch=batch,
                max_q=max_q,
                max_kv=kv,
                runner={
                    "mask": mask,
                    "ktype": t.GENERATION,
                    "sched": sched,
                    "mcta": mcta,
                },
                window_left=window,
                topk=topk,
                cum_q=cum_q,
            )
        )
    return out


def flashinfer_runner(t: Any, m: dict[str, Any], case: dict[str, Any]) -> bool:
    """Whether FlashInfer v0.6.9's launchers produce these runner fields."""
    r = case["runner"]
    layout = t.case_layout(m, case)
    if layout == t.PAGED_KV:
        if r["ktype"] == t.CONTEXT:
            return (r["mask"], r["sched"], r["mcta"]) == (t.CAUSAL, t.PERSISTENT, 0)
        return (r["mask"], r["sched"], r["mcta"]) == (t.CAUSAL, t.STATIC, 1)
    if layout == t.SEPARATE_QKV:
        return r["ktype"] == t.CONTEXT and (r["sched"], r["mcta"]) == (t.PERSISTENT, 0)
    return False


def group_key(t: Any, m: dict[str, Any]) -> tuple:
    """The row traits candidate generation depends on (rows sharing them
    share candidates; selection then picks among them)."""
    context = m["ktype"] == t.CONTEXT
    return (
        context,
        m["dtq"],
        m["dtkv"],
        m["dto"],
        m["layout"],
        m["hdQk"],
        m["hdV"],
        m["tpp"],
        m["mask"],
        m["skips"],
        m["sparse"],
        m["sched"] if context else None,
        m["mcta"] != 0 if context else None,
    )


def visible_keys(case: dict[str, Any], b: int) -> int:
    kv, q = case["kv_lens"][b], case["q_lens"][b]
    vis = kv
    if case.get("topk"):
        vis = min(vis, case["topk"])
    if case["window_left"] >= 0:
        vis = min(vis, case["window_left"] + q)
    return vis


def case_cost(t: Any, m: dict[str, Any], case: dict[str, Any]) -> tuple[float, int]:
    """(reference FLOPs, input bytes) of ``case``."""
    hq, hkv = case["hq"], case["hkv"]
    dqk, dv = m["hdQk"], m["hdV"]
    flops = sum(
        2.0 * q * hq * visible_keys(case, b) * (dqk + dv)
        for b, q in enumerate(case["q_lens"])
    )
    bits_kv = t.BITS[m["dtkv"]]
    sum_q, n_kv = sum(case["q_lens"]), sum(case["kv_lens"])
    nbytes = sum_q * hq * (dqk + dv) * 4  # q (worst case float) + o
    if t.case_layout(m, case) == t.PAGED_KV:
        heads = 1 if case["shared_kv"] else 2 * hkv
        d = max(dqk, dv)
        nbytes += case["num_pages"] * case["page_size"] * heads * d * bits_kv // 8
    else:
        nbytes += n_kv * hkv * (dqk + dv) * bits_kv // 8
    return flops, nbytes


def _log_lengths(
    rng: random.Random, n: int, lo: int, hi: int, minimal: bool
) -> list[int]:
    """``n`` log-uniform lengths in [lo, hi], one equal to ``hi`` and (n >= 3,
    ``minimal``) one equal to ``lo``: a few long requests among short ones."""
    lo = max(1, min(lo, hi))
    if n == 1:
        return [hi]
    vals = [
        int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))))) for _ in range(n)
    ]
    top = rng.randrange(n)
    vals[top] = hi
    if minimal and n >= 3:
        vals[(top + 1 + rng.randrange(n - 1)) % n] = lo
    return [min(hi, max(lo, v)) for v in vals]


def materialize(
    t: Any, m: dict[str, Any], skel: dict[str, Any], seed: int, serving: bool = False
) -> dict[str, Any] | None:
    """A generation skeleton with ragged lengths (same maxima, so the same
    kernel is selected): skewed KV lengths with a minimal request, variable
    query lengths when the case passes cumulative query lengths. ``serving``
    (throughput) keeps the KV lengths within [max / 4, max], no minimal
    request. None: no such lengths within the kernel's case restrictions."""
    rng = random.Random(seed)
    case = json.loads(json.dumps(skel))
    batch = len(case["q_lens"])
    max_q, max_kv = case["q_lens"][0], case["kv_lens"][0]
    if case.get("cum_q") and batch > 1:
        q_lens = [rng.randint(1, max_q) for _ in range(batch)]
        q_lens[rng.randrange(batch)] = max_q
    else:
        q_lens = [max_q] * batch
    lo = max(max_q, max_kv // 4) if serving else max_q
    if case.get("topk"):
        lo = max(lo, min(case["topk"], max_kv))
    kv = _log_lengths(rng, batch, lo, max_kv, minimal=not serving)
    kv = [max(k, q) for k, q in zip(kv, q_lens)]
    if sliding_tiles_restricted(t, m):
        fitted, kv = fit_last_tiles(m, q_lens, kv, max_kv, case["window_left"])
        # Without cumulative query lengths every request has max_q tokens.
        ragged_ok = bool(case.get("cum_q"))
        if max(kv) != max_kv or (fitted != q_lens and not ragged_ok):
            return None
        if max(fitted) != max_q:
            return None
        q_lens = fitted
    case["q_lens"] = q_lens
    case["kv_lens"] = kv
    for key in ("num_pages", "max_pages"):
        case.pop(key, None)
    return finish_case(t, m, case)


def context_designs(t: Any, m: dict[str, Any]) -> list[dict[str, Any]]:
    """Designed smoke shapes of a context kernel (selection does not depend
    on them, except that a sliding-window kernel needs window < max KV)."""
    mla = m["hdQk"] == 192
    layout = m["layout"]
    runner = {"ktype": t.CONTEXT, "sched": m["sched"], "mcta": int(m["mcta"] != 0)}
    masks = _runner_masks(t, m)
    designs = [
        # ragged batch: a long, a single-token and partial-tile requests
        ((8, 2), [300, 1, 129, 37, 64], [77, 0, 384, 5, 191], 127),
        # many query tiles over > 148 CTAs, chunked prefill with a long prefix
        ((32, 4), [517, 256, 130], [0, 1000, 37], 300),
        # MHA, short queries against long cached prefixes
        ((6, 6), [64, 200, 1, 17], [960, 0, 2047, 128], 127),
    ]
    out = []
    mask, sliding = masks[0]
    for (hq, hkv), q_lens, prefix, window in designs:
        if mla:
            hkv = hq
        if layout == t.PACKED_QKV:
            prefix = [0] * len(q_lens)
        kv_lens = [q + p for q, p in zip(q_lens, prefix)]
        case = {
            "hq": hq,
            "hkv": hkv,
            "q_lens": q_lens,
            "kv_lens": kv_lens,
            "window_left": window if sliding else -1,
            "skip_thr": SKIP_SCALE if m["skips"] else 0.0,
            "bmm1_scale": 1.0 / math.sqrt(m["hdQk"]),
            "bmm2_scale": 1.0,
            "runner": {**runner, "mask": mask},
        }
        out.append(finish_case(t, m, case))
    return out


def packed_design(t: Any, m: dict[str, Any]) -> dict[str, Any]:
    """A packed-QKV smoke case for a SeparateQkv kernel that serves its
    removed PackedQkv twin (self-attention: KV lengths = query lengths)."""
    mask, sliding = _runner_masks(t, m)[0]
    q_lens = [300, 1, 129, 37, 64]
    return {
        "hq": 8,
        "hkv": 2,
        "q_lens": q_lens,
        "kv_lens": list(q_lens),
        "window_left": 127 if sliding else -1,
        "skip_thr": SKIP_SCALE if m["skips"] else 0.0,
        "bmm1_scale": 1.0 / math.sqrt(m["hdQk"]),
        "bmm2_scale": 1.0,
        "runner": {
            "ktype": t.CONTEXT,
            "sched": m["sched"],
            "mcta": int(m["mcta"] != 0),
            "mask": mask,
        },
        "packed": True,
    }


# -- goals and features --------------------------------------------------------

GOALS = ("batch3", "hkv2", "group", "tiles3", "split", "waves", "spec", "ragged_q")


def goals(t: Any, m: dict[str, Any], case: dict[str, Any]) -> set[str]:
    """Hard paths a (materialized or skeleton) case exercises."""
    r = t.runner_params(m, case)
    launch = t.launch_config(m, r)
    ctas = math.prod(launch["grid"])
    out = set()
    if r["batch"] >= 3:
        out.add("batch3")
    if case["hkv"] >= 2:
        out.add("hkv2")
    if case["hq"] > case["hkv"]:
        out.add("group")
    vis = max(visible_keys(case, b) for b in range(r["batch"]))
    if vis >= 3 * m["tileKv"]:
        out.add("tiles3")
    if launch["max_ctas_kv"] >= 2:
        out.add("split")
    if ctas > t.NUM_SMS:
        out.add("waves")
    if r["max_q"] >= 2:
        out.add("spec")
    if case.get("cum_q") or m["ktype"] == t.CONTEXT:
        out.add("ragged_q")
    return out


def shape_traits(case: dict[str, Any]) -> set[tuple[str, Any]]:
    """Coarse shape attributes; later smoke picks prefer unseen ones."""
    batch = len(case["q_lens"])
    return {
        ("batch", min(batch, 9)),
        ("hkv", case["hkv"]),
        ("group", case["hq"] // case["hkv"]),
        ("max_q", max(case["q_lens"])),
        ("kv", max(case["kv_lens"]).bit_length()),
        ("runner", tuple(sorted(case["runner"].items()))),
        ("window", case["window_left"]),
    }


def use_skip_data(m: dict[str, Any], case: dict[str, Any]) -> None:
    """Skip-softmax tile pattern with threshold ``SKIP_SCALE`` (a window of at
    least one tile keeps a beacon in every partly visible high tile; the
    window is never changed, since it takes part in kernel selection). Rows
    whose window holds no high tile see only low keys (nothing to skip):
    there sinks would dominate the weight, so windows shorter than two tiles
    drop the sinks."""
    window = case["window_left"]
    if 0 <= window < m["tileKv"] - 1:
        case["skip_thr"] = 1e-30
        case.pop("skip_data", None)
        return
    if 0 <= window < 2 * m["tileKv"] - 1:
        case.pop("sinks", None)
    case["skip_thr"] = SKIP_SCALE
    case["skip_data"] = True


def fills_gpu(t: Any, m: dict[str, Any], case: dict[str, Any]) -> bool:
    r = t.runner_params(m, case)
    return math.prod(t.launch_config(m, r)["grid"]) >= t.NUM_SMS


def apply_profile(
    t: Any, m: dict[str, Any], case: dict[str, Any], index: int, seed: int
) -> dict[str, Any]:
    """Cycle the selection-neutral runtime features over a kernel's smoke
    cases: profile 0 switches (nearly) everything on, 1 everything off,
    2 a mix; skip-softmax kernels get the tile pattern (0, 2) and
    FlashInfer's tiny threshold (1)."""
    case = dict(case)
    layout = t.case_layout(m, case)
    paged = layout == t.PAGED_KV
    mla = is_mla(m)
    fp4_kv = m["dtkv"] == "e2m1"
    profile = index % 3
    rng = random.Random(seed)
    if profile == 0:
        case["sinks"] = not mla
        case["device_scales"] = True
        case["bmm2_scale"] = 0.8
        if paged and not mla and not fp4_kv:
            if not nhd_restricted(m) or case["hkv"] == 1:
                case["kv_layout"] = "NHD"
        if paged and not m["sparse"]:
            case["shared_idx"] = False
        if paged and not mla:
            case["q_noncontig"] = True
    elif profile == 1:
        case["bmm2_scale"] = 1.25
        if paged and len(case["kv_lens"]) > 1:
            case["prefix_pages"] = 2
    else:
        case["sinks"] = not mla
        if paged and not mla:
            case["kv_tuple"] = True
    if m["skips"]:
        if profile == 1:
            case["skip_thr"] = 1e-30
        else:
            use_skip_data(m, case)
    if m["dto"] == "e2m1" and profile != 1:
        sum_q = sum(case["q_lens"])
        case["sf_start"] = 1 + rng.randrange(100)
        case["sf_rows"] = _ceil_div(sum_q + case["sf_start"] + 1, 128) * 128
    for key in [k for k, v in case.items() if v is False]:
        if key != "shared_idx":
            del case[key]
    return finish_case(t, m, case)


# -- model and stress shapes ------------------------------------------------------


def model_heads(models: Any, hd: int) -> list[tuple[str, int, int, int]]:
    """(model, tp, Hq, Hkv) of the ``harness.models`` configurations with head
    dim ``hd``, tensor-parallel shards 1-8 (KV heads replicated below 1)."""
    out = []
    for name, spec in models.MODELS.items():
        if spec.get("kv_lora_rank") or spec.head_dim != hd:
            continue
        for tp in (1, 2, 4, 8):
            if spec.heads % tp:
                continue
            hq, hkv = spec.heads // tp, max(1, spec.kv_heads // tp)
            if hq % hkv == 0:
                out.append((name, tp, hq, hkv))
    return out


def model_candidates(
    t: Any, m: dict[str, Any], models: Any
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(case, source) model-shaped throughput candidates for ``m``'s group."""
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    rng = random.Random(hash_id(m) & 0xFFFF)
    masks = _runner_masks(t, m)
    mask, sliding = masks[0] if masks else (t.CAUSAL, False)
    mla = is_mla(m)

    def source(model: str, layer: str) -> dict[str, Any]:
        return {
            "kind": "model_shape",
            "model": model,
            "hf": models.MODELS[model].hf,
            "layer": layer,
        }

    def windows(model: str) -> list[int]:
        w = models.MODELS[model].get("sliding_window")
        return [w - 1] if w else []

    if m["ktype"] == t.CONTEXT:
        if m["layout"] == t.SEPARATE_QKV and m["hdQk"] == 192:
            configs = [
                (
                    name,
                    tp,
                    models.MODELS[name].heads // tp,
                    models.MODELS[name].heads // tp,
                )
                for name in ("deepseek_v3", "kimi_k2")
                for tp in (1, 8)
            ]
        else:
            configs = model_heads(models, m["hdQk"])
        shapes = [
            ("prefill 8k", [8192], [0]),
            ("prefill 16k", None, None),
            ("chunked prefill 8k+8k", [8192], [8192]),
        ]
        for (name, tp, hq, hkv), (label, q_lens, prefix) in itertools.product(
            configs, shapes
        ):
            if q_lens is None:
                q_lens = _log_lengths(rng, 6, 256, 8192, minimal=False)
                q_lens[-1] += 16384 - sum(q_lens) if sum(q_lens) < 16384 else 0
                prefix = [0] * len(q_lens)
            if m["layout"] == t.PACKED_QKV:
                prefix = [0] * len(q_lens)
            if sliding:
                ws = windows(name)
                if not ws:
                    continue
                window = ws[0]
            else:
                window = -1
            kv_lens = [q + p for q, p in zip(q_lens, prefix)]
            case = {
                "hq": hq,
                "hkv": hkv,
                "q_lens": list(q_lens),
                "kv_lens": kv_lens,
                "window_left": window,
                "skip_thr": SKIP_SCALE if m["skips"] else 0.0,
                "bmm1_scale": 1.0 / math.sqrt(m["hdQk"]),
                "bmm2_scale": 1.0,
                "runner": {
                    "ktype": t.CONTEXT,
                    "sched": m["sched"],
                    "mcta": int(m["mcta"] != 0),
                    "mask": mask,
                },
            }
            out.append(
                (finish_case(t, m, case), source(name, f"attention tp={tp} {label}"))
            )
        return out
    # Generation.
    if mla:
        names = ["deepseek_v3", "kimi_k2"]
        if m["sparse"]:
            names = ["deepseek_v3"]  # V3.2 sparse attention (index_topk)
        configs = [
            (n, tp, models.MODELS[n].heads // tp, 1)
            for n in names
            for tp in (1, 2, 4, 8)
        ]
        if m["hdQk"] == 320:
            configs = []
    else:
        configs = model_heads(models, m["hdQk"])
    shapes = [
        ("decode b64 kv8k", 64, 8192, 1),
        ("decode b128 kv8k", 128, 8192, 1),
        ("decode b256 kv8k", 256, 8192, 1),
        ("decode b64 kv16k", 64, 16384, 1),
        ("decode b128 kv16k", 128, 16384, 1),
        ("decode b64 kv32k", 64, 32768, 1),
        ("mtp decode b64 kv8k q2", 64, 8192, 2),
        ("spec decode b64 kv8k q4", 64, 8192, 4),
        ("long-context b1 kv128k", 1, 131072, 1),
        ("long-context b2 kv64k", 2, 65536, 1),
        ("long-context b4 kv64k", 4, 65536, 1),
        ("long-context b8 kv32k", 8, 32768, 1),
        ("long-context b16 kv32k", 16, 32768, 1),
        ("long-context b32 kv32k", 32, 32768, 1),
        ("long-context b16 kv64k", 16, 65536, 1),
    ]
    modes = [((t.STATIC, 1), ""), ((t.STATIC, 0), " (generic runner, one CTA per KV)")]
    for (
        (name, tp, hq, hkv),
        (label, batch, kv, q),
        ((sched, mcta), note),
    ) in itertools.product(configs, shapes, modes):
        window = -1
        if sliding:
            ws = windows(name)
            if not ws:
                continue
            window = ws[0]
        topk = models.MODELS[name].get("index_topk", 0) if m["sparse"] else 0
        if m["sparse"] and not topk:
            continue
        skel = skeleton(
            t,
            m,
            hq=hq,
            hkv=hkv,
            batch=batch,
            max_q=q,
            max_kv=kv,
            runner={"mask": mask, "ktype": t.GENERATION, "sched": sched, "mcta": mcta},
            window_left=window,
            topk=topk,
            cum_q=False,
        )
        out.append((skel, source(name, f"attention tp={tp} {label}{note}")))
    return out


def dsa_candidates(t: Any, m: dict[str, Any]) -> list[tuple[dict, dict]]:
    """Sparse MLA: the official DSA sparse-attention inventory rows (16 heads,
    top-k 2048, 64-token pages; lengths and indices are regenerated)."""
    from harness.throughput import inventory

    if not m["sparse"]:
        return []
    rows, manifest = inventory("dsa_attention")
    out, seen = [], set()
    for row in rows:
        axes = row["axes"]
        key = tuple(sorted(axes.items()))
        if key in seen:
            continue
        seen.add(key)
        batch = axes["num_tokens"]
        skel = skeleton(
            t,
            m,
            hq=16,
            hkv=1,
            batch=batch,
            max_q=1,
            max_kv=min(131072, axes["num_pages"] * 64 // batch),
            runner={
                "mask": t.CAUSAL,
                "ktype": t.GENERATION,
                "sched": t.STATIC,
                "mcta": 1,
            },
            topk=2048,
        )
        skel["page_size"] = 64
        # The softmax scale is replayed from the recorded row.
        scale = row["inputs"]["sm_scale"]["value"]
        skel["sm_scale"] = skel["bmm1_scale"] = scale
        source = {
            **manifest,
            "kind": "official_shape",
            "inventory": "dsa_attention",
            "uuid": row["uuid"],
            "axes": axes,
            "input_policy": "Seeded synthetic values, lengths and page mappings; "
            "not tensor-blob replay.",
        }
        out.append((finish_case(t, m, skel), source))
    return out


# -- exclusions --------------------------------------------------------------------

TWIN = "kSM_100f twin replaced by the kSM_100 kernel with the same hash"
REUSE_K = "mReuseSmemKForV is never set by kernel selection"
CONTEXT_TRAITS = "context selection never produces these tile/split traits"
MLA_TILE32 = "non-sparse MLA SwapsMmaAb selection caps tileSizeQ at 16"
# Exclusions that claim upstream selection can never pick the kernel.
UNREACHABLE = (TWIN, REUSE_K, CONTEXT_TRAITS, MLA_TILE32)
# Kernels upstream selects but that do not compute the attention correctly with
# KernelParams identical to upstream's (measured on a B200). FlashInfer's tests
# select none but the NVFP4-KV head-dim-256 context kernels (FP8 output and
# NVFP4 KV only at head dims 128 and 256; NVFP4 output only for KV lengths of
# one CTA per sequence); its NVFP4-KV prefill tolerances (rtol = atol = 0.5,
# 10% of elements unchecked) admit their error.
DEFECT_NVFP4_MCTA = (
    "defect: NVFP4 output with multi-CTA KV reduction (GmemReduction and "
    "CgaSmemReduction): wrong, up to saturated (448), output scale factors in 142 "
    "of these 291 kernels' cases on B200, for some only after an unrelated kernel "
    "ran (identical inputs and buffers); compute-sanitizer racecheck reports a "
    "shared-memory hazard"
)
DEFECT_FP8_H64 = (
    "defect: FP8 output, head dim 64, SwapsMmaAb tileQ >= 16 without multi-CTA KV: "
    "heads 8-15 of each tile are stored pairwise swapped (8<->9, 10<->11, ...)"
)
DEFECT_FP4KV_H256 = (
    "defect: NVFP4 KV, head dim 256 context kernel: reads the V scale factors of "
    "16-element blocks >= 8 from overlapping offsets (byte (t//4)*64 + (s-6)*4 + "
    "t%4 instead of s*4 for token t, block s), so no cache layout dequantizes V"
)
DEFECT_FP4KV_H64_P16 = (
    "defect: NVFP4 KV, head dim 64, 16-token pages: the kernel faults with a "
    "misaligned shared-memory address (compute-sanitizer memcheck)"
)
DUPLICATE = "duplicate: code identical to {survivor}, which serves packed QKV"


def defect(t: Any, m: dict[str, Any]) -> str | None:
    """Selectable row that does not compute attention correctly (None: fine)."""
    if m["dto"] == "e2m1" and m["mcta"] in (t.MCTA_GMEM, t.MCTA_CGA):
        return DEFECT_NVFP4_MCTA
    if (
        m["dto"] == "e4m3"
        and m["hdV"] == 64
        and m["ktype"] == t.SWAPS_AB
        and m["tileQ"] >= 16
        and m["mcta"] == t.MCTA_DISABLED
    ):
        return DEFECT_FP8_H64
    if m["dtkv"] == "e2m1" and m["hdV"] == 256 and m["ktype"] == t.CONTEXT:
        return DEFECT_FP4KV_H256
    if m["dtkv"] == "e2m1" and m["hdV"] == 64 and m["tpp"] == 16:
        return DEFECT_FP4KV_H64_P16
    return None


# Shape-dependent defects of kernels that compute every other shape correctly
# (measured on a B200 with KernelParams identical to upstream's, and
# reproduced with FlashInfer v0.6.9's own trtllm_batch_decode_with_kv_cache):
# their cases are restricted to the shapes they serve correctly instead of
# excluding the kernels. FlashInfer's tests use none of these shapes.
RESTRICT_MIXED_HEADS = (
    "restricted (defect): generation kernels with Q dtype != KV dtype compute only "
    "the first tile of query heads (grid y = 0) correctly when numHeadsQPerKv > "
    "stepQ; cases keep numHeadsQPerKv <= stepQ"
)
RESTRICT_SLIDING_TILES = (
    "restricted (defect): SlidingOrChunkedCausal generation kernels with one query "
    "token per CTA (groupsTokensHeadsQ false) take the KV tile range of query token "
    "i of q from position kv - 1 - 2 (q-1-i) instead of kv - 1 - (q-1-i): where that "
    "falls in an earlier tile, the token loses its last KV tile (or, none left, is "
    "not written) or, at the sliding-window start, also processes the preceding tile "
    "without the window mask; cases keep both positions in the same tile, at the end "
    "and at the window start, for every query token of every request"
)
RESTRICT_NHD_H64 = (
    "restricted (defect): FP16/BF16 Q with E4M3 KV, head dim 64: NHD caches with "
    "more than one KV head give wrong output; NHD cases keep one KV head"
)


def sliding_tiles_restricted(t: Any, m: dict[str, Any]) -> bool:
    return (
        m["ktype"] != t.CONTEXT
        and m["mask"] == t.SLIDING
        and not m["groupsTokensHeadsQ"]
    )


def tile_offset_defect(m: dict[str, Any], q: int, kv: int, window: int) -> bool:
    """A query token of a (q, kv) request whose KV tile range
    RESTRICT_SLIDING_TILES changes: its last position, or its window start
    (window >= 0 and in use), lies in another tile than the kernel's."""
    tile = m["tileKv"]
    for d in range(1, q):
        if (kv - 1 - d) // tile != (kv - 1 - 2 * d) // tile or kv - 1 - 2 * d < 0:
            return True
        start = kv - 1 - d - window
        if window >= 0 and start > 0 and max(0, start - d) // tile != start // tile:
            return True
    return False


def nhd_restricted(m: dict[str, Any]) -> bool:
    return m["dtq"] in ("fp16", "bf16") and m["dtkv"] == "e4m3" and m["hdQk"] == 64


def case_restriction(t: Any, m: dict[str, Any], case: dict[str, Any]) -> str | None:
    """The restriction ``case`` violates (None: the kernel serves it)."""
    if (
        m["ktype"] != t.CONTEXT
        and m["dtq"] != m["dtkv"]
        and case["hq"] // case["hkv"] > m["stepQ"]
    ):
        return RESTRICT_MIXED_HEADS
    if sliding_tiles_restricted(t, m) and any(
        tile_offset_defect(m, q, kv, case["window_left"])
        for q, kv in zip(case["q_lens"], case["kv_lens"])
    ):
        return RESTRICT_SLIDING_TILES
    if nhd_restricted(m) and case.get("kv_layout") == "NHD" and case["hkv"] > 1:
        return RESTRICT_NHD_H64
    return None


def fit_last_tiles(
    m: dict[str, Any],
    q_lens: list[int],
    kv_lens: list[int],
    max_kv: int,
    window: int,
) -> tuple[list[int], list[int]]:
    """Ragged lengths within RESTRICT_SLIDING_TILES: each request's KV length
    grows to the next admissible one up to ``max_kv``, else its query length
    shrinks (callers check that the maxima, which selection reads, hold)."""
    q_out, kv_out = [], []
    for q, kv in zip(q_lens, kv_lens):
        while tile_offset_defect(m, q, kv, window) and kv < max_kv:
            kv += 1
        while tile_offset_defect(m, q, kv, window):
            q -= 1
        q_out.append(q)
        kv_out.append(kv)
    return q_out, kv_out


def exclusion(t: Any, m: dict[str, Any], twins: set[str]) -> str | None:
    """Why row ``m`` can never be a workload (None: probe it)."""
    if m["name"] in twins:
        return TWIN
    if m["reuseK"]:
        return REUSE_K
    if m["mask"] == t.CUSTOM:
        return "custom (TRT-LLM packed tree) mask not modelled"
    if m["mcta"] == t.MCTA_GMEM_SEPARATE:
        return "GmemReductionWithSeparateKernel needs the separate reduction kernel"
    if m["layout"] == t.CONTIGUOUS_KV:
        return "ContiguousKv layout is not served by FlashInfer"
    if m["ktype"] == t.CONTEXT and (
        m["tileQ"] != 128
        or m["tileKv"] != 128
        or m["hdPerCtaV"] != m["hdV"]
        or m["twoCta"]
        or m["sparse"]
    ):
        return CONTEXT_TRAITS
    mla = m["hdQk"] == 576 and m["hdV"] == 512
    if mla and not m["sparse"] and m["ktype"] == t.SWAPS_AB and m["tileQ"] > 16:
        return MLA_TILE32
    return defect(t, m)


def code_hash(sections: dict[str, bytes]) -> str:
    """Hash of a stripped cubin's code: ``.text``, ``.nv.info``, shared and
    constant sections (kernel names excluded)."""
    texts = [v for k, v in sorted(sections.items()) if k.startswith(".text.")]
    info = [
        v
        for k, v in sorted(sections.items())
        if k.startswith((".nv.info.", ".nv.shared.", ".nv.constant"))
    ]
    return hashlib.sha256(b"".join(texts) + b"|" + b"|".join(info)).hexdigest()


# -- the build ---------------------------------------------------------------------


def probe_command(compiler: Any, output: Path) -> list[str]:
    flashinfer = compiler.resources / f"flashinfer-{FLASHINFER_V069}"
    artifact = compiler.resources / "trtllm-gen-artifacts" / ARTIFACT
    return [
        compiler.nvcc,
        "-std=c++17",
        compiler.optimization,
        "-x",
        "cu",
        "--expt-relaxed-constexpr",
        "-diag-suppress=20012",
        '-DTLLM_GEN_FMHA_CUBIN_PATH="probe"',
        '-DTLLM_GEN_FMHA_METAINFO_HASH="probe"',
        "-I" + str(artifact / "include"),
        "-I" + str(flashinfer / "include"),
        "-I" + str(flashinfer / "include/flashinfer/trtllm/fmha"),
        "-I" + str(flashinfer / "csrc"),
        "-I" + str(compiler.resources / CUTLASS / "include"),
        *compiler.flags,
        str(_compiler_module(compiler).PACKAGE / PROBE),
        "-o",
        str(output),
    ]


def run_probe(binary: Path, lines: list[str], jobs: int = 8) -> dict[int, dict]:
    """Probe records by case id (lines split over ``jobs`` processes)."""
    if not lines:
        return {}
    chunk = _ceil_div(len(lines), jobs)
    parts = [lines[i : i + chunk] for i in range(0, len(lines), chunk)]

    def one(part: list[str]) -> list[str]:
        result = subprocess.run(
            [str(binary)],
            input="\n".join(part) + "\n",
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()[1:]

    records: dict[int, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(len(parts)) as pool:
        for out in pool.map(one, parts):
            for line in out:
                rec = json.loads(line)
                records[int(rec["id"])] = rec
    return records


def probe_layout(binary: Path) -> dict[str, Any]:
    result = subprocess.run(
        [str(binary)], input="", capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout.splitlines()[0])["layout"]


def upstream_meta(t: Any, entry: dict[str, Any]) -> dict[str, Any]:
    """The meta fields ``runner_params`` reads, for an upstream entry."""
    dtq, dtkv, dto = entry["dtypes"]
    return {
        "dtq": dtq,
        "dtkv": dtkv,
        "dto": dto,
        "layout": entry["layout"],
        "hdQk": entry["hd"][0],
        "hdV": entry["hd"][1],
        "skips": entry["case"]["skip_thr"] != 0.0,
        "sparse": bool(entry["case"].get("topk")),
        "tpp": entry["case"].get("page_size", 0),
    }


def build(compiler: Any) -> dict[str, str]:
    helpers = _compiler_module(compiler)
    package: Path = helpers.PACKAGE
    root: Path = helpers.ROOT
    from harness import cubin_strip, models, workload

    t = helpers.load_workload_module(root, "trtllm")
    upstream = load_upstream(package)

    artifact = compiler.resources / "trtllm-gen-artifacts" / ARTIFACT
    meta_path = artifact / "include/flashInferMetaInfo.h"
    rows, vx = parse_meta_info(meta_path)
    checksums = {
        line.split()[1]: line.split()[0]
        for line in (artifact / "checksums.txt").read_text().splitlines()
        if line.strip()
    }
    source_dir = root / "cubins" / "fmha"
    sm100 = [r for r in rows if r["sm"] in ("100a", "100f")]
    present = {p.name for p in source_dir.glob("*.cubin")}
    expected = {r["name"] + ".cubin" for r in sm100} | {n + ".cubin" for n in vx}
    if present != expected:
        raise ValueError(
            f"cubins/fmha differs from the meta-info: {len(present - expected)} extra, "
            f"{len(expected - present)} missing"
        )
    for name in sorted(present):
        digest = helpers.sha256_file(source_dir / name)
        if checksums.get(name) != digest:
            raise ValueError(f"{name}: SHA256 differs from checksums.txt")
    by_name = {r["name"]: r for r in sm100}
    for r in sm100:
        if r["sha256"] != checksums[r["name"] + ".cubin"]:
            raise ValueError(
                f"{r['name']}: meta-info SHA256 differs from checksums.txt"
            )

    # loadKernels on kSM_100: per dtype triple, the kSM_100 kernel wins a hash.
    twins: set[str] = set()
    owners: dict[tuple, dict[str, Any]] = {}
    for r in sm100:
        key = (r["dtq"], r["dtkv"], r["dto"], hash_id(r))
        if key in owners:
            other = owners[key]
            if {other["sm"], r["sm"]} != {"100a", "100f"}:
                raise ValueError(f"hash conflict {other['name']} / {r['name']}")
            loser = r if r["sm"] == "100f" else other
            twins.add(loser["name"])
            owners[key] = other if loser is r else r
        else:
            owners[key] = r

    dead: dict[str, str] = {
        n: "Vx (Sage/int8-QK) table: no FlashInfer v0.6.9 code loads it" for n in vx
    }
    for r in sm100:
        reason = exclusion(t, r, twins)
        if reason:
            dead[r["name"]] = reason

    # Strip every remaining kernel; identical code (PackedQkv == SeparateQkv)
    # keeps the SeparateQkv kernel, which serves packed QKV.
    stripped: dict[str, bytes] = {}
    code: dict[str, list[str]] = defaultdict(list)
    for r in sm100:
        name = r["name"]
        if name in dead:
            continue
        source = (source_dir / f"{name}.cubin").read_bytes()
        found = cubin_strip.cubin_kernels(source)
        if sorted(found) != sorted([name, name + "GetSmemSize"]):
            raise ValueError(f"{name}: unexpected kernels {found}")
        image = cubin_strip.strip_cubin(source, name)
        cubin_strip.check_strip(source, image, name)
        stripped[name] = image
        code[code_hash(workload._sections(image))].append(name)
    _log(f"stripped {len(stripped)} kernels")
    dup_of: dict[str, str] = {}
    for names in code.values():
        if len(names) == 1:
            continue
        metas = [by_name[n] for n in names]
        separate = [m for m in metas if m["layout"] == t.SEPARATE_QKV]
        packed = [m for m in metas if m["layout"] == t.PACKED_QKV]
        ignore = ("name", "sha256", "layout", "smem")
        same = (
            all(
                {k: v for k, v in m.items() if k not in ignore}
                == {k: v for k, v in separate[0].items() if k not in ignore}
                for m in metas
            )
            if separate
            else False
        )
        if len(names) != 2 or len(separate) != 1 or len(packed) != 1 or not same:
            raise ValueError(f"unexpected identical-code kernels {names}")
        dup_of[packed[0]["name"]] = separate[0]["name"]
        dead[packed[0]["name"]] = DUPLICATE.format(survivor=separate[0]["name"])
    survivors = {v: k for k, v in dup_of.items()}

    live_rows = [r for r in sm100 if r["name"] not in dead]
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in live_rows:
        groups[group_key(t, r)].append(r)

    with tempfile.TemporaryDirectory(prefix="fmha-trtllm-") as tmp:
        binary = Path(tmp) / "trtllm_probe"
        subprocess.run(
            probe_command(compiler, binary), check=True, cwd=root, capture_output=True
        )
        layout = probe_layout(binary)
        result = _build_cases(
            t, models, upstream, binary, layout, by_name, dead, groups, survivors
        )
    index_cases, fixture_records, examples, upstream_map, stats = result

    for r in live_rows:
        if r["name"] not in index_cases:
            dead[r["name"]] = "not selected on sm_100 by any probed runner parameters"
    index_rows = [
        [r[c] for c in t.META_COLUMNS] for r in live_rows if r["name"] not in dead
    ]

    # Write the artifacts.
    out_dir = package / "cubins" / ARCH
    if out_dir.is_dir():
        for old in out_dir.glob("*.cubin"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    kernels: dict[str, Any] = {}
    names_seen: set[str] = set()
    for row in index_rows:
        kernel = row[0]
        wname = t.workload_name(kernel)
        if wname in names_seen:
            raise ValueError(f"duplicate workload name {wname}")
        names_seen.add(wname)
        image = stripped[kernel]
        arch, names, params = workload.cubin_info(image)
        if names != [kernel] or arch not in ("sm_100a", "sm_100f"):
            raise ValueError(f"{kernel}: stripped cubin is {arch} {names}")
        if [p.size for p in params] != [layout["size"]]:
            raise ValueError(f"{kernel}: parameters {params} != KernelParams")
        target = out_dir / f"{wname}.cubin"
        target.write_bytes(image)
        mapping[wname] = target.relative_to(package).as_posix()
        kernels[wname] = {
            "kernel": kernel,
            "source_sha256": checksums[kernel + ".cubin"],
            "sha256": helpers.sha256_bytes(image),
            "declared_arch": arch,
        }

    helpers.write_json(
        helpers.VARIANTS / f"{ARCH}.json",
        {"layout": layout, "kernels": index_rows, "cases": index_cases},
        indent=None,
    )
    helpers.write_json(
        package / "fixtures" / f"{ARCH}.json",
        {"records": fixture_records, "examples": examples, "upstream": upstream_map},
        indent=None,
    )
    reasons: dict[str, int] = defaultdict(int)
    for reason in dead.values():
        reasons[reason] += 1
    version = (
        subprocess.run(
            [compiler.nvcc, "--version"], capture_output=True, text=True, check=True
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    helpers.merge_provenance(
        ARCH,
        {
            "library": "trtllm-gen FMHA (flashinfer_cubin 0.6.8/0.6.9)",
            "source": {
                "cubins": "cubins/fmha",
                "artifact": ARTIFACT,
                "checksums": f"resources/trtllm-gen-artifacts/{ARTIFACT}/checksums.txt",
                "checksums_sha256": helpers.sha256_file(artifact / "checksums.txt"),
                "meta_info": f"resources/trtllm-gen-artifacts/{ARTIFACT}/include/"
                "flashInferMetaInfo.h",
                "meta_info_sha256": helpers.sha256_file(meta_path),
                "flashinfer_revision": FLASHINFER_V069,
                "host_code": [
                    "include/flashinfer/trtllm/fmha/fmhaKernels.cuh (TllmGenFmhaKernel::run)",
                    "include/flashinfer/trtllm/fmha/kernelParams.h (setKernelParams)",
                    "include/flashinfer/trtllm/fmha/fmhaRunnerParams.h",
                    "csrc/trtllm_fmha_kernel_launcher.cu",
                ],
                "cutlass": f"resources/{CUTLASS} (headers kernelParams.h includes)",
            },
            "strip": "harness.cubin_strip.strip_cubin (+ check_strip): drops <kernel>GetSmemSize",
            "probe": PROBE,
            "probe_sha256": helpers.sha256_file(package / PROBE),
            "probe_command": [
                helpers.rel(Path(p)) if p.startswith("/") else p
                for p in probe_command(compiler, Path("trtllm_probe"))
            ],
            "upstream_enumerator": "trtllm_upstream.py",
            "upstream_enumerator_sha256": helpers.sha256_file(
                package / "trtllm_upstream.py"
            ),
            "nvcc": version,
            "num_sms": t.NUM_SMS,
            **stats,
            "live": len(index_rows),
            "removed_duplicates": dict(sorted(dup_of.items())),
            "dead_counts": dict(sorted(reasons.items())),
            "dead": dict(sorted(dead.items())),
            "kernels": kernels,
        },
    )
    return mapping


def _build_cases(
    t: Any,
    models: Any,
    upstream: Any,
    binary: Path,
    layout: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    dead: dict[str, str],
    groups: dict[tuple, list[dict[str, Any]]],
    survivors: dict[str, str],
) -> tuple:
    """Selection sweep, case composition and the verifying probe pass."""
    # -- pass 1: selection -----------------------------------------------------
    lines: list[str] = []
    cases1: list[dict[str, Any]] = []
    kinds: list[tuple[str, Any]] = []  # (kind, payload)
    seen: dict[str, int] = {}

    def add(m: dict[str, Any], case: dict[str, Any], kind: str, payload: Any) -> None:
        line = t.probe_line("@", t.runner_params(m, case))
        if kind == "pool" and line in seen:
            return
        seen.setdefault(line, len(lines))
        lines.append(line.replace("id=@", f"id={len(lines)}", 1) + " emit=0")
        cases1.append(case)
        kinds.append((kind, payload))

    for gi, (key, members) in enumerate(
        sorted(groups.items(), key=lambda kv: str(kv[0]))
    ):
        m = members[0]
        if m["ktype"] == t.CONTEXT:
            continue  # designed cases, verified in pass 2
        for case in legacy_generation_grid(t, m):
            add(m, case, "pool", None)
        for case in generation_pool(t, m, seed=gi):
            add(m, case, "pool", None)
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        m = members[0]
        for case, source in model_candidates(t, m, models) + dsa_candidates(t, m):
            add(m, case, "model", source)
    entries = upstream.upstream_params()
    upstream_keys: dict[str, dict[str, Any]] = {}
    for entry in entries:
        um = upstream_meta(t, entry)
        case = finish_case(t, {**um, "layout": entry["layout"]}, dict(entry["case"]))
        entry["case"] = case
        skey = upstream.shape_key(case)
        full = json.dumps([entry["dtypes"], entry["hd"], entry["layout"], skey])
        digest = hashlib.sha256(full.encode()).hexdigest()[:16]
        entry["key"] = digest
        if digest not in upstream_keys:
            upstream_keys[digest] = entry
            add(um, case, "upstream", digest)
    _log(f"pass 1: probing {len(lines)} candidates")
    records = run_probe(binary, lines)
    _log("pass 1 done")

    selected: dict[str, list[int]] = defaultdict(list)
    model_sel: dict[str, list[int]] = defaultdict(list)
    upstream_map: dict[str, str] = {}
    for i, (kind, payload) in enumerate(kinds):
        rec = records.get(i)
        kernel = None
        if rec is not None and "error" not in rec and rec["kernel"] in by_name:
            kernel = rec["kernel"]
            if rec["reductions"]:
                kernel = None if kind != "upstream" else kernel
        if kind == "upstream":
            if kernel is None:
                why = rec.get("error", "no kernel") if rec else "no probe record"
                upstream_map[payload] = "error: " + why
            elif kernel in dead:
                upstream_map[payload] = "excluded: " + dead[kernel]
            else:
                upstream_map[payload] = kernel
            continue
        if kernel is None:
            continue
        if kernel in dead:
            if dead[kernel] in UNREACHABLE:
                raise ValueError(
                    f"kernel {kernel} is selected although: {dead[kernel]}"
                )
            continue
        (model_sel if kind == "model" else selected)[kernel].append(i)

    # -- compose the cases of every kernel ---------------------------------------
    final: list[tuple[str, dict[str, Any]]] = []  # (kernel, case)
    flags_needed: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    upstream_by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        kernel = upstream_map.get(entry["key"], "")
        if kernel in by_name:
            upstream_by_kernel[kernel].append(entry)
            for f, v in entry["flags"].items():
                if f in upstream.FLAG_KEYS and f != "sf_start":
                    flags_needed[kernel][f].add(v)

    # Candidates dropped by the case restrictions: reason -> kernel -> count.
    dropped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def admissible(m: dict[str, Any], case: dict[str, Any] | None) -> bool:
        # materialize returns None only for RESTRICT_SLIDING_TILES.
        reason = (
            RESTRICT_SLIDING_TILES if case is None else case_restriction(t, m, case)
        )
        if reason is not None:
            dropped[reason][m["name"]] += 1
        return reason is None

    goal_stats = {"achievable": defaultdict(int), "covered": defaultdict(int)}
    for members in groups.values():
        for m in members:
            name = m["name"]
            cases: list[dict[str, Any]] = []
            rng_seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            # Smoke.
            if m["ktype"] == t.CONTEXT:
                picks = context_designs(t, m)
                achievable = set().union(*(goals(t, m, c) for c in picks))
            else:
                sweep = [
                    materialize(t, m, cases1[i], rng_seed + i) for i in selected[name]
                ]
                pool = [c for c in sweep if admissible(m, c)]
                if not pool:
                    # Selected only by model or upstream shapes (or no sweep
                    # shape is admissible).
                    other = [
                        materialize(t, m, cases1[i], rng_seed + i)
                        for i in model_sel.get(name, [])
                    ] + [dict(e["case"]) for e in upstream_by_kernel.get(name, [])]
                    if not sweep and not other:
                        continue
                    pool = [c for c in other if admissible(m, c)]
                if not pool:
                    raise AssertionError(f"{name}: no selecting shape is admissible")
                cands = []
                for i, case in enumerate(pool):
                    flops, nbytes = case_cost(t, m, case)
                    cands.append((flops, nbytes, i, case, goals(t, m, case)))
                achievable = set().union(*(c[4] for c in cands))
                cheap = [
                    c for c in cands if c[0] <= SMOKE_FLOPS and c[1] <= SMOKE_BYTES
                ]
                if not cheap:
                    cheap = [min(cands, key=lambda c: (c[0], c[1]))]
                # Smoke shapes: small batches (the reference loops over
                # requests), new goals first, then shapes unlike the picks.
                small = [c for c in cheap if len(c[3]["q_lens"]) <= 32] or cheap
                picks, covered, traits = [], set(), set()
                for _ in range(SMOKE_PICKS):
                    best = max(
                        (c for c in small if c[3] not in picks),
                        key=lambda c: (
                            len(c[4] - covered),
                            len(shape_traits(c[3]) - traits),
                            -c[0],
                        ),
                        default=None,
                    )
                    if best is None:
                        break
                    picks.append(best[3])
                    covered |= best[4]
                    traits |= shape_traits(best[3])
            # Cheapest first: CPU reference checks take the first smoke case.
            picks.sort(key=lambda c: case_cost(t, m, c)[0])
            # Rare kernels with fewer distinct shapes still get every
            # feature profile (same shape, other features).
            for n in range(len(picks), SMOKE_PICKS if picks else 0):
                picks.append(json.loads(json.dumps(picks[n % len(picks)])))
            for g in achievable:
                goal_stats["achievable"][g] += 1
            covered = set().union(*(goals(t, m, c) for c in picks)) if picks else set()
            for g in covered:
                goal_stats["covered"][g] += 1
            for n, case in enumerate(picks):
                case = apply_profile(t, m, case, n, rng_seed + n)
                case.update(suite="smoke", label=f"smoke_{n}")
                cases.append(case)
            if name in survivors:
                case = apply_profile(t, m, packed_design(t, m), 1, rng_seed)
                case.update(suite="smoke", label="smoke_packed")
                cases.append(case)
            # Upstream: one case per distinct shape, flags covering the
            # kernel's upstream flag values.
            missing = {(f, v) for f, vs in flags_needed[name].items() for v in vs}
            by_shape: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for entry in upstream_by_kernel.get(name, []):
                by_shape[entry["key"]].append(entry)
            chosen: list[dict[str, Any]] = []
            for shape_entries in by_shape.values():
                best = max(
                    shape_entries,
                    key=lambda e: len(missing & set(e["flags"].items())),
                )
                missing -= set(best["flags"].items())
                chosen.append(best)
            for shape_entries in by_shape.values():
                for entry in shape_entries:
                    if missing & set(entry["flags"].items()):
                        missing -= set(entry["flags"].items())
                        chosen.append(entry)
            if missing:
                raise AssertionError(f"{name}: upstream flags {missing} uncovered")
            for n, entry in enumerate(chosen):
                case = json.loads(json.dumps(entry["case"]))
                flops, nbytes = case_cost(t, m, case)
                small = flops <= SMOKE_FLOPS and nbytes <= SMOKE_BYTES
                if m["dto"] == "e2m1" and n % 2:
                    case["sf_start"] = 5 + n
                    rows = sum(case["q_lens"]) + case["sf_start"] + 1
                    case["sf_rows"] = _ceil_div(rows, 128) * 128
                if m["dto"] == "e4m3":
                    case["bmm2_scale"] = 1.25  # FlashInfer: v_scale / o_scale
                case.update(
                    suite="smoke" if small else "throughput",
                    label=f"upstream_{n}",
                    source={
                        "kind": "upstream_test",
                        "test": entry["test"],
                        "revision": upstream.FLASHINFER,
                        "upstream_key": entry["key"],
                        "parametrizations": len(by_shape[entry["key"]]),
                    },
                )
                cases.append(case)
            # Throughput: model shapes (largest within budget, then another
            # regime), else the largest selecting stress shape.
            tp_cands = []
            for i in model_sel.get(name, []):
                kind, source = kinds[i]
                case = cases1[i]
                if m["ktype"] != t.CONTEXT:
                    case = materialize(t, m, case, rng_seed + i, serving=True)
                if not admissible(m, case):
                    continue
                if source.get("inventory") == "dsa_attention":
                    # The recorded page pool and token count (one query
                    # token per request).
                    axes = source["axes"]
                    if case["num_pages"] > axes["num_pages"]:
                        raise ValueError(f"{name}: DSA lengths exceed the pool")
                    case["num_pages"] = case["pages"] = axes["num_pages"]
                    case["tokens"] = axes["num_tokens"]
                if source.get("model") and models.MODELS[source["model"]].get(
                    "attention_sinks"
                ):
                    case["sinks"] = True
                flops, nbytes = case_cost(t, m, case)
                if flops <= THROUGHPUT_FLOPS and nbytes <= THROUGHPUT_BYTES:
                    tp_cands.append((nbytes + flops / 1e3, i, case, source))
            # Grids that fill the 148 SMs first (multi-CTA KV kernels fill
            # them by splitting the KV), then the largest working set.
            tp_cands.sort(key=lambda c: (-fills_gpu(t, m, c[2]), -c[0]))
            picks_tp: list[tuple] = []
            for cand in tp_cands:
                if picks_tp and m["mcta"] == 0 and not fills_gpu(t, m, cand[2]):
                    continue  # a second, latency-bound shape adds no stress
                regime = cand[3].get("layer", "").split(" ")[2:3]
                if any(
                    p[3].get("layer", "").split(" ")[2:3] == regime for p in picks_tp
                ):
                    continue
                picks_tp.append(cand)
                if len(picks_tp) == THROUGHPUT_PICKS:
                    break
            if not picks_tp:
                stress = []
                for i in selected.get(name, []):
                    case = materialize(t, m, cases1[i], rng_seed + 7 * i, serving=True)
                    if not admissible(m, case):
                        continue
                    flops, nbytes = case_cost(t, m, case)
                    if flops <= THROUGHPUT_FLOPS and nbytes <= THROUGHPUT_BYTES:
                        stress.append((nbytes + flops / 1e3, i, case, None))
                if m["ktype"] == t.CONTEXT:
                    big = context_designs(t, m)[1]
                    big = dict(big)
                    big["q_lens"] = [q * 8 for q in big["q_lens"]]
                    big["kv_lens"] = [k * 8 for k in big["kv_lens"]]
                    big.pop("num_pages", None)
                    big = finish_case(t, m, big)
                    stress.append((0, -1, big, None))
                if stress:
                    picks_tp.append(
                        max(stress, key=lambda c: (fills_gpu(t, m, c[2]), c[0]))
                    )
            for n, (_, _, case, source) in enumerate(picks_tp):
                case = dict(case)
                if m["skips"]:
                    use_skip_data(m, case)
                case.update(suite="throughput", label=f"trtllm_{n}")
                if source:
                    case["source"] = source
                cases.append(case)
            if name in survivors:
                case = context_designs(t, m)[1]
                q_lens = [q * 8 for q in case["q_lens"]]
                case = {
                    **packed_design(t, m),
                    "q_lens": q_lens,
                    "kv_lens": list(q_lens),
                }
                if m["skips"]:
                    use_skip_data(m, case)
                case.update(suite="throughput", label="trtllm_packed")
                cases.append(case)
            for case in cases:
                reason = case_restriction(t, m, case)
                if reason is not None:
                    raise AssertionError(f"{name} {case['label']}: {reason}")
                if (
                    m["ktype"] != t.CONTEXT
                    and len(set(case["q_lens"])) > 1
                    and not case.get("cum_q")
                ):
                    raise AssertionError(
                        f"{name} {case['label']}: ragged query lengths need cum_q"
                    )
                case["flashinfer"] = flashinfer_runner(t, m, case)
                final.append((name, case))

    # -- pass 2: every final case, full runner fields ---------------------------
    lines2 = []
    for i, (name, case) in enumerate(final):
        r = t.runner_params(by_name[name], case)
        lines2.append(t.probe_line(str(i), r))
    _log(f"pass 2: probing {len(lines2)} cases")
    records2 = run_probe(binary, lines2)
    _log("pass 2 done")
    pointers = {role: t.sentinel(role) for role in t.ROLES}
    index_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fixture_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    examples: dict[str, str] = {}
    for i, (name, case) in enumerate(final):
        m = by_name[name]
        rec = records2.get(i)
        if rec is None or "error" in rec:
            raise AssertionError(f"{name}: probe failed for case {case}: {rec}")
        want = survivors[name] if case.get("packed") else name
        if rec["kernel"] != want:
            raise AssertionError(
                f"{name}: case selects {rec['kernel']} instead: {case}"
            )
        r = t.runner_params(m, case)
        data, launch = t.build_params(
            m, r, {role: pointers[role] for role in r["ptrs"]}, t.fake_encode, layout
        )
        got = {
            "grid": launch["grid"],
            "cluster": launch["cluster"],
            "block": [m["threads"], 1, 1],
            "smem": m["smem"],
        }
        want_launch = {k: rec[k] for k in ("grid", "cluster", "block", "smem")}
        probe_bytes = bytes.fromhex(rec["params"])
        if got != want_launch or data != probe_bytes:
            fields = sorted(
                field
                for field, (offset, size, _) in layout["fields"].items()
                if data[offset : offset + size] != probe_bytes[offset : offset + size]
            )
            raise AssertionError(
                f"{name}: Python port differs from the probe: {got} {want_launch} "
                f"fields {fields}, case {case}"
            )
        # (max_dyn_smem is whatever kernel selection loaded last; the
        # launched kernel's attribute is set in loadKernel when it is first
        # loaded, as TrtllmFmha.configure does.)
        if rec["pdl"] != 1 or rec["policy"] != int(rec["cluster"][0] > 1):
            raise AssertionError(f"{name}: unexpected launch attributes {rec}")
        n = len(index_cases[name])
        index_cases[name].append(case)
        fixture_records[name].append(
            {
                "case": n,
                "grid": rec["grid"],
                "cluster": rec["cluster"],
                "block": rec["block"],
                "smem": rec["smem"],
                "params_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        digest = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        if n == 0 and len(examples) < EXAMPLES and digest % 97 < 5:
            examples[name] = rec["params"]

    counts: dict[str, int] = defaultdict(int)
    small_grid = 0
    for name, case in final:
        counts[case["suite"]] += 1
        if case["suite"] == "throughput":
            r = t.runner_params(by_name[name], case)
            if math.prod(t.launch_config(by_name[name], r)["grid"]) < t.NUM_SMS:
                small_grid += 1
    upstream_live = sum(1 for v in upstream_map.values() if v in by_name)
    by_test: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for entry in entries:
        func = entry["test"].split("::")[1].split("[")[0]
        by_test[func][0] += 1
        by_test[func][1] += upstream_map[entry["key"]] in by_name
    stats = {
        "candidates": len(lines),
        "cases": dict(sorted(counts.items())),
        "throughput_cases_below_148_ctas": small_grid,
        "case_restrictions": {
            reason: {
                "kernels": len(per_kernel),
                "candidates_dropped": sum(per_kernel.values()),
            }
            for reason, per_kernel in sorted(dropped.items())
        },
        "smoke_goals": {
            g: [goal_stats["covered"][g], goal_stats["achievable"][g]] for g in GOALS
        },
        "upstream": {
            "parametrizations": len(entries),
            "shapes": len(upstream_map),
            "shapes_on_live_kernels": upstream_live,
            "shapes_on_excluded_kernels": dict(
                sorted(
                    _count(
                        v[len("excluded: ") :]
                        for v in upstream_map.values()
                        if v.startswith("excluded: ")
                    ).items()
                )
            ),
            "parametrizations_by_test_on_live_kernels": {
                func: f"{live} of {total}"
                for func, (total, live) in sorted(by_test.items())
            },
            "shapes_without_kernel": sum(
                1 for v in upstream_map.values() if v.startswith("error: ")
            ),
        },
    }
    return dict(index_cases), dict(fixture_records), examples, upstream_map, stats


def _count(values: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for v in values:
        out[v] += 1
    return out

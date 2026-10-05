"""batched_gemm package: trtllm-gen batched/dense GEMMs (sm_100a), CUTLASS
segment GEMM with row-major weights (sm_86).

sm_100a -- trtllm-gen batched GEMM and dense GEMM cubins of FlashInfer v0.6.9
==============================================================================

``cubins/batched_gemm`` (1,405 ``Bmm_*_sm100f`` cubins) and ``cubins/gemm``
(74 ``Gemm_*_sm100f`` cubins) are the trtllm-gen artifacts FlashInfer v0.6.9
pins (``flashinfer/artifacts.py``: ``TRTLLM_GEN_BMM`` =
``39a9d282.../batched_gemm-5ee61af-2b9855b``, ``TRTLLM_GEN_GEMM`` =
``31e75d42.../gemm-4daf11e-1fddea2``); every file is verified against the
artifact's ``checksums.txt`` (itself pinned by ``CheckSumHash``), fetched with
its export headers and ``flashinferMetaInfo.h`` by ``scripts/fetch_resources.py
--only trtllm-gen``. Each cubin holds its kernel and a ``<kernel>GetSmemSize``
helper that no host code launches; ``harness.cubin_strip`` deletes the helper
(``check_strip`` verifies every kept byte) and writes
``cubins/sm_100a/<kernel>.cubin`` (generated, gitignored). Cubins declare
``sm_100f`` and are registered for ``sm_100a``.

Host-only probes compile FlashInfer v0.6.9's host code against the pinned
export headers, with every CUDA runtime/driver entry point interposed
(``kernels/probe_common.cuh``):

* ``kernels/bmm_probe.cu`` + ``kernels/bmm_moe_probe.cu`` (two translation
  units, as upstream builds them; ``kernels/moe_bridge.h``):
  ``csrc/trtllm_batched_gemm_runner.cu`` and ``csrc/trtllm_fused_moe_runner.cu``
  unmodified, plus the fused-MoE launchers' ``computeSelectedTileN`` and
  tile ladders copied verbatim from ``csrc/trtllm_fused_moe_kernel_launcher.cu``.
  A MoE case goes in as the MoE problem (tokens, top-k, local experts,
  hidden, intermediate, optional buffers): the probe reports the launcher's
  ladder and candidate tiles, the PermuteGemm1/Gemm2 runner's passing and
  ``isValidConfigIndex`` verdicts, whether ``MoE::Runner::
  getValidConfigIndices`` pairs the config with a valid FC1/FC2 config (what
  ``getValidConfigs`` hands FlashInfer's autotuner) and the launch
  ``PermuteGemm1/Gemm2::Runner::run`` makes. The runner-level problem the
  MoE runner derives is rebuilt and must reproduce the launched
  ``KernelParams``, and ``host.moe_problem``/``moe_buffers``/
  ``selected_tiles`` are checked against the probe for every case: the
  dimension mapping and tile selection come from upstream code, not from
  Python. Which optional buffers are non-null is the launchers' choice
  (TVM-FFI code that is not compiled here): ``host.moe_buffers`` restates
  it (bias: FP4 launcher; alpha/beta/clamp: FP4 and MxInt4 launchers;
  routing scales: FP8 per-tensor launcher with Llama4 routing; output
  scales per quantization mode) and the probe forwards them exactly as the
  MoE runners do.
* ``kernels/gemm_probe.cu``: the dense runners (copied verbatim in
  ``kernels/gemm_runners.cuh``): validity, ``getValidTactics`` (the
  autotuner's list) and the ``tactic = -1`` heuristic.

Both record, for every case registered here (smoke and throughput), the
launch upstream makes (grid, block, dynamic smem, launch attributes,
``KernelParams`` bytes, bytes upstream leaves indeterminate,
``cuTensorMapEncodeTiled`` arguments): ``fixtures/sm_100a.jsonl.gz``.

Cases (``CASE_POLICY``, recorded in ``provenance.json``) are problems the
runners accept for the kernel; each records in ``source`` whether
FlashInfer's tile selection offers the kernel's tile for it and its regime
(``host.moe_regime``: centre tile, K loop wrapping the pipeline, persistent
multi-tile, features). MoE kernels get ``smoke_tiny``, ``smoke_ragged``
(every runtime feature a launcher can switch on), ``smoke_dispatch``
(computeSelectedTileN centres on the tile), ``smoke_deep`` (K tiles >
stages, more tiles than resident CTAs for persistent schedulers),
``upstream_<i>`` and model-shape throughput cases (decode, prefill/batch
where the tile selection offers the tile, stress beyond it). Dense kernels
get ``smoke_tiny``/``smoke_rows``/``smoke_deep``, every exact upstream
shape and model linear layers (decode, prefill up to m = 16384).

Upstream tests are a subset of the cases: ``harness/workloads/batched_gemm/
upstream.py`` evaluates the parametrizations of FlashInfer v0.6.9's tests
that reach these kernels (``tests/moe/test_trtllm_gen_fused_moe.py``,
``tests/moe/test_trtllm_gen_routed_fused_moe.py``,
``tests/autotuner/test_trtllm_fused_moe_autotuner_integration.py``,
``tests/gemm/test_mm_fp4.py``, ``test_mm_mxfp8.py``,
``test_groupwise_scaled_gemm_fp8.py``, ``test_mm_fp8.py`` and, for sm_86,
``test_group_gemm.py``) with their skip rules into problems
(``variants/upstream.json``). The probe decides which kernels each problem
may launch (MoE: the tile is among the candidate tiles, or the test forces
every tile, and the MoE runner pairs the config with a valid one; dense:
in ``getValidTactics`` when autotuned, else the heuristic's choice);
``variants/sm_100a.json`` lists them per kernel (``upstream``). Every such
MoE problem has a case of the kernel in its regime -- the cheapest exact
problem of each regime is registered -- and every dense/segment problem is
an exact case; ``tests/test_batched_gemm.py`` enforces both and that the
inventory is current. ``tests/moe/test_dpsk_fused_moe_fp8.py`` contributes
nothing (its ``skip_checks`` call skips SwiGlu at hidden 7168), nor do the
``auto`` backends (never trtllm).

Registered but unreachable: the 24 E4m3 per-tensor configs with tileN 192
or 256 (16 FC1, 8 FC2) are not on ``Fp8PerTensorLauncher``'s ladder (8..128),
so ``computeSelectedTileN`` never offers them and no FlashInfer v0.6.9 path
launches them; they keep their stress cases (``unreachable_registered`` in
``provenance.json``).

Live/dead decisions (who launches each kernel in FlashInfer v0.6.9):

=====================================  ======  ===============================
kernels                                status  reason
=====================================  ======  ===============================
Bmm, dynamic batch, routed B           live    MoE FC1 (``PermuteGemm1``):
(route ldgsts/TMA), C = B dtype                ``TrtllmGenBatchedGemmRunner``
                                               with ``routeAct=true``,
                                               ``staticBatch=false``
Bmm, dynamic batch, no route,          live    MoE FC2 (``Gemm2``,
C = Bfloat16                                   ``dtypeOut=Bfloat16``)
Bmm, static batch (42)                 dead    no FlashInfer caller builds the
                                               runner with ``staticBatch=true``
                                               (``trtllm_fused_moe_runner.cu``
                                               hardcodes ``false``)
Gemm E2m1 -> Bf16 (34)                 live    ``mm_fp4`` trtllm backend
                                               (``trtllm_gemm``, autotuned)
Gemm MxE4m3 -> Bf16 (12)               live    ``mm_mxfp8`` trtllm backend
Gemm E4m3 DeepSeek FP8 (4)             live    ``gemm_fp8_nt_groupwise``
                                               trtllm backend: ``tactic=-1``,
                                               ``select_kernel_fp8`` by name
Gemm E4m3, split-K, shuffled (16)      dead    reachable only by an explicit
                                               tactic index; no FlashInfer
                                               caller passes one for E4m3
Gemm E4m3 BlockMajorK, shuffled (8)    live    ``trtllm_low_latency_gemm``
                                               (``mm_fp8`` low-latency backend)
GetSmemSize helpers (all cubins)       dead    never launched
=====================================  ======  ===============================

Which config a call uses depends on its shape; a live config without a valid
probe-verified ``smoke_tiny`` and ``smoke_ragged`` (``smoke_rows``) shape
would be excluded and listed in ``provenance.json``.

sm_86 -- FlashInfer CUTLASS segment GEMM, row-major weights
===========================================================

FlashInfer 0.7.0's sm80 JIT-cache wheel (``scripts/fetch_resources.py --only
jit-cache``) holds the only grouped/batched GEMM kernels of the cache in
``gemm/gemm.2.sm_80.cubin``: the 8 ``cutlass::Kernel<GemmGrouped<...>>``
instantiations of ``CutlassSegmentGEMMRun`` ({bf16, fp16} x {B RowMajor, B
ColumnMajor} x {2, 4 stages}). ``SegmentGEMMWrapper.run(...,
weight_column_major=False)`` dispatches the four RowMajor ones (weights
``[G, K, N]``), which this package owns; the ColumnMajor kernels belong to
deep_gemm. Each is cut out with ``harness.cubin_strip.strip_cubin``.

=========================  ======  ===========================================
kernel (B RowMajor)        status  reason
=========================  ======  ===========================================
bf16/fp16 MmaPipelined     live    ``DISPATCH_SMEM_CONFIG`` picks the 2-stage
                                   mainloop below 147968 B shared memory per
                                   SM (sm_86, sm_89): upstream's sm_86 kernel
bf16/fp16 MmaMultistage    live*   picked on sm_80/sm_87; its sm_80 SASS
                                   (64 KiB shared memory) runs unchanged on
                                   sm_86, registered so the local library
                                   covers both mainloops
B ColumnMajor (4)          other   deep_gemm package
=========================  ======  ===========================================

``GemmGrouped::Params`` (by value) is rebuilt in Python from the layout that
``kernels/segment_gemm_rowmajor_probe.cu`` records (CUTLASS b46b16d, as
pinned by FlashInfer 0.7.0), byte-compared with the probe's reference structs.

Cases (``segment_cases``, written to ``variants/sm_86.json``): three smoke
cases (tiny, empty segments with partial tiles, ragged experts), the 72
problems of FlashInfer's ``test_segment_gemm`` with the sm80 backend and
row-major weights (exact, except that problems whose weights exceed 2 GiB
keep their segments and dims with fewer segments, for the 6 GB sm_86
budget; those over 256 MiB in the throughput suite) and model-shape throughput cases (MoE expert projections
of Mixtral, Qwen3-30B and DeepSeek-V3 EP8 with skewed segment lengths,
multi-LoRA shrink/expand on Llama-3.1-8B), each within ~4 GB as the local
RTX 3090 runs them.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.workloads.batched_gemm import upstream  # noqa: E402

MANIFEST = PACKAGE / "kernels.json"
PROVENANCE = PACKAGE / "provenance.json"
VARIANTS = PACKAGE / "variants"
FIXTURES = PACKAGE / "fixtures"

FLASHINFER_V069 = "a1aa676196f798435248d9ea205c67674476f473"
FLASHINFER_V070 = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
CUTLASS_DIR = "cutlass-b46b16d003484063bca4ed365e44095c4c6ed633"
SM_COUNT = 148  # B200: what the runners read through cudaDeviceGetAttribute

ARTIFACTS = {
    "bmm": {
        "path": "39a9d28268f43475a757d5700af135e1e58c9849/batched_gemm-5ee61af-2b9855b",
        "checksums_sha256": (
            "db06db7f36a2a9395a2041ff6ac016fe664874074413a2ed90797f91ef17e0f6"
        ),
        "cubins": "cubins/batched_gemm",
        "export": "trtllmGen_bmm_export",
        "stage": "flashinfer/trtllm/batched_gemm",
        "probe": ("kernels/bmm_probe.cu", "kernels/bmm_moe_probe.cu"),
    },
    "gemm": {
        "path": "31e75d429ff3f710de1251afdd148185f53da44d/gemm-4daf11e-1fddea2",
        "checksums_sha256": (
            "64b7114a429ea153528dd4d4b0299363d7320964789eb5efaefec66f301523c7"
        ),
        "cubins": "cubins/gemm",
        "export": "trtllmGen_gemm_export",
        "stage": "flashinfer/trtllm/gemm",
        "probe": ("kernels/gemm_probe.cu",),
    },
}
PROBE_SOURCES = (
    "kernels/bmm_probe.cu",
    "kernels/bmm_moe_probe.cu",
    "kernels/moe_bridge.h",
    "kernels/gemm_probe.cu",
    "kernels/gemm_runners.cuh",
    "kernels/probe_common.cuh",
)
NV_INTERNAL_COMMON = (
    "envUtils.cpp",
    "logger.cpp",
    "stringUtils.cpp",
    "tllmException.cpp",
)
PROBE_DEFINES = (
    "-DTLLM_GEN_EXPORT_INTERFACE",
    "-DTLLM_GEN_EXPORT_FLASHINFER",
    "-DTLLM_ENABLE_CUDA",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-DENABLE_FP4",
    '-DTLLM_GEN_GEMM_CUBIN_PATH="probe"',
)

JIT_CACHE_CUBIN = "flashinfer-jit-cache-sm80/gemm/gemm.2.sm_80.cubin"
JIT_CACHE_SHA256 = "8a23d28908a7b0ee713646602dd17dfaf995c77b1410e1c36482739fa8909c3c"
SEGMENT_PROBE = "kernels/segment_gemm_rowmajor_probe.cu"
SEGMENT_KERNELS = {
    # workload: (probe key, dtype token in the symbol, pipelined?)
    "batched_gemm_segment_bf16_pipelined": ("bf16_pipelined", "bfloat16_t", True),
    "batched_gemm_segment_bf16_multistage": ("bf16_multistage", "bfloat16_t", False),
    "batched_gemm_segment_fp16_pipelined": ("fp16_pipelined", "half_t", True),
    "batched_gemm_segment_fp16_multistage": ("fp16_multistage", "half_t", False),
}
ROW_MAJOR_B = "37RowMajorTensorOpMultiplicandCongruous"
COLUMN_MAJOR_B = "40ColumnMajorTensorOpMultiplicandCrosswise"

# select_kernel_fp8 (csrc/trtllm_gemm_runner.cu): kernel -> (n, k) smoke and
# throughput shapes satisfying its N/K-ratio rule.
DS_DENSE_SHAPES = {
    "gemm_Bfloat16_E4m3E4m3_Fp32_t128x8x128u2_s6_et64x8_m64x8x32_c1x1x1_16dp256b_rM_TN_"
    "transOut_noShflA_dsFp8_schPd2x2x1x3_sm100f": ((8192, 256), (32768, 1024)),
    "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_16dp256b_rM_TN_"
    "transOut_noShflA_dsFp8_schedS_sm100f": ((256, 256), (4096, 4096)),
    "gemm_Bfloat16_E4m3E4m3_Fp32_t128x32x128u2_s6_et64x32_m64x32x32_c1x1x1_16dp256b_rM_TN_"
    "transOut_noShflA_dsFp8_schPd2x2x1x3_sm100f": ((20480, 1024), (24576, 2048)),
    "gemm_Bfloat16_E4m3E4m3_Fp32_t128x16x128u2_s6_et64x16_m64x16x32_c1x1x1_16dp256b_rM_TN_"
    "transOut_noShflA_dsFp8_schedS_sm100f": ((1024, 256), (8192, 2048)),
}


def _harness() -> Any:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from harness import cubin_strip, workload
    from harness.workloads.batched_gemm import host

    return workload, cubin_strip, host


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any, indent: int | None = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=indent) + "\n")
    os.replace(tmp, path)


def _lines_json(path: Path, entries: dict[str, Any]) -> None:
    """One compact JSON line per entry (diff-friendly, small)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(
        f" {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}"
        for k, v in sorted(entries.items())
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("{\n" + body + "\n}\n")
    os.replace(tmp, path)


def _write_gzip_lines(path: Path, records: Iterable[Any]) -> None:
    """Deterministic gzip (mtime 0, no file name) of JSON lines."""
    raw = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as f:
        f.write(raw.encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(buffer.getvalue())
    os.replace(tmp, path)


def _merge(path: Path, arch: str, value: Any) -> None:
    data = json.loads(path.read_text()) if path.is_file() else {}
    data[arch] = value
    _write_json(path, dict(sorted(data.items())))


def _portable(command: Sequence[str]) -> list[str]:
    root = str(ROOT) + os.sep
    return [str(part).replace(root, "") for part in command]


def _verbatim_blocks(text: str) -> list[tuple[str, int, int, str]]:
    """(path, first line, last line, body) of each VERBATIM block."""
    pattern = re.compile(
        r"^// VERBATIM BEGIN (\S+):(\d+)-(\d+)\n(.*?)^// VERBATIM END\n",
        re.M | re.S,
    )
    return [
        (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4))
        for m in pattern.finditer(text)
    ]


# --- case shapes ------------------------------------------------------------------


def lcm(*values: int) -> int:
    return math.lcm(*values)


def _k_unit(o: dict[str, Any], host: Any, dense: bool = False) -> int:
    """Smallest K every operand layout of the config accepts: whole K tiles
    for every split-K slice, whole SF tiles (rows of 16 entries for the MoE
    activations' linear SFs read through TMA, 4 SF columns for the swizzled
    layouts), 128-element DeepSeek blocks and whole BlockMajorK blocks."""
    d = host.DTYPES
    unit = o["mTileK"] * o["mNumSlicesForSplitK"]
    dtypes = (o["mDtypeA"], o["mDtypeB"])
    sf_cols = 4 if dense else 16
    if d["E2m1"] in dtypes:
        unit = lcm(unit, sf_cols * 16)
    if any(t in dtypes for t in (d["MxE2m1"], d["MxE4m3"], d["MxInt4"])):
        unit = lcm(unit, sf_cols * 32)
    if o["mUseDeepSeekFp8"]:
        unit = lcm(unit, 128)
    if o["mLayoutA"] == host.LAYOUT_BLOCK_MAJOR_K:
        unit = lcm(unit, o["mBlockK"])
    return lcm(unit, 64)


def _round_up(value: int, unit: int) -> int:
    return -(-value // unit) * unit


# Throughput models (harness.models): local experts per rank (expert
# parallelism where the full expert set does not fit the budget or a node
# would shard it), top-k, hidden and intermediate sizes. gpt-oss's 2880 is
# padded to 3072 (a multiple of 256): the trtllm-gen kernels need 128-row M
# tiles and whole 128..512-element K tiles, so serving stacks pad it so.
MOE_MODELS: tuple[tuple[str, int, int, int | None, int | None], ...] = (
    # (model, local experts, expert-parallel ranks, hidden, intermediate)
    ("deepseek_v3", 32, 8, None, None),
    ("qwen3_235b_a22b", 128, 1, None, None),
    ("qwen3_30b_a3b", 128, 1, None, None),
    ("qwen3_next_80b_a3b", 512, 1, None, None),
    ("llama4_scout", 16, 1, None, None),
    ("gpt_oss_120b", 128, 1, 3072, 3072),
    ("mixtral_8x7b", 8, 1, None, None),
    ("kimi_k2", 48, 8, None, None),
    ("llama4_maverick", 16, 8, None, None),
    ("gpt_oss_20b", 32, 1, 3072, 3072),
)
DECODE_TOKENS = (1, 4, 16, 64)
PREFILL_TOKENS = (16384, 8192, 4096, 2048, 1024, 256)
STRESS_TOKENS = (8192, 4096)
CASE_BUDGET = 40 << 30  # inputs + outputs + reference temporaries (B200)
SMOKE_BUDGET = 512 << 20  # smoke cases also run (references) on a 24 GB RTX 3090
DISPATCH_DIMS = (
    (1024, 1024),
    (2048, 1024),
    (1024, 2048),
    (512, 512),
    (2048, 2048),
    (4096, 1536),
    (7168, 2048),
    (1024, 512),
    (3072, 3072),
)


def moe_model(name: str, experts: int, ep: int, hidden: int | None, inter: int | None):
    from harness.models import MODELS

    spec = MODELS[name]
    params = {
        "experts": experts,
        "top_k": spec.top_k,
        "hidden": hidden or spec.hidden,
        "intermediate": inter or spec.moe_intermediate,
    }
    source = {"kind": "model_shape", "model": name, "hf": spec.hf,
              "experts_global": spec.experts, "expert_parallel": ep}  # fmt: skip
    if hidden or inter:
        source["padded"] = {
            "hidden": spec.hidden,
            "intermediate": spec.moe_intermediate,
        }
    return params, source


def moe_case_bytes(entry: dict[str, Any], params: dict[str, Any], host: Any) -> int:
    """Upper estimate of a MoE case's GPU memory: the stored weights, one
    expert's FP32 temporaries, activations (stored and FP32 copies), the FP32
    reference with its encoded copy and the validator's temporaries over the
    C rows."""
    o = entry["options"]
    p = host.moe_problem(o, params)
    rows = p.max_num_ctas * o["mTileN"]
    cols = p.m // 2 if o["mFusedAct"] else p.m
    weights = p.num_batches * p.m * p.k * host.dtype_bits(o["mDtypeA"]) // 8
    b_rows = params["tokens"] if host.moe_role(o) == "fc1" else rows
    return weights + 16 * p.m * p.k + 9 * b_rows * p.k + 28 * rows * cols


def dense_case_bytes(params: dict[str, Any]) -> int:
    m, n, k = params["m"], params["n"], params["k"]
    return 9 * (m + n) * k + 26 * m * n


def moe_smoke_candidates(
    entry: dict[str, Any], host: Any
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """(case name, MoE params, source) candidates in preference order: the
    first upstream-valid one per name is registered (``smoke_dispatch``
    additionally needs FlashInfer's tile selection to offer the tile)."""
    o = entry["options"]
    tm = o["mTileM"] * o["mClusterDimX"]
    unit = _k_unit(o, host)
    fc1 = host.moe_role(o) == "fc1"
    factor = host.moe_gate_factor(o)
    tile = o["mTileN"]
    ds = bool(o["mUseDeepSeekFp8"])
    features = {
        f: 1 for f in ("bias", "gated_act", "routing_scales") if host.moe_supports(o, f)
    }

    def dims(m: int, k: int) -> dict[str, int]:
        return (
            {"hidden": k, "intermediate": m // factor}
            if fc1
            else {"hidden": m, "intermediate": k}
        )

    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for scale_m, scale_k in ((2, 2), (2, 1), (4, 2), (2, 4), (4, 4)):
        m, k = scale_m * tm, scale_k * unit
        base = {"top_k": 2, "experts": 4, **dims(m, k)}
        out.append(("smoke_tiny", {**base, "tokens": 3}, {
            "kind": "synthetic_stress",
            "reason": "3 tokens over 4 skewed experts: 1-token CTAs, an empty expert",
        }))  # fmt: skip
        # Every runtime feature a FlashInfer launcher can switch on for this
        # config (routing scales come with top-1 routing, as Llama4's).
        ragged = {**base, "tokens": 2 * tile + 5, **features}
        if features.get("routing_scales"):
            ragged["top_k"] = 1
        out.append(("smoke_ragged", ragged, {
            "kind": "synthetic_stress",
            "reason": "2 * tileN + 5 tokens: full and partial CTAs; runtime features on: "
            + (", ".join(features) or "none"),
        }))  # fmt: skip
    if tile & (tile - 1) == 0:
        # avg tokens per expert = 0.75 * tileN: computeSelectedTileN centres
        # on this tile; dims that also give the paired FC1/FC2 config whole
        # tiles (MoE::Runner offers only valid pairs).
        for hidden, inter in DISPATCH_DIMS:
            out.append(("smoke_dispatch", {"hidden": hidden, "intermediate": inter,
                                           "experts": 16, "top_k": 4, "tokens": 3 * tile}, {
                "kind": "probe_verified_dispatch",
                "reason": "FlashInfer's tile heuristic centres on this config's tileN",
            }))  # fmt: skip
    # The K loop wraps the smem pipeline; a persistent scheduler gets more
    # tiles than resident CTAs; skewed routing with an empty expert.
    stages = host.k_stages(o)
    k_deep = _round_up((stages + 2) * o["mTileK"] * o["mNumSlicesForSplitK"], unit)
    for scale_k in (1, 2):
        for scale_m in (2, 4):
            m = scale_m * tm
            params = {**dims(m, k_deep * scale_k), "experts": 8, "top_k": 2,
                      "skew": 1.0, "empty": 1}  # fmt: skip
            if o["mTileScheduler"] in host.SCHEDULERS_PERSISTENT:
                m_tiles = host.ceil_div(m, o["mTileM"])
                target = host.resident_ctas(entry) // m_tiles + 8
                params["tokens"] = ((target - 8) * tile + 8) // 2 + tile + 1
            else:
                params["tokens"] = 4 * tile + 3
            out.append(("smoke_deep", params, {
                "kind": "synthetic_stress",
                "reason": f"K = {params['hidden' if fc1 else 'intermediate']} wraps the "
                f"{stages}-stage pipeline; persistent: more tiles than resident CTAs",
            }))  # fmt: skip
    del ds
    return out


def moe_throughput_candidates(
    entry: dict[str, Any], rotation: int
) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """(slot, case name, params, source) model-shape candidates: slot
    ``decode`` (smallest tokens first), ``prefill`` (largest first) and
    ``stress``; models rotate per kernel so that the family covers all."""
    fc1 = entry["role"] == "fc1"
    out = []
    n = len(MOE_MODELS)
    for slot, tokens_list, offset in (
        ("decode", DECODE_TOKENS, 0),
        ("prefill", PREFILL_TOKENS, 1),
        ("stress", STRESS_TOKENS, 2),
    ):
        for j in range(n):
            name, experts, ep, hidden, inter = MOE_MODELS[(rotation + offset + j) % n]
            params, source = moe_model(name, experts, ep, hidden, inter)
            source["layer"] = "expert_gate_up" if fc1 else "expert_down"
            for tokens in tokens_list:
                label = "batch" if slot == "prefill" and tokens < 4096 else slot
                case = f"throughput_{label}_{name}_t{tokens}"
                out.append((slot, case, {**params, "tokens": tokens, "skew": 0.5},
                            {**source, "tokens": tokens}))  # fmt: skip
    return out


def gemm_role(function: str, o: dict[str, Any], host: Any) -> str | None:
    """Dense runner that launches this config, or None (dead)."""
    d = host.DTYPES
    if o["mLayoutA"] == host.LAYOUT_BLOCK_MAJOR_K and o["mUseShuffledMatrix"]:
        return "gemm_low_latency"
    if o["mDtypeA"] == d["E2m1"]:
        return "gemm_fp4"
    if o["mDtypeA"] == d["MxE4m3"]:
        return "gemm_mxfp8"
    if o["mDtypeA"] == d["E4m3"] and function in DS_DENSE_SHAPES:
        return "gemm_fp8_blockscale"
    return None


def gemm_smoke_candidates(
    function: str, entry: dict[str, Any], host: Any
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    o, role = entry["options"], entry["role"]
    tm = o["mTileM"] * o["mClusterDimX"]
    unit = _k_unit(o, host, dense=True)
    src = {"kind": "synthetic_stress"}
    out: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    stages = host.k_stages(o)
    k_min = (stages + 2) * o["mTileK"] * o["mNumSlicesForSplitK"]
    persistent = o["mTileScheduler"] in host.SCHEDULERS_PERSISTENT
    if role == "gemm_fp8_blockscale":
        # Activation scales are [K / 128, m] (MN-major): m is kept a multiple
        # of 4 so that no row-stride padding convention matters.
        (n, k), _ = DS_DENSE_SHAPES[function]
        out.append(("smoke_tiny", {"m": 4, "n": n, "k": k}, {**src, "reason": "m = 4"}))
        out.append(
            (
                "smoke_rows",
                {"m": 72, "n": n, "k": k},
                {**src, "reason": "partial tiles"},
            )
        )
        for s in (1, 2, 4, 8, 16):
            for m in (72, 360, 1500):
                out.append(("smoke_deep", {"m": m, "n": n * s, "k": k * s}, {
                    **src, "reason": f"K loop beyond {stages} stages"}))  # fmt: skip
        return out
    # getValidTactics keeps the configs sorted by (tileK, split-K, unroll-2x)
    # only up to the first non-unrolled config after an unrolled one, so a
    # smaller tileK is offered only when K is no multiple of the larger
    # tiles: odd multiples of the unit cover those configs.
    multiples = (2, 1, 3, 5, 7, 4, 15, 31)
    for j in multiples:
        for scale_n in (2, 4):
            n, k = scale_n * tm, j * unit
            out.append(
                ("smoke_tiny", {"m": 3, "n": n, "k": k}, {**src, "reason": "m = 3"})
            )
            out.append(("smoke_rows", {"m": 70, "n": n, "k": k}, {
                **src, "reason": "m = 70: partial tiles"}))  # fmt: skip
    first = max(1, -(-k_min // unit))
    for j in range(first, first + 24):
        for scale_n in (4, 8):
            n = scale_n * tm
            m = 70
            if persistent:
                n_tiles = host.ceil_div(n, o["mTileM"])
                m = (host.resident_ctas(entry) // n_tiles + 4) * o["mTileN"] + 3
            out.append(("smoke_deep", {"m": m, "n": n, "k": j * unit}, {
                **src, "reason": f"K loop beyond {stages} stages"
                + ("; more tiles than resident CTAs" if persistent else "")}))  # fmt: skip
    return out


DENSE_DECODE_M = (1, 16, 64)
DENSE_PREFILL_M = (16384, 8192, 4096)
DENSE_BUDGET = 24 << 30


def dense_throughput_candidates(rotation: int):
    """(slot, name, params, source): every model's linear layers, decode
    (smallest m first) and prefill (largest m first)."""
    from harness.models import MODELS

    names = sorted(MODELS)
    out = []
    for slot, ms, offset in (
        ("decode", DENSE_DECODE_M, 0),
        ("prefill", DENSE_PREFILL_M, 3),
    ):
        for j in range(len(names)):
            model = names[(rotation + offset + j) % len(names)]
            spec = MODELS[model]
            for layer, (n, k) in spec.linear_shapes().items():
                for m in ms:
                    out.append((slot, f"throughput_{slot}_{model}_{layer}_m{m}",
                                {"m": m, "n": n, "k": k},
                                {"kind": "model_shape", "model": model, "hf": spec.hf,
                                 "layer": layer}))  # fmt: skip
    # Stress fallback for configs no model K reaches (getValidTactics offers
    # a smaller tileK only when K is no multiple of the larger tiles): model
    # (m, n) with K one 128-element step past the layer's.
    for j in range(len(names)):
        model = names[(rotation + j) % len(names)]
        spec = MODELS[model]
        for layer, (n, k) in spec.linear_shapes().items():
            for m in DENSE_PREFILL_M:
                out.append(("stress", f"throughput_stress_{model}_{layer}_m{m}",
                            {"m": m, "n": n, "k": k + 128},
                            {"kind": "synthetic_stress", "reason": f"{model} {layer} "
                             "with K + 128: the tactic list offers this config only "
                             "for K an odd multiple of its tileK", "model": model}))  # fmt: skip
    return out


def merge_problems(problems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct upstream problems (everything but the test id), each with the
    ids of every test parametrization that runs it."""
    merged: dict[str, dict[str, Any]] = {}
    for p in problems:
        key = json.dumps({k: v for k, v in p.items() if k != "test"}, sort_keys=True)
        if key not in merged:
            merged[key] = {**p, "tests": []}
            del merged[key]["test"]
        merged[key]["tests"].append(p["test"])
    return list(merged.values())


def chosen_of(chosen: dict[str, Any], entries: dict[str, Any]) -> dict[str, Any]:
    return {n: c for n, c in chosen.items() if n in entries}


def host_supports(entry: dict[str, Any], feature: str) -> bool:
    _, _, host = _harness()
    return bool(host.moe_supports(entry["options"], feature))


CASE_POLICY = {
    "moe_smoke": [
        "smoke_tiny: 3 tokens, 4 skewed experts (one empty), features off",
        "smoke_ragged: 2 * tileN + 5 tokens, every runtime feature a FlashInfer "
        "launcher can switch on for the config (bias, gated-activation alpha/beta/"
        "clamp, routing scales with top-1)",
        "smoke_dispatch: 3 * tileN tokens over 16 experts, top-4, so that "
        "computeSelectedTileN centres on tileN (absent when the launcher's ladder "
        "lacks tileN)",
        "smoke_deep: K tiles > pipeline stages; persistent schedulers get more "
        "tiles than resident CTAs; 8 log-normal-skewed experts, one empty",
        "upstream_<i>: per regime (FlashInfer's centre tile, K wrap, persistent "
        "multi-tile, features) of the upstream problems that may launch the "
        "kernel, the cheapest exact upstream problem (throughput suite when "
        "over 1 GiB)",
    ],
    "moe_throughput": [
        "decode: smallest tokens in (1, 4, 16, 64) of a model whose tile "
        "selection offers the kernel's tileN",
        "prefill/batch: largest tokens in (16384 .. 256) with the tile offered",
        "stress: 8192 (4096) tokens when no offered size reaches 1024",
        "models rotate per kernel: " + ", ".join(m[0] for m in MOE_MODELS),
    ],
    "dense_smoke": "smoke_tiny (m 3/4), smoke_rows (m 70/72), smoke_deep (K tiles "
    "> stages, more tiles than resident CTAs when persistent), every upstream "
    "shape whose runner may pick the kernel (autotuned: in getValidTactics; "
    "otherwise the tactic -1 heuristic)",
    "dense_throughput": "one decode (m 1..64) and two prefill (m 4096..16384) "
    "linear layers of harness.models whose runner may pick the kernel",
    "budgets": {"case_bytes": CASE_BUDGET, "smoke_bytes": SMOKE_BUDGET},
}


def coverage_summary(
    variants: dict[str, Any], moe: list[dict[str, Any]], dense: list[dict[str, Any]]
) -> dict[str, Any]:
    """Which upstream problems reach which kernels (counts)."""
    reached_moe = {i for v in variants.values() if v["role"] in ("fc1", "fc2")
                   for i in v["upstream"]}  # fmt: skip
    reached_dense = {i for v in variants.values() if v["role"].startswith("gemm")
                     for i in v["upstream"]}  # fmt: skip
    return {
        "moe_problems": len(moe),
        "moe_test_parametrizations": sum(len(p["tests"]) for p in moe),
        "moe_problems_reaching_a_kernel": len(reached_moe),
        "moe_kernels_reached": sum(1 for v in variants.values()
                                   if v["role"] in ("fc1", "fc2") and v["upstream"]),  # fmt: skip
        "dense_problems": len(dense),
        "dense_test_parametrizations": sum(len(p["tests"]) for p in dense),
        "dense_problems_reaching_a_kernel": len(reached_dense),
        "dense_kernels_reached": sum(1 for v in variants.values()
                                     if v["role"].startswith("gemm") and v["upstream"]),  # fmt: skip
        "unreached_moe_problems": [p["tests"][0] for i, p in enumerate(moe)
                                   if i not in reached_moe][:50],  # fmt: skip
    }


def unreachable(variants: dict[str, Any]) -> dict[str, str]:
    """Registered MoE kernels no FlashInfer tile selection ever offers (their
    tileN is not on the launcher's ladder): kept, launched only by the
    harness's stress cases."""
    out = {}
    for name, v in variants.items():
        if v["role"] in ("fc1", "fc2") and v["options"]["mTileN"] not in v.get(
            "ladder", []
        ):
            out[name] = (
                f"tileN {v['options']['mTileN']} is not on {v['launcher']}'s tile ladder "
                f"{v['ladder']}: computeSelectedTileN never offers it, so no FlashInfer "
                "v0.6.9 path launches this kernel"
            )
    return out


# --- sm_86 segment GEMM cases ------------------------------------------------------------

# Throughput: MoE expert GEMMs with row-major weights [G, K, N] (the segment
# GEMM serves grouped expert/LoRA projections), prefill-sized and skewed over
# the groups, plus multi-LoRA shrink/expand (rank 64, 32 adapters) on
# Llama-3.1-8B's fused QKV projection; each at most ~4 GB with the
# reference, as the local RTX 3090 runs them.
SEGMENT_MODELS = (
    # (model, layer, groups, k, n, rows)
    ("mixtral_8x7b", "expert_down", 8, 14336, 4096, 8192),
    ("qwen3_30b_a3b", "expert_gate_up", 128, 2048, 1536, 16384),
    ("deepseek_v3", "expert_down (EP8: 32 local experts)", 32, 2048, 7168, 8192),
    ("llama3_8b", "lora_shrink_qkv_r64", 32, 4096, 64, 8192),
    ("llama3_8b", "lora_expand_qkv_r64", 32, 64, 6144, 8192),
)
SEGMENT_SMOKE_WEIGHTS = 256 << 20
SEGMENT_MAX_WEIGHTS = 2 << 30


def segment_cases(resources: Path) -> list[dict[str, Any]]:
    from harness.models import MODELS
    from harness.throughput import skewed_lengths

    syn = {"kind": "synthetic_stress"}
    cases = [
        {"name": "four_segments", "params": {"lengths": [1, 2, 3, 4], "n": 256, "k": 128},
         "seed": 1, "suite": "smoke", "source": {**syn, "reason": "tiny segments"}},
        {"name": "empty_and_ragged",
         "params": {"lengths": [0, 37, 300, 0, 129], "n": 520, "k": 1032},
         "seed": 2, "suite": "smoke",
         "source": {**syn, "reason": "empty segments, partial N and K tiles"}},
        {"name": "eight_experts",
         "params": {"lengths": [256, 1, 640, 128, 0, 77, 513, 431], "n": 1024, "k": 2048},
         "seed": 3, "suite": "smoke",
         "source": {**syn, "reason": "ragged expert segments, long K loop"}},
    ]  # fmt: skip
    for i, u in enumerate(upstream.segment_problems(resources)):
        batch, rows = len(u["lengths"]), u["lengths"][0]
        per_group = u["k"] * u["n"] * 2
        source = {
            "kind": "upstream_test",
            "test": u["test"],
            "revision": upstream.REVISION,
        }
        if batch * per_group > SEGMENT_MAX_WEIGHTS:
            # Same segments and dims, fewer of them: the weights (doubled by
            # the input snapshot of the native test) must fit the 6 GB budget.
            source["batch_reduced_from"] = batch
            batch = SEGMENT_MAX_WEIGHTS // per_group
        weights = batch * per_group
        suite = "smoke" if weights <= SEGMENT_SMOKE_WEIGHTS else "throughput"
        name = f"upstream_b{len(u['lengths'])}_r{rows}_k{u['k']}_n{u['n']}"
        if batch != len(u["lengths"]):
            name += f"_as_b{batch}"
        cases.append({
            "name": ("throughput_" if suite == "throughput" else "") + name,
            "params": {"lengths": [rows] * batch, "n": u["n"], "k": u["k"]},
            "seed": 10 + i, "suite": suite, "source": source,
        })  # fmt: skip
    for i, (model, layer, groups, k, n, rows) in enumerate(SEGMENT_MODELS):
        spec = MODELS[model]
        lengths = skewed_lengths(rows, groups, seed=i, sigma=1.0, minimum=0)
        cases.append({
            "name": f"throughput_{model}_{layer.split(' ')[0]}",
            "params": {"lengths": lengths, "n": n, "k": k},
            "seed": 100 + i, "suite": "throughput",
            "source": {"kind": "model_shape", "model": model, "hf": spec.hf,
                       "layer": layer, "rows": rows},
        })  # fmt: skip
    return cases


# --- compiler -----------------------------------------------------------------------


class ImplCompiler:
    """Index/strip the batched_gemm single-kernel cubins of one arch."""

    supported_arches: tuple[str, ...] = ("sm_86", "sm_100a")

    def __init__(
        self,
        arch: str,
        *,
        optimization: str = "-O3",
        flags: Sequence[str] = (),
        nvcc: str | None = None,
        resources: Path | None = None,
    ):
        arch = "sm_" + str(arch).removeprefix("-arch=").removeprefix("sm_")
        if arch not in self.supported_arches:
            raise ValueError(f"batched_gemm supports {self.supported_arches}")
        if optimization not in ("-O0", "-O1", "-O2", "-O3"):
            raise ValueError(f"invalid optimization {optimization!r}")
        self.arch = arch
        # Precompiled kernels are never recompiled; optimization/flags only
        # apply to the host-only probes.
        self.optimization = optimization
        self.flags = tuple(flags)
        self.nvcc = nvcc or os.environ.get("NVCC") or shutil.which("nvcc") or "nvcc"
        self.resources = Path(os.path.abspath(resources or ROOT / "resources"))
        self.out_dir = PACKAGE / "cubins" / arch

    def compile(self) -> dict[str, str]:
        if self.arch == "sm_100a":
            return self._build_sm100()
        return self._build_sm86()

    # -- sm_100a: sources -----------------------------------------------------

    @property
    def flashinfer(self) -> Path:
        return self.resources / f"flashinfer-{FLASHINFER_V069}"

    def artifact(self, kind: str) -> Path:
        return self.resources / "trtllm-gen-artifacts" / ARTIFACTS[kind]["path"]

    def checksums(self, kind: str) -> dict[str, str]:
        """Verified ``checksums.txt``: file name -> SHA256."""
        path = self.artifact(kind) / "checksums.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; run scripts/fetch_resources.py --only trtllm-gen"
            )
        data = path.read_bytes()
        if _sha256(data) != ARTIFACTS[kind]["checksums_sha256"]:
            raise ValueError(f"{path}: SHA256 differs from FlashInfer's CheckSumHash")
        table = {}
        for line in data.decode().splitlines():
            digest, name = line.split()
            table[name] = digest
        for name, digest in table.items():
            if name.startswith("include/"):
                header = self.artifact(kind) / name
                if _sha256(header.read_bytes()) != digest:
                    raise ValueError(f"{header}: SHA256 differs from checksums.txt")
        return table

    def verify_runners(self) -> None:
        """The VERBATIM blocks of gemm_runners.cuh (dense runners) and
        bmm_moe_probe.cu (MoE launcher tile selection) equal the pinned
        sources."""
        for source in ("kernels/gemm_runners.cuh", "kernels/bmm_moe_probe.cu"):
            blocks = _verbatim_blocks((PACKAGE / source).read_text())
            if not blocks:
                raise ValueError(f"{source} has no VERBATIM blocks")
            for path, first, last, body in blocks:
                lines = (self.flashinfer / path).read_text().splitlines(keepends=True)
                if "".join(lines[first - 1 : last]) != body:
                    raise ValueError(f"{source} differs from {path}:{first}-{last}")

    def probe_command(self, kind: str, stage: Path, output: Path) -> list[str]:
        include = self.artifact(kind) / "include"
        common = self.flashinfer / "csrc/nv_internal/cpp/common"
        return [
            self.nvcc,
            self.optimization,
            "-std=c++17",
            "-w",
            *PROBE_DEFINES,
            "-gencode=arch=compute_100a,code=sm_100a",
            "--cudart=shared",
            "-I" + str(stage),
            "-I" + str(include),
            "-I" + str(self.flashinfer),
            "-I" + str(self.flashinfer / "include"),
            "-I" + str(self.flashinfer / "csrc/nv_internal"),
            "-I" + str(self.flashinfer / "csrc/nv_internal/include"),
            "-I" + str(self.resources / CUTLASS_DIR / "include"),
            *self.flags,
            *(str(PACKAGE / src) for src in ARTIFACTS[kind]["probe"]),
            *(str(common / name) for name in NV_INTERNAL_COMMON),
            "-o",
            str(output),
        ]

    def build_probe(self, kind: str, tmp: Path) -> Path:
        stage = tmp / f"stage_{kind}"
        link = stage / ARTIFACTS[kind]["stage"] / ARTIFACTS[kind]["export"]
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(self.artifact(kind) / "include" / ARTIFACTS[kind]["export"])
        probe = tmp / f"{kind}_probe"
        subprocess.run(
            self.probe_command(kind, stage, probe),
            check=True,
            cwd=ROOT,
            capture_output=True,
        )
        return probe

    @staticmethod
    def run_probe(probe: Path, mode: str, stdin: str = "") -> list[dict[str, Any]]:
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as out:
            subprocess.run(
                [str(probe), mode, out.name],
                input=stdin,
                text=True,
                check=True,
                capture_output=True,
            )
            return [
                json.loads(line) for line in Path(out.name).read_text().splitlines()
            ]

    # -- sm_100a: probes --------------------------------------------------------------

    @staticmethod
    def _launch_ok(record: dict[str, Any], function: str) -> bool:
        if record.get("error") or len(record.get("launches", ())) != 1:
            return False
        if record["workspace"]:
            return False
        calls = {e["call"] for e in record["events"]}
        if calls - {"cudaDeviceGetAttribute", "cuFuncSetAttribute"}:
            return False
        return bool(record["launches"][0]["kernel"] == function)

    @staticmethod
    def moe_line(
        case_id: str,
        entry: dict[str, Any],
        params: dict[str, Any],
        host: Any,
        record: int,
        act: str = "Swiglu",
    ) -> str:
        """A bmm_probe ``moe`` line; ``act`` is the FC1 activation an FC2
        config is paired with (FC1 configs use their own)."""
        buffers = ",".join(host.moe_buffers(entry["options"], params)) or "-"
        return (
            f"{case_id} {entry['index']} {act if entry['role'] == 'fc2' else '-'} "
            f"{params['tokens']} {params['top_k']} {params['experts']} "
            f"{params['hidden']} {params['intermediate']} {SM_COUNT} {buffers} {record}\n"
        )

    @staticmethod
    def gemm_line(case_id: str, entry: dict[str, Any], params: dict[str, Any],
                  record: int) -> str:  # fmt: skip
        buffers = ",".join(entry["buffers"]) or "-"
        return (
            f"{case_id} {entry['index']} {params['m']} {params['n']} {params['k']} "
            f"{SM_COUNT} {buffers} {record}\n"
        )

    def _probe_lines(
        self, probe: Path, mode: str, lines: list[str]
    ) -> list[dict[str, Any]]:
        records = self.run_probe(probe, mode, "".join(lines))
        if len(records) != len(lines):
            raise RuntimeError(f"{probe.name} returned {len(records)} of {len(lines)}")
        return records

    # -- sm_100a: MoE cases -------------------------------------------------------------

    @staticmethod
    def _moe_dispatch(entry: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        return {
            "launcher": record["launcher"],
            "selected_tiles": record["selected"],
            "tile_selected": entry["options"]["mTileN"] in record["selected"],
            "moe_valid": record["moe_valid"],
        }

    def _check_moe_record(self, entry, params, record, host) -> None:
        """The Python host mirror agrees with the probe's upstream code."""
        p = host.moe_problem(entry["options"], params)
        got = (record["n"], record["m"], record["k"], record["max_num_ctas"])
        if got != (p.m, p.n, p.k, p.max_num_ctas):
            raise ValueError(f"{entry['index']}: MoE runner problem {got} != {p}")
        if record["buffers"] != host.moe_buffers(entry["options"], params):
            raise ValueError(f"{entry['index']}: forwarded buffers {record['buffers']}")
        sel = host.selected_tiles(record["ladder"], params["tokens"], params["top_k"],
                                  params["experts"]) if record["ladder"] else []  # fmt: skip
        if sel != record["selected"]:
            raise ValueError(f"computeSelectedTileN port {sel} != {record['selected']}")

    def _moe_cases(
        self,
        probe: Path,
        entries: dict[str, dict[str, Any]],
        inventory: list[dict[str, Any]],
        host: Any,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
        """Screen every candidate with the probe and choose each MoE
        variant's cases: smoke (tiny, ragged with features, dispatch, deep),
        one exact upstream problem per regime of the upstream problems that
        may launch it, and model-shape throughput cases."""
        lines: list[str] = []
        keys: list[tuple[str, str, Any]] = []
        rotation = {name: i for i, name in enumerate(sorted(entries))}
        smoke = {n: moe_smoke_candidates(e, host) for n, e in entries.items()}
        model = {
            n: moe_throughput_candidates(e, rotation[n]) for n, e in entries.items()
        }
        upstream_params = [
            {"tokens": u["tokens"], "top_k": u["top_k"], "experts": u["experts"],
             "hidden": u["hidden"], "intermediate": u["inter"]}
            for u in inventory
        ]  # fmt: skip
        matches: dict[str, list[int]] = {n: [] for n in entries}
        for name, entry in entries.items():
            o = entry["options"]
            for i, (case, params, _) in enumerate(smoke[name]):
                lines.append(self.moe_line(f"s{len(keys)}", entry, params, host, 0))
                keys.append((name, "smoke", i))
            for i, (_, _, params, _) in enumerate(model[name]):
                lines.append(self.moe_line(f"t{len(keys)}", entry, params, host, 0))
                keys.append((name, "model", i))
            for i, u in enumerate(inventory):
                weights, act_dtype, ds = upstream.QUANT_DTYPES[u["quant"]]
                if (host.DTYPES[weights], host.DTYPES[act_dtype], ds) != (
                    o["mDtypeA"],
                    o["mDtypeB"],
                    bool(o["mUseDeepSeekFp8"]),
                ):
                    continue
                layout = (
                    host.LAYOUT_BLOCK_MAJOR_K if u["layout"] == "BlockMajorK" else 0
                )
                if (u["shuffled"], layout) != (
                    bool(o["mUseShuffledMatrix"]),
                    o["mLayoutA"],
                ):
                    continue
                if entry["role"] == "fc1" and host.moe_activation(o) != u["act"]:
                    continue
                params = {**upstream_params[i], **self._upstream_features(entry, u)}
                lines.append(
                    self.moe_line(f"u{len(keys)}", entry, params, host, 0, u["act"])
                )
                keys.append((name, "upstream", i))
        records = self._probe_lines(probe, "moe", lines)
        screened: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
            n: {"smoke": {}, "model": {}, "upstream": {}} for n in entries
        }
        for (name, kind, i), record in zip(keys, records):
            if not record["passing"]:
                raise ValueError(f"{name}: not among the MoE runner's passing configs")
            screened[name][kind][i] = record
        chosen: dict[str, list[dict[str, Any]]] = {}
        for name, entry in entries.items():
            o, tile = entry["options"], entry["options"]["mTileN"]
            entry["launcher"] = next(iter(screened[name]["smoke"].values()))["launcher"]
            entry["ladder"] = next(iter(screened[name]["smoke"].values()))["ladder"]
            cases: list[dict[str, Any]] = []
            names: set[str] = set()

            def add(case, params, suite, source, record):
                self._check_moe_record(entry, params, record, host)
                source = {**source, "dispatch": self._moe_dispatch(entry, record),
                          "regime": list(host.moe_regime(entry, params))}  # fmt: skip
                cases.append({"name": case, "params": params, "seed": 1 + len(cases),
                              "suite": suite, "source": source})  # fmt: skip
                names.add(case)

            for i, (case, params, source) in enumerate(smoke[name]):
                record = screened[name]["smoke"][i]
                if case in names or not record["valid"]:
                    continue
                if case == "smoke_dispatch" and not (
                    tile in record["selected"] and record["moe_valid"]
                    and host.tile_center(entry["ladder"], params["tokens"],
                                         params["top_k"], params["experts"]) == tile
                ):  # fmt: skip
                    continue
                add(case, params, "smoke", source, record)
            # Upstream problems that may launch this kernel: FlashInfer's tile
            # selection offers its tile (or the test forces every tile) and
            # the MoE runner pairs it with a valid config.
            groups: dict[str, list[tuple[int, int, dict[str, Any]]]] = {}
            for i, record in screened[name]["upstream"].items():
                u = inventory[i]
                if not (record["valid"] and record["moe_valid"]):
                    continue
                if not (u["all_tiles"] or tile in record["selected"]):
                    continue
                params = {**upstream_params[i], **self._upstream_features(entry, u)}
                matches[name].append(i)
                key = json.dumps(host.moe_regime(entry, params))
                groups.setdefault(key, []).append(
                    (moe_case_bytes(entry, params, host), i, params)
                )
            for j, key in enumerate(sorted(groups)):
                cost, i, params = min(groups[key], key=lambda g: (g[0], g[1]))
                suite = "smoke" if cost <= SMOKE_BUDGET else "throughput"
                prefix = "" if suite == "smoke" else "throughput_"
                add(f"{prefix}upstream_{j}", params, suite, {
                    "kind": "upstream_test", "test": inventory[i]["tests"][0],
                    "revision": upstream.REVISION, "inventory": i,
                    "upstream_problems_in_regime": len(groups[key]),
                }, screened[name]["upstream"][i])  # fmt: skip
            # Model shapes: decode, prefill (FlashInfer's tile selection must
            # offer this tile), stress when no prefill reaches 1024 tokens.
            best: dict[str, tuple[str, dict[str, Any], dict[str, Any], dict]] = {}
            for i, (slot, case, params, source) in enumerate(model[name]):
                record = screened[name]["model"][i]
                if slot in best or not (record["valid"]):
                    continue
                if moe_case_bytes(entry, params, host) > CASE_BUDGET:
                    continue
                selected = tile in record["selected"] and record["moe_valid"]
                if slot != "stress" and not selected:
                    continue
                if slot == "stress" and (
                    best.get("prefill") and best["prefill"][1]["tokens"] >= 1024
                ):
                    continue
                best[slot] = (case, params, source, record)
            for slot in ("decode", "prefill", "stress"):
                if slot in best:
                    case, params, source, record = best[slot]
                    if slot == "stress":
                        source = {**source, "kind": "model_shape",
                                  "stress": "large batch beyond FlashInfer's tile choice"}  # fmt: skip
                    add(case, params, "throughput", source, record)
            chosen[name] = cases
        return chosen, matches

    @staticmethod
    def _upstream_features(entry: dict[str, Any], u: dict[str, Any]) -> dict[str, int]:
        fc1 = entry["role"] == "fc1"
        out = {}
        if (u["bias1"] if fc1 else u["bias2"]) and host_supports(entry, "bias"):
            out["bias"] = 1
        if fc1 and u["routing_scales"] and host_supports(entry, "routing_scales"):
            out["routing_scales"] = 1
        if u["zero"]:
            out["zero"] = 1
        return out

    # -- sm_100a: dense cases -------------------------------------------------------------

    @staticmethod
    def _dense_ok(
        entry: dict[str, Any], record: dict[str, Any], autotune: bool = True
    ) -> bool:
        if not record["valid"]:
            return False
        if entry["role"] == "gemm_fp8_blockscale" or not autotune:
            return record["heuristic"] == entry["index"]
        return entry["index"] in record["tactics"]

    def _dense_cases(
        self,
        probe: Path,
        entries: dict[str, dict[str, Any]],
        inventory: list[dict[str, Any]],
        host: Any,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]]]:
        lines: list[str] = []
        keys: list[tuple[str, str, int]] = []
        rotation = {name: i for i, name in enumerate(sorted(entries))}
        smoke = {n: gemm_smoke_candidates(n, e, host) for n, e in entries.items()}
        model = {n: dense_throughput_candidates(rotation[n]) for n in entries}
        for name, entry in entries.items():
            for i, (_, params, _) in enumerate(smoke[name]):
                lines.append(self.gemm_line(f"s{len(keys)}", entry, params, 0))
                keys.append((name, "smoke", i))
            for i, (_, _, params, _) in enumerate(model[name]):
                lines.append(self.gemm_line(f"t{len(keys)}", entry, params, 0))
                keys.append((name, "model", i))
            for i, u in enumerate(inventory):
                if u["role"] != entry["role"]:
                    continue
                if (
                    u["sf_layout"] is not None
                    and u["sf_layout"] != entry["options"]["mSfLayoutB"]
                ):
                    continue
                params = {"m": u["m"], "n": u["n"], "k": u["k"]}
                lines.append(self.gemm_line(f"u{len(keys)}", entry, params, 0))
                keys.append((name, "upstream", i))
        records = self._probe_lines(probe, "run", lines)
        screened: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
            n: {"smoke": {}, "model": {}, "upstream": {}} for n in entries
        }
        for (name, kind, i), record in zip(keys, records):
            screened[name][kind][i] = record
        chosen: dict[str, list[dict[str, Any]]] = {}
        matches: dict[str, list[int]] = {n: [] for n in entries}
        for name, entry in entries.items():
            o = entry["options"]
            cases: list[dict[str, Any]] = []
            names: set[str] = set()

            def add(case, params, suite, source):
                cases.append({"name": case, "params": params, "seed": 1 + len(cases),
                              "suite": suite, "source": source})  # fmt: skip
                names.add(case)

            for i, (case, params, source) in enumerate(smoke[name]):
                record = screened[name]["smoke"][i]
                if case in names or not self._dense_ok(entry, record):
                    continue
                if case == "smoke_deep":
                    k_tiles = host.ceil_div(
                        params["k"], o["mTileK"] * o["mNumSlicesForSplitK"]
                    )
                    if k_tiles <= host.k_stages(o):
                        continue
                add(case, params, "smoke", source)
            seen = set()
            for i, record in screened[name]["upstream"].items():
                u = inventory[i]
                if not self._dense_ok(entry, record, u["autotune"]):
                    continue
                matches[name].append(i)
                shape = (u["m"], u["n"], u["k"])
                if shape in seen:
                    continue
                seen.add(shape)
                params = dict(zip("mnk", shape))
                suite = (
                    "smoke"
                    if dense_case_bytes(params) <= SMOKE_BUDGET // 2
                    else "throughput"
                )
                prefix = "" if suite == "smoke" else "throughput_"
                add(f"{prefix}upstream_m{shape[0]}_n{shape[1]}_k{shape[2]}", params, suite,
                    {"kind": "upstream_test", "test": u["tests"][0],
                     "revision": upstream.REVISION, "inventory": i})  # fmt: skip
            counts = {"decode": 0, "prefill": 0, "stress": 0}
            layers: set[tuple[int, int]] = set()
            for i, (slot, case, params, source) in enumerate(model[name]):
                record = screened[name]["model"][i]
                if counts[slot] >= (1 if slot == "decode" else 2) or case in names:
                    continue
                if slot == "stress" and counts["decode"] + counts["prefill"]:
                    continue
                if (params["n"], params["k"]) in layers and slot == "prefill":
                    continue
                if dense_case_bytes(params) > DENSE_BUDGET or not self._dense_ok(
                    entry, record
                ):
                    continue
                counts[slot] += 1
                layers.add((params["n"], params["k"]))
                add(case, params, "throughput", source)
            chosen[name] = cases
        return chosen, matches

    def _record(
        self,
        probe: Path,
        kind: str,
        entries: dict[str, dict[str, Any]],
        chosen: dict[str, list[dict[str, Any]]],
        host: Any,
    ) -> dict[str, dict[str, Any]]:
        """Record the launch of every chosen case (fixtures)."""
        lines, keys = [], []
        for name, cases in chosen.items():
            for case in cases:
                entry = entries[name]
                if kind == "bmm":
                    lines.append(
                        self.moe_line(f"r{len(keys)}", entry, case["params"], host, 1)
                    )
                else:
                    lines.append(
                        self.gemm_line(f"r{len(keys)}", entry, case["params"], 1)
                    )
                keys.append(f"{name}/{case['name']}")
        records = self._probe_lines(probe, "moe" if kind == "bmm" else "run", lines)
        fixtures = {}
        for key, record in zip(keys, records):
            if not self._launch_ok(record, key.split("/", 1)[0]):
                raise ValueError(f"{key}: probe launch {record.get('error', '')!r}")
            fixtures[key] = {k: record[k] for k in ("launches",)}
        return fixtures

    # -- sm_100a: build -------------------------------------------------------------

    def _build_sm100(self) -> dict[str, str]:
        workload, cubin_strip, host = _harness()
        checksums = {kind: self.checksums(kind) for kind in ARTIFACTS}
        self.verify_runners()
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        moe_inventory = merge_problems(upstream.moe_problems(self.resources))
        dense_inventory = merge_problems(upstream.dense_problems(self.resources))
        with tempfile.TemporaryDirectory(prefix="batched_gemm-") as tmp_name:
            tmp = Path(tmp_name)
            probes = {kind: self.build_probe(kind, tmp) for kind in ARTIFACTS}
            layouts = {
                kind: self.run_probe(probes[kind], "layout")[0] for kind in ARTIFACTS
            }
            configs = {
                kind: self.run_probe(probes[kind], "configs") for kind in ARTIFACTS
            }
            dead: dict[str, str] = {}
            entries: dict[str, dict[str, Any]] = {}
            for kind in ARTIFACTS:
                for config in configs[kind]:
                    self._plan(kind, config, checksums[kind], host, dead, entries)
            moe = {n: e for n, e in entries.items() if e["role"] in ("fc1", "fc2")}
            dense = {n: e for n, e in entries.items() if n not in moe}
            chosen, matches = self._moe_cases(probes["bmm"], moe, moe_inventory, host)
            c, m = self._dense_cases(probes["gemm"], dense, dense_inventory, host)
            chosen.update(c)
            matches.update(m)
            excluded: dict[str, str] = {}
            for name in entries:
                have = {case["name"] for case in chosen.get(name, [])}
                missing = {
                    "smoke_tiny",
                    "smoke_ragged" if name in moe else "smoke_rows",
                } - have
                if missing:
                    excluded[name] = (
                        "no probe-verified case shape found for "
                        + ", ".join(sorted(missing))
                    )
                    chosen.pop(name, None)
            fixtures = self._record(
                probes["bmm"], "bmm", moe, chosen_of(chosen, moe), host
            )
            fixtures.update(
                self._record(
                    probes["gemm"], "gemm", dense, chosen_of(chosen, dense), host
                )
            )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.out_dir.glob("*.cubin"):
            stale.unlink()
        mapping: dict[str, str] = {}
        variants: dict[str, dict[str, Any]] = {}
        records: dict[str, dict[str, Any]] = {}
        for name, entry in sorted(entries.items()):
            if name not in chosen:
                continue
            source = ROOT / entry.pop("source")
            image = source.read_bytes()
            stripped = cubin_strip.strip_cubin(image, name)
            cubin_strip.check_strip(image, stripped, name)
            arch, names, params = workload.cubin_info(stripped)
            layout = layouts["bmm" if entry["role"].startswith("fc") else "gemm"]
            if arch != "sm_100f" or names != [name]:
                raise ValueError(f"{name}: stripped cubin declares {arch} {names}")
            # The kernel parameter is sizeof(KernelParams) (batched), or the
            # fields rounded up to the CUtensorMap alignment (dense: 960 B on
            # the device while the host's sizeof is 1024); launches pass the
            # kernel's size, the tail is padding no host code sets.
            last = max(f["offset"] + f["size"] for f in layout["fields"].values())
            sizes = {layout["size"], -(-last // 64) * 64}
            if len(params) != 1 or params[0].size not in sizes:
                raise ValueError(f"{name}: kernel parameters {params} vs {sizes}")
            target = self.out_dir / f"{name}.cubin"
            target.write_bytes(stripped)
            mapping[name] = target.relative_to(PACKAGE).as_posix()
            entry["cases"] = chosen[name]
            entry["upstream"] = sorted(matches[name])
            entry["param_size"] = params[0].size
            variants[name] = entry
            records[name] = {
                "source": source.relative_to(ROOT).as_posix(),
                "source_sha256": _sha256(image),
                "stripped_sha256": _sha256(stripped),
                "config_index": entry["index"],
                "role": entry["role"],
            }
        fixture_rows = [
            {"key": key, **record}
            for key, record in sorted(fixtures.items())
            if key.split("/", 1)[0] in variants
        ]
        _lines_json(VARIANTS / "sm_100a.json", variants)
        _write_json(VARIANTS / "sm_100a_layouts.json", layouts)
        _write_json(
            VARIANTS / "upstream.json",
            {
                "revision": upstream.REVISION,
                "moe": moe_inventory,
                "dense": dense_inventory,
            },
            indent=None,
        )
        _write_gzip_lines(FIXTURES / "sm_100a.jsonl.gz", fixture_rows)
        _merge(MANIFEST, self.arch, dict(sorted(mapping.items())))
        roles: dict[str, int] = {}
        for entry in variants.values():
            roles[entry["role"]] = roles.get(entry["role"], 0) + 1
        self._merge_provenance(
            {
                "sources": {
                    kind: {
                        "artifact": "resources/trtllm-gen-artifacts/" + a["path"],
                        "checksums_sha256": a["checksums_sha256"],
                        "cubins": a["cubins"],
                        "flashinfer_cubin_wheel": "flashinfer_cubin 0.6.8/0.6.9",
                    }
                    for kind, a in ARTIFACTS.items()
                },
                "flashinfer_revision": FLASHINFER_V069,
                "host_code": [
                    "csrc/trtllm_batched_gemm_runner.cu",
                    "csrc/trtllm_fused_moe_runner.cu",
                    "csrc/trtllm_fused_moe_kernel_launcher.cu",
                    "csrc/trtllm_gemm_runner.cu",
                    "csrc/trtllm_low_latency_gemm_runner.cu",
                    "flashinfer/gemm/gemm_base.py",
                ],
                "probes": {
                    src: _sha256((PACKAGE / src).read_bytes()) for src in PROBE_SOURCES
                },
                "probe_commands": {
                    kind: _portable(
                        self.probe_command(kind, Path("stage"), Path(f"{kind}_probe"))
                    )
                    for kind in ARTIFACTS
                },
                "nvcc": version.strip().splitlines()[-1],
                "strip": "harness.cubin_strip.strip_cubin (+ check_strip): deletes "
                "<kernel>GetSmemSize",
                "sm_count": SM_COUNT,
                "registered": dict(sorted(roles.items())),
                "dead": dict(sorted(dead.items())),
                "excluded": dict(sorted(excluded.items())),
                "case_policy": CASE_POLICY,
                "upstream_coverage": coverage_summary(
                    variants, moe_inventory, dense_inventory
                ),
                "unreachable_registered": unreachable(variants),
                "kernels": records,
            }
        )
        return mapping

    def _plan(
        self,
        kind: str,
        config: dict[str, Any],
        checksums: dict[str, str],
        host: Any,
        dead: dict[str, str],
        entries: dict[str, dict[str, Any]],
    ) -> None:
        function = config["function"]
        o = config["options"]
        cubin = ARTIFACTS[kind]["cubins"] + "/" + function[0].upper() + function[1:]
        cubin += ".cubin"
        digest = checksums.get(Path(cubin).name)
        path = ROOT / cubin
        if digest is None or not path.is_file():
            raise FileNotFoundError(f"{cubin} missing or not in checksums.txt")
        data = path.read_bytes()
        if _sha256(data) != digest or config["sha256"] != digest:
            raise ValueError(f"{cubin}: SHA256 differs from checksums.txt / meta-info")
        if config["sm"] != 2:  # trtllm::gen::CudaArch::Sm100f
            raise ValueError(f"{function}: unexpected arch {config['sm']}")
        options = {k: o[k] for k in sorted(o) if k in HOST_OPTIONS}
        if kind == "bmm":
            if o["mIsStaticBatch"]:
                dead[function] = "static batch: no FlashInfer caller"
                return
            role = host.moe_role(o)
            d = host.DTYPES
            if not o["mTransposeMmaOutput"] or not o["mEnablesEarlyExit"]:
                raise ValueError(f"{function}: unexpected batched options")
            if role == "fc1" and o["mDtypeC"] != o["mDtypeB"]:
                dead[function] = "routed config with C != B dtype: no MoE caller"
                return
            if role == "fc2" and (
                o["mDtypeC"] != d["Bfloat16"] or o["mFusedAct"] or o["mEltwiseActType"]
            ):
                dead[function] = "unrouted config FC2 never builds (C must be BF16)"
                return
            # PermuteGemm1/Gemm2::getOptions: epilogueTileM = DeepSeek ? 64 : 128
            if o["mEpilogueTileM"] != (64 if o["mUseDeepSeekFp8"] else 128):
                dead[function] = "epilogue tile differs from the MoE runner's"
                return
            buffers = host.moe_buffers(o)
        else:
            maybe_role = gemm_role(function, o, host)
            if maybe_role is None:
                dead[function] = (
                    "E4m3 split-K/shuffled config: only reachable through an "
                    "explicit tactic index, which no FlashInfer caller passes"
                )
                return
            role = maybe_role
            buffers = list(host.GEMM_BUFFERS[role])
        entries[function] = {
            "role": role,
            "index": config["index"],
            "source": cubin,
            "shared_mem": config["shared_mem"],
            "threads": config["threads"],
            "buffers": buffers,
            "options": options,
        }

    def _merge_provenance(self, build: dict[str, Any]) -> None:
        provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.is_file() else {}
        builds = provenance.get("builds", {})
        builds[self.arch] = build
        _write_json(
            PROVENANCE,
            {
                "package": "batched_gemm",
                "license": "LICENSE (FlashInfer, Apache-2.0); trtllm-gen export "
                "headers and cubins: NVIDIA, Apache-2.0; CUTLASS_LICENSE "
                "(BSD-3-Clause)",
                "scope": "sm_100a: trtllm-gen batched GEMM (MoE FC1/FC2) and dense "
                "GEMM cubins of FlashInfer v0.6.9, one workload per live kernel; "
                "sm_86: FlashInfer CUTLASS segment GEMM with row-major weights",
                "builds": dict(sorted(builds.items())),
            },
        )

    # -- sm_86 -----------------------------------------------------------------------

    def segment_probe_command(self, output: Path) -> list[str]:
        return [
            self.nvcc,
            "-std=c++17",
            self.optimization,
            "-gencode=arch=compute_80,code=sm_80",
            "--expt-relaxed-constexpr",
            "-diag-suppress=20012",
            "-I" + str(self.resources / CUTLASS_DIR / "include"),
            *self.flags,
            str(PACKAGE / SEGMENT_PROBE),
            "-o",
            str(output),
        ]

    def _build_sm86(self) -> dict[str, str]:
        workload, cubin_strip, _ = _harness()
        source_path = self.resources / JIT_CACHE_CUBIN
        source = source_path.read_bytes()
        if _sha256(source) != JIT_CACHE_SHA256:
            raise ValueError(f"{JIT_CACHE_CUBIN}: unexpected SHA256")
        kernels = cubin_strip.cubin_kernels(source)
        if len(kernels) != 8:
            raise ValueError(f"expected 8 GemmGrouped kernels, found {len(kernels)}")
        row_major = [k for k in kernels if ROW_MAJOR_B in k and COLUMN_MAJOR_B not in k]
        if len(row_major) != 4:
            raise ValueError(f"expected 4 row-major-weight kernels, found {row_major}")
        version = subprocess.check_output([self.nvcc, "--version"], text=True)
        with tempfile.TemporaryDirectory(prefix="batched_gemm-") as tmp_name:
            tmp = Path(tmp_name)
            probe = tmp / "segment_gemm_rowmajor_probe"
            subprocess.run(self.segment_probe_command(probe), check=True, cwd=ROOT)
            report = json.loads(subprocess.check_output([str(probe)], text=True))
            subprocess.run(
                ["cuobjdump", "-xelf", "all", str(probe)],
                check=True,
                cwd=tmp,
                capture_output=True,
            )
            probe_kernels = {
                name
                for image in tmp.glob("*.cubin")
                for name in cubin_strip.cubin_kernels(image.read_bytes())
            }
        if set(row_major) - probe_kernels:
            raise ValueError("the probe's kernels differ from the JIT-cache kernels")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        mapping: dict[str, str] = {}
        variants: dict[str, dict[str, Any]] = {}
        records: dict[str, Any] = {}
        cases = segment_cases(self.resources)
        for name, (key, dtype, pipelined) in SEGMENT_KERNELS.items():
            (symbol,) = [
                k
                for k in row_major
                if f"NS_{len(dtype)}{dtype}E" in k
                and ("12MmaPipelined" in k) == pipelined
            ]
            layout = report[key]
            stripped = cubin_strip.strip_cubin(source, symbol)
            cubin_strip.check_strip(source, stripped, symbol)
            arch, names, params = workload.cubin_info(stripped)
            if arch != "sm_80" or names != [symbol]:
                raise ValueError(f"{name}: stripped cubin {arch} {names}")
            if [p.size for p in params] != [layout["params_size"]]:
                raise ValueError(f"{name}: Params size differs from the probe")
            target = self.out_dir / f"{name}.cubin"
            target.write_bytes(stripped)
            sidecar = {
                "workload": name,
                "arch": self.arch,
                "cubin": target.name,
                "cubin_sha256": _sha256(stripped),
                "symbol": symbol,
                **{k: v for k, v in layout.items() if k != "examples"},
                "threadblock_count": 4,
            }
            _write_json(self.out_dir / f"{name}.json", sidecar, indent=2)
            _write_json(
                self.out_dir / f"{name}.fixtures.json",
                {"workload": name, "examples": layout["examples"]},
                indent=2,
            )
            mapping[name] = target.relative_to(PACKAGE).as_posix()
            variants[name] = {
                "dtype": "bf16" if dtype == "bfloat16_t" else "fp16",
                "mainloop": "MmaPipelined" if pipelined else "MmaMultistage",
                "stages": layout["stages"],
                "upstream_dispatch": (
                    "sm_86/sm_89 (< 147968 B shared memory per SM)"
                    if pipelined
                    else "sm_80/sm_87 (>= 147968 B shared memory per SM)"
                ),
                "sidecar": f"cubins/{self.arch}/{name}.json",
                "cases": cases,
            }
            records[name] = {
                "cubin": mapping[name],
                "sha256": _sha256(stripped),
                "symbol": symbol,
                "sidecar": f"cubins/{self.arch}/{name}.json",
            }
        _lines_json(VARIANTS / "sm_86.json", variants)
        _merge(MANIFEST, self.arch, dict(sorted(mapping.items())))
        self._merge_provenance(
            {
                "source": {
                    "wheel": "flashinfer_jit_cache_sm80-0.7.0+cu130-cp39-abi3-"
                    "manylinux_2_28_x86_64.whl (scripts/fetch_resources.py)",
                    "cubin": f"resources/{JIT_CACHE_CUBIN} (from gemm/gemm.so)",
                    "cubin_sha256": JIT_CACHE_SHA256,
                    "flashinfer_revision": FLASHINFER_V070,
                    "host_code": "include/flashinfer/gemm/group_gemm.cuh "
                    "(CutlassSegmentGEMMRun), csrc/group_gemm.cu, "
                    "flashinfer/gemm/gemm_base.py (SegmentGEMMWrapper, sm80 "
                    "backend, weight_column_major=False), flashinfer/triton/gemm.py "
                    "(compute_sm80_group_gemm_args)",
                    "cutlass": f"resources/{CUTLASS_DIR}",
                },
                "strip": "harness.cubin_strip.strip_cubin (+ check_strip)",
                "probe": SEGMENT_PROBE,
                "probe_sha256": _sha256((PACKAGE / SEGMENT_PROBE).read_bytes()),
                "probe_command": _portable(
                    self.segment_probe_command(Path("segment_gemm_rowmajor_probe"))
                ),
                "nvcc": version.strip().splitlines()[-1],
                "live_dead": {
                    "B RowMajor MmaPipelined (bf16, fp16)": "live: upstream's "
                    "choice when the SM has < 147968 B shared memory (sm_86)",
                    "B RowMajor MmaMultistage (bf16, fp16)": "live on sm_80/sm_87 "
                    "(>= 147968 B); registered for sm_86, where its sm_80 SASS "
                    "(64 KiB shared memory) runs unchanged",
                    "B ColumnMajor (4 kernels)": "owned by the deep_gemm package",
                },
                "cases": {
                    "smoke": sum(c["suite"] == "smoke" for c in cases),
                    "throughput": sum(c["suite"] == "throughput" for c in cases),
                    "upstream_test": f"tests/gemm/test_group_gemm.py::test_segment_gemm "
                    f"(FlashInfer {upstream.REVISION}): sm80 backend, row-major "
                    "weights, FP16; every problem is an exact case of every kernel "
                    "(use_weight_indices only changes the weight pointers)",
                    "models": [list(m) for m in SEGMENT_MODELS],
                },
                "kernels": records,
            }
        )
        return mapping


# Options the Python host mirror and the references read.
HOST_OPTIONS = frozenset("""
    mActType mBiasType mBlockK mClampBeforeAct mClusterDimX mClusterDimY
    mClusterDimZ mDtypeA mDtypeB mDtypeC mEltwiseActType mEnablesDelayedEarlyExit
    mEnablesEarlyExit mEpilogueTileM mEpilogueTileN mFusedAct
    mGridWaitForPrimaryA mGridWaitForPrimaryB mGridWaitForPrimaryEarlyExit
    mGridWaitForPrimaryRouting mIsStaticBatch mLayoutA mLayoutB mMmaKind
    mMmaTileK mNumSlicesForSplitK mNumStages mNumStagesA mNumStagesB
    mNumStagesMma mRouteImpl mRouteSfsImpl mSfBlockSizeA
    mSfBlockSizeB mSfBlockSizeC mSfLayoutA mSfLayoutB mSfLayoutC
    mSfReshapeFactor mSliceK mSparsityA mSplitK mTileK mTileM mTileN
    mTileScheduler mTransposeMmaOutput mUseDeepSeekFp8 mUsePerTokenSfA
    mUsePerTokenSfB mUseShuffledMatrix mUseTmaOobOpt mUseTmaStore
    """.split())

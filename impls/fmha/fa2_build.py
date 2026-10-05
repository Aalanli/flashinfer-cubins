"""sm_86 build of the fmha package: FlashInfer FA2 kernels of the sm80 JIT cache.

Source: ``flashinfer_jit_cache_sm80-0.7.0+cu130`` (FlashInfer 0.7.0, revision
``aa7c67f2``), extracted by ``scripts/fetch_resources.py --only jit-cache`` into
``resources/flashinfer-jit-cache-sm80/<module>/<module>.<n>.sm_80.cubin``
(``index.json`` records the shared object and SHA256 of every cubin). sm_80
SASS runs unchanged on sm_86; the planners and dispatchers below are evaluated
for the RTX 3090 (82 SMs, 100 KiB shared memory per SM, 99 KiB per block).

Every live kernel is cut out of its module cubin with
``harness.cubin_strip.strip_cubin`` (verified by ``check_strip``) into
``cubins/sm_86/<workload>.cubin`` and registered by
``harness/workloads/fmha/fa2.py`` from the compact index
``variants/sm_86.json``. Host-only probes compiled against the
``*_config.inc`` the JIT templates render for these modules
(``kernels/fa2_probe_*.cu``) record the by-value ``Params``
layouts, reference structs built by the upstream host code, planner results
and the shared-memory dispatch; they go to ``fixtures/sm_86.json`` and the
tests compare the Python port byte for byte.

Kernel inventory and live/dead decisions (sm_86):

=================================  ======  =====================================
module / kernel                    status  reason
=================================  ======  =====================================
batch_prefill_with_kv_cache (48    live    ``DISPATCH_NUM_MMA_KV`` in
modules: {bf16, f16} Q x {16-bit,          ``BatchPrefillWith{Paged,Ragged}
e4m3} KV, head dim {64, 128, 256},         KVCacheDispatched`` selects exactly
sliding window, soft cap) cubins           one ``NUM_MMA_KV`` per
.1-.8: ``BatchPrefillWith{Paged,           ``CTA_TILE_Q`` from the sm_86 shared
Ragged}KVCacheKernel`` for mask            memory budget (probe ``dispatch``);
{none, causal, custom,                     that kernel is live for every
multi-item} x ``CTA_TILE_Q``               ``CTA_TILE_Q`` ``FA2DetermineCtaTileQ``
{16, 64, 128} x ``NUM_MMA_KV``             returns (16, 64; 128 only below
                                           head dim 256, where it also fits).
  other ``NUM_MMA_KV``             dead    only selected with more shared memory
                                           per SM (sm_80/sm_90) or never.
  ``CTA_TILE_Q`` 128, head dim     --      not instantiated (``FA2DetermineCtaTileQ``
  256                                      never returns it).
  ragged, multi-item scoring       dead    ``BatchPrefillWithRaggedKVCacheWrapper
  (cubin .8)                               .run`` only passes masks 0-2 (the
                                           ragged kernel has no multi-item path).
  ``PersistentVariableLength       live*   only launched after a split-KV
  MergeStatesKernel`` copies               prefill/decode; identical in every
  (also in the decode and sink             module, registered once per
  modules)                                 (dtype, head dim) from the first
                                           module (sorted) that holds it; the
                                           head-dim-512 instance is dispatched
                                           by none of these modules (dead).
  .9/.10                           --      no kernel.
batch_prefill_with_attention_sink  live    ``BatchAttentionWithAttentionSinkWrapper``
(4 modules, head dim 64)                   (the AOT instance of the template it
                                           compiles): mask none/causal, as the
                                           prefill rows above.
  custom/multi-item masks          dead    ``AttentionSink`` has no custom mask or
                                           multi-item support.
  ``use_swa_True`` modules         dup     byte-identical code to the
                                           ``use_swa_False`` twin (the variant
                                           reads ``window_left`` at run time):
                                           removed; the twin serves windowed and
                                           unwindowed cases (``code_hash``; the
                                           build fails on any other identical
                                           pair).
batch_decode_with_kv_cache (48     live    ``NUM_STAGES_SMEM = 2``: compute
modules) .1: ``BatchDecodeWith             capability >= 8 (sm_86); one kernel
PagedKVCacheKernel`` for GQA               per GQA group size {1, 2, 3, 4, 6, 8}.
group size {1,2,3,4,6,8} x
``NUM_STAGES_SMEM`` {1, 2}
  ``NUM_STAGES_SMEM = 1``          dead    compute capability < 8 (sm_75) only.
  .2/.3                            --      no kernel.
batch_attention_with_kv_cache      live    ``flashinfer.BatchAttention``: both
(36 modules) .1-.4:                        attention runners and the state
``PersistentKernelTemplate``               reduction in one cooperative launch
per mask; .5/.6: no kernel                 (``TwoStageHolisticPlan``); live for
                                           head dim 64, mask none/causal, Q and
                                           KV of the same type or FP8 KV.
  custom/multi-item masks          dead    ``BatchAttention.plan`` passes causal
                                           or non-causal only.
  head dim 128                     dead    the cooperative grid (2 CTAs per SM)
                                           needs 2 x 66-70 KB of shared memory.
  head dim 256                     dead    > 99 KiB of shared memory per block.
  bf16 Q / f16 KV, f16 Q / bf16 KV dead    KV reaches the MMA as Q's type
                                           without conversion (measured).
batch_mla_attention (bf16, f16)    live    ``DISPATCH_SMEM_CONFIG`` on sm_86: 1
.2: ``BatchMLAPagedAttentionKernel``       stage, ``CTA_TILE_KV`` 16, no QK
per causal x configuration;                sharding; causal and non-causal; one
.1/.3: no kernel                           cooperative launch (``MLAPlan``).
  2-stage configurations           dead    only with >= 147968 B of shared memory
                                           per SM (sm_80/sm_90).
cascade .1: ``MergeState``,        live    ``flashinfer.cascade`` merge APIs,
``MergeStateInPlace``,                     ``DISPATCH_HEAD_DIM`` {64, 128, 256,
``MergeStates``,                           512}: ``vec_size`` 8 serves head dims
``MergeStatesLargeNumIndexSets``           64-256, 16 serves 512; the large
kernels, {bf16, f16}                       variant is chosen when
                                           ``num_index_sets >= seq_len``.
=================================  ======  =====================================

Probes (``kernels/fa2_probe_*.cu``, compiled against the configs the JIT
templates render): ``prefill`` (also built with ``-DFA2_PROBE_SINK`` for the
sink modules; non-split, split-KV and CUDA-graph Params, and
``PrefillSplitQOKVIndptr`` for every plan option), ``decode`` (Params of
every plan kind, ``DecodePlanImpl`` below the occupancy query, decode and
merge launch arithmetic), ``persistent`` (``TwoStageHolisticPlan``, with
``v_scale``) and ``mla`` (``MLAPlan``). Decode kernels also record
``blocks_per_sm``: ``cuOccupancyMaxActiveBlocksPerMultiprocessor`` modelled
from the cubin's register count and shared memory (``blocks_per_sm``), the
grid ``DecodePlan`` sizes splits by; native launches assert the driver agrees.

Upstream tests: ``fa2_upstream.py`` maps the pinned FlashInfer tree's tests
that launch these kernels on sm_80/86 (batch prefill paged/ragged/custom
mask, tensor-core decode, batch-invariant, sliding window, packed inputs,
LSE base, KV scales, shared-prefix cascade, CUDA-core decode incl. CUDA
graphs, BatchAttention, DeepSeek MLA, the cascade merges) to launches; every
(kernel, regime) of them gets its cheapest parametrization as an upstream case
(``fixtures/sm_86_upstream.json``; counts in ``provenance.json``), and
``tests/test_fmha.py`` re-derives and checks the mapping.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ARCH = "sm_86"
FLASHINFER_V070 = "aa7c67f2b876b89be34c7a70ac022369a56e60d5"
JIT_DIR = "flashinfer-jit-cache-sm80"
# name -> (source, rendered-config set, extra defines)
PROBES = {
    "prefill": ("kernels/fa2_probe_prefill.cu", "default", ()),
    "prefill_sink": ("kernels/fa2_probe_prefill.cu", "sink", ("-DFA2_PROBE_SINK",)),
    "decode": ("kernels/fa2_probe_decode.cu", "default", ()),
    "persistent": ("kernels/fa2_probe_persistent.cu", "persistent", ()),
    "mla": ("kernels/fa2_probe_mla.cu", "mla", ()),
}
PROBE_COMMON = "kernels/fa2_probe_common.h"
# The probes' sm_86 device model (fa2_probe_common.h).
NUM_SMS = 82
MAX_SMEM_PER_BLOCK_OPTIN = 101376
DTYPES = {"__nv_bfloat16": "bf16", "__half": "f16", "__nv_fp8_e4m3": "e4m3"}
MASKS = {0: "none", 1: "causal", 2: "custom", 3: "multiitem"}
MODULE_RE = re.compile(
    r"dtype_q_(?P<q>\w+?)_dtype_kv_(?P<kv>\w+?)_dtype_o_(?P<o>\w+?)_dtype_idx_i32"
    r"_head_dim_qk_(?P<dqk>\d+)_head_dim_vo_(?P<dvo>\d+)_posenc_0"
    r"_use_swa_(?P<swa>True|False)_use_logits_cap_(?P<cap>True|False)"
)
PERSISTENT_RE = re.compile(
    r"batch_attention_with_kv_cache_dtype_q_(?P<q>\w+?)_dtype_kv_(?P<kv>\w+?)_dtype_o_(?P<o>\w+?)"
    r"_dtype_idx_i32_head_dim_qk_(?P<dqk>\d+)_head_dim_vo_(?P<dvo>\d+)_posenc_0"
    r"_use_logits_soft_cap_(?P<cap>true|false)_use_profiler_false"
)
# sm_86: 100 KiB of shared memory per SM, 1 KiB of it reserved per resident block.
MAX_SMEM_PER_SM = 102400
RESERVED_SMEM_PER_BLOCK = 1024
SINK_RE = re.compile(
    r"attention_sink_kv_cache_dtype_q_(?P<q>\w+?)_dtype_kv_(?P<kv>\w+?)_dtype_o_(?P<o>\w+?)"
    r"_dtype_idx_i32_head_dim_qk_(?P<dqk>\d+)_head_dim_vo_(?P<dvo>\d+)_use_swa_(?P<swa>True|False)_"
)


def _compiler_module(compiler: Any) -> Any:
    return sys.modules[type(compiler).__module__]


def _harness(root: Path) -> tuple[Any, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from harness import cubin_strip, workload

    return workload, cubin_strip


# -- demangled symbol parsing ---------------------------------------------------


def template_args(text: str) -> list[str]:
    """Top-level template arguments of the first ``<...>`` in ``text``."""
    start = text.index("<")
    depth, args, current = 0, [], ""
    for ch in text[start:]:
        if ch == "<":
            depth += 1
            if depth == 1:
                continue
        elif ch == ">":
            depth -= 1
            if depth == 0:
                args.append(current.strip())
                return args
        elif ch == "," and depth == 1:
            args.append(current.strip())
            current = ""
            continue
        current += ch
    raise ValueError(f"unbalanced template arguments: {text}")


def _int(arg: str) -> int:
    return int(re.sub(r"^\([\w: ]+\)", "", arg))


def _dtype(arg: str) -> str:
    return DTYPES[arg.replace("flashinfer::", "")]


def parse_kernel(demangled: str) -> dict[str, Any]:
    """Template arguments of one FA2 kernel (by kernel family)."""
    name = re.match(r"void (?:\w+::)*(\w+)<", demangled)
    if name is None:
        raise ValueError(f"unexpected kernel {demangled}")
    family = name.group(1)
    args = template_args(demangled)
    if family in (
        "BatchPrefillWithPagedKVCacheKernel",
        "BatchPrefillWithRaggedKVCacheKernel",
    ):
        paged = family == "BatchPrefillWithPagedKVCacheKernel"
        traits = template_args(args[1] if paged else args[0])
        sink = traits[14].replace("flashinfer::", "") == "AttentionSink"
        variant: str | list[bool] = (
            "sink" if sink else [bool(_int(v)) for v in template_args(traits[14])]
        )
        return {
            "family": "prefill",
            "kind": "paged" if paged else "ragged",
            "same_kv_strides": bool(_int(args[0])) if paged else None,
            "mask": _int(traits[0]),
            "cta_tile_q": _int(traits[1]),
            "num_mma_q": _int(traits[2]),
            "num_mma_kv": _int(traits[3]),
            "head_dim_qk": 16 * _int(traits[4]),
            "head_dim_vo": 16 * _int(traits[5]),
            "num_warps_q": _int(traits[6]),
            "num_warps_kv": _int(traits[7]),
            "pos_encoding": _int(traits[8]),
            "dtype_q": _dtype(traits[9]),
            "dtype_kv": _dtype(traits[10]),
            "dtype_o": _dtype(traits[11]),
            "variant": variant,
        }
    if family == "BatchMLAPagedAttentionKernel":
        traits = template_args(args[0])
        return {
            "family": "mla",
            "causal": bool(_int(traits[0])),
            "num_stages": _int(traits[1]),
            "qk_shard": bool(_int(traits[2])),
            "head_dim_ckv": _int(traits[3]),
            "head_dim_kpe": _int(traits[4]),
            "cta_tile_q": _int(traits[5]),
            "cta_tile_kv": _int(traits[6]),
            "dtype_q": _dtype(traits[7]),
            "dtype_kv": _dtype(traits[8]),
            "dtype_o": _dtype(traits[9]),
        }
    if family == "PersistentKernelTemplate":
        runners = [template_args(a) for a in args[:2]]
        runner_traits = [template_args(r[0]) for r in runners]
        standard = template_args(runner_traits[0][14])
        return {
            "family": "persistent",
            "mask": _int(runner_traits[0][0]),
            "cta_tile_q": [_int(t[1]) for t in runner_traits],
            "num_mma_kv": [_int(t[3]) for t in runner_traits],
            "head_dim_qk": 16 * _int(runner_traits[0][4]),
            "head_dim_vo": 16 * _int(runner_traits[0][5]),
            "dtype_q": _dtype(runner_traits[0][9]),
            "dtype_kv": _dtype(runner_traits[0][10]),
            "dtype_o": _dtype(runner_traits[0][11]),
            "softcap": bool(_int(standard[0])),
        }
    if family == "BatchDecodeWithPagedKVCacheKernel":
        default = template_args(args[7])
        return {
            "family": "decode",
            "pos_encoding": _int(args[0]),
            "num_stages_smem": _int(args[1]),
            "tile_size_per_bdx": _int(args[2]),
            "vec_size": _int(args[3]),
            "bdx": _int(args[4]),
            "bdy": _int(args[5]),
            "bdz": _int(args[6]),
            "variant": [bool(_int(v)) for v in default],
        }
    if family == "PersistentVariableLengthMergeStatesKernel":
        return {
            "family": "merge_varlen",
            "vec_size": _int(args[0]),
            "bdx": _int(args[1]),
            "bdy": _int(args[2]),
            "num_smem_stages": _int(args[3]),
            "dtype_in": _dtype(args[4]),
            "dtype_o": _dtype(args[5]),
        }
    if family in ("MergeStateKernel", "MergeStatesKernel"):
        return {
            "family": "merge_state" if family == "MergeStateKernel" else "merge_states",
            "vec_size": _int(args[0]),
            "dtype_in": _dtype(args[1]),
            "dtype_o": _dtype(args[2]),
        }
    if family == "MergeStateInPlaceKernel":
        return {
            "family": "merge_state_in_place",
            "vec_size": _int(args[0]),
            "dtype_in": _dtype(args[1]),
            "dtype_o": _dtype(args[1]),
        }
    if family == "MergeStatesLargeNumIndexSetsKernel":
        return {
            "family": "merge_states_large",
            "vec_size": _int(args[0]),
            "bdx": _int(args[1]),
            "bdy": _int(args[2]),
            "num_smem_stages": _int(args[3]),
            "dtype_in": _dtype(args[4]),
            "dtype_o": _dtype(args[5]),
        }
    raise ValueError(f"unexpected kernel family {family}")


def demangle(nvcc: str, symbols: Iterable[str]) -> dict[str, str]:
    symbols = list(symbols)
    filt = Path(nvcc).resolve().parent / "cu++filt"
    tool = str(filt) if filt.is_file() else (shutil.which("cu++filt") or "cu++filt")
    out = subprocess.run(
        [tool], input="\n".join(symbols), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    if len(out) != len(symbols):
        raise RuntimeError("cu++filt output does not match its input")
    return dict(zip(symbols, out))


# -- upstream dispatch on sm_86 ---------------------------------------------------


def cta_tile_q_values(head_dim: int, kv_bytes: int) -> list[int]:
    """Every value ``FA2DetermineCtaTileQ`` (utils.cuh) can return on sm_86
    for a symmetric head dim <= 256, by ``avg_packed_qo_len`` range."""
    values = [64]
    q_tile_smem = 16 * head_dim * 2
    kv_step_smem = 2 * head_dim * 16 * 4 * kv_bytes
    if q_tile_smem + kv_step_smem <= MAX_SMEM_PER_BLOCK_OPTIN:
        values.append(16)
    if head_dim < 256:
        values.append(128)
    return sorted(values)


def merge_config(dtype: str, head_dim: int) -> dict[str, int]:
    """VariableLengthMergeStates / MergeStatesLargeNumIndexSets template
    arguments for ``head_dim`` (cascade.cuh)."""
    size = 2  # bf16 / f16
    vec = max(16 // size, head_dim // 32)
    bdx = head_dim // vec
    return {"vec_size": vec, "bdx": bdx, "bdy": 128 // bdx, "num_smem_stages": 4}


# -- config rendering and probes ------------------------------------------------


def _upstream_module(flashinfer: Path, relative: str) -> Any:
    """Load one self-contained upstream Python file by path."""
    import importlib.util

    path = flashinfer / relative
    name = "fmha_upstream_" + Path(relative).stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream_utils(flashinfer: Path) -> Any:
    return _upstream_module(flashinfer, "flashinfer/jit/attention/utils.py")


def render_configs(flashinfer: Path, out: Path) -> dict[str, str]:
    """``batch_{prefill,decode}_config.inc`` exactly as the JIT generators
    (``flashinfer/jit/attention/modules.py``: ``_gen_batch_prefill_module``
    with the fa2 backend and ``paged_kv_stride_mode="equal"``,
    ``gen_batch_decode_module``, ``gen_batch_prefill_attention_sink_module``)
    render them, for bf16 Q/KV/O, head dim 128, no sliding window and no soft
    cap, into ``out/<set>/``. The Params layouts do not depend on these
    choices (the dtype fields are pointers); the probes evaluate the dispatch
    per dtype themselves. Returns {``<set>/<file>``: text}."""
    import jinja2

    utils = _upstream_utils(flashinfer)
    common = {
        "variant_decl": "#include<flashinfer/attention/variants.cuh>",
        "dtype_q": "nv_bfloat16",
        "dtype_kv": "nv_bfloat16",
        "dtype_o": "nv_bfloat16",
        "idtype": "int32_t",
        "head_dim_qk": 128,
        "head_dim_vo": 128,
        "pos_encoding_mode": "PosEncodingMode::kNone",
        "use_sliding_window": "false",
        "use_logits_soft_cap": "false",
    }
    decl, func, setter = utils.generate_additional_params(
        [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
            "maybe_k_cache_sf",
            "maybe_v_cache_sf",
        ],
        [
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
            "uint8_t",
            "uint8_t",
        ],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ],
        ["double", "double", "double", "double", "int64_t"],
    )
    prefill = jinja2.Template(
        (flashinfer / "csrc/batch_prefill_customize_config.jinja").read_text()
    ).render(
        **common,
        additional_params_decl=decl,
        additional_func_params=func,
        additional_params_setter=setter,
        variant_name="DefaultAttention<use_custom_mask, false, false, false>",
        use_fp16_qk_reduction="false",
        paged_kv_stride_mode="equal",
        require_fp4_kv_cache=False,
    )
    decl, func, setter = utils.generate_additional_params(
        ["maybe_alibi_slopes"],
        ["float"],
        ["logits_soft_cap", "sm_scale", "rope_rcp_scale", "rope_rcp_theta"],
        ["double", "double", "double", "double"],
    )
    decode = jinja2.Template(
        (flashinfer / "csrc/batch_decode_customize_config.jinja").read_text()
    ).render(
        **common,
        additional_params_decl=decl,
        additional_func_params=func,
        additional_params_setter=setter,
        variant_name="DefaultAttention<false, false, false, false>",
    )
    variants = _upstream_module(flashinfer, "flashinfer/jit/attention/variants.py")
    decl, func, setter = utils.generate_additional_params(
        ["sink"], ["float"], ["sm_scale"], ["double"]
    )
    sink = jinja2.Template(
        (flashinfer / "csrc/batch_prefill_customize_config.jinja").read_text()
    ).render(
        **{**common, "variant_decl": variants.attention_sink_decl["fa2"]},
        additional_params_decl=decl,
        additional_func_params=func,
        additional_params_setter=setter,
        variant_name="AttentionSink",
        use_fp16_qk_reduction="false",
        paged_kv_stride_mode="independent",
        require_fp4_kv_cache=False,
    )
    # gen_batch_attention_module / gen_customize_batch_attention_module (no
    # profiler): k/v cache scale-factor pointers are the only additional params.
    decl, func, _ = utils.generate_additional_params(
        ["maybe_k_cache_sf", "maybe_v_cache_sf"], ["uint8_t", "uint8_t"], [], []
    )
    persistent = jinja2.Template(
        (flashinfer / "csrc/batch_attention_customize_config.jinja").read_text()
    ).render(
        variant_decl="#include<flashinfer/attention/variants.cuh>",
        variant_name="StandardAttention<false>",
        dtype_q="nv_bfloat16",
        dtype_kv="nv_bfloat16",
        dtype_o="nv_bfloat16",
        idtype="int32_t",
        head_dim_qk=128,
        head_dim_vo=128,
        pos_encoding_mode="PosEncodingMode::kNone",
        use_logits_soft_cap="false",
        additional_params_decl=decl,
        additional_func_params=func,
        additional_params_setter="",
    )
    # gen_batch_mla_module (fa2 backend).
    mla = jinja2.Template(
        (flashinfer / "csrc/batch_mla_config.jinja").read_text()
    ).render(
        dtype_q="nv_bfloat16",
        dtype_kv="nv_bfloat16",
        dtype_o="nv_bfloat16",
        dtype_idx="int32_t",
        head_dim_ckv=512,
        head_dim_kpe=64,
    )
    rendered = {
        "mla/batch_mla_config.inc": mla,
        "default/batch_prefill_config.inc": prefill,
        "default/batch_decode_config.inc": decode,
        "sink/batch_prefill_config.inc": sink,
        "persistent/batch_attention_config.inc": persistent,
    }
    for name, text in rendered.items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(text)
    return rendered


def probe_command(
    compiler: Any,
    source: Path,
    include: Path,
    output: Path,
    defines: tuple[str, ...] = (),
) -> list[str]:
    flashinfer = compiler.resources / f"flashinfer-{FLASHINFER_V070}"
    return [
        compiler.nvcc,
        "-std=c++17",
        compiler.optimization,
        "-x",
        "cu",
        "-gencode=arch=compute_80,code=sm_80",
        "--expt-relaxed-constexpr",
        "-diag-suppress=20012",
        *defines,
        "-I" + str(include),
        "-I" + str(flashinfer / "include"),
        *compiler.flags,
        str(source),
        "-o",
        str(output),
    ]


def run_probes(
    compiler: Any, package: Path, root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    """({probe name: report}, rendered configs)."""
    flashinfer = compiler.resources / f"flashinfer-{FLASHINFER_V070}"
    reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="fmha-fa2-") as tmp:
        include = Path(tmp) / "gen"
        rendered = render_configs(flashinfer, include)
        for name, (source, config, defines) in PROBES.items():
            binary = Path(tmp) / f"fa2_probe_{name}"
            command = probe_command(
                compiler, package / source, include / config, binary, defines
            )
            subprocess.run(command, check=True, cwd=root, capture_output=True)
            reports[name] = json.loads(
                subprocess.run(
                    [str(binary)], check=True, capture_output=True, text=True
                ).stdout
            )
    return reports, rendered


# -- the build ------------------------------------------------------------------


def module_config(module: str) -> dict[str, Any] | None:
    sink = SINK_RE.search(module)
    match = sink or MODULE_RE.search(module)
    if match is None:
        return None
    return {
        "dtype_q": match["q"],
        "dtype_kv": match["kv"],
        "dtype_o": match["o"],
        "head_dim": int(match["dqk"]),
        "head_dim_vo": int(match["dvo"]),
        "swa": match["swa"] == "True",
        "softcap": False if sink else match["cap"] == "True",
        "sink": sink is not None,
    }


def layout_name(entry: dict[str, Any]) -> str:
    """The probe layout of a prefill/decode variant's Params."""
    if entry["family"] in ("decode", "persistent", "mla"):
        return str(entry["family"])
    return entry["kind"] + ("_sink" if entry.get("sink") else "")


DUPLICATE_SINK_SWA = (
    "sink prefill: the use_swa module's kernel is byte-identical to the non-swa "
    "module's (AttentionSink applies window_left at run time); the non-swa name "
    "serves both windowed and unwindowed cases"
)


def code_hash(workload: Any, image: bytes) -> str:
    """SHA256 of a one-kernel cubin's code: the ``.text`` section plus the
    kernel's info, shared-memory and constant sections (names excluded)."""
    import hashlib

    sections = workload._sections(image)
    parts = [v for k, v in sorted(sections.items()) if k.startswith(".text.")]
    parts += [
        v
        for k, v in sorted(sections.items())
        if k.startswith((".nv.info.", ".nv.shared.", ".nv.constant"))
    ]
    return hashlib.sha256(b"|".join(parts)).hexdigest()


EIATTR_REGCOUNT = 0x2F
# sm_86 per-SM limits (CUDA occupancy calculator): blocks, warps, registers
# (allocated per warp in units of 256, in 4 sub-partitions), shared memory
# (1 KiB reserved per block, 128-byte granularity).
MAX_BLOCKS_PER_SM, MAX_WARPS_PER_SM, REGS_PER_SM = 16, 48, 65536


def kernel_resources(image: bytes) -> tuple[int, int]:
    """(registers per thread, static shared memory bytes) of a one-kernel
    cubin: EIATTR_REGCOUNT of ``.nv.info`` and the size of its
    ``.nv.shared.<kernel>`` (NOBITS) section."""
    import struct

    shoff = struct.unpack_from("<Q", image, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 0x3A)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", image, shoff + i * shentsize)
        for i in range(shnum)
    ]
    names = image[headers[shstrndx][4] : headers[shstrndx][4] + headers[shstrndx][5]]
    regs, static = None, 0
    for name, _, _, _, offset, size, *_ in headers:
        label = names[name : names.index(b"\0", name)].decode()
        if label.startswith(".nv.shared."):
            static += size
        if label != ".nv.info":
            continue
        info, pos = image[offset : offset + size], 0
        while pos + 4 <= len(info):
            if info[pos] != 4:  # EIFMT_SVAL
                pos += 4
                continue
            length = struct.unpack_from("<H", info, pos + 2)[0]
            if info[pos + 1] == EIATTR_REGCOUNT:
                regs = struct.unpack_from("<I", info, pos + 8)[0]
            pos += 4 + length
    if regs is None:
        raise ValueError("no EIATTR_REGCOUNT")
    return regs, static


def blocks_per_sm(threads: int, regs: int, shared_mem: int) -> int:
    """``cuOccupancyMaxActiveBlocksPerMultiprocessor`` on sm_86 (checked
    against the driver at every native decode launch)."""
    warps = -(-threads // 32)
    regs_per_warp = -(-regs * 32 // 256) * 256
    # Registers are split over the 4 SM sub-partitions; a warp's registers
    # live in one of them.
    by_regs = 4 * (REGS_PER_SM // 4 // regs_per_warp) // warps
    smem = -(-(shared_mem + RESERVED_SMEM_PER_BLOCK) // 128) * 128
    by_smem = MAX_SMEM_PER_SM // smem
    return min(MAX_BLOCKS_PER_SM, MAX_WARPS_PER_SM // warps, by_regs, by_smem)


def upstream_table(helpers: Any, root: Path, resources: Path) -> dict[str, Any]:
    """The upstream-test representatives of every kernel (``fa2_upstream``),
    evaluated with the harness planners over the freshly written index."""
    import importlib.util
    from collections import Counter

    fa2 = helpers.load_workload_module(root, "fa2")
    fa2.load_index.cache_clear()
    fa2.probe.cache_clear()
    path = Path(__file__).with_name("fa2_upstream.py")
    spec = importlib.util.spec_from_file_location("impls_fmha_fa2_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    resolver = upstream.Resolver(fa2)
    counts = Counter(
        t.split("[")[0] for _, _, t in upstream.launches(resources, fa2, resolver)
    )
    return {
        "flashinfer_revision": FLASHINFER_V070,
        "launches_per_test": dict(sorted(counts.items())),
        # Launches whose kernel output is not attention (fa2.causal_mask_skipped):
        # these upstream parametrizations fail on sm_86 and are not cases.
        "defective_launches_per_test": dict(sorted(resolver.defects.items())),
        "kernels": upstream.representatives(resources, fa2),
    }


def _flags(swa: bool, softcap: bool) -> str:
    return ("_swa" if swa else "") + ("_softcap" if softcap else "")


def build(compiler: Any) -> dict[str, str]:
    helpers = _compiler_module(compiler)
    package: Path = helpers.PACKAGE
    root: Path = helpers.ROOT
    workload, cubin_strip = _harness(root)
    jit = compiler.resources / JIT_DIR
    index = json.loads((jit / "index.json").read_text())
    reports, rendered = run_probes(compiler, package, root)
    prefill_report, decode_report = reports["prefill"], reports["decode"]

    # Every kernel of every module this package serves.
    entries: list[tuple[str, str, str]] = []  # (cubin path, module, symbol)
    for relative in sorted(index["cubins"]):
        module = relative.split("/", 1)[0]
        if not (
            module.startswith(
                (
                    "batch_prefill_with_kv_cache_",
                    "batch_prefill_with_attention_sink_kv_cache_",
                    "batch_decode_with_kv_cache_",
                    "batch_mla_attention_",
                )
            )
            or PERSISTENT_RE.search(module) is not None
            or module == "cascade"
        ):
            continue
        data = (jit / relative).read_bytes()
        if helpers.sha256_bytes(data) != index["cubins"][relative]["sha256"]:
            raise ValueError(f"{relative}: SHA256 differs from index.json")
        for symbol in cubin_strip.cubin_kernels(data):
            entries.append((relative, module, symbol))
    names = demangle(compiler.nvcc, sorted({e[2] for e in entries}))

    dispatch = {
        (d["kind"], d["dtype_q"], d["dtype_kv"], d["head_dim"], d["cta_tile_q"]): d
        for d in prefill_report["dispatch"]
    }
    decode_dispatch = {
        (d["dtype_q"], d["dtype_kv"], d["head_dim"], d["group_size"]): d
        for d in decode_report["dispatch"]
    }
    mla_dispatch = {d["dtype"]: d for d in reports["mla"]["dispatch"]}
    persistent_dispatch = {
        (d["dtype_q"], d["dtype_kv"], d["head_dim"]): d
        for d in reports["persistent"]["dispatch"]
    }

    live: dict[str, dict[str, Any]] = {}  # workload -> variant entry (+ source)
    dead: dict[str, int] = {}
    seen_merge: set[str] = set()

    def mark_dead(reason: str) -> None:
        dead[reason] = dead.get(reason, 0) + 1

    for relative, module, symbol in entries:
        info = parse_kernel(names[symbol])
        family = info["family"]
        source = {"cubin": relative, "symbol": symbol}
        if family == "prefill":
            cfg = module_config(module)
            assert cfg is not None
            kv_bytes = 1 if cfg["dtype_kv"] == "e4m3" else 2
            if (
                info["dtype_kv"] != cfg["dtype_kv"]
                or info["head_dim_qk"] != cfg["head_dim"]
            ):
                raise ValueError(f"{relative}: kernel {info} disagrees with its module")
            expected_variant = (
                "sink"
                if cfg["sink"]
                else [info["mask"] == 2, cfg["swa"], cfg["softcap"], False]
            )
            if info["variant"] != expected_variant:
                raise ValueError(f"{relative}: unexpected variant {info['variant']}")
            selected = dispatch[
                (
                    info["kind"],
                    cfg["dtype_q"],
                    cfg["dtype_kv"],
                    cfg["head_dim"],
                    info["cta_tile_q"],
                )
            ]
            if info["kind"] == "ragged" and info["mask"] == 3:
                mark_dead("prefill: ragged multi-item scoring never dispatched")
                continue
            if cfg["sink"] and info["mask"] >= 2:
                mark_dead(
                    "sink prefill: AttentionSink has no custom-mask or multi-item support"
                )
                continue
            if info["cta_tile_q"] not in cta_tile_q_values(cfg["head_dim"], kv_bytes):
                mark_dead("prefill: CTA_TILE_Q never chosen on sm_86")
                continue
            if not selected["fits"] or selected["num_mma_kv"] != info["num_mma_kv"]:
                mark_dead("prefill: NUM_MMA_KV not selected on sm_86")
                continue
            if (
                selected["num_warps_q"],
                selected["num_warps_kv"],
                selected["num_mma_q"],
            ) != (
                info["num_warps_q"],
                info["num_warps_kv"],
                info["num_mma_q"],
            ):
                raise ValueError(f"{relative}: warp layout differs from the dispatcher")
            prefix = "sink" if cfg["sink"] else "prefill"
            name = (
                f"fmha_fa2_{prefix}_{info['kind']}_{cfg['dtype_q']}_{cfg['dtype_kv']}"
                f"_hd{cfg['head_dim']}_{MASKS[info['mask']]}"
                f"{_flags(cfg['swa'], cfg['softcap'])}_cta{info['cta_tile_q']}"
            )
            entry = {
                "family": "prefill",
                "kind": info["kind"],
                "dtype_q": cfg["dtype_q"],
                "dtype_kv": cfg["dtype_kv"],
                "dtype_o": cfg["dtype_o"],
                "head_dim": cfg["head_dim"],
                "mask": info["mask"],
                "swa": cfg["swa"],
                "softcap": cfg["softcap"],
                "sink": cfg["sink"],
                "cta_tile_q": info["cta_tile_q"],
                "num_mma_kv": info["num_mma_kv"],
                "block": [32, info["num_warps_q"], info["num_warps_kv"]],
                "shared_mem": selected["shared_mem"],
            }
        elif family == "mla":
            selected = mla_dispatch[info["dtype_kv"]]
            if (info["num_stages"], info["cta_tile_kv"], info["qk_shard"]) != (
                selected["num_stages"],
                selected["cta_tile_kv"],
                selected["qk_shard"],
            ):
                mark_dead(
                    "mla: DISPATCH_SMEM_CONFIG selects this configuration only with "
                    ">= 147968 B of shared memory per SM (sm_80/sm_90)"
                )
                continue
            if (info["head_dim_ckv"], info["head_dim_kpe"], info["cta_tile_q"]) != (
                512,
                64,
                64,
            ):
                raise ValueError(f"{relative}: unexpected MLA traits {info}")
            name = f"fmha_fa2_mla_{info['dtype_q']}_{'causal' if info['causal'] else 'none'}"
            entry = {
                "family": "mla",
                "dtype_q": info["dtype_q"],
                "dtype_kv": info["dtype_kv"],
                "dtype_o": info["dtype_o"],
                "head_dim_ckv": 512,
                "head_dim_kpe": 64,
                "causal": info["causal"],
                "block": selected["block"],
                "shared_mem": selected["shared_mem"],
            }
        elif family == "persistent":
            match = PERSISTENT_RE.search(module)
            assert match is not None
            if (
                info["dtype_q"],
                info["dtype_kv"],
                info["head_dim_qk"],
                info["softcap"],
            ) != (
                match["q"],
                match["kv"],
                int(match["dqk"]),
                match["cap"] == "true",
            ):
                raise ValueError(f"{relative}: kernel {info} disagrees with its module")
            selected = persistent_dispatch[
                (info["dtype_q"], info["dtype_kv"], info["head_dim_qk"])
            ]
            if (
                info["cta_tile_q"] != selected["cta_tile_q"]
                or info["num_mma_kv"] != selected["num_mma_kv"]
            ):
                raise ValueError(f"{relative}: tiles differ from the launcher")
            if info["mask"] >= 2:
                mark_dead(
                    "persistent: custom/multi-item mask never dispatched "
                    "(BatchAttention.plan: causal or not)"
                )
                continue
            if info["dtype_kv"] not in (info["dtype_q"], "e4m3"):
                # Measured on sm_86: outputs equal attention over the KV bits
                # reinterpreted as Q's type.
                mark_dead(
                    "persistent: 16-bit KV of another type than Q reaches the MMA "
                    "unconverted (not attention of the inputs)"
                )
                continue
            smem = selected["shared_mem"]
            ctas_per_sm = 1 if info["head_dim_qk"] >= 256 else 2  # TwoStageHolisticPlan
            if smem > MAX_SMEM_PER_BLOCK_OPTIN:
                mark_dead(
                    "persistent: needs more than 99 KiB of shared memory per block"
                )
                continue
            if ctas_per_sm * (smem + RESERVED_SMEM_PER_BLOCK) > MAX_SMEM_PER_SM:
                mark_dead(
                    "persistent: the cooperative grid (2 CTAs per SM) does not fit "
                    "sm_86 shared memory"
                )
                continue
            name = (
                f"fmha_fa2_persistent_{info['dtype_q']}_{info['dtype_kv']}"
                f"_hd{info['head_dim_qk']}_{MASKS[info['mask']]}"
                f"{'_softcap' if info['softcap'] else ''}"
            )
            entry = {
                "family": "persistent",
                "dtype_q": info["dtype_q"],
                "dtype_kv": info["dtype_kv"],
                "dtype_o": info["dtype_o"],
                "head_dim": info["head_dim_qk"],
                "mask": info["mask"],
                "softcap": info["softcap"],
                "block": [selected["num_threads"], 1, 1],
                "shared_mem": smem,
            }
        elif family == "decode":
            cfg = module_config(module)
            assert cfg is not None
            if info["num_stages_smem"] != 2:
                mark_dead("decode: NUM_STAGES_SMEM = 1 (compute capability < 8)")
                continue
            if info["variant"] != [False, cfg["swa"], cfg["softcap"], False]:
                raise ValueError(f"{relative}: unexpected variant {info['variant']}")
            group = info["bdy"]
            selected = decode_dispatch[
                (cfg["dtype_q"], cfg["dtype_kv"], cfg["head_dim"], group)
            ]
            for key in (
                "tile_size_per_bdx",
                "vec_size",
                "bdx",
                "bdz",
                "num_stages_smem",
            ):
                if selected[key] != info[key]:
                    raise ValueError(f"{relative}: {key} differs from the dispatcher")
            name = (
                f"fmha_fa2_decode_{cfg['dtype_q']}_{cfg['dtype_kv']}_hd{cfg['head_dim']}"
                f"_g{group}{_flags(cfg['swa'], cfg['softcap'])}"
            )
            entry = {
                "family": "decode",
                "dtype_q": cfg["dtype_q"],
                "dtype_kv": cfg["dtype_kv"],
                "dtype_o": cfg["dtype_o"],
                "head_dim": cfg["head_dim"],
                "group_size": group,
                "swa": cfg["swa"],
                "softcap": cfg["softcap"],
                "block": [info["bdx"], info["bdy"], info["bdz"]],
                "shared_mem": selected["shared_mem"],
            }
        elif family == "merge_varlen":
            head_dim = info["vec_size"] * info["bdx"]
            if merge_config(info["dtype_in"], head_dim) != {
                k: info[k] for k in ("vec_size", "bdx", "bdy", "num_smem_stages")
            }:
                raise ValueError(f"{relative}: unexpected merge configuration {info}")
            cfg = module_config(module)
            assert cfg is not None
            if head_dim != cfg["head_dim"]:
                mark_dead("merge_varlen: head dim not dispatched by its module")
                continue
            if symbol in seen_merge:
                mark_dead("merge_varlen: duplicate of an identical registered kernel")
                continue
            seen_merge.add(symbol)
            name = f"fmha_fa2_merge_varlen_{info['dtype_in']}_hd{head_dim}"
            entry = {
                "family": "merge_varlen",
                "dtype_in": info["dtype_in"],
                "dtype_o": info["dtype_o"],
                "head_dim": head_dim,
                "block": [info["bdx"], info["bdy"], 1],
                "shared_mem": 4 * info["bdy"] * head_dim * 2 + 128 * 4,
            }
        else:  # cascade
            if info["family"] == "merge_states_large":
                head_dim = info["vec_size"] * info["bdx"]
                if merge_config(info["dtype_in"], head_dim) != {
                    k: info[k] for k in ("vec_size", "bdx", "bdy", "num_smem_stages")
                }:
                    raise ValueError(
                        f"{relative}: unexpected merge configuration {info}"
                    )
                name = f"fmha_fa2_{info['family']}_{info['dtype_in']}_hd{head_dim}"
                entry = {
                    "family": info["family"],
                    "dtype_in": info["dtype_in"],
                    "dtype_o": info["dtype_o"],
                    "head_dims": [head_dim],
                    "block": [info["bdx"], info["bdy"], 1],
                    "shared_mem": 4 * info["bdy"] * head_dim * 2 + 128 * 4,
                }
            else:
                vec = info["vec_size"]
                dims = [d for d in (64, 128, 256, 512) if max(8, d // 32) == vec]
                name = f"fmha_fa2_{info['family']}_{info['dtype_in']}_vec{vec}"
                entry = {
                    "family": info["family"],
                    "dtype_in": info["dtype_in"],
                    "dtype_o": info["dtype_o"],
                    "head_dims": dims,
                    "vec_size": vec,
                    "shared_mem": 0,
                }
        if name in live:
            raise ValueError(f"duplicate workload {name}")
        live[name] = {**entry, "source": source}

    # Strip and write.
    out_dir = package / "cubins" / ARCH
    if out_dir.is_dir():
        for old in out_dir.glob("*.cubin"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    records: dict[str, Any] = {}
    cache: dict[str, bytes] = {}
    layouts = {
        "decode": decode_report["params"],
        "paged": prefill_report["paged"],
        "ragged": prefill_report["ragged"],
        "paged_sink": reports["prefill_sink"]["paged"],
        "ragged_sink": reports["prefill_sink"]["ragged"],
        "persistent": reports["persistent"]["params"],
        "mla": reports["mla"]["params"],
    }
    code_owner: dict[str, str] = {}  # code hash -> first workload with that code
    duplicates: dict[str, str] = {}  # removed workload -> kept twin
    for name, entry in sorted(live.items()):
        source = entry.pop("source")
        if source["cubin"] not in cache:
            cache.clear()
            cache[source["cubin"]] = (jit / source["cubin"]).read_bytes()
        original = cache[source["cubin"]]
        stripped = cubin_strip.strip_cubin(original, source["symbol"])
        cubin_strip.check_strip(original, stripped, source["symbol"])
        arch, kernels, params = workload.cubin_info(stripped)
        if arch != "sm_80" or kernels != [source["symbol"]]:
            raise ValueError(f"{name}: stripped cubin is {arch} {kernels}")
        digest = code_hash(workload, stripped)
        if digest in code_owner:
            kept = code_owner[digest]
            # AttentionSink reads window_left at run time in both module
            # flavours, so the use_swa module compiles to the same kernel.
            if not (entry.get("sink") and name.replace("_swa_", "_") == kept):
                raise ValueError(f"{name}: kernel code identical to {kept}")
            duplicates[name] = kept
            mark_dead(DUPLICATE_SINK_SWA)
            continue
        code_owner[digest] = name
        family = entry["family"]
        if family == "decode":
            regs, static = kernel_resources(stripped)
            entry["blocks_per_sm"] = blocks_per_sm(
                entry["block"][0] * entry["block"][1] * entry["block"][2],
                regs,
                entry["shared_mem"] + static,
            )
        if family in ("prefill", "decode", "persistent", "mla"):
            layout = layouts[layout_name(entry)]
            if family == "persistent":
                layout_params = [layout["param_size"]] * 2
            else:
                layout_params = [layout["param_size"]]
            if [p.size for p in params] != layout_params:
                raise ValueError(f"{name}: kernel parameters {params} != Params")
        target = out_dir / f"{name}.cubin"
        target.write_bytes(stripped)
        mapping[name] = target.relative_to(package).as_posix()
        entry["params"] = [p.size for p in params]
        records[name] = {
            "source": f"resources/{JIT_DIR}/{source['cubin']}",
            "source_sha256": index["cubins"][source["cubin"]]["sha256"],
            "shared_object": index["cubins"][source["cubin"]]["member"],
            "symbol": source["symbol"],
            "sha256": helpers.sha256_bytes(stripped),
        }

    for name in duplicates:
        del live[name]
    helpers.write_json(
        helpers.VARIANTS / f"{ARCH}.json",
        {
            "arch": ARCH,
            "layouts": layouts,
            "kernels": dict(sorted(live.items())),
        },
        indent=None,
    )
    helpers.write_json(
        package / "fixtures" / f"{ARCH}.json",
        {
            "prefill": {
                k: prefill_report[k] for k in ("examples", "plans", "dispatch")
            },
            "prefill_sink": {"examples": reports["prefill_sink"]["examples"]},
            "persistent": {
                k: reports["persistent"][k] for k in ("examples", "dispatch")
            },
            "mla": {k: reports["mla"][k] for k in ("examples", "dispatch")},
            "decode": {
                k: decode_report[k] for k in ("examples", "plans", "dispatch", "merge")
            },
        },
    )
    upstream = upstream_table(helpers, root, compiler.resources)
    helpers.write_json(
        package / "fixtures" / f"{ARCH}_upstream.json", upstream, indent=None
    )
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
            "library": "FlashInfer FA2 (sm80 JIT cache)",
            "source": {
                "wheel": index["wheel"],
                "wheel_sha256": index["sha256"],
                "extracted": f"resources/{JIT_DIR} (scripts/fetch_resources.py --only jit-cache)",
                "flashinfer_revision": FLASHINFER_V070,
                "host_code": [
                    (
                        "include/flashinfer/attention/prefill.cuh "
                        "(BatchPrefillWith{Paged,Ragged}KVCacheDispatched)"
                    ),
                    (
                        "include/flashinfer/attention/decode.cuh "
                        "(BatchDecodeWithPagedKVCacheDispatched)"
                    ),
                    (
                        "include/flashinfer/attention/scheduler.cuh "
                        "(PrefillPlan, PrefillSplitQOKVIndptr, DecodePlan)"
                    ),
                    "include/flashinfer/attention/cascade.cuh",
                    (
                        "csrc/batch_prefill.cu, csrc/batch_prefill_paged.cuh, csrc/batch_decode.cu, "
                        "csrc/cascade.cu"
                    ),
                    "flashinfer/prefill.py, flashinfer/decode.py, flashinfer/cascade.py",
                ],
            },
            "strip": "harness.cubin_strip.strip_cubin (+ check_strip)",
            "probes": {
                name: {
                    "source": source,
                    "sha256": helpers.sha256_file(package / source),
                    "config": config,
                    "command": [
                        helpers.rel(Path(p)) if p.startswith("/") else p
                        for p in probe_command(
                            compiler,
                            package / source,
                            Path(f"<gen>/{config}"),
                            Path(f"fa2_probe_{name}"),
                            defines,
                        )
                    ],
                }
                for name, (source, config, defines) in PROBES.items()
            },
            "probe_common_sha256": helpers.sha256_file(package / PROBE_COMMON),
            "rendered_configs_sha256": {
                name: helpers.sha256_bytes(text.encode())
                for name, text in rendered.items()
            },
            "nvcc": version,
            "dead": dict(sorted(dead.items())),
            "upstream_tests": {
                "source": f"resources/flashinfer-{FLASHINFER_V070}/tests",
                "mapping": "impls/fmha/fa2_upstream.py (TESTS, adapters, exclusions)",
                "cases": f"fixtures/{ARCH}_upstream.json",
                "launches_per_test": upstream["launches_per_test"],
                "defective_launches_per_test": upstream["defective_launches_per_test"],
                "kernels": len(upstream["kernels"]),
                "regimes": sum(len(rows) for rows in upstream["kernels"].values()),
            },
            "duplicates_removed": {
                "reason": DUPLICATE_SINK_SWA,
                "kept": dict(sorted(duplicates.items())),
            },
            "kernels": records,
        },
    )
    return mapping

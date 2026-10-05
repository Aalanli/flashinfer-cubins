"""FlashInfer v0.6.9 test parametrizations that reach the trtllm-gen FMHA kernels.

``upstream_params()`` enumerates every parametrization of the pinned tests
(``resources/flashinfer-<rev>/tests``) that launches a trtllm-gen FMHA kernel
on a B200, after the tests' own ``pytest.skip`` conditions, with the runner
fields FlashInfer's wrappers and launchers derive from it:

* ``attention/test_trtllm_gen_attention.py``: ``test_trtllm_batch_prefill``
  (+ ``_bs1``), ``test_trtllm_batch_decode`` (``backend="trtllm-gen"``),
  ``_bs1``, ``_head_dim_256``, ``_long_sequence_length``,
  ``test_trtllm_batch_decode_spec``, ``test_trtllm_gen_prefill`` (+ ``_bs1``,
  ragged ``trtllm_ragged_attention_deepseek``);
* ``attention/test_trtllm_gen_mla.py``: ``test_trtllm_batch_decode_mla``
  (``backend="trtllm-gen"``), ``_sparse``, ``_preallocated_out``;
* ``attention/test_attention_sink_blackwell.py``: decode and context sinks;
* ``attention/test_trtllm_ragged_kv_stride.py``: K/V numel > 2^31.

Sequence lengths are reproduced with the tests' seeds and CPU generator calls
where the tests draw them on the CPU (paged decode/prefill, MLA). The ragged
tests draw them on the CUDA generator; there the same calls on the CPU give
lengths of the same distribution, and the maxima (which the tests pass as
``max_q_len``/``max_kv_len`` and which select the kernel) are exact.

Each parametrization becomes ``{"id", "test", "dtypes", "hd", "layout",
"case", "flags"}``: ``case`` holds the harness case fields (shapes, mask,
runner modes, window, skip threshold, paging, feature flags), ``flags`` the
runtime features that do not affect kernel selection (KV layout, sinks,
shared page indices, non-contiguous query, device scales, separate K/V
tensors, uniform or cumulative query lengths, NVFP4 scale-factor offset).
PDL on/off is not a kernel input (launch attribute only) and is folded.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch

FLASHINFER = "a1aa676196f798435248d9ea205c67674476f473"
ATTN = "tests/attention/test_trtllm_gen_attention.py"
MLA = "tests/attention/test_trtllm_gen_mla.py"
SINK = "tests/attention/test_attention_sink_blackwell.py"
STRIDE = "tests/attention/test_trtllm_ragged_kv_stride.py"

# QkvLayout / mask / kernel type / scheduler values (fmhaRunnerParams.h).
SEPARATE_QKV, PAGED_KV = 0, 2
DENSE, CAUSAL = 0, 1
CONTEXT, GENERATION = 0, 1
STATIC, PERSISTENT = 0, 1
DT = {"bf16": "bf16", "fp16": "fp16", "fp8": "e4m3", "nvfp4": "e2m1"}
FLAG_KEYS = (
    "kv_layout",
    "sinks",
    "shared_idx",
    "q_noncontig",
    "device_scales",
    "kv_tuple",
    "cum_q",
    "sf_start",
)
# Runner fields of FlashInfer's launchers.
PAGED_CONTEXT = {"mask": CAUSAL, "ktype": CONTEXT, "sched": PERSISTENT, "mcta": 0}
PAGED_DECODE = {"mask": CAUSAL, "ktype": GENERATION, "sched": STATIC, "mcta": 1}


def _ragged_runner(causal: bool) -> dict[str, int]:
    return {
        "mask": CAUSAL if causal else DENSE,
        "ktype": CONTEXT,
        "sched": PERSISTENT,
        "mcta": 0,
    }


def _ints(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.flatten().tolist()]


def _prefill_lens(batch: int, max_q: int, max_in_kv: int) -> tuple[list, list]:
    """generate_seq_lens_prefill after torch.manual_seed(0)."""
    torch.manual_seed(0)
    q = torch.randint(1, max_q + 1, (batch,), dtype=torch.int32)
    q[-1] = max_q
    in_kv = torch.randint(0, max_in_kv + 1, (batch,), dtype=torch.int)
    in_kv[-1] = max_in_kv
    return _ints(q), _ints(q + in_kv)


def _decode_lens(
    batch: int, q_len: int | None, max_in_kv: int, max_q: int | None
) -> tuple[list, list]:
    """generate_seq_lens_decode after torch.manual_seed(0)."""
    torch.manual_seed(0)
    if q_len is not None:
        q = torch.full((batch,), q_len, dtype=torch.int32)
    else:
        q = torch.randint(1, max_q + 1, (batch,), dtype=torch.int32)
    in_kv = torch.randint(0, max_in_kv + 1, (batch,), dtype=torch.int)
    in_kv[-1] = max_in_kv
    return _ints(q), _ints(q + in_kv)


def _flags(**kw: Any) -> dict[str, Any]:
    flags = {
        "kv_layout": "HND",
        "sinks": False,
        "shared_idx": True,
        "q_noncontig": False,
        "device_scales": False,
        "kv_tuple": False,
        "cum_q": False,
        "sf_start": False,
    }
    flags.update(kw)
    return flags


def _entry(
    test: str,
    func: str,
    params: dict[str, Any],
    dtypes: tuple[str, str, str],
    hd: tuple[int, int],
    layout: int,
    case: dict[str, Any],
    flags: dict[str, Any],
) -> dict[str, Any]:
    label = ",".join(f"{k}={v}" for k, v in params.items())
    return {
        "id": f"{func}[{label}]",
        "test": f"{test}::{func}[{label}]",
        "dtypes": [DT.get(d, d) for d in dtypes],
        "hd": list(hd),
        "layout": layout,
        "case": case,
        "flags": flags,
    }


def _paged_case(
    *,
    hq: int,
    hkv: int,
    q_lens: list[int],
    kv_lens: list[int],
    page: int,
    runner: dict[str, int],
    window_left: int,
    skip: bool,
    bmm1: float,
    max_q: int | None = None,
    shared_kv: bool = False,
    topk: int = 0,
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "hq": hq,
        "hkv": hkv,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "window_left": window_left,
        "skip_thr": 1e-30 if skip else 0.0,
        "bmm1_scale": bmm1,
        "bmm2_scale": 1.0,
        "runner": dict(runner),
        "page_size": page,
        "shared_kv": shared_kv,
    }
    if max_q is not None and max_q != max(q_lens):
        case["max_q"] = max_q
    if topk:
        case["topk"] = topk
    return case


def _feature_cases(flags: dict[str, Any]) -> dict[str, Any]:
    """Harness case fields of the selection-neutral flags."""
    out: dict[str, Any] = {}
    if flags["kv_layout"] == "NHD":
        out["kv_layout"] = "NHD"
    for key in ("sinks", "q_noncontig", "device_scales", "kv_tuple", "cum_q"):
        if flags[key]:
            out[key] = True
    if not flags["shared_idx"]:
        out["shared_idx"] = False
    return out


def _paged_prefill(entries: list, func: str, grid: dict[str, list]) -> None:
    for values in itertools.product(*grid.values()):
        a = dict(zip(grid.keys(), values))
        batch, page, hkv, grp = a["config"]
        q_dt, kv_dt, o_dt = a["dtypes"]
        if a["skips_softmax"] and q_dt != kv_dt:
            continue
        if kv_dt == "nvfp4" and (q_dt != "fp8" or o_dt != "fp8"):
            continue
        q_lens, kv_lens = _prefill_lens(batch, a["max_q_len"], a["max_kv_len"])
        hd = a["head_dim"]
        device_scale = a.get("device_scale", kv_dt in ("fp8", "nvfp4"))
        flags = _flags(
            kv_layout=a["kv_layout"] if kv_dt != "nvfp4" else "HND",
            sinks=a["enable_sink"],
            shared_idx=a["uses_shared_paged_kv_idx"],
            q_noncontig=a.get("non_contiguous_query", False),
            device_scales=device_scale,
            sf_start=o_dt == "nvfp4",
        )
        case = _paged_case(
            hq=hkv * grp,
            hkv=hkv,
            q_lens=q_lens,
            kv_lens=kv_lens,
            page=page,
            runner=PAGED_CONTEXT,
            window_left=a["window_left"],
            skip=a["skips_softmax"],
            bmm1=1.0 / math.sqrt(hd),
        )
        params = {k: v for k, v in a.items() if k != "config"}
        params = {"batch,page,hkv,grp": a["config"], **params}
        case.update(_feature_cases(flags))
        entries.append(
            _entry(ATTN, func, params, a["dtypes"], (hd, hd), PAGED_KV, case, flags)
        )


DECODE_DTYPES = [
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("bf16", "fp8", "bf16"),
    ("fp16", "fp8", "fp16"),
    ("bf16", "fp8", "fp8"),
    ("fp16", "fp8", "fp8"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "fp8", "nvfp4"),
    ("fp8", "nvfp4", "fp8"),
]


def _paged_decode(entries: list, func: str, grid: dict[str, list]) -> None:
    for values in itertools.product(*grid.values()):
        a = dict(zip(grid.keys(), values))
        cfg = a["config"]
        if len(cfg) == 6:  # test_trtllm_batch_decode_spec
            batch, max_q, page, hkv, grp, hd = cfg
            q_len = None
        else:
            batch, q_len, page, hkv, grp = cfg
            max_q, hd = None, a["head_dim"]
        q_dt, kv_dt, o_dt = a["dtypes"]
        if a["skips_softmax"] and q_dt != kv_dt:
            continue
        if o_dt == "fp8" and q_dt != "fp8":
            continue
        if kv_dt == "nvfp4" and (q_dt != "fp8" or o_dt != "fp8"):
            continue
        q_lens, kv_lens = _decode_lens(batch, q_len, a["max_in_kv_len"], max_q)
        device_scale = a.get("device_scale", False)
        if "device_scale" not in a and func == "test_trtllm_batch_decode":
            device_scale = kv_dt in ("fp8", "nvfp4")
        flags = _flags(
            kv_layout=a.get("kv_layout", "HND") if kv_dt != "nvfp4" else "HND",
            sinks=a.get("enable_sink", False),
            shared_idx=a["uses_shared_paged_kv_idx"],
            q_noncontig=a.get("non_contiguous_query", False),
            device_scales=device_scale,
            cum_q=q_len is None,
            sf_start=o_dt == "nvfp4",
        )
        case = _paged_case(
            hq=hkv * grp,
            hkv=hkv,
            q_lens=q_lens,
            kv_lens=kv_lens,
            page=page,
            runner=PAGED_DECODE,
            window_left=a["window_left"],
            skip=a["skips_softmax"],
            bmm1=1.0 / math.sqrt(hd),
            max_q=max_q if q_len is None else q_len,
        )
        params = {k: v for k, v in a.items() if k != "config"}
        params = {"config": cfg, **params}
        case.update(_feature_cases(flags))
        entries.append(
            _entry(ATTN, func, params, a["dtypes"], (hd, hd), PAGED_KV, case, flags)
        )


def _ragged(entries: list, func: str, grid: dict[str, list], test: str = ATTN):
    dims = {"deepseek": (192, 128), "smaller": (128, 128)}
    for values in itertools.product(*grid.values()):
        a = dict(zip(grid.keys(), values))
        s_qo, s_kv = a["s_qo"], a["s_kv"]
        if s_qo > s_kv:
            continue
        batch = a["batch_size"]
        torch.manual_seed(0)
        q = torch.randint(1, s_qo + 1, (batch, 1, 1, 1), dtype=torch.int32)
        kv = torch.randint(s_qo, s_kv + 1, (batch, 1, 1, 1), dtype=torch.int32)
        q_lens, kv_lens = _ints(q), _ints(kv)
        # The tests pass s_qo / s_kv as max_q_len / max_kv_len.
        q_lens[-1], kv_lens[-1] = s_qo, s_kv
        hqk, hv = dims[a["mla_dimensions"]]
        hkv = a["num_kv_heads"]
        case = {
            "hq": hkv * a["head_grp_size"],
            "hkv": hkv,
            "q_lens": q_lens,
            "kv_lens": kv_lens,
            "window_left": -1,
            "skip_thr": 1e-30 if a["skips_softmax"] else 0.0,
            "bmm1_scale": 1.0 / math.sqrt(hqk),
            "bmm2_scale": 1.0,
            "runner": _ragged_runner(a["causal"]),
        }
        entries.append(
            _entry(
                test,
                func,
                a,
                ("bf16", "bf16", "bf16"),
                (hqk, hv),
                SEPARATE_QKV,
                case,
                _flags(),
            )
        )


def _mla_decode(entries: list, func: str, grid: dict[str, list]) -> None:
    layers = {
        "dsr1_128": (576, 512, 128),
        "glm5_64": (576, 512, 64),
        "smaller_32": (320, 256, 32),
    }
    for values in itertools.product(*grid.values()):
        a = dict(zip(grid.keys(), values))
        hqk, hv, heads = layers[a["layer_dimensions"]]
        batch, max_seq = a["batch_size"], a.get("max_seq_len", 1024)
        if func.endswith("preallocated_out"):
            kv_lens = [max_seq] * batch
        else:
            torch.manual_seed(42)
            kv_lens = [
                int(torch.randint(1, max_seq, (1,)).item()) for _ in range(batch)
            ]
            kv_lens[-1] = max_seq
        q_len = a["q_len_per_request"]
        dtype = a.get("dtype", "bf16")
        flags = _flags(shared_idx=a.get("uses_shared_paged_kv_idx", True))
        scale = a.get("scale", 1.0)
        bmm1 = scale / math.sqrt(128 + 64) if "scale" in a else 1.0 / math.sqrt(hqk)
        case = _paged_case(
            hq=heads,
            hkv=1,
            q_lens=[q_len] * batch,
            kv_lens=kv_lens,
            page=a.get("page_size", 64),
            runner=PAGED_DECODE,
            window_left=-1,
            skip=a.get("skips_softmax", False),
            bmm1=bmm1,
            shared_kv=True,
        )
        case.update(_feature_cases(flags))
        o = "bf16"
        entries.append(
            _entry(MLA, func, a, (dtype, dtype, o), (hqk, hv), PAGED_KV, case, flags)
        )


def _mla_sparse(entries: list, grid: dict[str, list]) -> None:
    for values in itertools.product(*grid.values()):
        a = dict(zip(grid.keys(), values))
        batch, topk = a["batch_size"], a["topk"]
        torch.manual_seed(42)
        if a["is_varlen"]:
            kv_lens = [
                max(topk, int(torch.distributions.Normal(4096, 2048).sample().item()))
                for _ in range(batch)
            ]
            kv_lens[-1] = 4096
            kv_lens = [min(s, 4096) for s in kv_lens]
        else:
            kv_lens = [4096] * batch
        q_len = a["q_len_per_request"]
        case = _paged_case(
            hq=a["num_attn_heads"],
            hkv=1,
            q_lens=[q_len] * batch,
            kv_lens=kv_lens,
            page=32,
            runner=PAGED_DECODE,
            window_left=-1,
            skip=False,
            bmm1=1.0 / math.sqrt(a["qk_nope_head_dim"] + 64),
            shared_kv=True,
            topk=topk,
        )
        dtype = a["dtype"]
        entries.append(
            _entry(
                MLA,
                "test_trtllm_batch_decode_mla_sparse",
                a,
                (dtype, dtype, "bf16"),
                (576, 512),
                PAGED_KV,
                case,
                _flags(),
            )
        )


def _sinks(entries: list) -> None:
    grid = {
        "dtype": ["fp16", "bf16"],
        "batch_size": [1, 4, 16],
        "page_size": [32],
        "seq_len": [32, 128, 1024],
        "num_qo_heads": [32],
        "num_kv_heads": [8, 32],
        "head_dim": [64, 128],
    }
    for func, context in (
        ("test_blackwell_trtllm_gen_decode_attention_sink", False),
        ("test_blackwell_trtllm_gen_context_attention_sink", True),
    ):
        for values in itertools.product(*grid.values()):
            a = dict(zip(grid.keys(), values))
            batch, seq = a["batch_size"], a["seq_len"]
            q_lens = [seq if context else 1] * batch
            flags = _flags(sinks=True, kv_tuple=True)
            case = _paged_case(
                hq=a["num_qo_heads"],
                hkv=a["num_kv_heads"],
                q_lens=q_lens,
                kv_lens=[seq] * batch,
                page=a["page_size"],
                runner=PAGED_CONTEXT if context else PAGED_DECODE,
                window_left=-1,
                skip=False,
                bmm1=1.0,
            )
            case.update(_feature_cases(flags))
            d = a["dtype"]
            hd = a["head_dim"]
            entries.append(
                _entry(SINK, func, a, (d, d, d), (hd, hd), PAGED_KV, case, flags)
            )


def _ragged_stride(entries: list) -> None:
    torch.manual_seed(42)
    q = torch.randint(50, 150, (16,), dtype=torch.int32)
    q_lens = _ints(q)
    case = {
        "hq": 128,
        "hkv": 128,
        "q_lens": q_lens,
        "kv_lens": [8192] * 16,
        "window_left": -1,
        "skip_thr": 0.0,
        "bmm1_scale": 1.0 / math.sqrt(192),
        "bmm2_scale": 1.0,
        "runner": _ragged_runner(True),
    }
    entries.append(
        _entry(
            STRIDE,
            "test_trtllm_ragged_kv_large_stride_overflow",
            {},
            ("bf16", "bf16", "bf16"),
            (192, 128),
            SEPARATE_QKV,
            case,
            _flags(),
        )
    )


PREFILL_CONFIGS = [
    (4, 16, 2, 1),
    (4, 32, 4, 5),
    (4, 64, 4, 8),
    (128, 16, 2, 5),
    (128, 32, 4, 1),
    (128, 64, 2, 8),
    (256, 16, 4, 8),
    (256, 32, 2, 8),
    (256, 64, 4, 1),
    (256, 64, 4, 5),
]
DECODE_CONFIGS = [
    (4, 1, 16, 2, 1),
    (4, 1, 32, 2, 5),
    (4, 2, 64, 2, 5),
    (4, 3, 32, 2, 5),
    (4, 3, 64, 2, 1),
    (4, 4, 64, 4, 1),
    (4, 5, 64, 4, 8),
    (128, 1, 64, 2, 5),
    (128, 2, 32, 4, 1),
    (128, 3, 16, 4, 8),
    (128, 4, 16, 2, 5),
    (128, 5, 16, 2, 5),
    (256, 1, 64, 4, 8),
    (256, 2, 16, 2, 8),
    (256, 3, 64, 4, 5),
    (256, 4, 32, 2, 8),
    (256, 5, 32, 2, 1),
]
SPEC_CONFIGS = [
    (4, 1, 16, 2, 1, 128),
    (4, 1, 32, 2, 5, 128),
    (4, 2, 64, 2, 5, 128),
    (4, 3, 32, 2, 5, 128),
    (4, 3, 64, 2, 1, 128),
    (4, 4, 64, 4, 1, 128),
    (4, 5, 64, 4, 8, 128),
    *[(bs, 4, 64, 4, 16, hd) for bs in [4, 8, 16, 32] for hd in [128, 256]],
    (128, 1, 64, 2, 5, 128),
    (128, 2, 32, 4, 1, 128),
    (128, 3, 16, 4, 8, 128),
    (128, 4, 16, 2, 5, 128),
    (128, 5, 16, 2, 5, 128),
    (256, 1, 64, 4, 8, 256),
    (256, 2, 16, 2, 8, 256),
    (256, 3, 64, 4, 5, 256),
    (256, 4, 32, 2, 8, 256),
    (256, 16, 32, 2, 8, 256),
]
TF = [True, False]


def upstream_params() -> list[dict[str, Any]]:
    """Every parametrization (see the module docstring), PDL folded."""
    entries: list[dict[str, Any]] = []
    prefill_dtypes = [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
        ("fp8", "nvfp4", "fp8"),
    ]
    _paged_prefill(
        entries,
        "test_trtllm_batch_prefill",
        {
            "kv_layout": ["HND", "NHD"],
            "config": PREFILL_CONFIGS,
            "window_left": [-1],
            "dtypes": prefill_dtypes,
            "enable_sink": TF,
            "max_q_len": [511],
            "max_kv_len": [2047],
            "head_dim": [128, 256],
            "non_contiguous_query": [False, True],
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _paged_prefill(
        entries,
        "test_trtllm_batch_prefill_bs1",
        {
            "kv_layout": ["HND", "NHD"],
            "config": [(1, 16, 8, 8)],
            "window_left": [-1],
            "dtypes": [("bf16", "bf16", "bf16")],
            "enable_sink": [False],
            "max_q_len": [8192],
            "max_kv_len": [8192],
            "head_dim": [128, 256],
            "device_scale": [False],
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _paged_decode(
        entries,
        "test_trtllm_batch_decode",
        {
            "kv_layout": ["HND", "NHD"],
            "config": DECODE_CONFIGS,
            "window_left": [-1, 127],
            "dtypes": DECODE_DTYPES,
            "enable_sink": TF,
            "max_in_kv_len": [110],
            "head_dim": [128, 256],
            "non_contiguous_query": [False, True],
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _paged_decode(
        entries,
        "test_trtllm_batch_decode_bs1",
        {
            "config": [(1, 1, 16, 8, 8), (1, 1, 32, 8, 8)],
            "window_left": [-1],
            "dtypes": [("fp8", "fp8", "fp8")],
            "max_in_kv_len": [4096, 8192],
            "head_dim": [128],
            "device_scale": TF,
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _paged_decode(
        entries,
        "test_trtllm_batch_decode_head_dim_256",
        {
            "config": [
                (4, 1, 16, 2, 1),
                (4, 1, 32, 2, 5),
                (4, 3, 64, 2, 1),
                (4, 4, 64, 4, 1),
                (128, 3, 16, 4, 8),
                (128, 4, 16, 2, 5),
                (256, 4, 32, 2, 8),
                (256, 5, 32, 2, 1),
            ],
            "window_left": [-1],
            "dtypes": [
                ("bf16", "bf16", "bf16"),
                ("fp16", "fp16", "fp16"),
                ("fp8", "fp8", "fp16"),
                ("fp8", "fp8", "fp8"),
                ("fp8", "fp8", "nvfp4"),
                ("fp8", "nvfp4", "fp8"),
            ],
            "max_in_kv_len": [110],
            "head_dim": [256],
            "device_scale": TF,
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _paged_decode(
        entries,
        "test_trtllm_batch_decode_long_sequence_length",
        {
            "config": [
                (1, 1, 16, 2, 1),
                (1, 1, 32, 2, 5),
                (1, 3, 64, 2, 1),
                (1, 4, 64, 4, 1),
                (32, 4, 16, 2, 8),
                (32, 8, 16, 2, 8),
                (32, 16, 16, 2, 8),
            ],
            "window_left": [-1],
            "dtypes": [("bf16", "bf16", "bf16"), ("fp8", "fp8", "fp8")],
            "max_in_kv_len": [4096, 8192, 16384, 32768, 65536, 131072],
            "head_dim": [128],
            "device_scale": TF,
            "skips_softmax": [False],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _paged_decode(
        entries,
        "test_trtllm_batch_decode_spec",
        {
            "kv_layout": ["HND", "NHD"],
            "config": SPEC_CONFIGS,
            "window_left": [-1, 127],
            "dtypes": DECODE_DTYPES[:10],
            "enable_sink": TF,
            "max_in_kv_len": [110],
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    for func, grid in (
        (
            "test_trtllm_gen_prefill",
            {
                "mla_dimensions": ["deepseek", "smaller"],
                "batch_size": [4, 128, 256],
                "s_qo": [32, 64, 87],
                "s_kv": [32, 64, 87],
                "num_kv_heads": [16, 32],
                "head_grp_size": [1, 5, 8],
                "causal": TF,
                "skips_softmax": [False, True],
            },
        ),
        (
            "test_trtllm_gen_prefill_bs1",
            {
                "mla_dimensions": ["deepseek", "smaller"],
                "batch_size": [1],
                "s_qo": [1024],
                "s_kv": [1024],
                "num_kv_heads": [128],
                "head_grp_size": [1],
                "causal": TF,
                "skips_softmax": [False, True],
            },
        ),
    ):
        _ragged(entries, func, grid)
    _mla_decode(
        entries,
        "test_trtllm_batch_decode_mla",
        {
            "layer_dimensions": ["dsr1_128", "glm5_64", "smaller_32"],
            "batch_size": [1, 2, 4, 16, 32, 64, 128, 256, 512, 768, 1024],
            "scale": [1.0, 0.5],
            "dtype": ["fp8", "bf16"],
            "page_size": [32, 64],
            "q_len_per_request": [1, 2],
            "skips_softmax": [False, True],
            "uses_shared_paged_kv_idx": TF,
        },
    )
    _mla_sparse(
        entries,
        {
            "batch_size": [1, 2, 4, 16, 32, 64, 128],
            "dtype": ["fp8", "bf16"],
            "q_len_per_request": [1, 2],
            "topk": [128, 2048],
            "is_varlen": [False, True],
            "qk_nope_head_dim": [128, 192],
            "num_attn_heads": [128, 64],
        },
    )
    _mla_decode(
        entries,
        "test_trtllm_batch_decode_mla_preallocated_out",
        {
            "layer_dimensions": ["dsr1_128"],
            "q_len_per_request": [1, 2, 4],
            "batch_size": [1, 4],
            "max_seq_len": [256],
        },
    )
    _sinks(entries)
    _ragged_stride(entries)
    ids = [e["id"] for e in entries]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate upstream parametrization ids")
    return entries


def shape_key(entry_case: dict[str, Any]) -> str:
    """The selection- and shape-relevant fields of a case (what an upstream
    parametrization and the harness case covering it must share)."""
    keys = (
        "hq",
        "hkv",
        "q_lens",
        "kv_lens",
        "window_left",
        "runner",
        "page_size",
        "shared_kv",
        "topk",
        "max_q",
    )
    parts = [f"{k}={entry_case.get(k)}" for k in keys]
    parts.append(f"skip={entry_case['skip_thr'] != 0.0}")
    return ";".join(parts)

"""FlashInfer's own tests of the sm_86 FA2 kernels, as harness launches.

The tests of the pinned FlashInfer tree (``resources/flashinfer-<rev>/tests``)
that launch one of this package's sm_86 kernels on an sm_80/sm_86 GPU are
listed in ``TESTS``. ``launches()`` parses each test's ``pytest.mark.
parametrize`` grid from the source (``ast``; no test code is imported), applies
the test's skips and hard-coded values, and turns every parametrization into
the kernel launches it makes on sm_86 (wrapper -> plan -> kernel), each as
``(workload name, harness case params, test id)``. The harness computes the
launch's *regime* (``fa2.regime``: page size, KV layout, GQA group, split-KV /
fixed-split / CUDA-graph plan, window active or not, LSE output, strides,
fully masked rows, special values, ...). The build records, per kernel and
regime, the cheapest upstream parametrization as that kernel's upstream case
(``fixtures/sm_86_upstream.json``); ``tests/test_fmha.py`` re-derives the
launches and checks that every one's regime is a regime of its kernel's
cases (and that the recorded table is current).

Not served by any sm_86 kernel of this package (excluded parametrizations):
ROPE_LLAMA / ALIBI position encodings (``posenc_0`` modules only), head dim
512 (decode, prefill, Gemma-4 FP8 tests), NVFP4 KV, asymmetric head dims,
unequal paged K/V strides (the equal-stride modules reject them; the
independent-stride module is the AttentionSink one), head dim 128 attention
sinks (``test_attention_sink.py``: the JIT cache's sink modules are head dim
64), multi-item scoring (only tested with ROPE_LLAMA), MLA without the RoPE
part (``head_dim_kpe`` 0), the MLA uint32 page-index overflow regression
(needs a 9.7 GB latent cache, beyond the 6 GB sm_86 case budget), BatchAttention
head dims 128/256 and NVFP4 (no sm_86 kernel), fa3/cutlass/trtllm/cudnn
backends, and error-path tests that launch nothing.
"""

from __future__ import annotations

import ast
import itertools
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

FLASHINFER = "flashinfer-aa7c67f2b876b89be34c7a70ac022369a56e60d5"
NHD, HND = 0, 1
DTYPE = {"float16": "f16", "half": "f16", "bfloat16": "bf16", "float8_e4m3fn": "e4m3"}
Launch = tuple[str, dict[str, Any], str]


class _Torch:
    """``torch.<dtype>`` in parametrize lists, as the harness dtype names."""

    def __getattr__(self, name: str) -> str:
        return DTYPE.get(name, name)

    @staticmethod
    def manual_seed(seed: int) -> None:
        """Seeds nothing: the configs only draw from numpy's seeded RNG."""


class _Pytest:
    @staticmethod
    def param(*values: Any, **_: Any) -> Any:
        return values[0] if len(values) == 1 else values


def _source(root: Path, file: str) -> tuple[str, ast.Module]:
    text = (root / FLASHINFER / "tests" / file).read_text()
    return text, ast.parse(text)


def parametrizations(root: Path, file: str, function: str) -> list[dict[str, Any]]:
    """Every parametrization of ``file::function`` (cartesian product of its
    ``pytest.mark.parametrize`` decorators), evaluated from the source."""
    _, tree = _source(root, file)
    namespace: dict[str, Any] = {
        "torch": _Torch(),
        "pytest": _Pytest(),
        "list": list,
        "range": range,
        "dict": dict,
        "tuple": tuple,
        # test_lse_base.py's backend lists on an sm_80/sm_86 GPU
        "_ragged_backends": lambda: ["fa2"],
        "_paged_backends": lambda: ["fa2"],
    }
    # Module-level helpers a decorator calls (e.g. _build_seq_len_configs).
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_build"):
            import numpy as np

            namespace["np"] = np
            exec(compile(ast.Module([node], []), file, "exec"), namespace)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function
    )
    axes: list[tuple[list[str], list[Any]]] = []
    for dec in fn.decorator_list:
        if not (
            isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and dec.func.attr == "parametrize"
        ):
            continue
        names = ast.literal_eval(dec.args[0])
        names = (
            [n.strip() for n in names.split(",")] if isinstance(names, str) else names
        )
        values = eval(compile(ast.Expression(dec.args[1]), file, "eval"), namespace)
        axes.append((list(names), list(values)))
    out = []
    for combo in itertools.product(*(values for _, values in axes)):
        cfg: dict[str, Any] = {}
        for (names, _), value in zip(axes, combo):
            if len(names) == 1:
                cfg[names[0]] = value
            else:
                cfg.update(zip(names, value))
        out.append(cfg)
    return out


def _test_id(file: str, function: str, cfg: dict[str, Any]) -> str:
    def short(value: Any) -> str:
        text = repr(value)
        return text if len(text) <= 40 else text[:37] + "..."

    params = ",".join(f"{k}={short(v)}" for k, v in cfg.items())
    return f"tests/{file}::{function}[{params}]"


# -- kernel names (impls/fmha/fa2_build.py naming) --------------------------------


def _flags(swa: bool, softcap: bool) -> str:
    return ("_swa" if swa else "") + ("_softcap" if softcap else "")


def prefill_name(
    kind: str, dq: str, dkv: str, d: int, mask: str, swa: bool, cap: bool, cta: int
) -> str:
    return f"fmha_fa2_prefill_{kind}_{dq}_{dkv}_hd{d}_{mask}{_flags(swa, cap)}_cta{cta}"


def decode_name(dq: str, dkv: str, d: int, group: int, swa: bool, cap: bool) -> str:
    return f"fmha_fa2_decode_{dq}_{dkv}_hd{d}_g{group}{_flags(swa, cap)}"


class Resolver:
    """Routes a launch to its kernel with the harness's own planners."""

    def __init__(self, fa2: Any):
        self.fa2 = fa2
        self.index = fa2.load_index()["kernels"]
        # Parametrizations whose launch hits the causal kernels' qo_len >
        # kv_len masking defect (``fa2.causal_mask_skipped``): the kernel's
        # output is not attention, so they fail upstream on sm_86 too and are
        # not harness cases. {test function: launches}
        self.defects: dict[str, int] = {}

    def prefill(
        self,
        kind: str,
        dq: str,
        dkv: str,
        d: int,
        mask: str,
        p: dict[str, Any],
    ) -> str | None:
        swa = p.get("window_left", -1) >= 0
        cap = p.get("logits_soft_cap", 0.0) > 0
        meta = {
            "kind": kind,
            "head_dim": d,
            "dtype_kv": dkv,
        }
        full = {**self.fa2.COMMON_DEFAULTS, **self.fa2.Fa2Prefill.defaults, **p}
        if kind == "ragged":
            full["page_size"] = 1
        plan = self.fa2.prefill_route(meta, full, self.fa2.NUM_SMS)
        name = prefill_name(kind, dq, dkv, d, mask, swa, cap, plan["cta_tile_q"])
        if name not in self.index:
            return None
        full["hq"], full["hkv"] = p["hq"], p["hkv"]
        if self.fa2.causal_mask_skipped(self.index[name], full, plan):
            return ""
        return name

    def decode(self, dq: str, dkv: str, d: int, p: dict[str, Any]) -> str | None:
        group = p["hq"] // p["hkv"]
        swa = p.get("window_left", -1) >= 0
        cap = p.get("logits_soft_cap", 0.0) > 0
        name = decode_name(dq, dkv, d, group, swa, cap)
        return name if name in self.index else None


# -- adapters: one per upstream test function ---------------------------------------
#
# An adapter maps one parametrization to its sm_86 launches (none when the
# test skips it or no sm_86 kernel serves it).


def _uniform(batch: int, n: int) -> list[int]:
    return [n] * batch


def _prefill(
    r: Resolver,
    kind: str,
    dq: str,
    dkv: str,
    d: int,
    mask: str,
    p: dict[str, Any],
    test: str,
    out: list[Launch],
) -> None:
    name = r.prefill(kind, dq, dkv, d, mask, p)
    if name == "":
        function = test.split("[")[0]
        r.defects[function] = r.defects.get(function, 0) + 1
    elif name is not None:
        out.append((name, p, test))


def _merge_after_split(
    r: Resolver, name: str, p: dict[str, Any], test: str, out: list[Launch]
) -> None:
    """The VariableLengthMergeStates launch after a split-KV prefill/decode."""
    w = r.fa2.probe(name)
    meta = w.meta
    plan = w.route(p)
    if not plan["split_kv"]:
        return
    dtype = meta["dtype_o"]
    merge = f"fmha_fa2_merge_varlen_{dtype}_hd{meta['head_dim']}"
    if merge not in r.index:
        return
    if meta["family"] == "decode":
        sets = [b - a for a, b in itertools.pairwise(plan["o_indptr"])]
        lse = p.get("return_lse", True)
        rows = None
    else:
        sets = [b - a for a, b in itertools.pairwise(plan["merge_indptr"])]
        lse = p.get("return_lse", True)
        rows = len(sets) if plan["cuda_graph"] else None
    heads = w.params(p)["hq"]
    params: dict[str, Any] = {"sets": sets, "heads": heads, "d": meta["head_dim"]}
    if not lse:
        params["return_lse"] = False
    if rows is not None:
        params["seq_len"] = rows
    out.append((merge, params, test))


def _with_merges(r: Resolver, launches: list[Launch]) -> list[Launch]:
    """``launches`` followed by the merge launch of each split-KV one."""
    merges: list[Launch] = []
    for name, params, test in launches:
        _merge_after_split(r, name, params, test, merges)
    return launches + merges


def batch_prefill_paged(
    r: Resolver, cfg: dict, test: str, tuple_kv: bool
) -> list[Launch]:
    if cfg["pos_encoding_mode"] != "NONE":
        return []
    causal = cfg.get("causal", False)
    if cfg["qo_len"] > cfg["kv_len"] and causal:
        return []
    bs = cfg["batch_size"]
    p = {
        "hq": cfg["num_qo_heads"],
        "hkv": cfg["num_kv_heads"],
        "q_lens": _uniform(bs, cfg["qo_len"]),
        "kv_lens": _uniform(bs, cfg["kv_len"]),
        "page_size": cfg["page_size"],
        "kv_layout": NHD,
        "pages": "identity",
        "kv_storage": "separate" if tuple_kv else "interleaved",
        "disable_split_kv": False,
        "cuda_graph": cfg["use_cuda_graph"],
    }
    out: list[Launch] = []
    mask = "causal" if causal else "none"
    _prefill(r, "paged", "f16", "f16", cfg["head_dim"], mask, p, test, out)
    return _with_merges(r, out)


def batch_prefill_custom(r: Resolver, cfg: dict, test: str, kind: str) -> list[Launch]:
    if cfg["pos_encoding_mode"] != "NONE":
        return []
    if kind == "paged" and cfg["qo_len"] > cfg["kv_len"]:
        return []
    bs = cfg["batch_size"]
    cap = cfg.get("logits_soft_cap", 0.0)
    p: dict[str, Any] = {
        "hq": cfg["num_qo_heads"],
        "hkv": cfg["num_kv_heads"],
        "q_lens": _uniform(bs, cfg["qo_len"]),
        "kv_lens": _uniform(bs, cfg["kv_len"]),
        "kv_layout": NHD,
        "disable_split_kv": False,
        "logits_soft_cap": cap,
        "return_lse": cfg.get("return_lse", True),
    }
    if kind == "paged":
        p |= {"page_size": cfg["page_size"], "pages": "identity"}
    out: list[Launch] = []
    _prefill(
        r,
        kind,
        "f16",
        "f16",
        cfg["head_dim"],
        "custom",
        {**p, "mask": "tril"},
        test,
        out,
    )
    _prefill(r, kind, "f16", "f16", cfg["head_dim"], "causal", p, test, out)
    return _with_merges(r, out)


def batch_prefill_ragged(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    if cfg["pos_encoding_mode"] != "NONE":
        return []
    if cfg["qo_len"] > cfg["kv_len"] and cfg["causal"]:
        return []
    bs = cfg["batch_size"]
    p = {
        "hq": cfg["num_qo_heads"],
        "hkv": cfg["num_kv_heads"],
        "q_lens": _uniform(bs, cfg["qo_len"]),
        "kv_lens": _uniform(bs, cfg["kv_len"]),
        "kv_layout": NHD,
        "disable_split_kv": False,
    }
    out: list[Launch] = []
    mask = "causal" if cfg["causal"] else "none"
    _prefill(r, "ragged", "f16", "f16", cfg["head_dim"], mask, p, test, out)
    return _with_merges(r, out)


def lazy_stride_router(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    # The equal-stride runs (k, v contiguous); the unequal-V run routes to a
    # module this package does not serve.
    layout = NHD if cfg["kv_layout"] == "NHD" else HND
    p = {
        "hq": 8,
        "hkv": 2,
        "q_lens": [17, 17],
        "kv_lens": [97, 97],
        "page_size": 16,
        "kv_layout": layout,
        "pages": "identity",
        "kv_storage": "separate",
        "disable_split_kv": False,
        "fixed_split_size": 2,
        "return_lse": False,
    }
    out: list[Launch] = []
    _prefill(r, "paged", "bf16", "bf16", cfg["head_dim"], "causal", p, test, out)
    return _with_merges(r, out)


def one_valid_key(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    dt = cfg["dtype"]
    p = {
        "hq": 32,
        "hkv": 8,
        "q_lens": [1],
        "kv_lens": [1],
        "values": "extreme",
        "sm_scale": 1 / math.sqrt(128),
        "disable_split_kv": False,
    }
    out: list[Launch] = []
    _prefill(r, "ragged", dt, dt, 128, "causal", p, test, out)
    return out


def fully_masked_rows(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    dt = cfg["dtype"]
    p = {
        "hq": 32,
        "hkv": 8,
        "q_lens": [34],
        "kv_lens": [1],
        "page_size": 1,
        "pages": "offset",
        "kv_storage": "separate",
        "values": "constant",
        "sm_scale": 1 / math.sqrt(128),
        "disable_split_kv": False,
    }
    out: list[Launch] = []
    _prefill(r, "paged", dt, dt, 128, "causal", p, test, out)
    return _with_merges(r, out)


def split_kv_empty_chunk(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    dt = cfg["dtype"]
    p = {
        "hq": 8,
        "hkv": 2,
        "q_lens": [2],
        "kv_lens": [129],
        "page_size": 16,
        "pages": "identity",
        "disable_split_kv": False,
    }
    out: list[Launch] = []
    _prefill(r, "paged", dt, dt, 128, "causal", p, test, out)
    return _with_merges(r, out)


def tensor_core_decode(r: Resolver, cfg: dict, test: str, graph: bool) -> list[Launch]:
    """Tensor-core decode: the paged prefill kernel with one query per
    request, non-causal; CUDA graphs plan with ``enable_cuda_graph``."""
    if cfg["pos_encoding_mode"] != "NONE":
        return []
    bs, hkv = cfg["batch_size"], cfg["num_kv_heads"]
    layout = NHD if cfg["kv_layout"] == "NHD" else HND
    out: list[Launch] = []
    batches = [bs]
    if "invariant_bs" in cfg:
        batches.append(cfg["invariant_bs"])
    for b in batches:
        p: dict[str, Any] = {
            "hq": hkv * cfg["group_size"],
            "hkv": hkv,
            "q_lens": _uniform(b, 1),
            "kv_lens": _uniform(b, cfg["kv_len"]),
            "page_size": cfg["page_size"],
            "kv_layout": layout,
            "pages": "identity",
            "disable_split_kv": cfg.get("disable_split_kv", False),
            "cuda_graph": graph,
        }
        if not p["disable_split_kv"] and "fixed_split_size" in cfg:
            p["fixed_split_size"] = cfg["fixed_split_size"]
        _prefill(r, "paged", "f16", "f16", cfg["head_dim"], "none", p, test, out)
    return _with_merges(r, out)


def tensor_core_prefill_invariant(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    if cfg["pos_encoding_mode"] != "NONE":
        return []
    hkv = cfg["num_kv_heads"]
    layout = NHD if cfg["kv_layout"] == "NHD" else HND
    out: list[Launch] = []
    for b in (cfg["batch_size"], cfg["invariant_bs"]):
        p: dict[str, Any] = {
            "hq": hkv * cfg["group_size"],
            "hkv": hkv,
            "q_lens": _uniform(b, cfg["qo_len"]),
            "kv_lens": _uniform(b, cfg["kv_len"]),
            "page_size": cfg["page_size"],
            "kv_layout": layout,
            "pages": "identity",
            "disable_split_kv": cfg["disable_split_kv"],
        }
        if not cfg["disable_split_kv"]:
            p["fixed_split_size"] = cfg["fixed_split_size"]
        _prefill(r, "paged", "f16", "f16", cfg["head_dim"], "none", p, test, out)
    return _with_merges(r, out)


def uniform_multi_token_decode(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    bs, q_len = cfg["batch_size"], cfg["q_len_per_req"]
    hkv = cfg["num_kv_heads"]
    out: list[Launch] = []
    for lengths in (
        [129] * bs,
        [(33, 17, 65, 129)[i % 4] for i in range(bs)],
        [(q_len, 128, 64, 100)[i % 4] for i in range(bs)],
    ):
        p = {
            "hq": hkv * cfg["gqa_group_size"],
            "hkv": hkv,
            "q_lens": _uniform(bs, q_len),
            "kv_lens": lengths,
            "page_size": 16,
            "pages": "identity",
            "disable_split_kv": False,
            "cuda_graph": True,
            "uniform_q_len": q_len,
            "return_lse": False,
        }
        _prefill(r, "paged", "bf16", "bf16", 128, "causal", p, test, out)
    return _with_merges(r, out)


def mlc_failed_case(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    base = {
        "hkv": 32,
        "kv_lens": [0, 129],
        "page_size": 16,
        "kv_layout": HND,
        "pages": "offset",
    }
    out: list[Launch] = []
    name = r.decode("f16", "f16", 128, {**base, "hq": 32})
    if name:
        out.append((name, base, test))
    p = {**base, "hq": 32, "q_lens": [1, 1], "disable_split_kv": False}
    _prefill(r, "paged", "f16", "f16", 128, "none", p, test, out)
    return _with_merges(r, out)


def kv_scale(
    r: Resolver, cfg: dict, test: str, n_ctx: int, scales: tuple
) -> list[Launch]:
    dt = cfg["dtype"]
    out: list[Launch] = []
    for k_scale in scales:
        p = {
            "hq": 1,
            "hkv": 1,
            "q_lens": [n_ctx],
            "kv_lens": [n_ctx],
            "page_size": 16,
            "pages": "identity",
            "kv_storage": "separate",
            "sm_scale": k_scale / math.sqrt(64),  # run() folds k_scale into sm_scale
            "disable_split_kv": False,
        }
        _prefill(r, "paged", dt, dt, 64, "causal", p, test, out)
    return out


def lse_base(r: Resolver, cfg: dict, test: str, kind: str) -> list[Launch]:
    p: dict[str, Any] = {
        "hq": 8,
        "hkv": 2,
        "q_lens": [5, 64, 17],
        "kv_lens": [37, 64, 90],
        "disable_split_kv": False,
    }
    if kind == "paged":
        p |= {"page_size": 16, "pages": "identity", "kv_storage": "separate"}
    out: list[Launch] = []
    mask = "causal" if cfg["causal"] else "none"
    _prefill(r, kind, "bf16", "bf16", 128, mask, p, test, out)
    return _with_merges(r, out)


def packed_prefill(r: Resolver, cfg: dict, test: str, kind: str) -> list[Launch]:
    hq, hkv = cfg["num_qo_heads"], cfg["num_kv_heads"]
    if hq % hkv:
        return []
    bs, n = cfg["batch_size"], cfg["seq_len"]
    p: dict[str, Any] = {
        "hq": hq,
        "hkv": hkv,
        "q_lens": _uniform(bs, n),
        "kv_lens": _uniform(bs, n),
        "disable_split_kv": False,
    }
    if kind == "ragged":
        p["kv_storage"] = "packed"
    else:
        p |= {
            "page_size": cfg["page_size"],
            "pages": "identity",
            "kv_storage": "separate",
            "q_storage": "packed",
        }
    p["return_lse"] = False
    out: list[Launch] = []
    mask = "causal" if cfg["causal"] else "none"
    _prefill(r, kind, "f16", "f16", cfg["head_dim"], mask, p, test, out)
    return _with_merges(r, out)


def sliding_window_prefill(
    r: Resolver, cfg: dict, test: str, kind: str
) -> list[Launch]:
    hq, hkv = cfg["num_qo_heads"], cfg["num_kv_heads"]
    if hq < hkv or hq % hkv or cfg["head_dim"] > 256:
        return []
    bs = cfg["batch_size"]
    p: dict[str, Any] = {
        "hq": hq,
        "hkv": hkv,
        "q_lens": _uniform(bs, cfg["qo_len"]),
        "kv_lens": _uniform(bs, cfg["kv_len"]),
        "window_left": cfg["window_left"],
        "disable_split_kv": False,
        "return_lse": False,
    }
    if kind == "paged":
        p |= {
            "page_size": cfg["page_size"],
            "pages": "identity",
            "kv_storage": "separate",
        }
    out: list[Launch] = []
    _prefill(r, kind, "f16", "f16", cfg["head_dim"], "causal", p, test, out)
    return _with_merges(r, out)


def shared_prefix_cascade(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    """MultiLevelCascadeAttentionWrapper(2): level 0 (all queries against the
    shared prefix), level 1 (each request's unique KV), merge_state_in_place;
    plus the decode-stage comparison path (CUDA-core decode + merge)."""
    bs, h, d = cfg["batch_size"], cfg["num_heads"], cfg["head_dim"]
    page = cfg["page_size"]
    unique, shared = cfg["unique_kv_len"], cfg["shared_kv_len"]
    q_per = 1 if cfg["stage"] == "decode" else unique
    common = {
        "hq": h,
        "hkv": h,
        "page_size": page,
        "disable_split_kv": False,
    }
    out: list[Launch] = []
    _prefill(
        r, "paged", "f16", "f16", d, "none",
        {**common, "q_lens": [bs * q_per], "kv_lens": [shared], "pages": "identity"},
        test, out,
    )  # fmt: skip
    _prefill(
        r, "paged", "f16", "f16", d, "none",
        {**common, "q_lens": _uniform(bs, q_per), "kv_lens": _uniform(bs, unique),
         "pages": "offset"},
        test, out,
    )  # fmt: skip
    if cfg["stage"] == "decode":
        p = {**common, "kv_lens": _uniform(bs, unique), "pages": "offset"}
        p.pop("disable_split_kv")
        name = r.decode("f16", "f16", d, p)
        if name:
            out.append((name, {k: v for k, v in p.items() if k != "hq"}, test))
    out = _with_merges(r, out)
    merge = "fmha_fa2_merge_state_in_place_f16_vec8"
    out.append((merge, {"seq": bs * q_per, "heads": h, "d": d, "mask": "none"}, test))
    return out


def batch_decode(
    r: Resolver, cfg: dict, test: str, tuple_kv: bool, graph: bool
) -> list[Launch]:
    if cfg["pos_encoding_mode"] != "NONE" or cfg["head_dim"] > 256:
        return []
    bs = cfg["batch_size"]
    lse = cfg.get("return_lse", True) and not graph
    p: dict[str, Any] = {
        "hkv": cfg["num_kv_heads"],
        "kv_lens": _uniform(bs, cfg["kv_len"]),
        "page_size": cfg["page_size"],
        "kv_layout": NHD,
        "pages": "identity",
        "kv_storage": "separate" if tuple_kv else "interleaved",
        "cuda_graph": graph,
        "return_lse": lse,
    }
    out: list[Launch] = []
    lengths = [cfg["kv_len"]]
    if graph:  # replays with 1..3 pages per request, then the real lengths
        pages = -(-cfg["kv_len"] // cfg["page_size"])
        lengths = [i * cfg["page_size"] for i in range(1, min(4, pages))] + lengths
    for n in lengths:
        q = {**p, "kv_lens": _uniform(bs, n)}
        name = r.decode(cfg["q_dtype"], cfg["kv_dtype"], cfg["head_dim"],
                        {**q, "hq": cfg["num_qo_heads"]})  # fmt: skip
        if name:
            out.append((name, q, test))
    return _with_merges(r, out)


def decode_noncontiguous_kv(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    p = {
        "hkv": 4,
        "kv_lens": _uniform(4, 54),
        "page_size": 8,
        "pages": "identity",
        "kv_storage": "padded",
        "return_lse": False,
    }
    out: list[Launch] = []
    name = r.decode("f16", "f16", 128, {**p, "hq": 4})
    if name:
        out.append((name, p, test))
    return _with_merges(r, out)


def decode_extreme_logits(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    dt = cfg["dtype"]
    p = {
        "hkv": 4,
        "kv_lens": [2],
        "page_size": 1,
        "pages": "offset",
        "kv_storage": "separate",
        "values": "extreme",
        "sm_scale": 1 / math.sqrt(128),
    }
    out: list[Launch] = []
    name = r.decode(dt, dt, 128, {**p, "hq": 32})
    if name:
        out.append((name, p, test))
    return _with_merges(r, out)


def sliding_window_decode(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    hq, hkv = cfg["num_qo_heads"], cfg["num_kv_heads"]
    if cfg["head_dim"] > 256 or hq % hkv:
        return []
    p = {
        "hkv": hkv,
        "kv_lens": _uniform(cfg["batch_size"], cfg["kv_len"]),
        "page_size": cfg["page_size"],
        "pages": "identity",
        "kv_storage": "separate",
        "window_left": cfg["window_left"],
        "return_lse": False,
    }
    out: list[Launch] = []
    name = r.decode("f16", "f16", cfg["head_dim"], {**p, "hq": hq})
    if name:
        out.append((name, p, test))
    return _with_merges(r, out)


def packed_decode(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    hq, hkv = cfg["num_qo_heads"], cfg["num_kv_heads"]
    if hq % hkv:
        return []
    p = {
        "hkv": hkv,
        "kv_lens": _uniform(cfg["batch_size"], cfg["seq_len"]),
        "page_size": cfg["page_size"],
        "pages": "identity",
        "kv_storage": "separate",
        "q_storage": "packed",
        "return_lse": False,
    }
    out: list[Launch] = []
    name = r.decode("f16", "f16", cfg["head_dim"], {**p, "hq": hq})
    if name:
        out.append((name, p, test))
    return _with_merges(r, out)


def batch_attention(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    """flashinfer.BatchAttention (persistent kernel), head dim 64 only."""
    if cfg["head_dim"] != 64:
        return []
    pairs = cfg["seq_len_pairs"]
    hkv = cfg["num_kv_heads"]
    dt = cfg["test_dtype"]
    cap = cfg["logits_soft_cap"]
    mask = "causal" if cfg["causal"] else "none"
    name = f"fmha_fa2_persistent_{dt}_{dt}_hd64_{mask}{'_softcap' if cap > 0 else ''}"
    if name not in r.index:
        return []
    p = {
        "hq": hkv * cfg["gqa_group_size"],
        "hkv": hkv,
        "q_lens": [int(q) for _, q in pairs],
        "kv_lens": [int(k) for k, _ in pairs],
        "page_size": cfg["page_block_size"],
        "kv_layout": NHD if cfg["layout"] == "NHD" else HND,
        "pages": "identity",
        "logits_soft_cap": cap,
        "v_scale": cfg["v_scale"] if cfg["v_scale"] is not None else 1.0,
    }
    return [(name, p, test)]


def batch_attention_noncontiguous(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    p = {
        "hq": 1,
        "hkv": 1,
        "q_lens": [146],
        "kv_lens": [146],
        "page_size": 1,
        "pages": "identity",
        "q_storage": "chunked",
    }
    return [("fmha_fa2_persistent_bf16_bf16_hd64_causal", p, test)]


def mla(
    r: Resolver, cfg: dict, test: str, kv_lens: list[int], q_len: int, poison: bool
) -> list[Launch]:
    causal = cfg["causal"]
    if causal and q_len > min(kv_lens):
        return []
    dt = cfg["dtype"]
    name = f"fmha_fa2_mla_{dt}_{'causal' if causal else 'none'}"
    if name not in r.index:
        return []
    p = {
        "heads": cfg["num_heads"],
        "q_lens": [q_len] * len(kv_lens),
        "kv_lens": kv_lens,
        "page_size": cfg["page_size"],
        "pages": "identity",
        "poison": poison,
    }
    return [(name, p, test)]


def merge_state_trace(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    s = cfg["shape_kwargs"]
    return [
        (
            "fmha_fa2_merge_state_f16_vec8",
            {"seq": s["seq_len"], "heads": s["num_heads"], "d": s["head_dim"]},
            test,
        )
    ]


def merge_state_in_place_trace(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    s = cfg["shape_kwargs"]
    p = {
        "seq": s["seq_len"],
        "heads": s["num_heads"],
        "d": s["head_dim"],
        "mask": "none",
    }
    return [("fmha_fa2_merge_state_in_place_f16_vec8", p, test)]


def merge_states_trace(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    s = cfg["shape_kwargs"]
    p = {
        "seq": s["seq_len"],
        "sets": s["num_states"],
        "heads": s["num_heads"],
        "d": s["head_dim"],
    }
    return [("fmha_fa2_merge_states_f16_vec8", p, test)]


def merge_in_place_with_mask(r: Resolver, cfg: dict, test: str) -> list[Launch]:
    name = "fmha_fa2_merge_state_in_place_f16_vec8"
    return [
        (name, {"seq": 512, "heads": 32, "d": 128, "mask": mask}, test)
        for mask in ("none", "ones", "zeros", "random")
    ]


def triton_cascade(r: Resolver, cfg: dict, test: str, kernel: str) -> list[Launch]:
    seq, h, d = cfg["seq_len"], cfg["num_heads"], cfg["head_dim"]
    if kernel == "merge_state":
        return [
            ("fmha_fa2_merge_state_f16_vec8", {"seq": seq, "heads": h, "d": d}, test)
        ]
    if kernel == "merge_state_in_place":
        p = {"seq": seq, "heads": h, "d": d, "mask": "none"}
        return [("fmha_fa2_merge_state_in_place_f16_vec8", p, test)]
    if kernel == "merge_states":
        p = {"seq": seq, "sets": cfg["num_states"], "heads": h, "d": d}
        return [("fmha_fa2_merge_states_f16_vec8", p, test)]
    # variable_length_merge_states reference: merge_states per row with
    # seq_len 1 and 1..511 index sets (unseeded): the large kernel.
    name = "fmha_fa2_merge_states_large_f16_hd128"
    return [
        (name, {"seq": 1, "sets": n, "heads": h, "d": d}, test) for n in (1, 255, 511)
    ]


AdapterFn = Callable[[Resolver, dict, str], list[Launch]]
A = "attention/"
TESTS: list[tuple[str, str, AdapterFn]] = [
    (A + "test_batch_prefill_kernels.py", "test_batch_prefill_with_paged_kv_cache",
     lambda r, c, t: batch_prefill_paged(r, c, t, tuple_kv=False)),
    (A + "test_batch_prefill_kernels.py", "test_batch_prefill_with_tuple_paged_kv_cache",
     lambda r, c, t: batch_prefill_paged(r, c, t, tuple_kv=True)),
    (A + "test_batch_prefill_kernels.py", "test_batch_prefill_with_paged_kv_cache_custom_mask",
     lambda r, c, t: batch_prefill_custom(r, c, t, "paged")),
    (A + "test_batch_prefill_kernels.py", "test_batch_prefill_with_ragged_kv_cache",
     batch_prefill_ragged),
    (A + "test_batch_prefill_kernels.py", "test_batch_prefill_with_ragged_kv_cache_custom_mask",
     lambda r, c, t: batch_prefill_custom(r, c, t, "ragged")),
    (A + "test_batch_prefill_kernels.py", "test_batch_prefill_lazy_stride_router_plan_reuse",
     lazy_stride_router),
    (A + "test_batch_prefill_kernels.py", "test_ragged_prefill_one_valid_key", one_valid_key),
    (A + "test_batch_prefill_kernels.py", "test_paged_prefill_fully_masked_rows",
     fully_masked_rows),
    (A + "test_batch_prefill_kernels.py", "test_paged_prefill_split_kv_empty_chunk",
     split_kv_empty_chunk),
    (A + "test_tensor_cores_decode.py", "test_batch_decode_tensor_cores",
     lambda r, c, t: tensor_core_decode(r, c, t, graph=False)),
    (A + "test_tensor_cores_decode.py", "test_batch_decode_tensor_cores_cuda_graph",
     lambda r, c, t: tensor_core_decode(r, c, t, graph=True)),
    (A + "test_tensor_cores_decode.py", "test_batch_decode_tensor_cores_with_fast_plan",
     lambda r, c, t: tensor_core_decode(r, c, t, graph=False)),
    (A + "test_tensor_cores_decode.py", "test_batch_fast_decode_tensor_cores_cuda_graph",
     lambda r, c, t: tensor_core_decode(r, c, t, graph=True)),
    (A + "test_batch_invariant_fa2.py", "test_batch_decode_tensor_cores",
     lambda r, c, t: tensor_core_decode(r, c, t, graph=False)),
    (A + "test_batch_invariant_fa2.py", "test_batch_prefill_tensor_cores",
     tensor_core_prefill_invariant),
    (A + "test_batch_decode_kernels.py", "test_cuda_graph_uniform_multi_token_decode_with_paged_kv_cache",
     uniform_multi_token_decode),
    (A + "test_decode_prefill_lse.py", "test_mlc_failed_case", mlc_failed_case),
    (A + "test_batch_prefill.py", "test_kv_scale_forwarding_effect",
     lambda r, c, t: kv_scale(r, c, t, 8, (0.1, 2.0))),
    (A + "test_batch_prefill.py", "test_kv_scale_forwarding_math_property",
     lambda r, c, t: kv_scale(r, c, t, 128, (1.0, 0.5))),
    (A + "test_lse_base.py", "test_ragged_lse_base", lambda r, c, t: lse_base(r, c, t, "ragged")),
    (A + "test_lse_base.py", "test_paged_lse_base", lambda r, c, t: lse_base(r, c, t, "paged")),
    (A + "test_non_contiguous_prefill.py", "test_batch_ragged_prefill_packed_input",
     lambda r, c, t: packed_prefill(r, c, t, "ragged")),
    (A + "test_non_contiguous_prefill.py", "test_batch_paged_prefill_packed_input",
     lambda r, c, t: packed_prefill(r, c, t, "paged")),
    (A + "test_sliding_window.py", "test_batch_paged_prefill_sliding_window",
     lambda r, c, t: sliding_window_prefill(r, c, t, "paged")),
    (A + "test_sliding_window.py", "test_batch_ragged_prefill_sliding_window",
     lambda r, c, t: sliding_window_prefill(r, c, t, "ragged")),
    (A + "test_shared_prefix_kernels.py", "test_batch_attention_with_shared_prefix_paged_kv_cache",
     shared_prefix_cascade),
    (A + "test_batch_decode_kernels.py", "test_batch_decode_with_paged_kv_cache",
     lambda r, c, t: batch_decode(r, c, t, tuple_kv=False, graph=False)),
    (A + "test_batch_decode_kernels.py", "test_batch_decode_with_paged_kv_cache_with_fast_plan",
     lambda r, c, t: batch_decode(r, c, t, tuple_kv=False, graph=False)),
    (A + "test_batch_decode_kernels.py", "test_batch_decode_with_tuple_paged_kv_cache",
     lambda r, c, t: batch_decode(r, c, t, tuple_kv=True, graph=False)),
    (A + "test_batch_decode_kernels.py", "test_cuda_graph_batch_decode_with_paged_kv_cache",
     lambda r, c, t: batch_decode(r, c, t, tuple_kv=False, graph=True)),
    (A + "test_batch_decode_kernels.py", "test_batch_decode_rejects_unequal_kv_strides_nvfp4_contract",
     decode_noncontiguous_kv),
    (A + "test_batch_decode_kernels.py", "test_paged_decode_extreme_negative_logits",
     decode_extreme_logits),
    (A + "test_sliding_window.py", "test_batch_decode_sliding_window", sliding_window_decode),
    (A + "test_non_contiguous_decode.py", "test_batch_paged_decode_packed_input", packed_decode),
    (A + "test_batch_attention.py", "test_batch_attention_correctness", batch_attention),
    (A + "test_batch_attention.py", "test_batch_attention_with_noncontiguous_q",
     batch_attention_noncontiguous),
    (A + "test_deepseek_mla.py", "test_batch_mla_varlen_page_attention",
     lambda r, c, t: mla(r, c, t, [c["kv_len_0"], c["kv_len_1"], c["kv_len_2"]] * c["batch_size"],
                         c["qo_len"], False) if c["backend"] == "fa2" else []),
    (A + "test_deepseek_mla.py", "test_batch_mla_oob_kv_nan",
     lambda r, c, t: mla(r, c, t, [c["kv_len"]] * c["batch_size"], c["qo_len"], True)
     if c["backend"] == "fa2" else []),
    (A + "test_deepseek_mla.py", "test_batch_mla_page_attention",
     lambda r, c, t: mla(r, c, t, [c["kv_len"]] * c["batch_size"], c["qo_len"], False)
     if c["backend"] == "fa2" else []),
    ("trace/test_merge_state_reference_correctness.py", "test_merge_state_reference_correctness",
     merge_state_trace),
    ("trace/test_merge_state_in_place_reference_correctness.py",
     "test_merge_state_in_place_reference_correctness", merge_state_in_place_trace),
    ("trace/test_merge_states_reference_correctness.py", "test_merge_states_reference_correctness",
     merge_states_trace),
    (A + "test_shared_prefix_kernels.py", "test_merge_state_in_place_with_mask",
     merge_in_place_with_mask),
    ("utils/test_triton_cascade.py", "test_merge_state",
     lambda r, c, t: triton_cascade(r, c, t, "merge_state")),
    ("utils/test_triton_cascade.py", "test_merge_state_in_place",
     lambda r, c, t: triton_cascade(r, c, t, "merge_state_in_place")),
    ("utils/test_triton_cascade.py", "test_merge_states",
     lambda r, c, t: triton_cascade(r, c, t, "merge_states")),
    ("utils/test_triton_cascade.py", "test_variable_length_merge_states",
     lambda r, c, t: triton_cascade(r, c, t, "varlen")),
]  # fmt: skip


def launches(
    root: Path, fa2: Any, resolver: Resolver | None = None
) -> Iterator[Launch]:
    """Every sm_86 kernel launch of the ``TESTS`` parametrizations (the
    defective ones are counted in ``resolver.defects`` instead)."""
    resolver = resolver or Resolver(fa2)
    for file, function, adapter in TESTS:
        for cfg in parametrizations(root, file, function):
            yield from adapter(resolver, cfg, _test_id(file, function, cfg))


def representatives(root: Path, fa2: Any) -> dict[str, list[dict[str, Any]]]:
    """Per kernel, the cheapest upstream launch of every regime it is tested
    in (``fa2.regime``), with its suite (``fa2.case_suite``)."""
    best: dict[tuple[str, str], tuple[float, dict[str, Any], str]] = {}
    for name, params, test in launches(root, fa2):
        key = (name, fa2.regime_key(name, params))
        cost = fa2.case_cost(name, params)
        if key not in best or cost < best[key][0]:
            best[key] = (cost, params, test)
    table: dict[str, list[dict[str, Any]]] = {}
    for (name, _), (cost, params, test) in sorted(best.items()):
        table.setdefault(name, []).append(
            {"test": test, "params": params, "suite": fa2.case_suite(name, params)}
        )
    return table

"""FlashInfer v0.6.9 tests that launch the batched_gemm kernels, as problems.

The pinned test files (``resources/flashinfer-<rev>/tests``) are read, not
imported: :func:`parametrizations` evaluates each test function's
``pytest.mark.parametrize`` decorators symbolically from the AST (names and
enum members become their dotted names, ``pytest.param`` keeps its id) and
expands the cross product. Each ``*_problems`` function then binds a test's
parametrizations to the problems its body hands to FlashInfer, applying the
test's own skip rules (``skip_checks`` of ``tests/moe/utils.py`` is ported
line by line in :func:`moe_skipped`, for an SM100 device), so the result is
the set of problems the upstream suite runs on a B200:

* :func:`moe_problems`: fused-MoE calls (quantization mode, activation,
  weight preparation, tokens, hidden, intermediate, local experts, top-k and
  the runtime features: GEMM biases, routing scales on the input, all-zero
  hidden states, autotuning). Which batched-GEMM configs FlashInfer may then
  launch for a problem is decided by its own code at compile time
  (``impls/batched_gemm/kernels/bmm_moe_probe.cu``: tile selection and the
  MoE runners' valid configs).
* :func:`dense_problems`: ``mm_fp4`` / ``mm_mxfp8`` / ``gemm_fp8_nt_groupwise``
  with ``backend="trtllm"`` and ``mm_fp8`` (low latency).
* :func:`segment_problems`: ``SegmentGEMMWrapper`` (sm80 backend, row-major
  weights).

Calls that reach no kernel of this package are not listed: other backends,
``auto`` (FlashInfer's heuristics never pick trtllm for ``mm_fp4`` /
``mm_mxfp8``), ``tests/autotuner``'s ``test_fp4_moe_autotune`` (E2m1 weights
with BF16 activations: no trtllm-gen config has these dtypes) and every
``tests/moe/test_dpsk_fused_moe_fp8.py`` case (its ``skip_checks`` call skips
SwiGlu with hidden size 7168 > 1024).
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Iterator
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

REVISION = "a1aa676196f798435248d9ea205c67674476f473"  # FlashInfer v0.6.9


def tests_root(resources: Path) -> Path:
    return resources / f"flashinfer-{REVISION}" / "tests"


# --- symbolic evaluation of parametrize decorators ---------------------------------


class Param(NamedTuple):
    id: str
    values: tuple[Any, ...]


class Call(NamedTuple):
    func: str
    kwargs: tuple[tuple[str, Any], ...]

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self.kwargs).get(key, default)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    raise ValueError(f"unsupported expression {ast.dump(node)}")


def sym(node: ast.AST) -> Any:
    """Literal value of ``node``; names/attributes become dotted strings
    (``ActivationType.Swiglu.value`` -> ``"ActivationType.Swiglu"``), calls
    :class:`Call` (``pytest.param``: :class:`Param`)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(sym(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {sym(k): sym(v) for k, v in zip(node.keys, node.values) if k is not None}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -sym(node.operand)
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _dotted(node).removesuffix(".value")
    if isinstance(node, ast.Call):
        func = _dotted(node.func)
        kwargs = {k.arg: sym(k.value) for k in node.keywords if k.arg}
        if func == "pytest.param":
            ident = kwargs.get("id")
            return Param(ident, tuple(sym(a) for a in node.args))
        return Call(func, tuple(sorted(kwargs.items())))
    if isinstance(node, ast.Lambda):
        return None  # ids=lambda ...
    raise ValueError(f"unsupported expression {ast.dump(node)}")


@cache
def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in _module(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"{path.name} has no {name}")


def _assignment(path: Path, name: str) -> ast.AST:
    for node in _module(path).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise KeyError(f"{path.name} has no module-level {name}")


def decorator_grid(path: Path, function: str) -> list[tuple[tuple[str, ...], list]]:
    """[(argnames, [(id, values), ...])] of each parametrize decorator,
    innermost (applied first) first."""
    grid = []
    for deco in reversed(_function(path, function).decorator_list):
        if not (
            isinstance(deco, ast.Call) and _dotted(deco.func).endswith("parametrize")
        ):
            continue
        names = tuple(n.strip() for n in sym(deco.args[0]).split(","))
        entries = []
        values_node = deco.args[1]
        if isinstance(values_node, ast.Name):  # a module-level list
            values_node = _assignment(path, values_node.id)
        for i, value in enumerate(sym(values_node)):
            if isinstance(value, Param):
                values = value.values if len(names) > 1 else value.values[:1]
                ident = value.id
            else:
                values = tuple(value) if len(names) > 1 else (value,)
                ident = None
            if ident is None:
                scalar = len(names) == 1 and isinstance(values[0], (int, float, str))
                ident = str(values[0]) if scalar else f"{names[0]}{i}"
            entries.append((ident, values))
        grid.append((names, entries))
    return grid


def parametrizations(path: Path, function: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """(pytest-style id, {argname: value}) for every grid point."""
    grid = decorator_grid(path, function)
    for combo in itertools.product(*(entries for _, entries in grid)):
        params: dict[str, Any] = {}
        ids = []
        for (names, _), (ident, values) in zip(grid, combo):
            params.update(zip(names, values))
            ids.append(str(ident))
        yield "-".join(ids), params


# --- fused MoE -------------------------------------------------------------------------

# QuantMode (tests/moe/utils.py) -> (weight dtype, activation dtype, DeepSeek
# FP8) of the batched GEMMs the launcher builds.
QUANT_DTYPES = {
    "FP4_NVFP4_NVFP4": ("E2m1", "E2m1", False),
    "FP4_MXFP4_MXFP8": ("MxE2m1", "MxE4m3", False),
    "FP4_MXFP4_Bf16": ("MxE2m1", "Bfloat16", False),
    "FP8_BLOCK_SCALE_DEEPSEEK": ("E4m3", "E4m3", True),
    "FP8_BLOCK_SCALE_MXFP8": ("MxE4m3", "MxE4m3", False),
    "FP8_PER_TENSOR": ("E4m3", "E4m3", False),
    "BF16": ("Bfloat16", "Bfloat16", False),
    "MXINT4_BF16_BF16": ("MxInt4", "Bfloat16", False),
}
NON_GATED_QUANT = ("FP4_NVFP4_NVFP4", "FP8_BLOCK_SCALE_MXFP8", "FP8_PER_TENSOR", "BF16")
FP32_LOGITS_QUANT = (
    "FP4_NVFP4_NVFP4",
    "FP8_PER_TENSOR",
    "FP8_BLOCK_SCALE_DEEPSEEK",
    "FP8_BLOCK_SCALE_MXFP8",
    "BF16",
)
GATED = ("Swiglu", "Geglu")


def _last(name: Any) -> str:
    return str(name).rsplit(".", 1)[-1]


def moe_impl(value: Any) -> tuple[str, str]:
    """(implementation class, QuantMode) of a test's ``moe_impl`` value."""
    if not isinstance(value, Call):
        raise ValueError(f"unexpected moe_impl {value!r}")
    fixed = {
        "BF16Moe": "BF16",
        "FP8PerTensorMoe": "FP8_PER_TENSOR",
        "MxInt4BlockScaleMoe": "MXINT4_BF16_BF16",
    }
    if value.func in fixed:
        return value.func, fixed[value.func]
    mode = value.get("fp8_quantization_type") or value.get("quant_mode")
    return value.func, _last(mode)


def moe_skipped(
    impl: str,
    quant: str,
    routing: dict[str, Any],
    weights: dict[str, Any],
    act: str,
    tokens: int,
    hidden: int,
    inter: int,
    logits_fp32: bool,
    zero: bool,
) -> bool:
    """``skip_checks`` (tests/moe/utils.py) on an SM100 device."""
    if zero and impl != "FP8BlockScaleMoe":
        return True
    method = _last(routing["routing_method_type"])
    if act == "Geglu" and (
        impl != "FP4Moe"
        or quant != "FP4_NVFP4_NVFP4"
        or method != "TopK"
        or tokens > 128
    ):
        return True
    if act == "Swiglu" and (hidden > 1024 or inter > 1024):
        return True
    acts = routing.get("compatible_activation_types")
    if acts is not None and act not in {_last(a) for a in acts}:
        return True
    if act not in GATED and quant not in NON_GATED_QUANT:
        return True
    if routing["num_experts"] > 512 and inter > 512:
        return True
    if impl not in routing["compatible_moe_impls"]:
        return True
    if impl not in weights["compatible_moe_impls"]:
        return True
    if quant == "FP8_BLOCK_SCALE_MXFP8" and not weights["use_shuffled_weight"]:
        return True
    if quant == "FP8_BLOCK_SCALE_MXFP8" and _last(weights["layout"]) != "MajorK":
        return True
    if inter not in routing["compatible_intermediate_size"]:
        return True
    if quant == "MXINT4_BF16_BF16" and (inter % 256 or hidden % 256):
        return True
    return logits_fp32 and quant not in FP32_LOGITS_QUANT


def moe_problem(
    test: str,
    quant: str,
    act: str,
    shuffled: bool,
    layout: str,
    tokens: int,
    hidden: int,
    inter: int,
    experts: int,
    top_k: int,
    *,
    bias1: bool = False,
    bias2: bool = False,
    routing_scales: bool = False,
    zero: bool = False,
    autotune: bool = True,
    all_tiles: bool = False,
) -> dict[str, Any]:
    return {
        "test": test,
        "quant": quant,
        "act": act,
        "shuffled": bool(shuffled),
        "layout": _last(layout),
        "tokens": tokens,
        "hidden": hidden,
        "inter": inter,
        "experts": experts,
        "top_k": top_k,
        "bias1": bias1,
        "bias2": bias2,
        "routing_scales": routing_scales,
        "zero": zero,
        "autotune": bool(autotune),
        "all_tiles": all_tiles,
    }


def _run_moe_test(
    test: str, p: dict[str, Any], **extra: Any
) -> Iterator[dict[str, Any]]:
    """``run_moe_test``'s problem (or nothing when ``skip_checks`` skips)."""
    impl, quant = moe_impl(p["moe_impl"])
    routing, weights = p["routing_config"], p["weight_processing"]
    act = _last(p["activation_type"])
    tokens, hidden, inter = p["num_tokens"], p["hidden_size"], p["intermediate_size"]
    zero = bool(p.get("zero_hidden_states", False))
    logits_fp32 = p.get("routing_logits_dtype", "torch.bfloat16") == "torch.float32"
    if moe_skipped(
        impl, quant, routing, weights, act, tokens, hidden, inter, logits_fp32, zero
    ):
        return
    yield moe_problem(
        test,
        quant,
        act,
        weights["use_shuffled_weight"],
        weights["layout"],
        tokens,
        hidden,
        inter,
        routing["num_experts"],
        routing["top_k"],
        # run_moe_test: use_routing_scales_on_input for Llama4 routing.
        routing_scales=_last(routing["routing_method_type"]) == "Llama4",
        zero=zero,
        autotune=routing.get("enable_autotune", True),
        **extra,
    )


MOE_FILE = "moe/test_trtllm_gen_fused_moe.py"
ROUTED_FILE = "moe/test_trtllm_gen_routed_fused_moe.py"
DPSK_FILE = "moe/test_dpsk_fused_moe_fp8.py"
AUTOTUNER_FILE = "autotuner/test_trtllm_fused_moe_autotuner_integration.py"

# Tests whose parametrize grid holds run_moe_test's own arguments.
RUN_MOE_TESTS = (
    "test_renormalize_routing",
    "test_deepseekv3_routing",
    "test_topk_routing",
    "test_llama4_routing",
    "test_dyn_block_kernel_routing",
    "test_tier_1024_experts_routing",
    "test_routing_dtype_flexibility",
)
SHUFFLED_MAJOR_K = {
    "use_shuffled_weight": True,
    "layout": "WeightLayout.MajorK",
    "compatible_moe_impls": ("FP4Moe", "FP8PerTensorMoe", "FP8BlockScaleMoe"),
}


def _routing(experts, top_k, method, impls, inters, acts=None, autotune=True):
    out = {
        "num_experts": experts,
        "top_k": top_k,
        "routing_method_type": method,
        "compatible_moe_impls": impls,
        "compatible_intermediate_size": inters,
        "enable_autotune": autotune,
    }
    if acts is not None:
        out["compatible_activation_types"] = acts
    return out


def moe_problems(resources: Path) -> list[dict[str, Any]]:
    """Every fused-MoE problem the upstream tests run (see module docstring)."""
    root = tests_root(resources)
    out: list[dict[str, Any]] = []

    def grid(file: str, function: str) -> Iterator[tuple[str, dict[str, Any]]]:
        for ident, p in parametrizations(root / file, function):
            yield f"tests/{file}::{function}[{ident}]", p

    for function in RUN_MOE_TESTS:
        for test, p in grid(MOE_FILE, function):
            out += _run_moe_test(test, p)
    for test, p in grid(MOE_FILE, "test_nvfp4_moe_gemm_bias"):
        p = {
            **p,
            "moe_impl": Call("FP4Moe", (("quant_mode", "QuantMode.FP4_NVFP4_NVFP4"),)),
            "routing_config": _routing(
                8,
                2,
                "RoutingMethodType.Renormalize",
                ("FP4Moe",),
                (512, 768, 1024, 2048),
            ),
            "weight_processing": SHUFFLED_MAJOR_K,
            "activation_type": "ActivationType.Swiglu",
        }
        out += _run_moe_test(
            test, p, bias1="gemm1" in p["bias"], bias2="gemm2" in p["bias"]
        )
    mxfp8 = Call(
        "FP8BlockScaleMoe",
        (("fp8_quantization_type", "QuantMode.FP8_BLOCK_SCALE_MXFP8"),),
    )
    for test, p in grid(MOE_FILE, "test_mxfp8_block_scale_moe_relu2_non_gated"):
        p = {**p, "moe_impl": mxfp8, "activation_type": "ActivationType.Relu2"}
        out += _run_moe_test(test, p)
    single = {
        "num_tokens": 128,
        "hidden_size": 1024,
        "intermediate_size": 512,
        "moe_impl": mxfp8,
        "routing_config": _routing(
            512,
            22,
            "RoutingMethodType.DeepSeekV3",
            ("FP8BlockScaleMoe",),
            (512,),
            ("ActivationType.Relu2",),
            autotune=False,
        ),
        "weight_processing": {
            **SHUFFLED_MAJOR_K,
            "compatible_moe_impls": ("FP8BlockScaleMoe",),
        },
        "activation_type": "ActivationType.Relu2",
        "routing_logits_dtype": "torch.float32",
    }
    out += _run_moe_test(
        f"tests/{MOE_FILE}::test_mxfp8_block_scale_moe_relu2_deepseekv3_topk22", single
    )
    for function, impl_name, logits in (
        (
            "test_fp8_block_scale_autotune_valid_configs",
            "FP8BlockScaleMoe",
            "torch.float32",
        ),
        (
            "test_fp8_per_tensor_autotune_valid_configs_nonefp8",
            "FP8PerTensorMoe",
            "torch.bfloat16",
        ),
    ):
        for test, p in grid(MOE_FILE, function):
            case = p["autotune_case"]
            impl = (
                Call(impl_name, (("fp8_quantization_type", case["quant_mode"]),))
                if impl_name == "FP8BlockScaleMoe"
                else Call(impl_name, ())
            )
            q = {
                "num_tokens": case["num_tokens"],
                "hidden_size": case["hidden_size"],
                "intermediate_size": case["intermediate_size"],
                "moe_impl": impl,
                "routing_config": _routing(
                    case["num_experts"],
                    case["top_k"],
                    case.get("routing_method_type", "RoutingMethodType.Renormalize"),
                    (impl_name,),
                    (case["intermediate_size"],),
                    (case["activation_type"],),
                ),
                "weight_processing": {
                    **SHUFFLED_MAJOR_K,
                    "compatible_moe_impls": (impl_name,),
                },
                "activation_type": case["activation_type"],
                "routing_logits_dtype": logits,
            }
            out += _run_moe_test(test, q)
    # Direct calls without skip_checks.
    out.append(
        moe_problem(
            f"tests/{MOE_FILE}::test_fp8_block_scale_routed_activation_type_relu2_smoke",
            "FP8_BLOCK_SCALE_MXFP8",
            "Relu2",
            True,
            "MajorK",
            32,
            512,
            512,
            64,
            8,
            autotune=False,
        )
    )
    modes = {
        "NvFP4xNvFP4": "FP4_NVFP4_NVFP4",
        "MxFP4xMxFP8": "FP4_MXFP4_MXFP8",
        "MxFP4xBf16": "FP4_MXFP4_Bf16",
    }
    for test, p in grid(ROUTED_FILE, "test_trtllm_gen_routed_fused_moe"):
        out.append(
            moe_problem(
                test,
                modes[p["quant_mode"]],
                "Swiglu",
                True,
                "MajorK",
                p["num_tokens"],
                p["hidden_size"],
                p["intermediate_size"],
                p["num_experts"],
                p["top_k"],
                autotune=False,
            )
        )
    for function, quant, shuffled, layout in (
        (
            "test_trtllm_gen_fp8_routed_fused_moe",
            "FP8_BLOCK_SCALE_DEEPSEEK",
            False,
            "MajorK",
        ),
        ("test_trtllm_gen_bf16_routed_fused_moe", "BF16", True, "BlockMajorK"),
        (
            "test_fp8_block_scale_moe_routing_replay",
            "FP8_BLOCK_SCALE_DEEPSEEK",
            False,
            "MajorK",
        ),
    ):
        for test, p in grid(ROUTED_FILE, function):
            out.append(
                moe_problem(
                    test,
                    quant,
                    "Swiglu",
                    shuffled,
                    layout,
                    p["num_tokens"],
                    p["hidden_size"],
                    p["intermediate_size"],
                    p["num_experts"],
                    p["top_k"],
                    autotune=False,
                )
            )
    for test, p in grid(
        ROUTED_FILE, "test_trtllm_gen_fp8_mxfp8_routed_activation_parity"
    ):
        out.append(
            moe_problem(
                test,
                "FP8_BLOCK_SCALE_MXFP8",
                _last(p["activation_type"]),
                True,
                "MajorK",
                32,
                512,
                512,
                64,
                8,
                autotune=False,
            )
        )
    # test_correctness_dpsk_fp8_fused_moe: skip_checks(hidden_size=7168, Swiglu).
    for test, p in grid(DPSK_FILE, "test_correctness_dpsk_fp8_fused_moe"):
        impl = Call(
            "FP8BlockScaleMoe",
            (("fp8_quantization_type", "QuantMode.FP8_BLOCK_SCALE_DEEPSEEK"),),
        )
        routing = {
            **p["routing_config"],
            "routing_method_type": "RoutingMethodType.DeepSeekV3",
            "compatible_moe_impls": ("FP8BlockScaleMoe",),
        }
        weights = {
            **p["weight_processing"],
            "compatible_moe_impls": ("FP8BlockScaleMoe",),
        }
        q = {
            "num_tokens": p["seq_len"],
            "hidden_size": 7168,
            "intermediate_size": p["intermediate_size"],
            "moe_impl": impl,
            "routing_config": routing,
            "weight_processing": weights,
            "activation_type": "ActivationType.Swiglu",
            "routing_logits_dtype": "torch.float32",
        }
        found = list(_run_moe_test(test, q))
        if found:
            raise AssertionError(f"{test} is no longer skipped: bind its local experts")
    # tests/autotuner: the BF16 test forces every supported tile (its
    # profiling bias), tuned at 256 tokens and run at 500.
    for test, p in grid(
        AUTOTUNER_FILE, "test_bf16_moe_all_supported_tile_n_inference_succeed"
    ):
        for tokens in (p["tune_num_tokens"], p["infer_num_tokens"]):
            out.append(
                moe_problem(
                    test,
                    "BF16",
                    "Swiglu",
                    True,
                    "BlockMajorK",
                    tokens,
                    1024,
                    1024,
                    p["num_experts"],
                    p["top_k"],
                    all_tiles=True,
                )
            )
    for test, p in grid(AUTOTUNER_FILE, "test_fp8_moe_autotune"):
        out.append(
            moe_problem(
                test,
                "FP8_BLOCK_SCALE_DEEPSEEK",
                "Swiglu",
                False,
                "MajorK",
                p["num_tokens"],
                512,
                512,
                p["num_experts"],
                p["top_k"],
            )
        )
    return out


# --- dense GEMMs ---------------------------------------------------------------------


def dense_problems(resources: Path) -> list[dict[str, Any]]:
    """Dense trtllm-gen GEMM calls: role, runner (m, n, k), the activations'
    SF layout (``SfLayout`` value: 1 = R8c4, 3 = R128c4; None: no SFs) and
    whether the autotuner runs every valid tactic."""
    root = tests_root(resources)
    out = []

    def add(test, role, m, n, k, sf_layout, autotune):
        out.append(
            {
                "test": test,
                "role": role,
                "m": m,
                "n": n,
                "k": k,
                "sf_layout": sf_layout,
                "autotune": bool(autotune),
            }
        )

    file = "gemm/test_mm_fp4.py"
    for ident, p in parametrizations(root / file, "test_mm_fp4"):
        # trtllm: BF16 output, NVFP4 only (mx_fp4 is cudnn/cute-dsl/auto only).
        if p["backend"] != "trtllm" or p["res_dtype"] != "torch.bfloat16":
            continue
        if p["fp4_type"] != "nvfp4":
            continue
        add(
            f"tests/{file}::test_mm_fp4[{ident}]",
            "gemm_fp4",
            p["m"],
            p["n"],
            p["k"],
            3 if p["use_128x4_sf_layout"] else 1,
            p["auto_tuning"],
        )
    file = "gemm/test_mm_mxfp8.py"
    for function in ("test_mm_mxfp8", "test_mm_mxfp8_large_dimensions"):
        for ident, p in parametrizations(root / file, function):
            if p["backend"] != "trtllm" or not p["is_sf_swizzled_layout"]:
                continue
            if p["k"] % 256 or p["out_dtype"] != "torch.bfloat16":
                continue
            # use_8x4_sf_layout_for_a=backend == "trtllm"
            add(
                f"tests/{file}::{function}[{ident}]",
                "gemm_mxfp8",
                p["m"],
                p["n"],
                p["k"],
                1,
                p.get("auto_tuning", False),
            )
    file = "gemm/test_groupwise_scaled_gemm_fp8.py"
    for ident, p in parametrizations(root / file, "test_fp8_groupwise_gemm"):
        if p["backend"] != "trtllm" or p["scale_major_mode"] != "MN" or p["k"] < 256:
            continue
        add(
            f"tests/{file}::test_fp8_groupwise_gemm[{ident}]",
            "gemm_fp8_blockscale",
            p["m"],
            p["n"],
            p["k"],
            None,
            False,
        )
    file = "gemm/test_mm_fp8.py"
    for ident, p in parametrizations(root / file, "test_mm_fp8"):
        add(
            f"tests/{file}::test_mm_fp8[{ident}]",
            "gemm_low_latency",
            p["m"],
            p["n"],
            p["k"],
            None,
            True,
        )
    return out


# --- segment GEMM (sm_86) ----------------------------------------------------------------


def segment_problems(resources: Path) -> list[dict[str, Any]]:
    """``test_segment_gemm`` with the sm80 backend and row-major weights
    (FP16 only): segment lengths, n = d_out, k = d_in. ``use_weight_indices``
    only changes which weight each segment's pointer addresses (``w[i %
    1024]`` of a 1024-weight pool, i < 199): the kernel reads the same
    per-segment pointer arrays, so both values map to one problem."""
    file = "gemm/test_group_gemm.py"
    out = {}
    for ident, p in parametrizations(tests_root(resources) / file, "test_segment_gemm"):
        if p["backend"] != "sm80" or p["column_major"] or p["dtype"] != "torch.float16":
            continue
        if p["batch_size"] * p["num_rows_per_batch"] > 8192:
            continue
        key = (p["batch_size"], p["num_rows_per_batch"], p["d_in"], p["d_out"])
        out.setdefault(key, f"tests/{file}::test_segment_gemm[{ident}]")
    return [
        {"test": test, "lengths": [rows] * batch, "n": n, "k": k}
        for (batch, rows, k, n), test in out.items()
    ]

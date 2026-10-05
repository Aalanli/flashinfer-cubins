"""Model architectures that throughput cases draw their shapes from.

Values are the published Hugging Face ``config.json`` fields of each model
(attention heads, KV heads, head dims, hidden/intermediate sizes, routed
experts and top-k). A case that uses one of them records it with
``harness.throughput.model_case`` so benchmark output names the model a shape
comes from. Kernels whose head counts or dims are compile-time constants (the
official contest definitions) take only batch, sequence and token axes from
here; the remaining dims are fixed by the definition.

Linear layers per decoder layer, as GEMM (n, k) with tokens as m:

* attention: ``qkv`` (fused q/k/v or MLA ``q_a``/``kv_a``), ``o``;
* dense MLP: ``gate_up`` (2 x intermediate), ``down``;
* MoE expert: ``gate_up`` (2 x moe_intermediate), ``down``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Model:
    name: str
    hf: str
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int = 0  # dense MLP; 0 when every layer is MoE
    experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0
    extra: tuple[tuple[str, Any], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        return dict(self.extra).get(key, default)

    @property
    def group(self) -> int:
        """Query heads per KV head (GQA group size)."""
        return self.heads // self.kv_heads

    def linear_shapes(self) -> dict[str, tuple[int, int]]:
        """GEMM (n, k) of each linear layer (tokens are m)."""
        shapes: dict[str, tuple[int, int]] = {}
        if self.get("kv_lora_rank"):
            q_lora, kv_lora = self.get("q_lora_rank"), self.get("kv_lora_rank")
            rope, nope, v = (
                self.get("qk_rope_head_dim"),
                self.get("qk_nope_head_dim"),
                self.get("v_head_dim"),
            )
            shapes["q_a_kv_a"] = (q_lora + kv_lora + rope, self.hidden)
            shapes["q_b"] = (self.heads * (nope + rope), q_lora)
            shapes["kv_b"] = (self.heads * (nope + v), kv_lora)
            shapes["o"] = (self.hidden, self.heads * v)
        else:
            qkv = (self.heads + 2 * self.kv_heads) * self.head_dim
            shapes["qkv"] = (qkv, self.hidden)
            shapes["o"] = (self.hidden, self.heads * self.head_dim)
        if self.intermediate:
            shapes["gate_up"] = (2 * self.intermediate, self.hidden)
            shapes["down"] = (self.hidden, self.intermediate)
        if self.experts:
            shapes["expert_gate_up"] = (2 * self.moe_intermediate, self.hidden)
            shapes["expert_down"] = (self.hidden, self.moe_intermediate)
        return shapes


MODELS: dict[str, Model] = {
    m.name: m
    for m in (
        Model(
            "deepseek_v3",
            "deepseek-ai/DeepSeek-V3",
            hidden=7168,
            heads=128,
            kv_heads=128,
            head_dim=192,
            intermediate=18432,  # first 3 dense layers
            experts=256,
            top_k=8,
            moe_intermediate=2048,
            extra=(
                ("q_lora_rank", 1536),
                ("kv_lora_rank", 512),
                ("qk_nope_head_dim", 128),
                ("qk_rope_head_dim", 64),
                ("v_head_dim", 128),
                ("n_group", 8),
                ("topk_group", 4),
                ("routed_scaling_factor", 2.5),
                ("shared_experts", 1),
                # DeepSeek-V3.2 sparse attention indexer.
                ("index_n_heads", 64),
                ("index_head_dim", 128),
                ("index_topk", 2048),
            ),
        ),
        Model(
            "kimi_k2",
            "moonshotai/Kimi-K2-Instruct",
            hidden=7168,
            heads=64,
            kv_heads=64,
            head_dim=192,
            intermediate=18432,
            experts=384,
            top_k=8,
            moe_intermediate=2048,
            extra=(
                ("q_lora_rank", 1536),
                ("kv_lora_rank", 512),
                ("qk_nope_head_dim", 128),
                ("qk_rope_head_dim", 64),
                ("v_head_dim", 128),
                ("n_group", 1),
                ("topk_group", 1),
            ),
        ),
        Model(
            "qwen3_235b_a22b",
            "Qwen/Qwen3-235B-A22B",
            hidden=4096,
            heads=64,
            kv_heads=4,
            head_dim=128,
            experts=128,
            top_k=8,
            moe_intermediate=1536,
        ),
        Model(
            "qwen3_30b_a3b",
            "Qwen/Qwen3-30B-A3B",
            hidden=2048,
            heads=32,
            kv_heads=4,
            head_dim=128,
            experts=128,
            top_k=8,
            moe_intermediate=768,
        ),
        Model(
            "qwen3_32b",
            "Qwen/Qwen3-32B",
            hidden=5120,
            heads=64,
            kv_heads=8,
            head_dim=128,
            intermediate=25600,
        ),
        Model(
            "qwen3_next_80b_a3b",
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            hidden=2048,
            heads=16,
            kv_heads=2,
            head_dim=256,
            experts=512,
            top_k=10,
            moe_intermediate=512,
            extra=(
                # Gated DeltaNet (linear attention) layers.
                ("linear_num_key_heads", 16),
                ("linear_num_value_heads", 32),
                ("linear_key_head_dim", 128),
                ("linear_value_head_dim", 128),
                ("shared_expert_intermediate", 512),
            ),
        ),
        Model(
            "llama3_8b",
            "meta-llama/Llama-3.1-8B",
            hidden=4096,
            heads=32,
            kv_heads=8,
            head_dim=128,
            intermediate=14336,
        ),
        Model(
            "llama3_70b",
            "meta-llama/Llama-3.1-70B",
            hidden=8192,
            heads=64,
            kv_heads=8,
            head_dim=128,
            intermediate=28672,
        ),
        Model(
            "llama3_405b",
            "meta-llama/Llama-3.1-405B",
            hidden=16384,
            heads=128,
            kv_heads=8,
            head_dim=128,
            intermediate=53248,
        ),
        Model(
            "llama4_scout",
            "meta-llama/Llama-4-Scout-17B-16E",
            hidden=5120,
            heads=40,
            kv_heads=8,
            head_dim=128,
            intermediate=16384,  # shared expert
            experts=16,
            top_k=1,
            moe_intermediate=8192,
        ),
        Model(
            "llama4_maverick",
            "meta-llama/Llama-4-Maverick-17B-128E",
            hidden=5120,
            heads=40,
            kv_heads=8,
            head_dim=128,
            intermediate=16384,
            experts=128,
            top_k=1,
            moe_intermediate=8192,
        ),
        Model(
            "mixtral_8x7b",
            "mistralai/Mixtral-8x7B-v0.1",
            hidden=4096,
            heads=32,
            kv_heads=8,
            head_dim=128,
            experts=8,
            top_k=2,
            moe_intermediate=14336,
        ),
        Model(
            "gpt_oss_120b",
            "openai/gpt-oss-120b",
            hidden=2880,
            heads=64,
            kv_heads=8,
            head_dim=64,
            experts=128,
            top_k=4,
            moe_intermediate=2880,
            extra=(("sliding_window", 128), ("attention_sinks", True)),
        ),
        Model(
            "gpt_oss_20b",
            "openai/gpt-oss-20b",
            hidden=2880,
            heads=64,
            kv_heads=8,
            head_dim=64,
            experts=32,
            top_k=4,
            moe_intermediate=2880,
            extra=(("sliding_window", 128), ("attention_sinks", True)),
        ),
        Model(
            "gemma2_9b",
            "google/gemma-2-9b",
            hidden=3584,
            heads=16,
            kv_heads=8,
            head_dim=256,
            intermediate=14336,
            extra=(("sliding_window", 4096), ("attn_logit_softcapping", 50.0)),
        ),
        Model(
            "gemma3_27b",
            "google/gemma-3-27b-pt",
            hidden=5376,
            heads=32,
            kv_heads=16,
            head_dim=128,
            intermediate=21504,
            extra=(("sliding_window", 1024),),
        ),
    )
}

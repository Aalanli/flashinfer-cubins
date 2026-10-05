# LLM kernel benchmark references

## Five FlashInfer competition kernels

| Workload | Definition and reference |
|---|---|
| Routed FP8 MoE | [moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048](https://bench.flashinfer.ai/kernels/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048) |
| DSA FP8 top-k indexer | [dsa_topk_indexer_fp8_h64_d128_topk2048_ps64](https://bench.flashinfer.ai/kernels/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64) |
| DSA sparse attention | [dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64](https://bench.flashinfer.ai/kernels/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64) |
| Gated Delta Net decode | [gdn_decode_qk4_v8_d128_k_last](https://bench.flashinfer.ai/kernels/gdn_decode_qk4_v8_d128_k_last) |
| Gated Delta Net prefill | [gdn_prefill_qk4_v8_d128_k_last](https://bench.flashinfer.ai/kernels/gdn_prefill_qk4_v8_d128_k_last) |

Shared resources:

- [Contest dataset: definitions, workloads and baseline solutions](https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest)
- [Official evaluation guide and baseline solution names](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit/blob/main/EVALUATION.md)
- [Contest starter kit](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit)
- [Competition website and results](https://mlsys26.flashinfer.ai/)

## Paged GQA decode

- [Definition and reference: gqa_paged_decode_h32_kv4_d128_ps1](https://bench.flashinfer.ai/kernels/gqa_paged_decode_h32_kv4_d128_ps1)
- [Paged GQA benchmark documentation](https://bench.flashinfer.ai/docs/op-types/gqa-paged)
- [FlashInfer trace dataset: definitions, workloads and available solutions](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)

## RMSNorm

- [Definition and reference: rmsnorm_h7168](https://bench.flashinfer.ai/kernels/rmsnorm_h7168)
- [FlashInfer trace dataset: definitions, workloads and available solutions](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)

## FP8 GEMM

- [FlashInfer groupwise FP8 GEMM API](https://docs.flashinfer.ai/generated/flashinfer.gemm.gemm_fp8_nt_groupwise.html)
- [FlashInfer GEMM API overview](https://docs.flashinfer.ai/api/gemm.html)
- [DeepGEMM optimized kernels](https://github.com/deepseek-ai/DeepGEMM)
- [DeepGEMM FP8/FP4 correctness and benchmark tests](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_fp8_fp4.py)
- [DeepGEMM input generators](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/generators.py)

## NVIDIA NVFP4 GEMM kernels

| Workload | Reference and input generator | Task and benchmark resources |
|---|---|---|
| Dense NVFP4 GEMM | [reference.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemm/reference.py) | [task.yml](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_gemm/task.yml) |
| Dual NVFP4 GEMM | [reference.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_dual_gemm/reference.py) | [Problem directory](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia/nvfp4_dual_gemm) |
| Grouped NVFP4 GEMM | [reference.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_group_gemm/reference.py) | [task.yml](https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/nvfp4_group_gemm/task.yml) |

Shared resources:

- [NVIDIA competition problem collection](https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia)
- [CUTLASS repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS/CuTe DSL persistent block-scaled GEMM](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py)
- [CUTLASS reference and benchmark utilities](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/utils.py)
- [FlashInfer FP4 GEMM API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)
- [FlashInfer GEMM API overview](https://docs.flashinfer.ai/api/gemm.html)

## General FlashInfer resources

- [FlashInfer optimized kernel library](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer API documentation](https://docs.flashinfer.ai/)
- [FlashInfer-Bench repository](https://github.com/flashinfer-ai/flashinfer-bench)
- [FlashInfer-Bench documentation](https://bench.flashinfer.ai/docs)

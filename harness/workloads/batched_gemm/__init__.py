"""batched_gemm single-kernel workloads, registered programmatically.

* ``trtllm`` (sm_100a): the trtllm-gen batched GEMM kernels of
  ``cubins/batched_gemm`` (FlashInfer's fused-MoE FC1/FC2) and the dense GEMM
  kernels of ``cubins/gemm`` (``mm_fp4``, ``mm_mxfp8``,
  ``gemm_fp8_nt_groupwise``, ``trtllm_low_latency_gemm``), one workload per
  live kernel, generated from ``impls/batched_gemm/variants/sm_100a.json``.
* ``segment`` (sm_86): FlashInfer's CUTLASS segment GEMM with row-major
  weights (``SegmentGEMMWrapper``, sm80 backend), from the FlashInfer 0.7.0
  sm80 JIT cache.

``host`` mirrors the trtllm-gen host code (KernelParams, TMA descriptors,
grids) and FlashInfer's dispatch policy; ``formats`` holds the element and
scale-factor formats. ``impls/batched_gemm/compiler.py`` writes the variant
indexes; nothing parses a cubin at import.
"""

from . import segment, trtllm  # noqa: F401  (imported for registration)

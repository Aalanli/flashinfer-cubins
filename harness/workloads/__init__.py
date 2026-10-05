"""Single-kernel workloads; each module registers its classes with @register."""

from . import (  # noqa: F401  (imported for registration)
    batched_gemm,
    deep_gemm,
    dsa_attention,
    dsa_indexer,
    fmha,
    fp8_gemm,
    gdn,
    gqa_decode,
    moe,
    nvfp4_dual_gemm,
    nvfp4_gemm,
    nvfp4_group_gemm,
    rmsnorm,
)

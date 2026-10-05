"""fmha single-kernel workloads, registered programmatically.

* ``fa2``: FlashInfer FA2 batch prefill (paged/ragged, with and without
  attention sinks), batch decode, persistent ``BatchAttention``, MLA and
  attention-state merge kernels (sm_86, from the FlashInfer 0.7.0 sm80 JIT
  cache).
* ``trtllm``: trtllm-gen FMHA kernels of ``cubins/fmha`` (sm_100a).

Variants come from the compact indexes ``impls/fmha/variants/<arch>.json``
written by ``impls/fmha/compiler.py``; nothing parses a cubin at import.
"""

from . import fa2, trtllm  # noqa: F401  (imported for registration)

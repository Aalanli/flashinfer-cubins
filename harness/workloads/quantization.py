"""Architecture-independent NVFP4 storage, input distributions and references.

Packed E2M1 is stored as uint8 (two values per byte, low nibble first) so CPU
and Ampere need no native FP4 support. Axes match NVIDIA's ``(rows, K/2, L)``
views of contiguous ``(L, rows, K/2)`` storage; scales are UE4M3
(``float8_e4m3fn`` bytes, sign bit clear), one per 16 K elements.

Input distributions (``make_fp4``), named after the official generators
(``resources/nvfp4_*/reference.py``) that use them:

* values ``"full"``: every byte, i.e. all 16 E2M1 codes (0, +-0.5, ..., +-6):
  nvfp4_gemm's ``randint(-128, 128)``.
* values ``"restricted"``: bytes masked with ``0xBB`` (``0b1011_1011``): each
  nibble keeps its sign bit and its two low bits and loses the high exponent
  bit, so the magnitudes are {0, 0.5, 1, 1.5}: the dual and group GEMM tasks'
  ``create_fp4_tensors``.
* scales ``"int0_3"``: integers 0..3, i.e. zero scales included (nvfp4_gemm);
  ``"int1_2"``: {1, 2} (nvfp4_group_gemm); ``"unit"``: U[0, 1) rounded to
  E4M3, i.e. zero (below 2**-10) and subnormal (below 2**-6) scales included
  (nvfp4_dual_gemm); ``"uniform"``: U[0.125, 0.875), three binades of normal
  scales (the harness's former only distribution).

Exactness: every E2M1 x UE4M3 value and every product of two is exact in
FP32. With integer scales each product is a multiple of 2**-2 (``full``) and
the FP32 accumulation is exact, in any order, while partial sums stay below
2**22, so a kernel and ``fp4_matmul`` agree bit for bit before the output
rounding. With ``unit``/``uniform`` scales the summation order matters, by far
less than one FP16 output ulp (see the workloads' ``validate``).

CUTLASS's blocked scale-factor layout (``Sm1xxBlkScaledConfig``, cuBLAS's
block-scaling layout, the NVIDIA tasks' ``to_blocked``) is produced once, by
``to_blocked_scales``: contiguous ``(L, ceil(rows/128), ceil(cols/4), 32, 4,
4)`` storage, where logical scale ``(r, c, l)`` sits at ``[l, r // 128, c // 4,
r % 32, (r // 32) % 4, c % 4]`` and the padding is zero. Each workload takes
the view its kernel's argument convention uses (the task's 6-D permuted view,
the previous package's flat ``[L, rows' * cols']`` or one group's 5-D block).
"""

from __future__ import annotations

import torch

SF_VEC = 16  # K elements per scale factor
SF_ROWS = 128  # rows (M or N) per scale-factor atom
SF_COLS = 4  # scale columns per atom (64 K elements)

E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
VALUES = ("full", "restricted")
SCALES = ("int0_3", "int1_2", "unit", "uniform")
RESTRICTED_MASK = 0xBB

# Elements of the FP32 temporaries the references materialize at a time, so
# that the largest model shapes stay within a few GiB of scratch.
CHUNK = 1 << 26


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def make_fp4(
    workload, g, rows, k, batches, *, values="full", scales="uniform"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded packed E2M1 ``(rows, k/2, l)`` and logical UE4M3 scales
    ``(rows, k/16, l)`` (views of contiguous batch-major storage) drawn from the
    named distributions (module docstring)."""
    if values not in VALUES or scales not in SCALES:
        raise ValueError(f"unknown distribution {values!r}/{scales!r}")
    device = workload.device
    data = torch.randint(
        0, 256, (batches, rows, k // 2), device=device, dtype=torch.uint8, generator=g
    )
    if values == "restricted":
        data &= RESTRICTED_MASK
    shape = (batches, rows, k // SF_VEC)
    if scales == "int0_3":
        raw = torch.randint(0, 4, shape, device=device, generator=g).float()
    elif scales == "int1_2":
        raw = torch.randint(1, 3, shape, device=device, generator=g).float()
    elif scales == "unit":
        raw = torch.rand(shape, device=device, generator=g)
    else:
        raw = torch.rand(shape, device=device, generator=g) * 0.75 + 0.125
    return data.permute(1, 2, 0), raw.to(torch.float8_e4m3fn).permute(1, 2, 0)


# -- blocked scale-factor layout ----------------------------------------------------


def to_blocked_scales(scales: torch.Tensor) -> torch.Tensor:
    """Logical ``(rows, cols, l)`` scales (any strides) -> CUTLASS's blocked
    layout as a fresh contiguous ``(l, rb, cb, 32, 4, 4)`` tensor of the same
    dtype, zero padded to ``rb * 128`` rows and ``cb * 4`` columns."""
    rows, cols, batches = scales.shape
    rb, cb = ceil_div(rows, SF_ROWS), ceil_div(cols, SF_COLS)
    padded = torch.zeros(
        (batches, rb * SF_ROWS, cb * SF_COLS), dtype=torch.uint8, device=scales.device
    )
    padded[:, :rows, :cols] = scales.view(torch.uint8).permute(2, 0, 1)
    # row = r4 * 32 + r32 inside an atom: (l, rb, r4, r32, cb, c4) -> (l, rb, cb, r32, r4, c4)
    blocked = padded.view(batches, rb, 4, 32, cb, SF_COLS).permute(0, 1, 4, 3, 2, 5)
    return blocked.contiguous().view(scales.dtype)


def from_blocked_scales(blocked: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of ``to_blocked_scales`` for its ``(l, rb, cb, 32, 4, 4)``
    result: the logical ``(rows, cols, l)`` scales (a view, padding dropped)."""
    batches, rb, cb = blocked.shape[:3]
    logical = blocked.permute(0, 1, 4, 3, 2, 5).reshape(
        batches, rb * SF_ROWS, cb * SF_COLS
    )
    return logical[:, :rows, :cols].permute(1, 2, 0)


# -- references ---------------------------------------------------------------------


def dequant_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 values ``(rows, k, l)`` (a view of contiguous ``(l, rows, k)``) of
    packed E2M1 ``(rows, k/2, l)`` times logical scales ``(rows, k/16, l)``.
    Exact: every product fits FP32's mantissa. Decoded in row chunks."""
    rows, half_k, batches = packed.shape
    k = 2 * half_k
    lut = torch.tensor(E2M1, device=packed.device, dtype=torch.float32)
    source = packed.permute(2, 0, 1).reshape(batches * rows, half_k)
    sf = scales.permute(2, 0, 1).reshape(batches * rows, k // SF_VEC)
    out = torch.empty((batches * rows, k), device=packed.device, dtype=torch.float32)
    step = max(1, CHUNK // max(k, 1))
    for start in range(0, batches * rows, step):
        part = source[start : start + step]
        nibbles = torch.stack((part & 15, part >> 4), dim=2).flatten(1)
        out[start : start + step] = lut[nibbles.long()] * sf[
            start : start + step
        ].float().repeat_interleave(SF_VEC, dim=1)
    return out.view(batches, rows, k).permute(1, 2, 0)


def fp4_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    *,
    alpha: float = 1.0,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``alpha * (A * SA) @ (B * SB)^T`` per batch, accumulated in FP32 and
    rounded once to ``out_dtype``, as an ``[m, n, l]`` view of contiguous
    ``[l, m, n]`` storage (the kernels' D layout, ``empty_fp4_output``).

    ``a``/``b`` are packed ``(rows, k/2, l)``, ``sa``/``sb`` logical scales.
    B is dequantized in row chunks so that FP32 copies of the largest weights
    are never materialized whole; the per-element result does not depend on
    the chunking.
    """
    m, half_k, batches = a.shape
    n, k = b.shape[0], 2 * half_k
    lhs = dequant_fp4(a, sa).permute(2, 0, 1)  # (l, m, k), contiguous
    out = torch.empty((batches, m, n), device=a.device, dtype=out_dtype)
    step = max(1, CHUNK // max(k, m, 1))
    for start in range(0, n, step):
        rhs = dequant_fp4(b[start : start + step], sb[start : start + step])
        part = torch.bmm(lhs, rhs.permute(2, 1, 0))  # (l, m, rows)
        if alpha != 1.0:
            part *= alpha
        out[:, :, start : start + step] = part
    return out.permute(1, 2, 0)


def empty_fp4_output(a, b, dtype=torch.float16):
    """Uninitialized ``[m, n, l]`` output in batch-major storage."""
    return torch.empty(
        (a.shape[2], a.shape[0], b.shape[0]), device=a.device, dtype=dtype
    ).permute(1, 2, 0)

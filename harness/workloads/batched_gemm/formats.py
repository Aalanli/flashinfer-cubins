"""Element and scale-factor formats of the trtllm-gen kernels, in torch.

Everything works on CPU and CUDA tensors and is exact: inputs are generated
directly as representable values (codes), dequantization multiplies codes by
power-of-two-friendly scale factors, and the layouts are pure index maps.

* E2m1 (NVFP4/MXFP4 elements): 4-bit codes, two per byte, the even element
  in the low nibble (``fp4_quantize`` / ``mxint4_quantize`` byte order).
* MxInt4: signed 4-bit integers, same packing.
* E4m3 elements and E4m3 scale factors: ``torch.float8_e4m3fn``.
* UE8m0 scale factors: biased exponent bytes, value ``2 ** (e - 127)``.
* Scale-factor layouts (``trtllm/gen/SfLayoutDecl.h``): Linear ``[m, n/b]``,
  R8c4 ``[m/8, n/b/4, 8, 4]`` and R128c4 ``[m/128, n/b/4, 32, 4, 4]``
  (FlashInfer's ``block_scale_interleave``), rows/columns zero-padded.
* Weight preparation (``flashinfer/utils.py``, ``fused_moe/core.py``): the
  gated-activation row interleave (``reorder_rows_for_gated_act_gemm``), the
  epilogue row shuffle (``shuffle_matrix_a``: blocks of 16 or 32 rows) and
  ``convert_to_block_layout`` (BlockMajorK).
"""

from __future__ import annotations

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
E2M1_MAX = 6.0


def e2m1_table(device: torch.device) -> torch.Tensor:
    pos = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)
    return torch.cat([pos, -pos])


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes [..., n] (n even, values < 16) -> bytes [..., n / 2]."""
    codes = codes.to(torch.uint8)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """bytes [..., n / 2] -> uint8 codes [..., n]."""
    packed = packed.view(torch.uint8)
    out = torch.stack([packed & 0xF, packed >> 4], dim=-1)
    return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def e2m1_decode(codes: torch.Tensor) -> torch.Tensor:
    return e2m1_table(codes.device)[codes.long()]


def int4_decode(codes: torch.Tensor) -> torch.Tensor:
    c = codes.to(torch.int16)
    return torch.where(c >= 8, c - 16, c).to(torch.float32)


def e2m1_encode(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even E2m1 codes of float values (saturating)."""
    table = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=x.device)
    mag = x.abs().clamp(max=E2M1_MAX).float()
    # midpoints between representable magnitudes; ties go to the even code
    lo = torch.searchsorted(table, mag.contiguous(), right=False).clamp(1, 7) - 1
    hi = lo + 1
    d_lo, d_hi = mag - table[lo], table[hi] - mag
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = torch.where(pick_hi, hi, lo)
    code = torch.where(mag >= table[7], torch.full_like(code, 7), code)
    sign = (x < 0) & (code > 0)
    return (code + sign.long() * 8).to(torch.uint8)


def ue8m0_decode(exponents: torch.Tensor) -> torch.Tensor:
    return torch.exp2(exponents.view(torch.uint8).float() - 127.0)


def e4m3_decode(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.float8_e4m3fn).float()


def e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Saturating round-to-nearest E4m3 (``cvt.rn.satfinite``)."""
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


# --- scale-factor layouts ---------------------------------------------------------

SF_LINEAR, SF_R8C4, SF_R128C4 = 0, 1, 3


def sf_padded_shape(rows: int, cols: int, layout: int) -> tuple[int, int]:
    if layout == SF_R128C4:
        return -(-rows // 128) * 128, -(-cols // 4) * 4
    if layout == SF_R8C4:
        return -(-rows // 8) * 8, -(-cols // 4) * 4
    if layout == SF_LINEAR:
        return rows, cols
    raise ValueError(f"unsupported SF layout {layout}")


def sf_offsets(rows: int, cols: int, layout: int, device: torch.device) -> torch.Tensor:
    """int64 [rows, cols]: storage offset of SF (row, col) in ``layout``."""
    r = torch.arange(rows, device=device)[:, None]
    c = torch.arange(cols, device=device)[None, :]
    _, padded_cols = sf_padded_shape(rows, cols, layout)
    if layout == SF_R128C4:
        tiles = padded_cols // 4
        return (
            (r // 128) * (tiles * 512)
            + (c // 4) * 512
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + c % 4
        )
    if layout == SF_R8C4:
        tiles = padded_cols // 4
        return (r // 8) * (tiles * 32) + (c // 4) * 32 + (r % 8) * 4 + c % 4
    return r * cols + c


def sf_storage_size(rows: int, cols: int, layout: int) -> int:
    pr, pc = sf_padded_shape(rows, cols, layout)
    return pr * pc


def sf_to_layout(sf: torch.Tensor, layout: int) -> torch.Tensor:
    """[rows, cols] scale factors -> flat storage in ``layout`` (padding 0)."""
    rows, cols = sf.shape
    flat = torch.zeros(
        sf_storage_size(rows, cols, layout), dtype=sf.dtype, device=sf.device
    )
    flat.view(-1)[sf_offsets(rows, cols, layout, sf.device).flatten()] = sf.flatten()
    return flat


def sf_from_layout(
    flat: torch.Tensor, rows: int, cols: int, layout: int
) -> torch.Tensor:
    """Flat storage in ``layout`` -> [rows, cols]."""
    return flat.reshape(-1)[sf_offsets(rows, cols, layout, flat.device)]


# --- weight preparation (row permutations) -----------------------------------------

_BLOCK16 = (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)
_BLOCK32 = (
    0, 8, 16, 24, 1, 9, 17, 25, 2, 10, 18, 26, 3, 11, 19, 27,
    4, 12, 20, 28, 5, 13, 21, 29, 6, 14, 22, 30, 7, 15, 23, 31,
)  # fmt: skip


def shuffle_rows(rows: int, epilogue_tile_m: int) -> torch.Tensor:
    """get_shuffle_matrix_a_row_indices: new row i takes old row idx[i]."""
    block = 32 if epilogue_tile_m % 128 == 0 else 16
    row_map = torch.tensor(_BLOCK32 if block == 32 else _BLOCK16, dtype=torch.long)
    if rows % block:
        raise ValueError(f"rows must be a multiple of {block}")
    old = torch.arange(rows, dtype=torch.long)
    new = (old // block) * block + row_map[old % block]
    idx = torch.empty(rows, dtype=torch.long)
    idx[new] = old
    return idx


def gated_rows(rows: int) -> torch.Tensor:
    """get_reorder_rows_for_gated_act_gemm_row_indices: [r0, rN/2, r1, ...]."""
    if rows % 2:
        raise ValueError("gated weights need an even row count")
    idx = torch.empty(rows, dtype=torch.long)
    idx[0::2] = torch.arange(rows // 2)
    idx[1::2] = torch.arange(rows // 2, rows)
    return idx


def to_block_major_k(w: torch.Tensor, block: int) -> torch.Tensor:
    """convert_to_block_layout per batch: [..., M, K] -> [..., K/b, M, b]."""
    *batch, m, k = w.shape
    if k % block:
        raise ValueError("K must be a multiple of blockK")
    return w.reshape(*batch, m, k // block, block).transpose(-3, -2).contiguous()


def from_block_major_k(w: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_block_major_k`: [..., K/b, M, b] -> [..., M, K]."""
    *batch, kb, m, b = w.shape
    return w.transpose(-3, -2).reshape(*batch, m, kb * b)


def spacing(values: torch.Tensor, fmt: str) -> torch.Tensor:
    """Distance from |value| to the next representable magnitude of ``fmt``
    ("e4m3" or "e2m1", scale 1): one quantization step."""
    mag = values.abs().float()
    if fmt == "e2m1":
        return torch.where(mag < 2, 0.5, torch.where(mag < 4, 1.0, 2.0))
    exponent = torch.floor(torch.log2(mag.clamp(min=2.0**-6)))
    return torch.exp2(exponent - 3)

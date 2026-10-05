"""Host-side CUTLASS arithmetic shared by workloads that rebuild Params in Python."""

from __future__ import annotations

from collections.abc import Sequence

from . import cuda_driver


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def fast_divmod(divisor: int) -> tuple[int, int, int]:
    """cutlass::FastDivmod(divisor) as (divisor, multiplier, shift_right)."""
    if divisor == 1:
        return 1, 0, 0
    log2 = (divisor - 1).bit_length()  # cutlass::find_log2: ceil(log2(divisor))
    p = 31 + log2
    return divisor, ((1 << p) + divisor - 1) // divisor, p - 32


def cutlass_small_tensor(nbytes: int) -> bool:
    """CUTLASS ``make_tma_copy_desc``'s test for clearing bit 21 of descriptor
    word 1 on drivers <= 13.1: ``bits_to_bytes(cosize * sizeof_bits) <
    131072``. ``cutlass::bits_to_bytes`` computes in ``int`` (its default
    ``R``): the bit count is truncated to 32 bits (two's complement), so
    tensors of 2**28 bytes or more whose truncated bit count is negative or
    below 1 MiBit also qualify."""

    def int32(value: int) -> int:
        return (value + (1 << 31)) % (1 << 32) - (1 << 31)

    total = int32(int32(nbytes * 8) + 7)
    return (total // 8 if total >= 0 else -(-total // 8)) < 131072


def driver_encode(
    data_type: int,
    address: int,
    dims: Sequence[int],
    strides: Sequence[int],
    box: Sequence[int],
    element_strides: Sequence[int],
    interleave: int,
    swizzle: int,
    l2_promotion: int,
    oob_fill: int,
) -> bytes:
    """cuTensorMapEncodeTiled through libcuda as 128 bytes (needs sm_90+).

    Same signature as each workload's recording ``fake_encode`` for tests.
    """
    return bytes(
        cuda_driver.encode_tensor_map_tiled(
            data_type,
            address,
            dims,
            strides,
            box,
            element_strides,
            interleave=interleave,
            swizzle=swizzle,
            l2_promotion=l2_promotion,
            oob_fill=oob_fill,
        )
    )

"""Ordinary grouped traversal and GF(2) shared/register layout helpers."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def grouped_program_coordinates(
    pid, count_m: gl.constexpr, count_n: gl.constexpr, GROUP_M: gl.constexpr
):
    # The caller launches count_m * count_n consecutive program IDs.
    gl.static_assert(count_m > 0 and count_n > 0 and GROUP_M > 0)
    # Program IDs in the supported launch grid are nonnegative.
    pid = pid.to(gl.uint32)
    group_width: gl.constexpr = GROUP_M * count_n
    if count_n == 1:
        # Keep rotated tail-row ordering; the single column is always zero.
        if count_m > GROUP_M and count_m % GROUP_M != 0:
            first_tail: gl.constexpr = (count_m // GROUP_M) * GROUP_M
            tail_rows: gl.constexpr = count_m % GROUP_M
            pm = gl.where(pid >= first_tail, first_tail + pid % tail_rows, pid)
            return pm.to(gl.int32), gl.cast(0, gl.int32)
        return pid.to(gl.int32), gl.cast(0, gl.int32)
    elif count_m <= GROUP_M:
        return (pid % count_m).to(gl.int32), (pid // count_m).to(gl.int32)
    elif count_m % GROUP_M == 0:
        return ((pid // group_width) * GROUP_M + pid % GROUP_M).to(gl.int32), (
            (pid % group_width) // GROUP_M
        ).to(gl.int32)
    else:
        full_groups: gl.constexpr = count_m // GROUP_M
        full_span: gl.constexpr = full_groups * group_width
        tail_m: gl.constexpr = count_m % GROUP_M
        in_tail = pid >= full_span
        full_m = (pid // group_width) * GROUP_M + pid % GROUP_M
        full_n = (pid % group_width) // GROUP_M
        # Preserve the original absolute modulo, including rotated tail rows.
        tail_m_index = full_groups * GROUP_M + pid % tail_m
        tail_n = (pid - full_span) // tail_m
        return gl.where(in_tail, tail_m_index, full_m).to(gl.int32), gl.where(
            in_tail, tail_n, full_n
        ).to(gl.int32)


@g.constexpr_function
def column_layout(layout, pack_registers=False):
    assert layout.rank == 2 and layout.shape[1] >= 32 and layout.shape[1] % 32 == 0

    def rotate(bases):
        return [[r, (n & ~31) | ((n & 15) << 1) | ((n >> 4) & 1)] for r, n in bases]

    registers = rotate(layout.reg_bases)
    if pack_registers and [0, 1] in registers:
        # A pure unit column basis is a pair owned by this same thread.
        # Move it to the least-significant register bit so vector stores see
        # adjacent BF16 values; lane/warp ownership and every value stay fixed.
        index = registers.index([0, 1])
        registers = [registers[index]] + registers[:index] + registers[index + 1 :]
    return gl.DistributedLinearLayout(
        registers,
        rotate(layout.lane_bases),
        rotate(layout.warp_bases),
        rotate(layout.block_bases),
        layout.shape,
    )


@g.constexpr_function
def tile_shared_layout(shape, minor_axis, bit_order, bank_xors):
    """Build a bijective physical-offset -> logical-element shared chart.

    bit_order permutes bits of the row-major storage ordinal. bank_xors
    mix physical offset bits into bank-selecting bits without carry. Neither
    a copy instruction nor source ownership is part of this ordinary layout.
    """
    if len(shape) != 2 or minor_axis not in (0, 1):
        raise ValueError("requires a rank-two tile and a minor axis")
    if any(n <= 0 or n & (n - 1) for n in shape):
        raise ValueError("tile dimensions must be powers of two")
    bits = (shape[0] * shape[1]).bit_length() - 1
    order = tuple(bit for bit in bit_order if bit < bits)
    if sorted(order) != list(range(bits)):
        raise ValueError("bit order must cover every storage bit once")
    width = shape[minor_axis]
    bases = []
    for bit in order:
        major, minor = divmod(1 << bit, width)
        bases.append([major, minor] if minor_axis == 1 else [minor, major])
    original = [v[:] for v in bases]
    for high, low in bank_xors:
        if not (0 <= low < high):
            raise ValueError("bank mixing must be a triangular XOR")
        if high < bits:
            bases[high] = [a ^ b for a, b in zip(bases[high], original[low])]
    return gl.SharedLinearLayout(bases, alignment=16)


def row_bit_order(group_bits=2):
    """Interleave row groups after the low six contiguous storage bits."""
    if not 0 <= group_bits <= 6:
        raise ValueError("row grouping must fit the 12-bit tile footprint")
    split = 6 + group_bits
    return (
        tuple(range(6))
        + tuple(range(split, min(split + 3, 12)))
        + tuple(range(6, split))
        + tuple(range(min(split + 3, 12), 12))
    )


# Column-major matrix reads prefer different register/warp bit orders as the
# contiguous N axis grows. These are tile layouts, independent of M/N/K inputs.
_COLUMN_BIT_ORDERS = {
    64: tuple(range(6)) + (7, 9, 10, 6, 8, 11),
    128: tuple(range(6)) + (6, 8, 10, 7, 11, 9),
    256: tuple(range(6)) + (6, 7, 9, 8, 10, 11),
}


def operand_shared_layouts(tile, *, transposed_b, row_group_bits=2):
    bm, bn, bk = tile
    a = tile_shared_layout(
        (bm, bk), 1, row_bit_order(row_group_bits), ((9, 3), (10, 4))
    )
    if transposed_b:
        b = tile_shared_layout((bk, bn), 0, row_bit_order(), ((9, 3), (10, 4)))
    else:
        # Short-K MMAC operands have four values per lane instead of eight.
        # Preserve their two low N groups before interleaving the K bits.
        order = (
            _COLUMN_BIT_ORDERS[256]
            if bk < 32
            else _COLUMN_BIT_ORDERS.get(bn, row_bit_order())
        )
        b = tile_shared_layout((bk, bn), 1, order, ((9, 5),))
    return a, b


@g.constexpr_function
def shared_frame_layout(elements):
    """Identity storage for one allocation-backed, power-of-two tile pair."""
    assert elements > 0 and elements & (elements - 1) == 0
    return gl.SharedLinearLayout(
        [[1 << bit] for bit in range(elements.bit_length() - 1)],
        alignment=min(32768, 2 * elements),
    )


@g.constexpr_function
def cooperative_shared_layout(transpose):
    """A complete 32KiB chart matched to the cooperative MMAC operand reads."""
    order = (0, 1, 2, 3, 4, 5, 8, 9, 10, 6, 7, 11, 12, 13)
    return tile_shared_layout(
        (64, 256) if transpose else (256, 64),
        0 if transpose else 1,
        order,
        ((9, 4), (10, 5)),
    )

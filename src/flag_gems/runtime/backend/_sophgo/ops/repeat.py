import logging

import torch
import triton
from triton import language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

# repeat kernel: copy the whole input as one period along each dimension,
# out_shape[i] = in_shape[i] * count[i].
#
# Grid is split into (num_rows, copies_last, cdiv(in_last, BLOCK)):
#   row_id : index over the flattened outer dims (first rank-1 dims)
#   k      : index over the copies of the last dim (0 .. copies_last-1)
#   blk    : index over BLOCK-sized chunks within one copy of the last dim
#
# Outer dims (rank-1) are decoded by i32 scalar divmod into in_outer_base.
# The last dim uses scalar + arange(BLOCK) offsets, which are purely affine,
# so load/store lower to contiguous copies rather than gather/scatter.
#
# Only the last dim is read contiguously from input and written contiguously to
# output (in_stride_last == 1). The outer index wraps modulo the input shape
# because the input is repeated along every dim.
#
# repeat requires len(sizes) >= inp.ndim (matches PyTorch), and left-pads the
# input shape with 1s when sizes has more dims.

MAX_RANK = 8
MAX_OUTER_RANK = MAX_RANK - 1  # = 7


@libentry()
@triton.jit
def repeat_kernel(
    in_ptr,
    out_ptr,
    # outer (rank-1) dims of the output shape, padded with 1 when unused
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    # input shape of the outer dims, padded with 1 when unused
    in_s0,
    in_s1,
    in_s2,
    in_s3,
    in_s4,
    in_s5,
    in_s6,
    # input stride of the outer dims, padded with 0 when unused
    in_stride0,
    in_stride1,
    in_stride2,
    in_stride3,
    in_stride4,
    in_stride5,
    in_stride6,
    # last dim
    in_last,  # = in_shape[-1]
    out_last,  # = in_last * copies_last
    OUTER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tle.program_id(0)
    k = tle.program_id(1)
    blk = tle.program_id(2)

    # scalar decode of outer dims: row_id -> in_outer_base, all i32 scalars.
    # peel from the innermost outer dim (index OUTER_RANK-1) outward.
    in_outer_base = 0
    t = row_id
    if OUTER_RANK == 0:
        in_outer_base = 0
    elif OUTER_RANK == 1:
        i0 = t
        in_outer_base = (i0 % in_s0) * in_stride0
    elif OUTER_RANK == 2:
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (i0 % in_s0) * in_stride0 + (i1 % in_s1) * in_stride1
    elif OUTER_RANK == 3:
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
        )
    elif OUTER_RANK == 4:
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
        )
    elif OUTER_RANK == 5:
        i4 = t % s4
        t = t // s4
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
            + (i4 % in_s4) * in_stride4
        )
    elif OUTER_RANK == 6:
        i5 = t % s5
        t = t // s5
        i4 = t % s4
        t = t // s4
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
            + (i4 % in_s4) * in_stride4
            + (i5 % in_s5) * in_stride5
        )
    else:  # OUTER_RANK == 7
        i6 = t % s6
        t = t // s6
        i5 = t % s5
        t = t // s5
        i4 = t % s4
        t = t // s4
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
            + (i4 % in_s4) * in_stride4
            + (i5 % in_s5) * in_stride5
            + (i6 % in_s6) * in_stride6
        )

    # contiguous copy of the last dim.
    j = blk * BLOCK + tl.arange(0, BLOCK)
    mask = j < in_last

    # input is contiguous (wrapper calls .contiguous().reshape(...)), so
    # in_stride_last == 1 and in_outer_base + j is purely affine.
    in_off = in_outer_base + j
    x = tl.load(in_ptr + in_off, mask=mask)

    # out_off = (row_id * out_last + k * in_last) + j, also scalar + arange.
    out_row_base = row_id * out_last + k * in_last
    out_off = out_row_base + j
    tl.store(out_ptr + out_off, x, mask=mask)


def _choose_block(in_last: int) -> int:
    # 128..512 tiles are the sweet spot for the TPU
    return min(512, max(1, triton.next_power_of_2(in_last)))


def repeat(inp: torch.Tensor, sizes) -> torch.Tensor:
    logging.debug("SOPHGO GEMS REPEAT")

    in0_shape = list(inp.shape)
    sizes_shape = list(sizes)
    in_rank = len(in0_shape)
    sizes_rank = len(sizes_shape)

    # repeat requires len(sizes) >= inp.ndim (matches PyTorch behavior)
    if sizes_rank < in_rank:
        raise RuntimeError(
            "Number of dimensions of repeat dims can not be smaller than"
            " number of dimensions of tensor"
        )
    # left-pad the input shape with 1s when sizes has more dims
    if sizes_rank > in_rank:
        in0_shape = [1] * (sizes_rank - in_rank) + in0_shape
    rank = max(in_rank, sizes_rank, 1)  # at least 1 dim, also handles 0-dim

    if rank > MAX_RANK:
        raise NotImplementedError(
            f"repeat only supports rank up to {MAX_RANK}, got {rank}"
        )

    # build out_shape and short-circuit on empty
    is_empty = False
    out_shape = []
    for i in range(rank):
        d = sizes_shape[i]
        assert d >= 0, f"repeat sizes must be >= 0, got {d}"
        if d == 0:
            is_empty = True
        out_shape.append(in0_shape[i] * d)

    out0 = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
    if is_empty:
        return out0

    # make input contiguous so reshape is zero-copy and stride_last == 1
    in0 = inp.contiguous().reshape(in0_shape)

    # split into outer dims (first rank-1) and the last dim
    outer_rank = rank - 1
    in_last = in0_shape[-1]
    out_last = out_shape[-1]
    num_rows = 1
    for i in range(outer_rank):
        num_rows *= out_shape[i]
    copies_last = sizes_shape[-1]

    block = _choose_block(in_last)
    grid = (num_rows, copies_last, triton.cdiv(in_last, block))

    # pad to MAX_OUTER_RANK
    in0_strides = list(in0.stride())
    pad = MAX_OUTER_RANK - outer_rank
    s_args = out_shape[:outer_rank] + [1] * pad
    in_s_args = in0_shape[:outer_rank] + [1] * pad
    in_stride_args = in0_strides[:outer_rank] + [0] * pad

    with torch_device_fn.device(inp.device.index):
        repeat_kernel[grid](
            in0,
            out0,
            *s_args,
            *in_s_args,
            *in_stride_args,
            in_last,
            out_last,
            OUTER_RANK=outer_rank,
            BLOCK=block,
            num_warps=4,
        )
    return out0

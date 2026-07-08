import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle


@libentry()
@triton.jit
def flip_generic_rank5_kernel(
    inp_ptr,
    out_ptr,
    outer_num,
    size0,
    size1,
    size2,
    size3,
    size4,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    inp_stride3,
    inp_stride4,
    flip0,
    flip1,
    flip2,
    flip3,
    flip4,
    BLOCK_OUTER: tl.constexpr,
    BLOCK_INNER: tl.constexpr,
):
    outer = tle.program_id(0) * BLOCK_OUTER + tl.arange(0, BLOCK_OUTER)
    inner = tle.program_id(1) * BLOCK_INNER + tl.arange(0, BLOCK_INNER)
    outer_mask = outer < outer_num
    inner_mask = inner < size4

    tmp = outer
    coord3 = tmp % size3
    tmp = tmp // size3
    coord2 = tmp % size2
    tmp = tmp // size2
    coord1 = tmp % size1
    tmp = tmp // size1
    coord0 = tmp % size0

    src0 = tl.where(flip0 != 0, size0 - 1 - coord0, coord0)
    src1 = tl.where(flip1 != 0, size1 - 1 - coord1, coord1)
    src2 = tl.where(flip2 != 0, size2 - 1 - coord2, coord2)
    src3 = tl.where(flip3 != 0, size3 - 1 - coord3, coord3)
    src4 = tl.where(flip4 != 0, size4 - 1 - inner, inner)

    src_offs = (
        src0[:, None] * inp_stride0
        + src1[:, None] * inp_stride1
        + src2[:, None] * inp_stride2
        + src3[:, None] * inp_stride3
        + src4[None, :] * inp_stride4
    )
    mask = outer_mask[:, None] & inner_mask[None, :]
    vals = tl.load(inp_ptr + src_offs, mask=mask, other=0)
    out_offs = outer[:, None] * size4 + inner[None, :]
    tl.store(out_ptr + out_offs, vals, mask=mask)


@libentry()
@triton.jit
def flip_dim0_2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    inp_stride0,
    inp_stride1,
    out_stride0,
    out_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    valid_rows = tl.minimum(rows, M - 1)
    src_rows = M - 1 - valid_rows
    vals = tl.load(
        inp_ptr + src_rows * inp_stride0 + cols * inp_stride1,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + rows * out_stride0 + cols * out_stride1,
        vals,
        mask=mask,
    )


@libentry()
@triton.jit
def flip_dim1_2d_kernel(
    inp_ptr,
    out_ptr,
    M,
    N,
    inp_stride0,
    inp_stride1,
    out_stride0,
    out_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = tle.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)
    valid_cols = tl.minimum(cols, N - 1)
    src_cols = N - 1 - valid_cols
    vals = tl.load(
        inp_ptr + rows * inp_stride0 + src_cols * inp_stride1,
        mask=mask,
        other=0.0,
    )
    tl.store(
        out_ptr + rows * out_stride0 + cols * out_stride1,
        vals,
        mask=mask,
    )


def flip(A: torch.Tensor, dims) -> torch.Tensor:
    logging.debug("GEMS FLIP")
    normalized_dims = []
    for dim in dims:
        assert (
            dim >= -A.dim() and dim < A.dim()
        ), "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
            -A.dim(), A.dim() - 1, dim
        )
        normalized_dim = dim % A.dim()
        assert (
            normalized_dim not in normalized_dims
        ), "dim {} appears multiple times in the list of dims".format(dim)
        normalized_dims.append(normalized_dim)

    if (
        A.ndim == 2
        and A.is_contiguous()
        and len(normalized_dims) == 1
        and normalized_dims[0] in (0, 1)
    ):
        out = torch.empty_like(A)
        M, N = A.shape
        grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
        if normalized_dims[0] == 0:
            flip_dim0_2d_kernel[grid](
                A,
                out,
                M,
                N,
                A.stride(0),
                A.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=16,
                BLOCK_N=64,
            )
        else:
            flip_dim1_2d_kernel[grid](
                A,
                out,
                M,
                N,
                A.stride(0),
                A.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_M=16,
                BLOCK_N=64,
            )
        return out

    if A.numel() <= 1 or all(A.size(dim) <= 1 for dim in normalized_dims):
        return A.clone()

    if A.ndim <= 5:
        padded_shape = (1,) * (5 - A.ndim) + tuple(A.shape)
        padded_strides = (0,) * (5 - A.ndim) + tuple(A.stride())
        flip_flags = [0] * 5
        dim_offset = 5 - A.ndim
        for dim in normalized_dims:
            flip_flags[dim_offset + dim] = 1

        out = torch.empty(A.shape, device=A.device, dtype=A.dtype)
        outer_num = A.numel() // padded_shape[4]
        grid = (triton.cdiv(outer_num, 16), triton.cdiv(padded_shape[4], 64))
        flip_generic_rank5_kernel[grid](
            A,
            out,
            outer_num,
            padded_shape[0],
            padded_shape[1],
            padded_shape[2],
            padded_shape[3],
            padded_shape[4],
            padded_strides[0],
            padded_strides[1],
            padded_strides[2],
            padded_strides[3],
            padded_strides[4],
            flip_flags[0],
            flip_flags[1],
            flip_flags[2],
            flip_flags[3],
            flip_flags[4],
            BLOCK_OUTER=16,
            BLOCK_INNER=64,
        )
        return out

    out = A
    for dim in normalized_dims:
        if out.size(dim) <= 1:
            continue
        index = torch.arange(
            out.size(dim) - 1,
            -1,
            -1,
            device=out.device,
            dtype=torch.int64,
        )
        out = torch.index_select(out, dim, index)
    return out

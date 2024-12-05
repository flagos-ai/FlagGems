import logging

import torch
import triton
import triton.language as tl

from .. import runtime
from ..utils import libentry, offsetCalculator, restride_dim
from ..utils import triton_lang_extension as tle


@libentry()
@triton.autotune(configs=runtime.get_triton_config("select_scatter"), key=["M", "N"])
@triton.jit
def slice_scatter_kernel(
    inp,
    inp_indices,
    src,
    src_offsets,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    rows_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rows_mask = rows_offsets < M

    for off in range(0, N, BLOCK_N):
        cols_offsets = off + tl.arange(0, BLOCK_N)[None, :]
        cols_mask = cols_offsets < N

        offsets = rows_offsets * N + cols_offsets
        mask = rows_mask and cols_mask

        indices = tl.load(inp_indices + offsets, mask=mask, other=0)
        src_indices = tl.load(src_offsets + offsets, mask=mask, other=0)
        cur_src = tl.load(src + src_indices, mask=mask, other=0)

        tl.store(inp + indices, cur_src, mask=mask)


def slice_scatter(inp, src, dim=0, start=None, end=None, step=1):
    logging.debug("GEMS SLICE_SCATTER")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim
    out = inp.clone().contiguous()
    src = src.contiguous()
    size_dim = inp.size(dim)

    if start is None:
        start = 0
    if end is None:
        end = size_dim

    range = end - start
    if end < start:
        range = 0
    elif (end - start) > size_dim:
        range = size_dim
        start = 0
        end = size_dim

    if range == 0:
        return out

    valid_shape = list(inp.shape)
    valid_shape[dim] = (range + (step - 1)) // step
    assert (
        list(src.shape) == valid_shape
    ), "Expected src to have a size equal to the slice of self"

    storage_offset = out.storage_offset() + start * out.stride(dim)
    out_strided = restride_dim(out, dim, valid_shape, step, storage_offset)
    idx = torch.arange(0, src.numel(), device=inp.device).reshape(valid_shape)
    strides = list(out.stride())
    strides[dim] *= step
    indices = (
        offsetCalculator(out_strided, idx, strides, dim, isInp=False) + storage_offset
    )
    src_offsets = offsetCalculator(src, idx, src.stride(), dim, isInp=False)

    N = valid_shape[src.ndim - 1]
    M = src.numel() // N

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    slice_scatter_kernel[grid](out, indices, src, src_offsets, M, N)

    return out


def simplify(x, retain_dim, ordered_dims=None):
    # This helper function tries to create a new view of the input such that
    # the original dims other than the retain_dim are coalesced into one outer dim
    # and/or one inner dim.
    ordered_dims = ordered_dims or sorted(range(x.ndim), key=lambda i: x.stride(i))
    assert x.ndim == len(ordered_dims)
    if len(ordered_dims) == 1:
        return x, ordered_dims

    size_list = [x.size(dim) for dim in ordered_dims]
    stride_list = [x.stride(dim) for dim in ordered_dims]

    n = ordered_dims.index(retain_dim)

    # Try to merge into a 3d tensor, retain_dim is kept in the middle
    new_sizes = [1, size_list[n], 1]
    new_dim = 0
    for i in range(x.ndim):
        if i == n:
            new_dim += 2
            continue
        new_sizes[new_dim] *= size_list[i]
        if 0 < i < n or i > n + 1:
            if stride_list[i] != stride_list[i - 1] * size_list[i]:
                # cannot merge!
                return None, None

    K, N, M = new_sizes
    K_stride, N_stride = stride_list[0], stride_list[n]
    M_stride = stride_list[n + 1] if n < x.ndim - 1 else None

    if n == 0:
        new_x = x.as_strided((M, N), (M_stride, N_stride))
    elif n == x.ndim - 1:
        new_x = x.as_strided((N, K), (N_stride, K_stride))
    else:
        new_x = x.as_strided((M, N, K), (M_stride, N_stride, K_stride))

    return new_x, ordered_dims


@triton.jit(
    do_not_specialize=[
        "N1",
        "N2",
        "K",
        "am_stride",
        "an_stride",
        "ak_stride",
        "bm_stride",
        "bn_stride",
        "bk_stride",
        "start",
        "end",
        "step",
    ]
)
def scatter_by_row_kernel(
    A,
    B,
    A_out,
    # mid dim size of A and A_out
    N1,
    # mid dim size of B
    N2,
    # inner dim size of inp, src and out
    K,
    # strides of A and A_out
    am_stride,
    an_stride,
    ak_stride,
    # strides of B
    bm_stride,
    bn_stride,
    bk_stride,
    # slice start index
    start,
    # slice end index
    end,
    # slice step
    step,
    BLOCK: tl.constexpr,
):
    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    am_idx = pidx // N1
    an_idx = pidx % N1

    # inp and out share the same layout
    offset = BLOCK * pidy + tl.arange(0, BLOCK)
    mask = offset < K

    if (start <= an_idx) & (an_idx < end) & ((an_idx - start) % step == 0):
        # mid dim) % step == 0 size): of src
        tl.device_assert(N2 == (end - start) // step)
        # slice index in B
        bn_idx = (an_idx - start) // step
        B_start = B + am_idx * bm_stride + bn_idx * bn_stride
        slice = tl.load(B_start + offset, mask=mask)
    else:
        # slice index in A
        A_start = A + am_idx * am_stride + an_idx * an_stride
        slice = tl.load(A_start + offset, mask=mask)

    A_out_start = A_out + am_idx * am_stride + an_idx * an_stride
    tl.store(A_out_start + offset, slice, mask=mask)


def scatter_by_row(A, B, A_out, start, end, step):
    assert A_out.ndim in (2, 3)
    if A_out.ndim == 2:
        A_out = A_out.unsqueeze(0)
        B = B.unsqueeze(0)
    M, N1, K = A_out.size()
    N2 = B.size(1)
    am_stride, an_stride, ak_stride = A_out.stride()
    bm_stride, bn_stride, bk_stride = B.stride()

    grid = lambda meta: (M * N1, triton.cdiv(K, meta["BLOCK"]))
    scatter_by_row_kernel[grid](
        A,
        B,
        A_out,
        N1,
        N2,
        K,
        am_stride,
        an_stride,
        ak_stride,
        bm_stride,
        bn_stride,
        bk_stride,
        start,
        end,
        step,
        BLOCK=512,
    )


@triton.jit(
    do_not_specialize=[
        "N1",
        "N2",
        "K",
        "am_stride",
        "an_stride",
        "ak_stride",
        "bm_stride",
        "bn_stride",
        "bk_stride",
        "start",
        "end",
        "step",
    ]
)
def scatter_3d_mid_kernel(
    A,
    B,
    A_out,
    # mid dim size of A and A_out
    N1,
    # mid dim size of B
    N2,
    # inner dim size of inp, src and out
    K,
    # strides of A and A_out
    am_stride,
    an_stride,
    ak_stride,
    # strides of B
    bm_stride,
    bn_stride,
    bk_stride,
    # slice start index
    start,
    # slice end index
    end,
    # slice step
    step,
    NBLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,
):
    # Each cta processes one [1, NBLOCK, KBLOCK] chunk for input and output
    # The src chunk size is dynamically determined

    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    am_idx = pidx
    an_idx = pidy * NBLOCK

    # Offsets into inp and out chunks
    n_idx_tile = an_idx + tl.arange(0, NBLOCK)[:, None]
    k_idx_tile = tl.arange(0, KBLOCK)[None, :]
    a_offset = am_idx * am_stride + n_idx_tile * an_stride + k_idx_tile * ak_stride
    n_mask = n_idx_tile < N1
    k_mask = k_idx_tile < K

    # Offsets into src
    b_offset = am_idx * bm_stride
    b_offset += (n_idx_tile - start) * bn_stride // step
    b_offset += k_idx_tile * bk_stride
    # This mask applies to the [NBLOCK, KBLOCK]
    bn_mask = start <= n_idx_tile
    bn_mask &= n_idx_tile < end
    bn_mask &= (n_idx_tile - start) % step == 0

    # merge inp and src then write back
    inp = tl.load(A + a_offset, mask=n_mask & k_mask)
    src = tl.load(B + b_offset, mask=bn_mask & k_mask)
    out = tl.where(bn_mask & k_mask, src, inp)
    tl.store(A_out + a_offset, out, mask=n_mask & k_mask)
    # tl.store(A_out + a_offset, src, mask=bn_mask & k_mask)


@libentry()
@triton.autotune(
    configs=runtime.get_triton_config("select_scatter_inner"),
    key=["strided", "pivoted"],
)
@triton.heuristics(
    values={
        "predicate_load": lambda args: args["x_ncol"] > 0.5 * args["y_ncol"],
    }
)
@triton.jit(
    do_not_specialize=[
        "nrow",
        "x_ncol",
        "y_ncol",
        "x_si",
        "x_sj",
        "y_si",
        "y_sj",
        "start",
        "end",
        "step",
    ]
)
def scatter_2d_inner_kernel(
    X,
    Y,
    X_out,
    # sizes
    nrow,
    x_ncol,
    y_ncol,
    # strides
    x_si,
    x_sj,
    y_si,
    y_sj,
    # slice
    start,
    end,
    step,
    strided: tl.constexpr,
    pivoted: tl.constexpr,
    predicate_load: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    i0 = tle.program_id(0) * R
    j0 = tle.program_id(1) * C
    ii = i0 + tl.arange(0, R)[:, None]
    jj = j0 + tl.arange(0, C)[None, :]

    if predicate_load:
        # predicate then load
        px = X + ii * x_si + jj * x_sj
        if (j0 + C < start) | (j0 >= end):
            p = px
        else:
            py = Y + ii * y_si + (jj - start) * y_sj // step
            mask = (start <= jj) & (jj < end) & ((jj - start) % step == 0)
            p = tl.where((ii < nrow) & mask, py, px)
        tmp = tl.load(p, mask=(ii < nrow) & (jj < x_ncol))
        tl.store(X_out + ii * x_si + jj * x_sj, tmp, mask=(ii < nrow) & (jj < x_ncol))
    else:
        # load then predicate
        x = tl.load(X + ii * x_si + jj * x_sj, mask=(ii < nrow) & (jj < x_ncol))
        if (j0 + C < start) | (j0 >= end):
            tl.store(X_out + ii * x_si + jj * x_sj, x, mask=(ii < nrow) & (jj < x_ncol))
        else:
            mask = (start <= jj) & (jj < end) & ((jj - start) % step == 0)
            y = tl.load(
                Y + ii * y_si + (jj - start) * y_sj // step, mask=(ii < nrow) & mask
            )
            z = tl.where((ii < nrow) & mask, y, x)
            tl.store(X_out + ii * x_si + jj * x_sj, z, mask=(ii < nrow) & (jj < x_ncol))


@triton.jit(
    do_not_specialize=[
        "nrow_A",
        "nrow_B",
        "ncol",
        "a_stride0",
        "a_stride1",
        "b_stride0",
        "b_stride1",
        "start",
        "end",
        "step",
    ]
)
def scatter_2d_outer_kernel(
    A,
    B,
    A_out,
    nrow_A,
    nrow_B,
    ncol,
    a_stride0,
    a_stride1,
    b_stride0,
    b_stride1,
    start,
    end,
    step,
    NROW: tl.constexpr,
    NCOL: tl.constexpr,
):
    pidx = tle.program_id(0)
    pidy = tle.program_id(1)

    row_idx = pidx * NROW
    col_idx = pidy * NCOL

    # Offsets into inp and out chunks
    row_idx = row_idx + tl.arange(0, NROW)[:, None]
    col_idx = col_idx + tl.arange(0, NCOL)[None, :]
    a_offset = row_idx * a_stride0 + col_idx * a_stride1
    a_row_mask = row_idx < nrow_A
    col_mask = col_idx < ncol

    # Offsets into src
    b_offset = (row_idx - start) * b_stride0 // step
    b_offset += col_idx * b_stride1
    # This mask applies to the [NROW, NCOL]
    b_row_mask = start <= row_idx
    b_row_mask &= row_idx < end
    b_row_mask &= (row_idx - start) % step == 0

    # merge inp and src then write back
    inp = tl.load(A + a_offset, mask=a_row_mask & col_mask)
    src = tl.load(B + b_offset, mask=b_row_mask & col_mask)
    out = tl.where(b_row_mask & col_mask, src, inp)
    tl.store(A_out + a_offset, out, mask=a_row_mask & col_mask)


def scatter_2d_inner(x, y, x_out, start, end, step):
    nrow, x_ncol = x_out.size()
    y_ncol = y.size(1)
    x_stride_i, x_stride_j = x_out.stride()
    y_stride_i, y_stride_j = y.stride()
    strided = step > 1
    pivoted = y_stride_i < y_stride_j
    grid = lambda meta: (
        triton.cdiv(nrow, meta["R"]),
        triton.cdiv(x_ncol, meta["C"]),
    )
    scatter_2d_inner_kernel[grid](
        x,
        y,
        x_out,
        nrow,
        x_ncol,
        y_ncol,
        x_stride_i,
        x_stride_j,
        y_stride_i,
        y_stride_j,
        start,
        end,
        step,
        strided=strided,
        pivoted=pivoted,
    )


def scatter_2d_outer(A, B, A_out, start, end, step):
    nrow_A, ncol = A_out.size()
    nrow_B = B.size(0)
    a_stride0, a_stride1 = A_out.stride()
    b_stride0, b_stride1 = B.stride()
    grid = lambda meta: (
        triton.cdiv(nrow_A, meta["NROW"]),
        triton.cdiv(ncol, meta["NCOL"]),
    )
    scatter_2d_outer_kernel[grid](
        A,
        B,
        A_out,
        nrow_A,
        nrow_B,
        ncol,
        a_stride0,
        a_stride1,
        b_stride0,
        b_stride1,
        start,
        end,
        step,
        NROW=64,
        NCOL=64,
    )


def scatter_3d_mid(A, B, A_out, start, end, step):
    M, N1, K = A_out.size()
    N2 = B.size(1)
    am_stride, an_stride, ak_stride = A_out.stride()
    bm_stride, bn_stride, bk_stride = B.stride()

    grid = lambda meta: (M, triton.cdiv(N1, meta["NBLOCK"]))
    scatter_3d_mid_kernel[grid](
        A,
        B,
        A_out,
        N1,
        N2,
        K,
        am_stride,
        an_stride,
        ak_stride,
        bm_stride,
        bn_stride,
        bk_stride,
        start,
        end,
        step,
        NBLOCK=512,
        KBLOCK=triton.next_power_of_2(K),
    )


def slice_scatter_v2(inp, src, dim=0, start=None, end=None, step=1):
    logging.debug("GEMS SLICE_SCATTER")
    assert src.device == inp.device, "inp and src reside on different devices."
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert step > 0, "slice step must be positive"
    dim = dim % inp.ndim

    start = start or 0
    end = end or inp.size(dim)

    out = torch.empty_strided(
        inp.size(), inp.stride(), dtype=inp.dtype, device=inp.device
    )

    # Look for a permute of dims so that the outer dims and inner dims relative to dim
    # after permute can be coalesced.
    # But this can be difficult so we're resorting to a suffcient and not necessary condition.
    new_out, ordered_dims = simplify(out, dim)
    if new_out is not None:
        new_src, _ = simplify(src, dim, ordered_dims)
    else:
        new_src = None

    if new_out is not None and new_src is not None:
        if dim == ordered_dims[0]:
            if len(ordered_dims) == 1:
                new_out = new_out.unsqueeze(0)
                new_src = new_src.unsqueeze(0)
            # slices on inner dim
            scatter_2d_inner(inp, new_src, new_out, start, end, step)
        elif new_src.stride(-1) == new_out.stride(-1) == 1 and new_src.size(-1) >= 128:
            # slices on outer dims while inner dims are contiguous
            scatter_by_row(inp, new_src, new_out, start, end, step)
        elif dim == ordered_dims[-1]:
            # slices on outer dims while inner dims may not be contiguous
            scatter_2d_outer(inp, new_src, new_out, start, end, step)
        else:
            # slices on middle dims
            scatter_3d_mid(inp, new_src, new_out, start, end, step)
        return out
    # Fall back
    return slice_scatter(inp, src, dim, start, end, step)

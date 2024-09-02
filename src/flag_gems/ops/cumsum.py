import logging

import torch
import triton
import triton.language as tl

from ..utils import libentry


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def cumsum_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offset = m_offset[:, None, None] * N * K + n_offset[None, :, None] * K + pid_k
    mask = m_offset[:, None, None] < M and n_offset[None, :, None] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask).to(tl.float32)
    result = tl.cumsum(inp_vals, axis=1)
    out_ptrs = out + offset
    tl.store(out_ptrs, result, mask=mask)


def cumsum(inp, dim=1, *, dtype=None):
    logging.debug("GEMS CUMSUM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int64
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch.cuda.device(inp.device):
        cumsum_kernel[grid](inp, out, M, N, K)
    return out


@libentry()
@triton.jit(do_not_specialize=["K"])
def normed_cumsum_kernel(inp, out, K, BLOCK: tl.constexpr):
    row_start = tl.program_id(0) * K
    row_off = tl.arange(0, BLOCK)
    x = tl.load(inp + row_start + row_off, mask=row_off < K, other=0)
    if x.dtype.is_fp16():
        x = x.to(tl.float32)
    y_sum = tl.sum(x, 0)
    y = tl.cumsum(x, 0)
    y = y / y_sum
    tl.store(out + row_start + row_off, y, mask=row_off < K)


@libentry()
@triton.jit(
    do_not_specialize=[
        "r",
        "t",
        "R",
        "K",
        "r_stride",
        "out_r_stride",
    ]
)
def block_cumsum_kernel(
    inp,
    out,
    sums,
    r,
    t,
    R,
    K,
    r_stride,
    k_stride,
    out_r_stride,
    out_k_stride,
    OUTPUT_SUMS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HAS_OUT_LAYOUT: tl.constexpr,
    TILE: tl.constexpr,
):
    # One CTA processes a (r, t*tile) chunk
    # rows = [ grid.y, grid.y + r )
    # cols = [ grid.x * t * tile, (grid.x + 1) * t * tile )
    gridx = tl.program_id(0).to(tl.int64)
    gridy = tl.program_id(1).to(tl.int64)
    n_chunks = tl.num_programs(0)

    for row in range(gridy * r, min((gridy + 1) * r, R)):
        curr_cumsum = tl.zeros((1,), tl.float32)
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for ti in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            if x.dtype.is_fp16() | x.dtype.is_bf16():
                x = x.to(tl.float32)
            tile_sum = tl.sum(x, 0)[None]
            tile_cumsum = tl.cumsum(x, 0) + curr_cumsum
            curr_cumsum += tile_sum
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, tile_cumsum, mask=cols < K)
            if OUTPUT_SUMS:
                tl.store(sums + row * n_chunks + gridx[None], curr_cumsum)
            cols += TILE
        if NORMALIZE:
            cols = gridx * t * TILE + tl.arange(0, TILE)
            for _ in range(0, t):
                cols_offset = cols * k_stride
                if HAS_OUT_LAYOUT:
                    cols_offset = cols * out_k_stride
                    row_offset = row * out_r_stride
                x = tl.load(out + row_offset + cols_offset, mask=cols < K, other=0)
                if x.dtype.is_fp16() | x.dtype.is_bf16():
                    x = x.to(tl.float32)
                x = x / curr_cumsum
                tl.store(out + row_offset + cols_offset, x, mask=cols < K)
                cols += TILE


@libentry()
@triton.jit(
    do_not_specialize=[
        "r",
        "t",
        "R",
        "K",
        "r_stride",
        "out_r_stride",
    ]
)
def block_update_kernel(
    inp,
    base,
    rscale_ptr,
    out,
    r,
    t,
    R,
    K,
    r_stride,
    k_stride,
    out_r_stride,
    out_k_stride,
    rscale_stride,
    HAS_OUT_LAYOUT: tl.constexpr,
    TILE: tl.constexpr,
):
    # One CTA processes a (r, t*tile) chunk
    # rows = [ grid.y, grid.y + r )
    # cols = [ grid.x * t * tile, (grid.x + 1) * t * tile )
    gridx = tl.program_id(0).to(tl.int64)
    gridy = tl.program_id(1).to(tl.int64)
    n_gridx = tl.num_programs(1)

    base += gridy * n_gridx + gridx
    rscale_ptr += gridy * rscale_stride

    for row in range(gridy, min(gridy + r, R)):
        d = tl.load(base)
        rscale = tl.load(rscale_ptr)
        base += gridx
        rscale_ptr += rscale_stride
        row_offset = row * r_stride
        cols = gridx * t * TILE + tl.arange(0, TILE)
        for _ in range(0, t):
            cols_offset = cols * k_stride
            x = tl.load(inp + row_offset + cols_offset, mask=cols < K, other=0)
            x += d
            x /= rscale
            if HAS_OUT_LAYOUT:
                cols_offset = cols * out_k_stride
                row_offset = row * out_r_stride
            tl.store(out + row_offset + cols_offset, x, mask=cols < K)
            cols += TILE


GRID_Y_LIMIT = 65535


def normed_cumsum(inp, dim=-1):
    logging.debug("GEMS NORMED_CUMSUM")
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    dim = dim % inp.ndim
    N = inp.numel()
    K = inp.size(dim)
    # inp = inp.contiguous()
    # First and last dims are easier to handle, but transpose the middle dim to the last
    ranked_dims = sorted(range(inp.ndim), key=lambda i: inp.stride(i), reverse=True)
    is_mid_dim = dim not in (ranked_dims[0], ranked_dims[-1])
    if is_mid_dim:
        inp = inp.transpose(dim, -1).contiguous()
        dim = -1
    out = torch.empty_like(inp)
    with torch.cuda.device(inp.device.index):
        # Pass one, scan a (batch, n_tiles * TILE) sized block within each cta
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        TILE = 2048
        # Each row is split into n_chunks of chunks where each chunk is compised of
        # n_tiles of tiles. Different chunks are assigned to different ctas.
        n_rows = N // K
        n_chunks = min(triton.cdiv(num_sms, n_rows), triton.cdiv(K, TILE))
        n_tiles = triton.cdiv(triton.cdiv(K, TILE), n_chunks)
        k_stride = inp.stride(dim)
        r_stride = inp.size(dim) if k_stride == 1 else 1
        if n_rows > GRID_Y_LIMIT:
            batch = triton.cdiv(n_rows, GRID_Y_LIMIT)
            n_batch = triton.cdiv(n_rows, batch)
        else:
            batch = 1
            n_batch = n_rows

        grid = (n_chunks, n_batch)
        if n_chunks == 1:
            block_cumsum_kernel[grid](
                inp,
                out,
                0,
                batch,
                n_tiles,
                n_rows,
                K,
                r_stride,
                k_stride,
                r_stride,
                k_stride,
                OUTPUT_SUMS=False,
                NORMALIZE=True,
                HAS_OUT_LAYOUT=False,
                TILE=TILE,
            )
            return out

        if inp.dtype != torch.float64:
            acc_dtype = torch.float32
        sums = torch.empty((n_rows, n_chunks), dtype=acc_dtype, device="cuda")
        cumsums = torch.empty_like(sums)
        block_cumsum_kernel[grid](
            inp,
            out,
            sums,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            OUTPUT_SUMS=True,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        # Pass two, scan partial cumsums
        block_cumsum_kernel[(1, n_batch)](
            sums,
            cumsums,
            0,
            batch,
            1,
            n_rows,
            n_chunks,
            n_chunks,
            1,
            n_chunks,
            1,
            OUTPUT_SUMS=False,
            NORMALIZE=False,
            HAS_OUT_LAYOUT=True,
            TILE=TILE,
        )
        # print(sums)
        rscale = cumsums[..., -1]
        block_update_kernel[grid](
            out,
            cumsums - sums,
            rscale,
            out,
            batch,
            n_tiles,
            n_rows,
            K,
            r_stride,
            k_stride,
            r_stride,
            k_stride,
            n_chunks,
            HAS_OUT_LAYOUT=False,
            TILE=TILE,
        )
        return out

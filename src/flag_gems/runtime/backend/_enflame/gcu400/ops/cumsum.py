import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import get_device_properties, libentry

_NP2 = triton.next_power_of_2
_CDIV = triton.cdiv
_BS = 1024
_MAX_GRID_X = 65535
_MAX_GRID_Z = 255
_NW = 1

device = device.name


@triton.jit
def _scan_part_k(inp, out, partial_sum, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(inp + offset, mask=mask, other=0.0)
    if (
        tl.constexpr(x.dtype.is_int64())
        or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    y = tl.cumsum(x, axis=0)
    tl.store(out + offset, y, mask=mask)
    tl.store(partial_sum + pid, tl.sum(x))


@triton.jit
def _add_base_k(out, partial_sum, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        return
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    vals = tl.load(out + offset, mask=mask)
    base = tl.load(partial_sum + pid - 1)
    tl.store(out + offset, (vals + base).to(vals.dtype), mask=mask)


@triton.jit
def _row_cumsum_k(inp, out, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(inp + base + cols, mask=mask, other=0.0)
    if (
        tl.constexpr(x.dtype.is_int64())
        or tl.constexpr(x.dtype.is_uint64())
    ) or tl.constexpr(x.dtype.is_fp64()):
        x = x
    elif tl.constexpr(x.dtype.is_int()):
        x = x.to(tl.int32)
    else:
        x = x.to(tl.float32)
    y = tl.cumsum(x, axis=0)
    tl.store(out + base + cols, y, mask=mask)


@triton.jit
def _row_scan_part_k(inp, out, partial_sum, M, N, m_grid, part_num, BLOCK_SIZE: tl.constexpr):
    blk = tl.program_id(0)
    pid_m = tl.program_id(1)
    row = pid_m
    while row < M:
        base = row * N
        cols = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(inp + base + cols, mask=mask, other=0.0)
        if (
            tl.constexpr(x.dtype.is_int64())
            or tl.constexpr(x.dtype.is_uint64())
        ) or tl.constexpr(x.dtype.is_fp64()):
            x = x
        elif tl.constexpr(x.dtype.is_int()):
            x = x.to(tl.int32)
        else:
            x = x.to(tl.float32)
        y = tl.cumsum(x, axis=0)
        tl.store(out + base + cols, y, mask=mask)
        tl.store(partial_sum + row * part_num + blk, tl.sum(x))
        row += m_grid


@triton.jit
def _row_add_base_k(out, partial_sum, M, N, m_grid, part_num, BLOCK_SIZE: tl.constexpr):
    blk = tl.program_id(0)
    pid_m = tl.program_id(1)
    if blk == 0:
        return
    row = pid_m
    while row < M:
        base = row * N
        cols = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        vals = tl.load(out + base + cols, mask=mask)
        base_sum = tl.load(partial_sum + row * part_num + blk - 1)
        tl.store(out + base + cols, (vals + base_sum).to(vals.dtype), mask=mask)
        row += m_grid


@triton.jit
def _scan_abc_k(inp, out, partial_sum, A, B, C, grid_a, c_grid, part_num, BLOCK_SIZE: tl.constexpr):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    a_idx = pid_a
    while a_idx < A:
        c_idx = pid_c
        while c_idx < C:
            b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            offset = a_idx * B * C + b_idx * C + c_idx
            mask = b_idx < B
            x = tl.load(inp + offset, mask=mask, other=0.0)
            if (
                tl.constexpr(x.dtype.is_int64())
                or tl.constexpr(x.dtype.is_uint64())
            ) or tl.constexpr(x.dtype.is_fp64()):
                x = x
            elif tl.constexpr(x.dtype.is_int()):
                x = x.to(tl.int32)
            else:
                x = x.to(tl.float32)
            y = tl.cumsum(x, axis=0)
            tl.store(out + offset, y, mask=mask)
            part_offset = a_idx * part_num * C + pid_b * C + c_idx
            tl.store(partial_sum + part_offset, tl.sum(x))
            c_idx += c_grid
        a_idx += grid_a


@triton.jit
def _add_abc_k(out, partial_sum, A, B, C, grid_a, c_grid, part_num, BLOCK_SIZE: tl.constexpr):
    pid_a = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_c = tl.program_id(2)
    a_idx = pid_a
    while a_idx < A:
        if pid_b > 0:
            c_idx = pid_c
            while c_idx < C:
                b_idx = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                offset = a_idx * B * C + b_idx * C + c_idx
                mask = b_idx < B
                vals = tl.load(out + offset, mask=mask)
                last_offset = a_idx * part_num * C + (pid_b - 1) * C + c_idx
                base_val = tl.load(partial_sum + last_offset)
                tl.store(out + offset, (vals + base_val).to(vals.dtype), mask=mask)
                c_idx += c_grid
        a_idx += grid_a


def _scan_1d(inp, out, N, dtype):
    bs = min(_BS, _NP2(N))
    pn = _CDIV(N, bs)
    ps = torch.empty(pn, dtype=dtype, device=inp.device)
    _scan_part_k[(pn,)](inp, out, ps, N, bs, num_warps=_NW)
    if pn >= 2:
        _scan_1d(ps, ps, pn, dtype)
        _add_base_k[(pn,)](out, ps, N, bs)


def _scan_rows(inp, out, M, N, dtype):
    ROW_BS = 4096
    if N <= ROW_BS:
        bs = _NP2(N)
        _row_cumsum_k[(M,)](inp, out, N, bs, num_warps=_NW)
        return
    bs = ROW_BS
    pn = _CDIV(N, bs)
    ps = torch.empty(M, pn, dtype=dtype, device=inp.device)
    m_grid = min(M, _MAX_GRID_Z)
    _row_scan_part_k[(pn, m_grid)](inp, out, ps, M, N, m_grid, pn, bs, num_warps=_NW)
    if pn >= 2:
        _scan_rows(ps.view(M, pn), ps.view(M, pn), M, pn, dtype)
        _row_add_base_k[(pn, m_grid)](out, ps, M, N, m_grid, pn, bs)


def _scan_abc(inp, out, A, B, C, dtype):
    bs = 4096
    if B <= 4096:
        bs = _NP2(B)
    pn = _CDIV(B, bs)
    ps = torch.empty(A, pn, C, dtype=dtype, device=inp.device)
    grid_a = min(A, _MAX_GRID_X)
    c_grid = min(C, _MAX_GRID_Z)
    grid = (grid_a, pn, c_grid)
    _scan_abc_k[grid](inp, out, ps, A, B, C, grid_a, c_grid, pn, bs, num_warps=_NW)
    if pn >= 2:
        _scan_abc(ps, ps, A, pn, C, dtype)
        _add_abc_k[grid](out, ps, A, B, C, grid_a, c_grid, pn, bs)


def _cumsum_impl(inp, dim, dtype=None, out=None):
    if inp.dtype == torch.int64:
        inp = inp.to(torch.int32)
    if dtype == torch.int64:
        dtype = torch.int32

    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = 1
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            dtype = torch.int32
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    compute_dtype = out.dtype
    if inp.dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32

    if K == 1:
        if M == 1:
            if N <= _BS:
                _row_cumsum_k[(1,)](inp, out, N, _NP2(N), num_warps=_NW)
            else:
                _scan_1d(inp, out, N, compute_dtype)
        else:
            _scan_rows(inp.view(M, N), out.view(M, N), M, N, compute_dtype)
    else:
        _scan_abc(inp, out, M, N, K, compute_dtype)
    return out


def cumsum(inp, dim=1, *, dtype=None):
    return _cumsum_impl(inp, dim, dtype)


def cumsum_out(inp, dim=1, *, dtype=None, out):
    return _cumsum_impl(inp, dim, dtype, out)


# --- normed_cumsum (kept with original structure) ---

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
    gridx = tl.program_id(0)
    gridy = tl.program_id(1)
    n_chunks = tl.num_programs(0)

    gridx = tl.program_id(0)
    gridy = tl.program_id(1)
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
    gridx = tl.program_id(0)
    gridy = tl.program_id(1)
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


GRID_Y_LIMIT = 255


def normed_cumsum(inp, dim=-1):
    assert inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    dim = dim % inp.ndim
    N = inp.numel()
    K = inp.size(dim)
    ranked_dims = sorted(range(inp.ndim), key=lambda i: inp.stride(i), reverse=True)
    is_mid_dim = dim not in (ranked_dims[0], ranked_dims[-1])
    if is_mid_dim:
        inp = inp.transpose(dim, -1).contiguous()
        dim = -1
    out = torch.empty_like(inp)
    with torch_device_fn.device(inp.device.index):
        num_sms = get_device_properties(device).multi_processor_count
        TILE = 2048
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
                inp, out, 0, batch, n_tiles, n_rows, K,
                r_stride, k_stride, r_stride, k_stride,
                OUTPUT_SUMS=False, NORMALIZE=True,
                HAS_OUT_LAYOUT=False, TILE=TILE,
            )
            return out

        if inp.dtype != torch.float64:
            acc_dtype = torch.float32
        sums = torch.empty((n_rows, n_chunks), dtype=acc_dtype, device=device)
        cumsums = torch.empty_like(sums)
        block_cumsum_kernel[grid](
            inp, out, sums, batch, n_tiles, n_rows, K,
            r_stride, k_stride, r_stride, k_stride,
            OUTPUT_SUMS=True, NORMALIZE=False,
            HAS_OUT_LAYOUT=False, TILE=TILE,
        )
        block_cumsum_kernel[(1, n_batch)](
            sums, cumsums, 0, batch, 1, n_rows, n_chunks,
            n_chunks, 1, n_chunks, 1,
            OUTPUT_SUMS=False, NORMALIZE=False,
            HAS_OUT_LAYOUT=True, TILE=TILE,
        )
        rscale = cumsums[..., -1]
        block_update_kernel[grid](
            out, cumsums - sums, rscale, out,
            batch, n_tiles, n_rows, K,
            r_stride, k_stride, r_stride, k_stride, n_chunks,
            HAS_OUT_LAYOUT=False, TILE=TILE,
        )
        return out

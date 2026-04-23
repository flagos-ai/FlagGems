import torch
import triton
import triton.language as tl

from flag_gems.utils import dim_compress

_NP2 = triton.next_power_of_2
_CDIV = triton.cdiv
_NPROGS = 48
_NW = 1
_BS = 8192


@triton.jit
def _count_flat_small_k(
    x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr,
):
    cnt = tl.zeros((), dtype=tl.int32)
    for off in range(0, numel, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        cnt += tl.sum((x != 0).to(tl.int32), axis=0)
    tl.store(out_ptr, cnt)


@triton.jit
def _count_flat_k(
    x_ptr, out_ptr, numel, num_progs, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cnt = tl.zeros((), dtype=tl.int32)
    blk = pid
    while blk * BLOCK_SIZE < numel:
        offsets = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        cnt += tl.sum((x != 0).to(tl.int32), axis=0)
        blk += num_progs
    tl.store(out_ptr + pid, cnt)


@triton.jit
def _reduce_sum_k(
    in_ptr, out_ptr, N: tl.constexpr,
):
    idx = tl.arange(0, N)
    v = tl.load(in_ptr + idx)
    s = tl.sum(v, axis=0)
    tl.store(out_ptr, s)


@triton.jit
def _count_dim_k(
    x_ptr, out_ptr, N, M, num_progs, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid
    while row < M:
        cnt = tl.zeros((), dtype=tl.int32)
        row_start = row * N
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(x_ptr + row_start + cols, mask=mask, other=0)
            cnt += tl.sum((x != 0).to(tl.int32), axis=0)
        tl.store(out_ptr + row, cnt)
        row += num_progs


@triton.jit
def _count_dim_batch_k(
    x_ptr, out_ptr, N, M, num_progs, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    tile_id = pid
    while tile_id < num_tiles:
        m_off = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_off < M
        cnt = tl.zeros([BLOCK_M], dtype=tl.int32)
        for n in range(0, N, 1):
            offsets = m_off * N + n
            x = tl.load(x_ptr + offsets, mask=m_mask, other=0)
            cnt += (x != 0).to(tl.int32)
        tl.store(out_ptr + m_off, cnt, m_mask)
        tile_id += num_progs


@triton.jit(do_not_specialize=["N", "M", "inner_size"])
def _count_dim_strided_k(
    x_ptr, out_ptr, N, M, inner_size, num_progs, BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    stride_n = inner_size
    tile_id = pid
    while tile_id < num_tiles:
        m_off = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_off < M
        m_high = m_off // inner_size
        m_low = m_off % inner_size
        base = m_high * (N * inner_size) + m_low
        cnt = tl.zeros([BLOCK_M], dtype=tl.int32)
        for n in range(0, N, 1):
            offsets = base + n * stride_n
            x = tl.load(x_ptr + offsets, mask=m_mask, other=0)
            cnt += (x != 0).to(tl.int32)
        tl.store(out_ptr + m_off, cnt, m_mask)
        tile_id += num_progs


def count_nonzero(x, dim=None):
    if dim is not None:
        dim = dim % x.ndim
        shape = x.shape
        N = shape[dim]
        M = x.numel() // N

        out = torch.empty(M, dtype=torch.int32, device=x.device)

        if dim == x.ndim - 1 and x.is_contiguous():
            x_flat = x.reshape(-1)
            if N <= 64 and M > 1024:
                BM = 512
                grid_x = min(_CDIV(M, BM), _NPROGS)
                _count_dim_batch_k[(grid_x,)](
                    x_flat, out, N, M, grid_x, BLOCK_M=BM, num_warps=_NW
                )
            else:
                grid_x = min(M, _NPROGS)
                BS = min(_NP2(N), _BS) if N <= _BS else _BS
                _count_dim_k[(grid_x,)](
                    x_flat, out, N, M, grid_x, BLOCK_SIZE=BS, num_warps=_NW
                )
        elif N <= 1024:
            x_contig = x.contiguous()
            x_flat = x_contig.flatten()
            inner_size = 1
            for i in range(dim + 1, x.ndim):
                inner_size *= shape[i]
            BM = 512
            grid_x = min(_CDIV(M, BM), _NPROGS)
            _count_dim_strided_k[(grid_x,)](
                x_flat, out, N, M, inner_size, grid_x, BLOCK_M=BM, num_warps=_NW
            )
        else:
            x_flat = dim_compress(x, dim).contiguous().flatten()
            grid_x = min(M, _NPROGS)
            BS = min(_NP2(N), _BS) if N <= _BS else _BS
            _count_dim_k[(grid_x,)](
                x_flat, out, N, M, grid_x, BLOCK_SIZE=BS, num_warps=_NW
            )

        out_shape = list(shape)
        del out_shape[dim]
        return out.view(out_shape).to(torch.int64)
    else:
        x = x.contiguous().flatten()
        numel = x.numel()

        if numel <= _BS:
            out = torch.empty(1, dtype=torch.int32, device=x.device)
            _count_flat_small_k[(1,)](
                x, out, numel, BLOCK_SIZE=_NP2(numel), num_warps=_NW
            )
            return out.to(torch.int64)[0]

        NP = _NP2(_NPROGS)
        out_t = torch.zeros(NP, dtype=torch.int32, device=x.device)
        _count_flat_k[(_NPROGS,)](
            x, out_t, numel, _NPROGS, BLOCK_SIZE=_BS, num_warps=_NW
        )

        out = torch.empty(1, dtype=torch.int32, device=x.device)
        _reduce_sum_k[(1,)](out_t, out, N=NP, num_warps=_NW)
        return out.to(torch.int64)[0]

import torch
import triton
import triton.language as tl

_BLOCK_C = 4096  # count kernel block size (split into 1024 sub-blocks)
_SPLIT = 4  # count sub-blocks per count block (-> 1024 granularity)
_BLOCK_S = 1024  # scatter kernel block size
_SB = 1024  # chunk size (in blocks) for the two-level scan
_FOLD_G = 1024  # max grid for the fused (scan-free) scatter
_SINGLE_N = 4096  # max numel handled by the single-block kernel


@triton.jit
def _count_kernel(
    x_ptr, counts_ptr, n_elements, BLOCK: tl.constexpr, SPLIT: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    nz = (x != 0) & mask
    nz32 = nz.to(tl.int32)
    if SPLIT == 1:
        tl.store(counts_ptr + pid, tl.sum(nz32))
    else:
        c4 = tl.sum(tl.reshape(nz32, (SPLIT, BLOCK // SPLIT)), axis=1)
        tl.store(counts_ptr + pid * SPLIT + tl.arange(0, SPLIT), c4)


@triton.jit
def _single_kernel(
    x_ptr,
    outs_ptr,
    total_ptr,
    n_elements,
    NDIM: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    nz = (x != 0) & mask
    local = tl.cumsum(nz.to(tl.int32), axis=0) - 1
    gpos = local.to(tl.int64)
    if NDIM == 1:
        tl.store(outs_ptr + gpos, offs.to(tl.int64), mask=nz)
    elif NDIM == 2:
        r = (offs // ST0).to(tl.int64)
        cc = (offs - r.to(tl.int32) * ST0).to(tl.int64)
        tl.store(outs_ptr + gpos, r, mask=nz)
        tl.store(outs_ptr + n_elements + gpos, cc, mask=nz)
    else:
        i0 = (offs // ST0).to(tl.int64)
        rem = offs - i0.to(tl.int32) * ST0
        i1 = (rem // ST1).to(tl.int64)
        i2 = (rem - i1.to(tl.int32) * ST1).to(tl.int64)
        tl.store(outs_ptr + gpos, i0, mask=nz)
        tl.store(outs_ptr + n_elements + gpos, i1, mask=nz)
        tl.store(outs_ptr + 2 * n_elements + gpos, i2, mask=nz)
    tl.store(total_ptr, tl.sum(nz.to(tl.int32)).to(tl.int64))


@triton.jit
def _scatter_fused_kernel(
    x_ptr,
    outs_ptr,
    counts_ptr,
    n_blocks,
    n_elements,
    NDIM: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    BLOCK: tl.constexpr,
    GB: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    nz = (x != 0) & mask
    nz32 = nz.to(tl.int32)
    local = tl.cumsum(nz32, axis=0) - 1
    offs_b = tl.arange(0, GB)
    cnt = tl.load(counts_ptr + offs_b, mask=offs_b < n_blocks, other=0).to(tl.int64)
    base = tl.sum(tl.where(offs_b < pid, cnt, 0))
    gpos = base + local.to(tl.int64)
    if NDIM == 1:
        tl.store(outs_ptr + gpos, offs.to(tl.int64), mask=nz)
    elif NDIM == 2:
        r = (offs // ST0).to(tl.int64)
        cc = (offs - r.to(tl.int32) * ST0).to(tl.int64)
        tl.store(outs_ptr + gpos, r, mask=nz)
        tl.store(outs_ptr + n_elements + gpos, cc, mask=nz)
    else:
        i0 = (offs // ST0).to(tl.int64)
        rem = offs - i0.to(tl.int32) * ST0
        i1 = (rem // ST1).to(tl.int64)
        i2 = (rem - i1.to(tl.int32) * ST1).to(tl.int64)
        tl.store(outs_ptr + gpos, i0, mask=nz)
        tl.store(outs_ptr + n_elements + gpos, i1, mask=nz)
        tl.store(outs_ptr + 2 * n_elements + gpos, i2, mask=nz)
    if pid == n_blocks - 1:
        tl.store(counts_ptr + n_blocks, (base + tl.sum(nz32).to(tl.int64)).to(tl.int32))


@triton.jit
def _chunkscan_kernel(
    counts_ptr, offsets_ptr, chunk_totals_ptr, n_blocks, SB: tl.constexpr
):
    c = tl.program_id(0)
    offs = c * SB + tl.arange(0, SB)
    m = offs < n_blocks
    cnt = tl.load(counts_ptr + offs, mask=m, other=0).to(tl.int64)
    cs = tl.cumsum(cnt, axis=0) - cnt
    tl.store(offsets_ptr + offs, cs, mask=m)
    tl.store(chunk_totals_ptr + c, tl.sum(cnt, axis=0))


@triton.jit
def _scatter_chunk_kernel(
    x_ptr,
    outs_ptr,
    offsets_ptr,
    chunk_totals_ptr,
    counts_ptr,
    n_blocks,
    n_elements,
    NDIM: tl.constexpr,
    ST0: tl.constexpr,
    ST1: tl.constexpr,
    BLOCK: tl.constexpr,
    SB: tl.constexpr,
    CB: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    nz = (x != 0) & mask
    nz32 = nz.to(tl.int32)
    local = tl.cumsum(nz32, axis=0) - 1
    c = pid // SB
    tc_offs = tl.arange(0, CB)
    tc = tl.load(chunk_totals_ptr + tc_offs, mask=tc_offs < c, other=0)
    base = tl.sum(tc, axis=0) + tl.load(offsets_ptr + pid)
    gpos = base + local.to(tl.int64)
    if NDIM == 1:
        tl.store(outs_ptr + gpos, offs.to(tl.int64), mask=nz)
    elif NDIM == 2:
        r = (offs // ST0).to(tl.int64)
        cc = (offs - r.to(tl.int32) * ST0).to(tl.int64)
        tl.store(outs_ptr + gpos, r, mask=nz)
        tl.store(outs_ptr + n_elements + gpos, cc, mask=nz)
    else:
        i0 = (offs // ST0).to(tl.int64)
        rem = offs - i0.to(tl.int32) * ST0
        i1 = (rem // ST1).to(tl.int64)
        i2 = (rem - i1.to(tl.int32) * ST1).to(tl.int64)
        tl.store(outs_ptr + gpos, i0, mask=nz)
        tl.store(outs_ptr + n_elements + gpos, i1, mask=nz)
        tl.store(outs_ptr + 2 * n_elements + gpos, i2, mask=nz)
    if pid == n_blocks - 1:
        tl.store(counts_ptr + n_blocks, (base + tl.sum(nz32).to(tl.int64)).to(tl.int32))


def run(inp):
    x = inp
    ndim = x.dim()
    if ndim == 0:
        x = x.view(1)
        ndim = 1
    elif not x.is_contiguous():
        x = x.contiguous()
    shape = x.shape
    n = x.numel()
    dev = x.device
    if n == 0:
        empty = torch.empty(0, dtype=torch.int64, device=dev)
        return (empty,) * ndim

    if ndim == 1:
        st0 = 1
        st1 = 1
    elif ndim == 2:
        st0 = shape[1]
        st1 = 1
    elif ndim == 3:
        st0 = shape[1] * shape[2]
        st1 = shape[2]
    else:
        st0 = 1
        st1 = 1
        for d in range(1, ndim):
            st0 *= shape[d]
        for d in range(2, ndim):
            st1 *= shape[d]

    if n <= _SINGLE_N:
        blk = triton.next_power_of_2(n)
        nw = 8 if blk >= 1024 else (4 if blk >= 128 else 1)
        outs = torch.empty((ndim, n), dtype=torch.int64, device=dev)
        total = torch.empty(1, dtype=torch.int64, device=dev)
        _single_kernel[(1,)](
            x, outs, total, n, NDIM=ndim, ST0=st0, ST1=st1, BLOCK=blk, num_warps=nw
        )
        n_out = int(total.item())
        if n_out == n:
            return tuple(outs[d] for d in range(ndim))
        return tuple(outs[d][:n_out] for d in range(ndim))

    # count with 4096-element blocks, split into 1024-element sub-counts;
    # slot [G4] of the counts buffer holds the total (written by the last
    # scatter block), so no separate zeroed accumulator is needed.
    G = triton.cdiv(n, _BLOCK_C)
    G4 = G * _SPLIT
    counts = torch.empty(G4 + 1, dtype=torch.int32, device=dev)
    outs = torch.empty((ndim, n), dtype=torch.int64, device=dev)
    _count_kernel[(G,)](x, counts, n, BLOCK=_BLOCK_C, SPLIT=_SPLIT, num_warps=4)
    if ndim <= 3 and G4 <= _FOLD_G:
        _scatter_fused_kernel[(G4,)](
            x,
            outs,
            counts,
            G4,
            n,
            NDIM=ndim,
            ST0=st0,
            ST1=st1,
            BLOCK=_BLOCK_S,
            GB=_FOLD_G,
            num_warps=8,
        )
    else:
        nchunks = triton.cdiv(G4, _SB)
        offsets = torch.empty(G4, dtype=torch.int64, device=dev)
        chunk_totals = torch.empty(nchunks, dtype=torch.int64, device=dev)
        _chunkscan_kernel[(nchunks,)](counts, offsets, chunk_totals, G4, SB=_SB)
        _scatter_chunk_kernel[(G4,)](
            x,
            outs,
            offsets,
            chunk_totals,
            counts,
            G4,
            n,
            NDIM=ndim,
            ST0=st0,
            ST1=st1,
            BLOCK=_BLOCK_S,
            SB=_SB,
            CB=triton.next_power_of_2(nchunks),
            num_warps=8,
        )
    n_out = int(counts[G4].item())
    if n_out == n:
        return tuple(outs[d] for d in range(ndim))
    return tuple(outs[d][:n_out] for d in range(ndim))


# Alias for FlagGems import convention
nonzero_numpy = run

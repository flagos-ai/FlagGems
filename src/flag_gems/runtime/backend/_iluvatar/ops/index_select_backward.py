import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_INT32_MAX = 2147483647


@triton.jit
def _isb_row_kernel(
    out_ptr,
    grad_ptr,
    index_ptr,
    n_rows,
    M,
    L,
    inner,
    nk,
    s_o,
    s_j,
    s_k,
    s_i,
    ACC_F32: tl.constexpr,
    INT32: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """inner > 1: one program per (outer*M destination row, BLOCK_K column tile).

    The index vector is walked sequentially so no cross-lane reduction is
    emitted; a non-matching entry predicates its load away. The scan is
    statically unrolled to amortise loop and address bookkeeping.
    """
    pid = tl.program_id(0)
    oi = pid // nk
    kt = pid - oi * nk
    o = oi // M
    i = oi - o * M
    valid = oi < n_rows

    k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
    kmask = (k < inner) & valid

    if INT32:
        rowbase = o * s_o + k * s_k
    else:
        rowbase = o.to(tl.int64) * s_o + k.to(tl.int64) * s_k

    if ACC_F32:
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_K], dtype=tl.int64)

    ngroups = L // 4
    for grp in range(0, ngroups):
        j0 = grp * 4
        for t in tl.static_range(4):
            j = j0 + t
            idx = tl.load(index_ptr + j * s_i)
            m = kmask & (idx == i)
            if INT32:
                g = tl.load(grad_ptr + rowbase + j * s_j, mask=m, other=0)
            else:
                g = tl.load(grad_ptr + rowbase + j.to(tl.int64) * s_j, mask=m, other=0)
            if ACC_F32:
                acc += g.to(tl.float32)
            else:
                acc += g.to(tl.int64)
    for j in range(ngroups * 4, L):
        idx = tl.load(index_ptr + j * s_i)
        m = kmask & (idx == i)
        if INT32:
            g = tl.load(grad_ptr + rowbase + j * s_j, mask=m, other=0)
        else:
            g = tl.load(grad_ptr + rowbase + j.to(tl.int64) * s_j, mask=m, other=0)
        if ACC_F32:
            acc += g.to(tl.float32)
        else:
            acc += g.to(tl.int64)

    if INT32:
        off = oi * inner + k
    else:
        off = oi.to(tl.int64) * inner + k.to(tl.int64)
    tl.store(out_ptr + off, acc.to(out_ptr.dtype.element_ty), mask=kmask)


@triton.jit
def _isb_vec_kernel(
    out_ptr,
    grad_ptr,
    index_ptr,
    outer,
    M,
    L,
    nI,
    s_o,
    s_j,
    s_i,
    ACC_F32: tl.constexpr,
    INT32: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """inner == 1: one program per (outer slice, BLOCK_I destination rows).

    Every lane owns one destination row and accumulates sequentially over j,
    statically unrolled, so no cross-lane reduction is required.
    """
    pid = tl.program_id(0)
    o = pid // nI
    it = pid - o * nI
    ib = it * BLOCK_I + tl.arange(0, BLOCK_I)
    imask = (ib < M) & (o < outer)

    if INT32:
        gbase = o * s_o
        obase = o * M
    else:
        gbase = o.to(tl.int64) * s_o
        obase = o.to(tl.int64) * M

    if ACC_F32:
        acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_I], dtype=tl.int64)

    ngroups = L // 4
    for grp in range(0, ngroups):
        j0 = grp * 4
        for t in tl.static_range(4):
            j = j0 + t
            idx = tl.load(index_ptr + j * s_i)
            if INT32:
                g = tl.load(grad_ptr + gbase + j * s_j)
            else:
                g = tl.load(grad_ptr + gbase + j.to(tl.int64) * s_j)
            if ACC_F32:
                acc += tl.where(ib == idx, g.to(tl.float32), 0.0)
            else:
                acc += tl.where(ib == idx, g.to(tl.int64), 0)
    for j in range(ngroups * 4, L):
        idx = tl.load(index_ptr + j * s_i)
        if INT32:
            g = tl.load(grad_ptr + gbase + j * s_j)
        else:
            g = tl.load(grad_ptr + gbase + j.to(tl.int64) * s_j)
        if ACC_F32:
            acc += tl.where(ib == idx, g.to(tl.float32), 0.0)
        else:
            acc += tl.where(ib == idx, g.to(tl.int64), 0)

    tl.store(out_ptr + obase + ib, acc.to(out_ptr.dtype.element_ty), mask=imask)


@triton.jit
def _isb_ocol_kernel(
    out_ptr,
    grad_ptr,
    index_ptr,
    outer,
    M,
    L,
    nO,
    s_o,
    s_j,
    s_i,
    ACC_F32: tl.constexpr,
    INT32: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    """inner == 1 with many outer slices: one program per (destination index,
    BLOCK_O outer slices).

    The destination index lives in scalar program space, so an index entry that
    does not select this destination costs no lane work and the vector work
    collapses to O(n_g); the outer slices fill the lanes.
    """
    pid = tl.program_id(0)
    i = pid // nO
    it = pid - i * nO
    io = it * BLOCK_O + tl.arange(0, BLOCK_O)
    omask = (io < outer) & (i < M)

    if INT32:
        gbase = io * s_o
        wbase = io * M + i
    else:
        gbase = io.to(tl.int64) * s_o
        wbase = io.to(tl.int64) * M + i

    if ACC_F32:
        acc = tl.zeros([BLOCK_O], dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_O], dtype=tl.int64)

    ngroups = L // 4
    for grp in range(0, ngroups):
        j0 = grp * 4
        for t in tl.static_range(4):
            j = j0 + t
            idx = tl.load(index_ptr + j * s_i)
            m = omask & (idx == i)
            if INT32:
                g = tl.load(grad_ptr + gbase + j * s_j, mask=m, other=0)
            else:
                g = tl.load(grad_ptr + gbase + j.to(tl.int64) * s_j, mask=m, other=0)
            if ACC_F32:
                acc += g.to(tl.float32)
            else:
                acc += g.to(tl.int64)
    for j in range(ngroups * 4, L):
        idx = tl.load(index_ptr + j * s_i)
        m = omask & (idx == i)
        if INT32:
            g = tl.load(grad_ptr + gbase + j * s_j, mask=m, other=0)
        else:
            g = tl.load(grad_ptr + gbase + j.to(tl.int64) * s_j, mask=m, other=0)
        if ACC_F32:
            acc += g.to(tl.float32)
        else:
            acc += g.to(tl.int64)

    tl.store(out_ptr + wbase, acc.to(out_ptr.dtype.element_ty), mask=omask)


def index_select_backward(grad, self_sizes, dim, index):
    logger.debug("GEMS_ILUVATAR INDEX_SELECT_BACKWARD")
    sizes = [int(s) for s in self_sizes]
    nd = len(sizes)
    d = dim + nd if dim < 0 else dim

    M = sizes[d]
    outer = 1
    for t in range(d):
        outer *= sizes[t]
    inner = 1
    for t in range(d + 1, nd):
        inner *= sizes[t]

    L = int(index.numel())
    out = torch.empty(sizes, dtype=grad.dtype, device=grad.device)

    if grad.is_contiguous():
        s_o = L * inner
        s_j = inner
        s_k = 1
    else:
        gv = grad.view(outer, L, inner)
        s_o = gv.stride(0)
        s_j = gv.stride(1)
        s_k = gv.stride(2)
    if d == 0:
        s_o = 0
    if d == nd - 1:
        s_k = 0
    s_i = index.stride(0) if index.dim() > 0 else 0

    acc_f32 = grad.dtype.is_floating_point
    int32 = grad.numel() <= _INT32_MAX and out.numel() <= _INT32_MAX

    if inner == 1 and outer >= 512:
        # Enough outer slices to keep the grid populated once the destination
        # index moves into scalar program space.
        bo = triton.next_power_of_2(outer)
        if bo > 256:
            bo = 256
        while bo > 32 and M * ((outer + bo - 1) // bo) < 128:
            bo //= 2
        no = (outer + bo - 1) // bo
        grid = (max(1, M * no),)
        _isb_ocol_kernel[grid](
            out,
            grad,
            index,
            outer,
            M,
            L,
            no,
            s_o,
            s_j,
            s_i,
            ACC_F32=acc_f32,
            INT32=int32,
            BLOCK_O=bo,
            num_warps=1 if bo <= 128 else 2,
        )
    elif inner == 1:
        bi = triton.next_power_of_2(M) if M > 0 else 1
        if bi > 64:
            bi = 64
        ni = (M + bi - 1) // bi if M > 0 else 1
        grid = (max(1, outer * ni),)
        _isb_vec_kernel[grid](
            out,
            grad,
            index,
            outer,
            M,
            L,
            ni,
            s_o,
            s_j,
            s_i,
            ACC_F32=acc_f32,
            INT32=int32,
            BLOCK_I=bi,
            num_warps=1,
        )
    else:
        bk = min(128, triton.next_power_of_2(inner))
        nk = (inner + bk - 1) // bk
        n_rows = outer * M
        ms = M if M > 0 else 1
        grid = (max(1, n_rows * nk),)
        _isb_row_kernel[grid](
            out,
            grad,
            index,
            n_rows,
            ms,
            L,
            inner,
            nk,
            s_o,
            s_j,
            s_k,
            s_i,
            ACC_F32=acc_f32,
            INT32=int32,
            BLOCK_K=bk,
            num_warps=1 if bk <= 64 else 2,
        )
    return out

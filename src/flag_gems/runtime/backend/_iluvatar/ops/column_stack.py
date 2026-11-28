"""Triton implementation of aten::column_stack.

Architecture: a single fused kernel launch copies all inputs into the contiguous
output.  The grid is (column-blocks, row-blocks, n_inputs); each program resolves
its input through a uniform branch on program_id(2), so there is exactly one
launch regardless of the number of inputs.

Two fused kernels cover the common cases with different argument counts:
  * _fused_uniform_kernel - all inputs have the same row stride, the same flat
    column stride and the same column count (the classic "stack equal-shaped
    inputs" form); the column prefix of input k is k*C.
  * _fused_kernel        - per-input row stride, column stride, column prefix and
    column count.
Inputs whose column dims cannot be expressed as one (size, stride) group, or a
number of inputs above 4, fall back to one launch per input (_copynd_kernel).

Tiling has two regimes: streaming shapes get a wide column tile (BLOCK_C up to
1024, BLOCK_R 1) so each thread owns 4 contiguous elements (16 B per lane), while
small shapes keep many narrow tiles to minimise latency.  When C % BLOCK_C == 0
and R % BLOCK_R == 0 every tile is fully in bounds, so the bounds masks are
dropped at compile time, which also lets Triton emit vector accesses.

Output geometry: out = (R, sum_i colsize_i, *dims[2:]) contiguous, therefore the
output row length is ROWO = sum_cols * dpost with dpost = prod(dims[2:]); input i
owns flat columns [P_i, P_i + C_i) of every row, with C_i = colsize_i * dpost and
P_i = (sum_{j<i} colsize_j) * dpost.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _slot_store(
    out_ptr,
    in_ptr,
    s0,
    t1,
    P,
    C,
    R,
    ROWO,
    offs_r,
    offs_c,
    OUT_DT: tl.constexpr,
    I64: tl.constexpr,
    NO_MASK: tl.constexpr,
):
    if I64:
        src = offs_r.to(tl.int64)[:, None] * s0 + offs_c.to(tl.int64)[None, :] * t1
        dst = (
            out_ptr
            + offs_r.to(tl.int64)[:, None] * ROWO
            + P
            + offs_c.to(tl.int64)[None, :]
        )
    else:
        src = offs_r[:, None] * s0 + offs_c[None, :] * t1
        dst = out_ptr + offs_r[:, None] * ROWO + P + offs_c[None, :]
    if NO_MASK:
        v = tl.load(in_ptr + src)
        tl.store(dst, v.to(OUT_DT))
    else:
        mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)
        v = tl.load(in_ptr + src, mask=mask)
        tl.store(dst, v.to(OUT_DT), mask=mask)


@triton.jit
def _fused_uniform_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    R,
    C,
    ROWO,
    s0,
    t1,
    OUT_DT: tl.constexpr,
    I64: tl.constexpr,
    NO_MASK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_k = tl.program_id(2)
    offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    if pid_k == 0:
        _slot_store(
            out_ptr, p0, s0, t1, 0, C, R, ROWO, offs_r, offs_c, OUT_DT, I64, NO_MASK
        )
    elif pid_k == 1:
        _slot_store(
            out_ptr, p1, s0, t1, C, C, R, ROWO, offs_r, offs_c, OUT_DT, I64, NO_MASK
        )
    elif pid_k == 2:
        _slot_store(
            out_ptr,
            p2,
            s0,
            t1,
            2 * C,
            C,
            R,
            ROWO,
            offs_r,
            offs_c,
            OUT_DT,
            I64,
            NO_MASK,
        )
    else:
        _slot_store(
            out_ptr,
            p3,
            s0,
            t1,
            3 * C,
            C,
            R,
            ROWO,
            offs_r,
            offs_c,
            OUT_DT,
            I64,
            NO_MASK,
        )


@triton.jit
def _fused_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    P0,
    P1,
    P2,
    P3,
    C0,
    C1,
    C2,
    C3,
    s0_0,
    t1_0,
    s0_1,
    t1_1,
    s0_2,
    t1_2,
    s0_3,
    t1_3,
    R,
    ROWO,
    OUT_DT: tl.constexpr,
    I64: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_k = tl.program_id(2)
    offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    if pid_k == 0:
        _slot_store(
            out_ptr,
            p0,
            s0_0,
            t1_0,
            P0,
            C0,
            R,
            ROWO,
            offs_r,
            offs_c,
            OUT_DT,
            I64,
            False,
        )
    elif pid_k == 1:
        _slot_store(
            out_ptr,
            p1,
            s0_1,
            t1_1,
            P1,
            C1,
            R,
            ROWO,
            offs_r,
            offs_c,
            OUT_DT,
            I64,
            False,
        )
    elif pid_k == 2:
        _slot_store(
            out_ptr,
            p2,
            s0_2,
            t1_2,
            P2,
            C2,
            R,
            ROWO,
            offs_r,
            offs_c,
            OUT_DT,
            I64,
            False,
        )
    else:
        _slot_store(
            out_ptr,
            p3,
            s0_3,
            t1_3,
            P3,
            C3,
            R,
            ROWO,
            offs_r,
            offs_c,
            OUT_DT,
            I64,
            False,
        )


@triton.jit
def _copynd_kernel(
    out_ptr,
    in_ptr,
    R,
    C,
    ROWO,
    s0,
    P,
    d1,
    t1,
    d2,
    t2,
    d3,
    t3,
    d4,
    t4,
    NEXTRA: tl.constexpr,
    OUT_DT: tl.constexpr,
    I64: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)

    off_c = tl.zeros([BLOCK_C], tl.int32)
    if NEXTRA == 1:
        off_c = offs_c * t1
    elif NEXTRA == 2:
        c2 = offs_c % d2
        c1 = offs_c // d2
        off_c = c1 * t1 + c2 * t2
    elif NEXTRA == 3:
        c3 = offs_c % d3
        rem = offs_c // d3
        c2 = rem % d2
        c1 = rem // d2
        off_c = c1 * t1 + c2 * t2 + c3 * t3
    else:
        c4 = offs_c % d4
        rem = offs_c // d4
        c3 = rem % d3
        rem = rem // d3
        c2 = rem % d2
        c1 = rem // d2
        off_c = c1 * t1 + c2 * t2 + c3 * t3 + c4 * t4

    if I64:
        src = offs_r.to(tl.int64)[:, None] * s0 + off_c.to(tl.int64)[None, :]
        dst = (
            out_ptr
            + offs_r.to(tl.int64)[:, None] * ROWO
            + P
            + offs_c.to(tl.int64)[None, :]
        )
    else:
        src = offs_r[:, None] * s0 + off_c[None, :]
        dst = out_ptr + offs_r[:, None] * ROWO + P + offs_c[None, :]

    v = tl.load(in_ptr + src, mask=mask)
    tl.store(dst, v.to(OUT_DT), mask=mask)


_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.bool: tl.int1,
}

_MAX_SLOTS = 4
_MAX_GRID_YZ = 65535
_MAX_BLOCK_C = 1024
# Above this per-input element count the copy is streaming (bandwidth/issue
# bound) and a wide column tile is used; below it the copy is latency bound and
# many narrow tiles are used instead.
_STREAM_ELEMS = 262144


def _merge_groups(shape, stride, nd):
    """Collapse dims 1..nd-1 into <=4 (size, stride) groups, outermost first."""
    groups = []
    cur_size = 1
    cur_stride = 0
    started = False
    for i in range(nd - 1, 0, -1):
        sz = shape[i]
        stp = stride[i]
        if not started:
            cur_size = sz
            cur_stride = stp
            started = True
        elif stp == cur_size * cur_stride:
            cur_size *= sz
        else:
            groups.append((cur_size, cur_stride))
            cur_size = sz
            cur_stride = stp
    if started:
        groups.append((cur_size, cur_stride))
    else:
        groups.append((1, 0))
    groups.reverse()
    return groups


def _blocks(R, C, contig):
    """Pick (BLOCK_R, BLOCK_C).  Streaming shapes get 4 contiguous elements per
    thread (16 B per lane); small shapes get many narrow tiles."""
    if contig and R * C >= _STREAM_ELEMS:
        bc = 1
        while bc < C and bc < _MAX_BLOCK_C:
            bc *= 2
        br = 1024 // bc
        if br < 1:
            br = 1
        if br > R:
            br = 1 << (R.bit_length() - 1) if R > 0 else 1
        while -(-R // br) > _MAX_GRID_YZ:
            br *= 2
        return br, bc

    bc = 1
    while bc < C and bc < 128:
        bc *= 2
    ng_c = -(-C // bc)
    ng_r_target = max(1, -(-32 // ng_c))
    br = 1024 // bc
    if br < 1:
        br = 1
    want = -(-R // ng_r_target)
    if br > want:
        br = want
    if br < 1:
        br = 1
    br = 1 << (br.bit_length() - 1)
    while -(-R // br) > _MAX_GRID_YZ:
        br *= 2
    return br, bc


def column_stack(tensors, out=None):
    logger.debug("GEMS_ILUVATAR COLUMN_STACK")
    if len(tensors) == 0:
        raise RuntimeError("column_stack expects a non-empty TensorList")

    meta = []
    ref_shape = None
    ref_rank = 0
    dtype = None
    device = tensors[0].device
    total_cols = 0
    dpost = 1
    big = False

    for idx, t in enumerate(tensors):
        nd = t.dim()
        shape = t.shape
        if nd == 0:
            R = 1
            colsize = 1
            dpost = 1
            s0 = 0
            pshape = (1, 1)
            groups = ((1, 0),)
        elif nd == 1:
            R = shape[0]
            colsize = 1
            dpost = 1
            s0 = t.stride(0)
            pshape = (R, 1)
            groups = ((1, 0),)
        else:
            R = shape[0]
            colsize = shape[1]
            dpost = 1
            for d in shape[2:]:
                dpost *= d
            s0 = t.stride(0)
            pshape = tuple(shape)
            groups = _merge_groups(shape, t.stride(), nd)
            if len(groups) > 4:
                raise NotImplementedError(
                    "column_stack: unsupported column layout (>4 stride groups)"
                )

        praw = 2 if nd <= 1 else nd
        if idx == 0:
            ref_shape = pshape
            ref_rank = praw
        else:
            if praw != ref_rank:
                raise RuntimeError(
                    "Tensors must have same number of dimensions: got "
                    f"{ref_rank} and {praw}"
                )
            for j in range(praw):
                if j != 1 and pshape[j] != ref_shape[j]:
                    raise RuntimeError(
                        f"Sizes of tensors must match except in dimension 1. "
                        f"Expected size {ref_shape[j]} but got size {pshape[j]} "
                        f"for tensor number {idx} in the list."
                    )

        dt = t.dtype
        dtype = dt if dtype is None else torch.promote_types(dtype, dt)
        C = colsize * dpost
        if t.numel() >= 2147483647 or C > 2147483647:
            big = True
        meta.append((t, R, C, s0, groups))
        total_cols += colsize

    out_shape = [ref_shape[0], total_cols]
    out_shape.extend(ref_shape[2:])
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=device)
    else:
        out = out.view(out_shape)

    if out.numel() == 0:
        return out
    if out.numel() >= 2147483647:
        big = True

    rowo = total_cols * dpost
    tl_dt = _TL_DTYPES[dtype]
    n = len(meta)
    R = meta[0][1]
    i64 = bool(big)

    # Fused single-launch paths: one column group per input, at most _MAX_SLOTS
    # inputs, all inputs exposing the same row count.
    fused = n <= _MAX_SLOTS and R > 0
    contig = True
    if fused:
        for _t, r_i, c_i, _s0, groups in meta:
            if r_i != R or c_i == 0 or len(groups) != 1:
                fused = False
                break
            if groups[0][1] != 1:
                contig = False

    if fused:
        maxc = max(m[2] for m in meta)
        br, bc = _blocks(R, maxc, contig)
        grid = (-(-maxc // bc), -(-R // br), n)
        s0_ref = meta[0][3]
        t1_ref = meta[0][4][0][1]
        c_ref = meta[0][2]
        uniform = True
        for _t, _r, c_i, s0_i, groups in meta:
            if s0_i != s0_ref or groups[0][1] != t1_ref or c_i != c_ref:
                uniform = False
                break
        ptrs = [out]
        for k in range(_MAX_SLOTS):
            ptrs.append(meta[k][0] if k < n else meta[0][0])
        if uniform:
            _fused_uniform_kernel[grid](
                *ptrs,
                R,
                c_ref,
                rowo,
                s0_ref,
                t1_ref,
                OUT_DT=tl_dt,
                I64=i64,
                NO_MASK=bool(c_ref % bc == 0 and R % br == 0),
                BLOCK_R=br,
                BLOCK_C=bc,
            )
        else:
            args = list(ptrs)
            for k in range(_MAX_SLOTS):
                args.append(sum(m[2] for m in meta[:k]) if k < n else 0)
            for k in range(_MAX_SLOTS):
                args.append(meta[k][2] if k < n else 1)
            for k in range(_MAX_SLOTS):
                args.append(meta[k][3] if k < n else 0)
                args.append(meta[k][4][0][1] if k < n else 0)
            args.append(R)
            args.append(rowo)
            _fused_kernel[grid](
                *args,
                OUT_DT=tl_dt,
                I64=i64,
                BLOCK_R=br,
                BLOCK_C=bc,
            )
        return out

    # General fallback: one launch per input.
    prefix = 0
    for t, r_i, c_i, s0_i, groups in meta:
        if r_i != 0 and c_i != 0:
            br, bc = _blocks(r_i, c_i, groups[0][1] == 1 and len(groups) == 1)
            grid = (-(-c_i // bc), -(-r_i // br))
            g = []
            for gsz, gst in groups:
                g.append(gsz)
                g.append(gst)
            while len(g) < 8:
                g.append(0)
            _copynd_kernel[grid](
                out,
                t,
                r_i,
                c_i,
                rowo,
                s0_i,
                prefix,
                g[0],
                g[1],
                g[2],
                g[3],
                g[4],
                g[5],
                g[6],
                g[7],
                NEXTRA=len(groups),
                OUT_DT=tl_dt,
                I64=i64,
                BLOCK_R=br,
                BLOCK_C=bc,
            )
        prefix += c_i
    return out

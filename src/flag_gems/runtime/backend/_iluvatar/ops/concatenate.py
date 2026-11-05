import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

MAXDIM = 8
_BLOCK = 1024
_BLOCK_R = 4
_BLOCK_Q = 128
_FUSED_MAX = 65536  # elements; below this, fuse all inputs into one launch


@triton.jit
def _cat_fused_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    in3_ptr,
    out_ptr,
    o_dim_inner: tl.int32,  # O[dim] * inner
    inner: tl.int32,
    total: tl.int32,
    n: tl.int32,
    c0: tl.int32,
    c1: tl.int32,
    c2: tl.int32,
    c3: tl.int32,
    s0: tl.int32,
    s1: tl.int32,
    s2: tl.int32,
    s3: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    u = pid * BLOCK + tl.arange(0, BLOCK)
    m = u < total
    r = u // o_dim_inner
    rem = u % o_dim_inner
    c = rem // inner
    t = rem % inner
    sel0 = (c >= c0) & (c < c0 + s0) & (0 < n)
    sel1 = (c >= c1) & (c < c1 + s1) & (1 < n)
    sel2 = (c >= c2) & (c < c2 + s2) & (2 < n)
    sel3 = (c >= c3) & (c < c3 + s3) & (3 < n)
    off0 = r * (s0 * inner) + (c - c0) * inner + t
    off1 = r * (s1 * inner) + (c - c1) * inner + t
    off2 = r * (s2 * inner) + (c - c2) * inner + t
    off3 = r * (s3 * inner) + (c - c3) * inner + t
    v0 = tl.load(in0_ptr + off0.to(tl.int64), mask=m & sel0, other=0.0)
    v1 = tl.load(in1_ptr + off1.to(tl.int64), mask=m & sel1, other=0.0)
    v2 = tl.load(in2_ptr + off2.to(tl.int64), mask=m & sel2, other=0.0)
    v3 = tl.load(in3_ptr + off3.to(tl.int64), mask=m & sel3, other=0.0)
    value = tl.where(sel0, v0, tl.where(sel1, v1, tl.where(sel2, v2, v3)))
    tl.store(out_ptr + u.to(tl.int64), value, mask=m)


@triton.jit
def _cat_copy_kernel(
    in_ptr,
    out_ptr,
    L: tl.int64,  # numel of this input tensor
    out_base: tl.int64,  # offset_k * inner (element offset into output)
    out_row_stride: tl.int64,  # O[dim] * inner (element stride between outer rows)
    VEC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    qblocks = (L + BLOCK - 1) // BLOCK
    r = pid // qblocks
    qb = pid % qblocks
    i = tl.arange(0, BLOCK).to(tl.int64)
    mask = qb * BLOCK + i < L
    base_in = r * L + qb * BLOCK
    base_out = out_base + r * out_row_stride + qb * BLOCK
    if VEC > 1:
        base_in = tl.multiple_of(base_in, VEC)
        base_out = tl.multiple_of(base_out, VEC)
    v = tl.load(in_ptr + base_in + i, mask=mask)
    tl.store(out_ptr + base_out + i, v, mask=mask)


@triton.jit
def _cat_general_kernel(
    in_ptr,
    out_ptr,
    out_base: tl.int64,
    out_row_stride: tl.int64,
    L: tl.int64,
    outer: tl.int64,
    s0: tl.int64,
    s1: tl.int64,
    s2: tl.int64,
    s3: tl.int64,
    s4: tl.int64,
    s5: tl.int64,
    s6: tl.int64,
    s7: tl.int64,
    k0: tl.int64,
    k1: tl.int64,
    k2: tl.int64,
    k3: tl.int64,
    k4: tl.int64,
    k5: tl.int64,
    k6: tl.int64,
    k7: tl.int64,
    BLOCK_R: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    MAXD: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    sizes = (s0, s1, s2, s3, s4, s5, s6, s7)
    strs = (k0, k1, k2, k3, k4, k5, k6, k7)
    qblocks = (L + BLOCK_Q - 1) // BLOCK_Q
    r0 = (pid // qblocks) * BLOCK_R
    qb = pid % qblocks
    r_vec = r0 + tl.arange(0, BLOCK_R).to(tl.int64)
    q_vec = qb * BLOCK_Q + tl.arange(0, BLOCK_Q).to(tl.int64)
    # Row-major flat logical index of this input.
    u = r_vec[:, None] * L + q_vec[None, :]
    uu = u
    in_off = tl.zeros([BLOCK_R, BLOCK_Q], dtype=tl.int64)
    for j in tl.static_range(MAXD - 1, -1, -1):
        sz = sizes[j]
        idx = uu % sz
        uu = uu // sz
        in_off += idx * strs[j]
    out_off = out_base + r_vec[:, None] * out_row_stride + q_vec[None, :]
    rm = r_vec < outer
    cm = q_vec < L
    m = rm[:, None] & cm[None, :]
    v = tl.load(in_ptr + in_off, mask=m)
    tl.store(out_ptr + out_off, v, mask=m)


def run(A, dim=0, out=None):
    logger.debug("GEMS_ILUVATAR CONCATENATE")
    if not A:
        raise ValueError("concatenate expects a non-empty tensor sequence")
    n = len(A)
    ndim = max(t.dim() for t in A)
    if ndim == 0:
        raise ValueError("zero-dimensional tensor cannot be concatenated")
    if not -ndim <= dim < ndim:
        raise IndexError(f"concatenate dimension {dim} is out of range for rank {ndim}")
    d = dim if dim >= 0 else dim + ndim

    dtype = A[0].dtype
    device = A[0].device
    shape_ref = next((t for t in A if not (t.dim() == 1 and t.numel() == 0)), A[0])
    for t in A:
        legacy_empty = t.dim() == 1 and t.numel() == 0
        if t.dim() != ndim and not legacy_empty:
            raise ValueError("all tensors must have the same number of dimensions")
        if t.dtype != dtype:
            raise TypeError("all tensors must have the same dtype")
        if t.device != device:
            raise ValueError("all tensors must be on the same device")
        for j in range(ndim):
            if not legacy_empty and j != d and t.shape[j] != shape_ref.shape[j]:
                raise ValueError(
                    "tensor sizes must match except in the concatenate dimension"
                )

    # Output shape: non-cat dims come from a non-empty input (all agree);
    # the cat dim sums the per-input sizes; empty tensors contribute 0.
    out_shape = [1] * ndim
    ref = None
    for t in A:
        if t.numel() > 0:
            ref = t
            break
    if ref is not None:
        for j in range(ndim):
            if j != d and j < ref.dim():
                out_shape[j] = ref.shape[j]
    else:
        for j in range(ndim):
            if j != d and j < A[0].dim():
                out_shape[j] = A[0].shape[j]
    cat_total = 0
    for t in A:
        cat_total += t.shape[d] if (t.numel() > 0 and t.dim() > d) else 0
    out_shape[d] = cat_total

    total = 1
    for s in out_shape:
        total *= s
    itemsize = A[0].element_size()
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=device)
    else:
        if out.dtype != dtype or out.device != device:
            raise ValueError("out must have the same dtype and device as the inputs")
        if not out.is_contiguous():
            raise ValueError("the Iluvatar concatenate out tensor must be contiguous")
        if tuple(out.shape) != tuple(out_shape):
            out.resize_(out_shape)
    if total == 0:
        return out

    outer = 1
    for j in range(d):
        outer *= out_shape[j]
    inner = 1
    for j in range(d + 1, ndim):
        inner *= out_shape[j]
    out_row_stride = out_shape[d] * inner

    vec = 16 // itemsize
    use_vec = vec if inner % vec == 0 else 1

    contig = all(t.is_contiguous() for t in A)

    # Collect the non-empty inputs that actually contribute data.
    launches = []
    offset = 0
    for t in A:
        if t.numel() == 0:
            continue
        L = t.shape[d] * inner
        if L > 0:
            launches.append((t, L, offset * inner))
        offset += t.shape[d]

    if contig:
        # One 128-bit vector per thread: 4 warps * 32 lanes * (16 // itemsize)
        # elements per program.  Measured sweet spot on BI-V150 for all dtypes.
        block = 128 * (16 // itemsize)
        if total <= _FUSED_MAX and n <= 4:
            # Pass each allocation as its own kernel argument.  This retains a
            # single launch without constructing pointers outside a tensor's
            # storage.  Missing/empty slots use a valid input pointer but are
            # masked by their zero segment length.
            ptrs = list(A) + [A[0]] * (4 - n)
            cums = [0] * 4
            ss = [0] * 4
            acc = 0
            for j, t in enumerate(A):
                sk = t.shape[d] if t.numel() > 0 else 0
                cums[j] = acc
                ss[j] = sk
                acc += sk
            grid = ((total + 255) // 256,)
            _cat_fused_kernel[grid](
                ptrs[0],
                ptrs[1],
                ptrs[2],
                ptrs[3],
                out,
                out_row_stride,
                inner,
                total,
                n,
                cums[0],
                cums[1],
                cums[2],
                cums[3],
                ss[0],
                ss[1],
                ss[2],
                ss[3],
                BLOCK=256,
            )
        else:
            for t, L, out_base in launches:
                qblocks = (L + block - 1) // block
                grid = (outer * qblocks,)
                _cat_copy_kernel[grid](
                    t,
                    out,
                    L,
                    out_base,
                    out_row_stride,
                    VEC=use_vec,
                    BLOCK=block,
                )
    else:
        for t, L, out_base in launches:
            shape = t.shape
            stride = t.stride()
            sargs = [shape[j] if j < ndim else 1 for j in range(MAXDIM)]
            kargs = [stride[j] if j < ndim else 0 for j in range(MAXDIM)]
            rblocks = (outer + _BLOCK_R - 1) // _BLOCK_R
            qblocks = (L + _BLOCK_Q - 1) // _BLOCK_Q
            grid = (rblocks * qblocks,)
            _cat_general_kernel[grid](
                t,
                out,
                out_base,
                out_row_stride,
                L,
                outer,
                sargs[0],
                sargs[1],
                sargs[2],
                sargs[3],
                sargs[4],
                sargs[5],
                sargs[6],
                sargs[7],
                kargs[0],
                kargs[1],
                kargs[2],
                kargs[3],
                kargs[4],
                kargs[5],
                kargs[6],
                kargs[7],
                BLOCK_R=_BLOCK_R,
                BLOCK_Q=_BLOCK_Q,
                MAXD=MAXDIM,
            )
    return out

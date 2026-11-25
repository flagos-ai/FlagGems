"""TLE implementation of FlagGems ``_index_put_impl_`` (in-place scatter).

Three device kernels carry all value-producing work; the wrapper only performs
metadata reads, zero-copy views, scalar/grid math and the JIT launch.

``_scatter_flat_kernel`` -- the index tensors already cover every dimension of
``inp``, so the trailing volume is one element.  One lane per index position and
every address a pure affine function of the lane.  This replaces the degenerate
``[B0, 1]`` tile whose masked 16-bit ``atomic_add`` collapsed a whole program's
lanes onto one address in round 1.

``_scatter_inner_kernel`` -- one index tensor over a contiguous ``inp``.  The
innermost ``inp`` dimension is a contiguous run of ``INNER`` elements in ``inp``;
the outer trailing coordinates are decoded once per program (scalar), so the tile
address is ``base + tl.arange(0, INNER)``: a plain affine unit-stride run the
backend can lower to wide vector movement instead of one computed scalar store
per element (round 1: 258 ms for a 512^3 input).

``_index_put_impl_kernel`` -- general fallback (several index tensors with a
trailing volume, or a strided ``inp``).  A multi-dimensional scatter destination
and its gathered ``values`` address are not expressible as a bulk tile transfer,
which is exactly the masked per-lane movement the implementation profile permits
plain ``triton.language`` for.  ``tle`` remains the admitted DSL entry point.
"""

import logging

import torch
import triton
import triton.experimental.tle as tle  # noqa: F401  (admitted DSL entry point)
import triton.language as tl

logger = logging.getLogger(__name__)

_MAX_RANK = 4
_FLAT_BLOCK = 1024
_MAX_INNER = 4096
_RUN_BLOCK = 8192


@triton.jit
def _scatter_flat_kernel(
    inp_ptr,
    val_ptr,
    i0_ptr,
    i1_ptr,
    i2_ptr,
    i3_ptr,
    is0,
    is1,
    is2,
    is3,
    it0,
    it1,
    it2,
    it3,
    id0,
    id1,
    id2,
    id3,
    s0_0,
    s0_1,
    s0_2,
    s0_3,
    s1_0,
    s1_1,
    s1_2,
    s1_3,
    s2_0,
    s2_1,
    s2_2,
    s2_3,
    s3_0,
    s3_1,
    s3_2,
    s3_3,
    vs0,
    vs1,
    vs2,
    vs3,
    M,
    IDX_LEN: tl.constexpr,
    IDX_RANK: tl.constexpr,
    B: tl.constexpr,
    ACC: tl.constexpr,
    SKIP_FIRST: tl.constexpr,
):
    """One lane per index position; every address is affine in the lane.

    With ``ACC`` the first grid entry is spent on a fully masked-off tile.  The
    FIRST entry to reach a 16-bit ``tl.atomic_add`` on this backend replays its
    whole tile onto its own addresses: with B = 1024 exactly the 1024 targets of
    program 0 came back over-added by ~16 extra copies of their own value (B =
    64 gave 64 such targets), while the identical tile one program later was
    exact.  A masked-off entry 0 absorbs the replay; every lane it could touch
    resolves to address 0 with value 0, so nothing observable changes.
    """
    pid0 = tl.program_id(axis=0)
    if SKIP_FIRST:
        pid0 = pid0 - 1
    o = pid0 * B + tl.arange(0, B)

    a0 = tl.zeros([B], tl.int32)
    a1 = tl.zeros([B], tl.int32)
    a2 = tl.zeros([B], tl.int32)
    a3 = tl.zeros([B], tl.int32)
    if IDX_RANK >= 1:
        r0 = o
        if IDX_RANK >= 4:
            a3 = r0 % id3
            r0 = r0 // id3
        if IDX_RANK >= 3:
            a2 = r0 % id2
            r0 = r0 // id2
        if IDX_RANK >= 2:
            a1 = r0 % id1
            r0 = r0 // id1
        a0 = r0 % id0

    voff = a0 * vs0
    if IDX_RANK >= 2:
        voff = voff + a1 * vs1
    if IDX_RANK >= 3:
        voff = voff + a2 * vs2
    if IDX_RANK >= 4:
        voff = voff + a3 * vs3

    m0 = o < M
    if SKIP_FIRST:
        m0 = m0 & (pid0 >= 0)

    o_i0 = a0 * s0_0
    if IDX_RANK >= 2:
        o_i0 = o_i0 + a1 * s0_1
    if IDX_RANK >= 3:
        o_i0 = o_i0 + a2 * s0_2
    if IDX_RANK >= 4:
        o_i0 = o_i0 + a3 * s0_3
    c0 = tl.load(i0_ptr + o_i0, mask=m0, other=0)
    ok = m0 & (c0 >= 0) & (c0 < is0)
    ioff = c0 * it0

    if IDX_LEN >= 2:
        o_i1 = a0 * s1_0
        if IDX_RANK >= 2:
            o_i1 = o_i1 + a1 * s1_1
        if IDX_RANK >= 3:
            o_i1 = o_i1 + a2 * s1_2
        if IDX_RANK >= 4:
            o_i1 = o_i1 + a3 * s1_3
        c1 = tl.load(i1_ptr + o_i1, mask=m0, other=0)
        ok = ok & (c1 >= 0) & (c1 < is1)
        ioff = ioff + c1 * it1

    if IDX_LEN >= 3:
        o_i2 = a0 * s2_0
        if IDX_RANK >= 2:
            o_i2 = o_i2 + a1 * s2_1
        if IDX_RANK >= 3:
            o_i2 = o_i2 + a2 * s2_2
        if IDX_RANK >= 4:
            o_i2 = o_i2 + a3 * s2_3
        c2 = tl.load(i2_ptr + o_i2, mask=m0, other=0)
        ok = ok & (c2 >= 0) & (c2 < is2)
        ioff = ioff + c2 * it2

    if IDX_LEN >= 4:
        o_i3 = a0 * s3_0
        if IDX_RANK >= 2:
            o_i3 = o_i3 + a1 * s3_1
        if IDX_RANK >= 3:
            o_i3 = o_i3 + a2 * s3_2
        if IDX_RANK >= 4:
            o_i3 = o_i3 + a3 * s3_3
        c3 = tl.load(i3_ptr + o_i3, mask=m0, other=0)
        ok = ok & (c3 >= 0) & (c3 < is3)
        ioff = ioff + c3 * it3

    v = tl.load(val_ptr + voff, mask=m0)
    if ACC:
        tl.atomic_add(inp_ptr + ioff, v, mask=ok)
    else:
        tl.store(inp_ptr + ioff, v, mask=ok)


@triton.jit
def _scatter_inner_kernel(
    inp_ptr,
    val_ptr,
    i0_ptr,
    inp_stride0,
    is0,
    id0,
    id1,
    id2,
    id3,
    s0_0,
    s0_1,
    s0_2,
    s0_3,
    vs0,
    vs1,
    vs2,
    vs3,
    out_shape1,
    out_shape2,
    out_stride1,
    out_stride2,
    vt1,
    vt2,
    vinner,
    M,
    IDX_RANK: tl.constexpr,
    B0: tl.constexpr,
    INNER: tl.constexpr,
    ACC: tl.constexpr,
):
    """``idx_len == 1`` over contiguous ``inp``: the last dim is unit-stride.

    Grid axis 0 walks index positions, axis 1 walks the outer trailing
    coordinates; those are decoded once per program (scalar), so the tile
    address stays affine in the lane and the inner axis is a bare
    ``tl.arange(0, INNER)`` run.
    """
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    i2 = pid1 % out_shape2
    i1 = (pid1 // out_shape2) % out_shape1
    oout = i1 * out_stride1 + i2 * out_stride2
    vout = i1 * vt1 + i2 * vt2

    o0 = pid0 * B0 + tl.arange(0, B0)[:, None]
    j = tl.arange(0, INNER)[None, :]

    a0 = tl.zeros([B0, 1], tl.int32)
    a1 = tl.zeros([B0, 1], tl.int32)
    a2 = tl.zeros([B0, 1], tl.int32)
    a3 = tl.zeros([B0, 1], tl.int32)
    if IDX_RANK >= 1:
        r0 = o0
        if IDX_RANK >= 4:
            a3 = r0 % id3
            r0 = r0 // id3
        if IDX_RANK >= 3:
            a2 = r0 % id2
            r0 = r0 // id2
        if IDX_RANK >= 2:
            a1 = r0 % id1
            r0 = r0 // id1
        a0 = r0 % id0

    rowoff = a0 * vs0
    if IDX_RANK >= 2:
        rowoff = rowoff + a1 * vs1
    if IDX_RANK >= 3:
        rowoff = rowoff + a2 * vs2
    if IDX_RANK >= 4:
        rowoff = rowoff + a3 * vs3

    o_i0 = a0 * s0_0
    if IDX_RANK >= 2:
        o_i0 = o_i0 + a1 * s0_1
    if IDX_RANK >= 3:
        o_i0 = o_i0 + a2 * s0_2
    if IDX_RANK >= 4:
        o_i0 = o_i0 + a3 * s0_3

    m0 = o0 < M
    c0 = tl.load(i0_ptr + o_i0, mask=m0, other=0)
    mask = m0 & (c0 >= 0) & (c0 < is0)

    dst = c0 * inp_stride0 + oout + j
    src = rowoff + vout + j * vinner
    v = tl.load(val_ptr + src, mask=mask)
    if ACC:
        tl.atomic_add(inp_ptr + dst, v, mask=mask)
    else:
        tl.store(inp_ptr + dst, v, mask=mask)


@triton.jit
def _scatter_run_kernel(
    inp_ptr,
    val_ptr,
    i0_ptr,
    inp_stride0,
    is0,
    id0,
    id1,
    id2,
    id3,
    s0_0,
    s0_1,
    s0_2,
    s0_3,
    vs0,
    vs1,
    vs2,
    vs3,
    out_shape1,
    out_shape2,
    out_stride1,
    out_stride2,
    vt1,
    vt2,
    vinner,
    IDX_RANK: tl.constexpr,
    INNER: tl.constexpr,
    CHUNK: tl.constexpr,
    VT_FLAT: tl.constexpr,
):
    """``idx_len == 1`` store path: one index position per program, scalar row base.

    The whole tile address of the previous formulation varied its row base inside
    one program (the gathered ``c0 * inp_stride0`` multiplied a per-lane row), so
    the backend scalarised the tile into nested per-element stores.  Here the
    index coordinate is decoded from ``program_id`` alone -- a SCALAR -- so each
    of the ``CHUNK`` stores is a unit-stride affine run ``base + tl.arange(0, INNER)``
    that lowers to wide vector movement.  ``CHUNK`` is a power of two dividing the
    trailing volume so no masking is needed on the write side.
    """
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    a3 = 0
    a2 = 0
    a1 = 0
    p = pid0
    if IDX_RANK >= 4:
        a3 = p % id3
        p = p // id3
    if IDX_RANK >= 3:
        a2 = p % id2
        p = p // id2
    if IDX_RANK >= 2:
        a1 = p % id1
        p = p // id1
    a0 = p % id0

    rowoff = a0 * vs0
    if IDX_RANK >= 2:
        rowoff = rowoff + a1 * vs1
    if IDX_RANK >= 3:
        rowoff = rowoff + a2 * vs2
    if IDX_RANK >= 4:
        rowoff = rowoff + a3 * vs3

    ioff = a0 * s0_0
    if IDX_RANK >= 2:
        ioff = ioff + a1 * s0_1
    if IDX_RANK >= 3:
        ioff = ioff + a2 * s0_2
    if IDX_RANK >= 4:
        ioff = ioff + a3 * s0_3

    c0 = tl.load(i0_ptr + ioff)
    base = c0 * inp_stride0

    j = tl.arange(0, INNER)
    qbase = pid1 * CHUNK
    if (c0 >= 0) & (c0 < is0):
        if VT_FLAT:
            # ``vt1 == vt2 == 0`` means the value row does not depend on the
            # trailing coordinates at all: the CHUNK stores all read the SAME
            # INNER-element row.  Loading it once therefore removes CHUNK - 1
            # full GM reads per program (profiled: MTE2 was saturated at 0.9982
            # with ~16x redundant load volume on CHUNK = 16).
            vrow = tl.load(val_ptr + rowoff + j * vinner)
            for r in tl.static_range(CHUNK):
                q = qbase + r
                i2 = q % out_shape2
                i1 = (q // out_shape2) % out_shape1
                dst = base + (i1 * out_stride1 + i2 * out_stride2) + j
                tl.store(inp_ptr + dst, vrow)
        else:
            for r in tl.static_range(CHUNK):
                q = qbase + r
                i2 = q % out_shape2
                i1 = (q // out_shape2) % out_shape1
                dst = base + (i1 * out_stride1 + i2 * out_stride2) + j
                src = rowoff + (i1 * vt1 + i2 * vt2) + j * vinner
                v = tl.load(val_ptr + src)
                tl.store(inp_ptr + dst, v)


@triton.jit
def _index_put_impl_kernel(
    inp_ptr,
    val_ptr,
    i0_ptr,
    i1_ptr,
    i2_ptr,
    i3_ptr,
    is0,
    is1,
    is2,
    is3,
    it0,
    it1,
    it2,
    it3,
    id0,
    id1,
    id2,
    id3,
    s0_0,
    s0_1,
    s0_2,
    s0_3,
    s1_0,
    s1_1,
    s1_2,
    s1_3,
    s2_0,
    s2_1,
    s2_2,
    s2_3,
    s3_0,
    s3_1,
    s3_2,
    s3_3,
    vs0,
    vs1,
    vs2,
    vs3,
    vt0,
    vt1,
    vt2,
    vt3,
    M,
    N,
    IN_RANK: tl.constexpr,
    IDX_LEN: tl.constexpr,
    IDX_RANK: tl.constexpr,
    ACC: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    o0 = pid0 * B0 + tl.arange(0, B0)[:, None]
    o1 = pid1 * B1 + tl.arange(0, B1)[None, :]

    # --- decode the flattened index position into index-tensor coordinates ---
    z0 = tl.zeros([B0, 1], tl.int32)
    a0 = z0
    a1 = z0
    a2 = z0
    a3 = z0
    if IDX_RANK >= 1:
        r0 = o0
        if IDX_RANK >= 4:
            a3 = r0 % id3
            r0 = r0 // id3
        if IDX_RANK >= 3:
            a2 = r0 % id2
            r0 = r0 // id2
        if IDX_RANK >= 2:
            a1 = r0 % id1
            r0 = r0 // id1
        a0 = r0 % id0

    # --- decode the flattened trailing (slice) position ---
    z1 = tl.zeros([1, B1], tl.int32)
    b1 = z1
    b2 = z1
    b3 = z1
    if IN_RANK > 1:
        r1 = o1
        if IN_RANK > 3:
            if IDX_LEN <= 3:
                b3 = r1 % is3
                r1 = r1 // is3
        if IN_RANK > 2:
            if IDX_LEN <= 2:
                b2 = r1 % is2
                r1 = r1 // is2
        if IDX_LEN <= 1:
            b1 = r1 % is1
            r1 = r1 // is1

    m0 = o0 < M

    # --- gather the index coordinates for the leading dimensions ---
    zc = tl.zeros([B0, 1], tl.int32)
    c0 = zc
    c1 = zc
    c2 = zc
    c3 = zc

    o_i0 = a0 * s0_0
    if IDX_RANK >= 2:
        o_i0 = o_i0 + a1 * s0_1
    if IDX_RANK >= 3:
        o_i0 = o_i0 + a2 * s0_2
    if IDX_RANK >= 4:
        o_i0 = o_i0 + a3 * s0_3
    c0 = tl.load(i0_ptr + o_i0, mask=m0, other=0)
    im = (c0 >= 0) & (c0 < is0)

    if IDX_LEN >= 2:
        o_i1 = a0 * s1_0
        if IDX_RANK >= 2:
            o_i1 = o_i1 + a1 * s1_1
        if IDX_RANK >= 3:
            o_i1 = o_i1 + a2 * s1_2
        if IDX_RANK >= 4:
            o_i1 = o_i1 + a3 * s1_3
        c1 = tl.load(i1_ptr + o_i1, mask=m0, other=0)
        im = im & (c1 >= 0) & (c1 < is1)

    if IDX_LEN >= 3:
        o_i2 = a0 * s2_0
        if IDX_RANK >= 2:
            o_i2 = o_i2 + a1 * s2_1
        if IDX_RANK >= 3:
            o_i2 = o_i2 + a2 * s2_2
        if IDX_RANK >= 4:
            o_i2 = o_i2 + a3 * s2_3
        c2 = tl.load(i2_ptr + o_i2, mask=m0, other=0)
        im = im & (c2 >= 0) & (c2 < is2)

    if IDX_LEN >= 4:
        o_i3 = a0 * s3_0
        if IDX_RANK >= 2:
            o_i3 = o_i3 + a1 * s3_1
        if IDX_RANK >= 3:
            o_i3 = o_i3 + a2 * s3_2
        if IDX_RANK >= 4:
            o_i3 = o_i3 + a3 * s3_3
        c3 = tl.load(i3_ptr + o_i3, mask=m0, other=0)
        im = im & (c3 >= 0) & (c3 < is3)

    m1 = o1 < N
    m = im & m0 & m1

    # --- destination offset inside ``inp`` (in elements) ---
    ioff = c0 * it0
    if IDX_LEN >= 2:
        ioff = ioff + c1 * it1
    if IDX_LEN >= 3:
        ioff = ioff + c2 * it2
    if IDX_LEN >= 4:
        ioff = ioff + c3 * it3
    if IDX_LEN <= 1:
        if IN_RANK >= 2:
            ioff = ioff + b1 * it1
    if IDX_LEN <= 2:
        if IN_RANK >= 3:
            ioff = ioff + b2 * it2
    if IDX_LEN <= 3:
        if IN_RANK >= 4:
            ioff = ioff + b3 * it3

    # --- source offset inside ``values`` (in elements, broadcast-aware) ---
    voff = a0 * vs0
    if IDX_RANK >= 2:
        voff = voff + a1 * vs1
    if IDX_RANK >= 3:
        voff = voff + a2 * vs2
    if IDX_RANK >= 4:
        voff = voff + a3 * vs3
    if IDX_LEN <= 1:
        if IN_RANK >= 2:
            voff = voff + b1 * vt1
    if IDX_LEN <= 2:
        if IN_RANK >= 3:
            voff = voff + b2 * vt2
    if IDX_LEN <= 3:
        if IN_RANK >= 4:
            voff = voff + b3 * vt3

    cur_value = tl.load(val_ptr + voff, mask=m)
    if ACC:
        tl.atomic_add(inp_ptr + ioff, cur_value, mask=m)
    else:
        tl.store(inp_ptr + ioff, cur_value, mask=m)


def _max_rank_shape(indices):
    tensor_indices = [idx for idx in indices if idx is not None]
    if len(tensor_indices) == 0:
        return []
    max_rank = 0
    for index in tensor_indices:
        if len(index.shape) > max_rank:
            max_rank = len(index.shape)
    shape = [0 for _ in range(max_rank)]
    for i in range(max_rank):
        max_num = 0
        for index in tensor_indices:
            axis = len(index.shape) - 1 - i
            if axis >= 0 and index.shape[axis] > max_num:
                max_num = index.shape[axis]
        shape[max_rank - 1 - i] = max_num
    return shape


def _broadcast_indices(indices, target_shape):
    target = tuple(target_shape)
    for i in range(len(indices)):
        index = indices[i]
        if index is not None and tuple(index.shape) != target:
            indices[i] = torch.broadcast_to(index, target)


def _pick_chunk(n_outer, inner):
    """Largest power of two that divides ``n_outer`` within the run budget.

    The kernel walks ``CHUNK`` trailing coordinates per index position, so an
    exact divisor keeps the write runs unmasked, and ``CHUNK * inner`` bounds the
    live value/address vectors of one program.
    """
    limit = max(1, _RUN_BLOCK // inner)
    chunk = 1
    while chunk * 2 <= limit and (n_outer % (chunk * 2)) == 0:
        chunk *= 2
    return chunk


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _pad(seq, n):
    out = [int(x) for x in list(seq)[:n]]
    while len(out) < n:
        out.append(0)
    return out


_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    logger.debug("GEMS _INDEX_PUT_IMPL_")
    if inp.dtype not in _SUPPORTED_DTYPES:
        from flag_gems.ops.index_put import _index_put_impl_ as _generic

        return _generic(inp, indices, values, accumulate, unsafe)

    idx_list = list(indices)

    if len(idx_list) == 1:
        first = idx_list[0]
        if first is not None and first.dtype == torch.bool:
            mask = first
            if mask.device != inp.device:
                mask = mask.to(inp.device)
            idx_list = list(torch.where(mask))
            k = idx_list[0].numel()
            target_shape = (k,) + inp.shape[len(idx_list) :]
            if values.numel() == 1:
                values = torch.full(
                    target_shape, values.item(), dtype=inp.dtype, device=inp.device
                )
            elif values.numel() == k:
                values = values.reshape((k,)).expand(target_shape)

    idx_list = [
        (
            index.to(inp.device)
            if index is not None and index.device != inp.device
            else index
        )
        for index in idx_list
    ]

    target_shape = _max_rank_shape(idx_list)
    _broadcast_indices(idx_list, target_shape)
    target_shape = target_shape + list(inp.shape[len(idx_list) :])

    tensor_indices = [index for index in idx_list if index is not None]
    if not tensor_indices:
        raise ValueError("At least one non-None index tensor is required")

    if values.device != inp.device:
        values = values.to(inp.device)
    values = torch.broadcast_to(values, target_shape)

    in_rank = inp.ndim
    idx_len = len(tensor_indices)
    idx_rank = tensor_indices[0].ndim
    if in_rank > _MAX_RANK or idx_rank > _MAX_RANK:
        raise NotImplementedError("index_put_impl: rank above 4 is not supported")

    m_total = int(tensor_indices[0].numel())
    n_total = 1
    for s in inp.shape[idx_len:]:
        n_total *= int(s)
    if m_total <= 0 or n_total <= 0:
        return inp

    acc = bool(accumulate)
    val_stride = list(values.stride())
    vs = _pad(values.stride(), _MAX_RANK)
    idx_shape = _pad(tensor_indices[0].shape, _MAX_RANK)
    in_shape = _pad(inp.shape, _MAX_RANK)
    in_stride = _pad(inp.stride(), _MAX_RANK)

    idx_stride = []
    for i in range(_MAX_RANK):
        if i < idx_len:
            idx_stride.extend(_pad(tensor_indices[i].stride(), _MAX_RANK))
        else:
            idx_stride.extend([0] * _MAX_RANK)

    idx_args = []
    for i in range(_MAX_RANK):
        idx_args.append(tensor_indices[i] if i < idx_len else inp)

    # --- fast path 1: the index tensors already cover every dimension ---
    if n_total == 1:
        # The first-entry replay that a 16-bit ``tl.atomic_add`` showed on this
        # backend is absorbed by the kernel's own masked no-op entry 0 (see
        # ``_scatter_flat_kernel``), so the accumulate tile does not have to be
        # small for correctness -- but it does have to be small for occupancy.
        # A wide tile covers every position in one program, and then a single
        # vector core performs the whole atomic scatter (profiled: one block,
        # scalar ~0.51 / store ~0.55, neither pipe saturated), which is why the
        # small and mid accumulate workloads cost the same regardless of dtype
        # or shape.  Spread them over roughly 16 blocks; the large flat case
        # keeps the wide tile because per-program cost dominates there.
        if acc:
            b = min(_FLAT_BLOCK, max(16, _next_pow2(-(-m_total // 16))))
        else:
            b = min(_next_pow2(m_total), _FLAT_BLOCK)
        _scatter_flat_kernel[(-(-m_total // b) + (1 if acc else 0),)](
            inp,
            values,
            idx_args[0],
            idx_args[1],
            idx_args[2],
            idx_args[3],
            in_shape[0],
            in_shape[1],
            in_shape[2],
            in_shape[3],
            in_stride[0],
            in_stride[1],
            in_stride[2],
            in_stride[3],
            idx_shape[0],
            idx_shape[1],
            idx_shape[2],
            idx_shape[3],
            idx_stride[0],
            idx_stride[1],
            idx_stride[2],
            idx_stride[3],
            idx_stride[4],
            idx_stride[5],
            idx_stride[6],
            idx_stride[7],
            idx_stride[8],
            idx_stride[9],
            idx_stride[10],
            idx_stride[11],
            idx_stride[12],
            idx_stride[13],
            idx_stride[14],
            idx_stride[15],
            vs[0],
            vs[1],
            vs[2],
            vs[3],
            m_total,
            IDX_LEN=idx_len,
            IDX_RANK=idx_rank,
            B=b,
            ACC=acc,
            SKIP_FIRST=acc,
        )
        return inp

    # --- fast path 2: one index tensor over a contiguous ``inp`` ---
    inner = int(inp.shape[in_rank - 1])
    if (
        idx_len == 1
        and in_rank >= 2
        and bool(inp.is_contiguous())
        and 16 <= inner <= _MAX_INNER
        and (inner & (inner - 1)) == 0
    ):
        n_outer = n_total // inner
        out_shape1 = int(inp.shape[1]) if in_rank >= 3 else 1
        out_shape2 = int(inp.shape[2]) if in_rank >= 4 else 1
        out_stride1 = in_stride[1] if in_rank >= 3 else 0
        out_stride2 = in_stride[2] if in_rank >= 4 else 0
        vt1 = int(val_stride[idx_rank]) if in_rank >= 3 else 0
        vt2 = int(val_stride[idx_rank + 1]) if in_rank >= 4 else 0
        vinner = int(val_stride[idx_rank + in_rank - 2])
        b0 = 2 if acc else max(1, min(16, _MAX_INNER // inner))
        # ``vt1 == vt2 == 0`` selects the form where the value row is loaded once
        # and stored CHUNK times, instead of one full-row load per store.
        vt_flat = (vt1 == 0) and (vt2 == 0)
        chunk = _pick_chunk(n_outer, inner)
        n_run = -(-n_outer // chunk)
        if not acc and m_total <= 65535 and n_run <= 65535:
            _scatter_run_kernel[(m_total, n_run)](
                inp,
                values,
                tensor_indices[0],
                in_stride[0],
                in_shape[0],
                idx_shape[0],
                idx_shape[1],
                idx_shape[2],
                idx_shape[3],
                idx_stride[0],
                idx_stride[1],
                idx_stride[2],
                idx_stride[3],
                vs[0],
                vs[1],
                vs[2],
                vs[3],
                out_shape1,
                out_shape2,
                out_stride1,
                out_stride2,
                vt1,
                vt2,
                vinner,
                IDX_RANK=idx_rank,
                INNER=inner,
                CHUNK=chunk,
                VT_FLAT=vt_flat,
            )
            return inp
        _scatter_inner_kernel[(max(1, -(-m_total // b0)), max(1, n_outer))](
            inp,
            values,
            tensor_indices[0],
            in_stride[0],
            in_shape[0],
            idx_shape[0],
            idx_shape[1],
            idx_shape[2],
            idx_shape[3],
            idx_stride[0],
            idx_stride[1],
            idx_stride[2],
            idx_stride[3],
            vs[0],
            vs[1],
            vs[2],
            vs[3],
            out_shape1,
            out_shape2,
            out_stride1,
            out_stride2,
            vt1,
            vt2,
            vinner,
            m_total,
            IDX_RANK=idx_rank,
            B0=b0,
            INNER=inner,
            ACC=acc,
        )
        return inp

    # --- general path ---
    if n_total == 1:
        b1 = 1
        b0 = 1024 if m_total >= 1024 else _next_pow2(m_total)
    else:
        b1 = 1024 if n_total >= 1024 else _next_pow2(n_total)
        b0 = 4096 // b1
        if b0 < 1:
            b0 = 1
        if b0 > 8:
            b0 = 8

    vt = [0] * _MAX_RANK
    k = idx_len
    while k < in_rank:
        vt[k] = int(val_stride[idx_rank + (k - idx_len)])
        k += 1

    grid = (-(-m_total // b0), -(-n_total // b1))

    _index_put_impl_kernel[grid](
        inp,
        values,
        idx_args[0],
        idx_args[1],
        idx_args[2],
        idx_args[3],
        in_shape[0],
        in_shape[1],
        in_shape[2],
        in_shape[3],
        in_stride[0],
        in_stride[1],
        in_stride[2],
        in_stride[3],
        idx_shape[0],
        idx_shape[1],
        idx_shape[2],
        idx_shape[3],
        idx_stride[0],
        idx_stride[1],
        idx_stride[2],
        idx_stride[3],
        idx_stride[4],
        idx_stride[5],
        idx_stride[6],
        idx_stride[7],
        idx_stride[8],
        idx_stride[9],
        idx_stride[10],
        idx_stride[11],
        idx_stride[12],
        idx_stride[13],
        idx_stride[14],
        idx_stride[15],
        vs[0],
        vs[1],
        vs[2],
        vs[3],
        vt[0],
        vt[1],
        vt[2],
        vt[3],
        m_total,
        n_total,
        IN_RANK=in_rank,
        IDX_LEN=idx_len,
        IDX_RANK=idx_rank,
        ACC=acc,
        B0=b0,
        B1=b1,
    )
    return inp

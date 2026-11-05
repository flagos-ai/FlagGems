import torch
import triton
import triton.language as tl

MAX_BATCH = 8
BLOCK_CLONE = 1024
BLOCK_SCATTER = 256
BLOCK_FUSED = 1024


@triton.jit
def _clone_kernel(in_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    v = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _scatter_kernel(
    src_ptr,
    out_ptr,
    n_diag,
    diag_len,
    offset,
    st1,
    st2,
    src_st_t,
    bp0,
    sz0,
    out_st0,
    src_st0,
    bp1,
    sz1,
    out_st1,
    src_st1,
    bp2,
    sz2,
    out_st2,
    src_st2,
    bp3,
    sz3,
    out_st3,
    src_st3,
    bp4,
    sz4,
    out_st4,
    src_st4,
    bp5,
    sz5,
    out_st5,
    src_st5,
    bp6,
    sz6,
    out_st6,
    src_st6,
    bp7,
    sz7,
    out_st7,
    src_st7,
    N_BATCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_diag
    b_idx = offs // diag_len
    t = offs % diag_len

    base_out = tl.zeros([BLOCK], dtype=tl.int64)
    base_src = tl.zeros([BLOCK], dtype=tl.int64)
    if N_BATCH >= 1:
        c = (b_idx // bp0) % sz0
        base_out += c * out_st0
        base_src += c * src_st0
    if N_BATCH >= 2:
        c = (b_idx // bp1) % sz1
        base_out += c * out_st1
        base_src += c * src_st1
    if N_BATCH >= 3:
        c = (b_idx // bp2) % sz2
        base_out += c * out_st2
        base_src += c * src_st2
    if N_BATCH >= 4:
        c = (b_idx // bp3) % sz3
        base_out += c * out_st3
        base_src += c * src_st3
    if N_BATCH >= 5:
        c = (b_idx // bp4) % sz4
        base_out += c * out_st4
        base_src += c * src_st4
    if N_BATCH >= 6:
        c = (b_idx // bp5) % sz5
        base_out += c * out_st5
        base_src += c * src_st5
    if N_BATCH >= 7:
        c = (b_idx // bp6) % sz6
        base_out += c * out_st6
        base_src += c * src_st6
    if N_BATCH >= 8:
        c = (b_idx // bp7) % sz7
        base_out += c * out_st7
        base_src += c * src_st7

    row = t + tl.where(offset >= 0, 0, -offset)
    col = t + tl.where(offset >= 0, offset, 0)
    out_flat = base_out + row * st1 + col * st2
    src_flat = base_src + t * src_st_t
    v = tl.load(src_ptr + src_flat, mask=mask)
    v = v.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + out_flat, v, mask=mask)


@triton.jit
def _fused_kernel(
    in_ptr,
    src_ptr,
    out_ptr,
    numel,
    log2_s1,
    log2_s2,
    s1_m1,
    s2_m1,
    s2,
    offset,
    diag_len,
    SRC_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fast path: input contiguous, diagonal over the last two dims,
    # both sizes powers of two. Index decode uses shifts/masks only.
    pid = tl.program_id(0)
    x = pid * BLOCK + tl.arange(0, BLOCK)
    mask = x < numel

    v = tl.load(in_ptr + x, mask=mask)

    q = x >> log2_s2
    c2 = x & s2_m1
    c1 = q & s1_m1
    b = q >> log2_s1
    on_diag = (c2 - c1) == offset

    if SRC_MODE == 0:
        # src is the diagonal view of input: element t sits at x + shift
        src_flat = x + tl.where(offset >= 0, -offset, offset * s2)
    else:
        # src is contiguous with shape (batch..., diag_len)
        t = tl.where(offset >= 0, c1, c2)
        src_flat = b * diag_len + t

    sv = tl.load(src_ptr + src_flat, mask=mask & on_diag)
    v = tl.where(on_diag, sv, v)
    tl.store(out_ptr + x, v, mask=mask)


def _try_fused(input, src, out, offset, d1, d2, ndim, sizes, diag_len):
    if ndim < 2 or d1 != ndim - 2 or d2 != ndim - 1:
        return False
    numel = input.numel()
    if numel >= (1 << 31):
        return False
    s1 = sizes[d1]
    s2 = sizes[d2]
    if s1 == 0 or s2 == 0:
        return False
    if (s1 & (s1 - 1)) != 0 or (s2 & (s2 - 1)) != 0:
        return False
    # input must be fully contiguous so flat index == storage offset
    cs = [1] * ndim
    p = 1
    for d in range(ndim - 1, -1, -1):
        cs[d] = p
        p *= sizes[d]
    if tuple(input.stride()) != tuple(cs):
        return False

    batch_dims = [d for d in range(ndim) if d != d1 and d != d2]
    diag_shape = tuple(sizes[d] for d in batch_dims) + (diag_len,)
    if tuple(src.shape) != diag_shape:
        return False
    src_strides = tuple(src.stride())
    src_cs = [1] * src.ndim
    p = 1
    for j in range(src.ndim - 1, -1, -1):
        src_cs[j] = p
        p *= src.shape[j]
    if src_strides == tuple(src_cs):
        mode = 1  # contiguous src
    else:
        # diagonal-view src: strides (batch strides..., s2+1) and storage aligned
        expected = tuple(input.stride(d) for d in batch_dims) + (s2 + 1,)
        if src_strides != expected:
            return False
        shift = offset if offset >= 0 else -offset * s2
        if src.storage_offset() != input.storage_offset() + shift:
            return False
        mode = 0

    log2_s1 = s1.bit_length() - 1
    log2_s2 = s2.bit_length() - 1
    _fused_kernel[(triton.cdiv(numel, BLOCK_FUSED),)](
        input,
        src,
        out,
        numel,
        log2_s1,
        log2_s2,
        s1 - 1,
        s2 - 1,
        s2,
        offset,
        diag_len,
        SRC_MODE=mode,
        BLOCK=BLOCK_FUSED,
        num_warps=4,
    )
    return True


def run(input, src, offset=0, dim1=0, dim2=1):
    ndim = input.dim()
    d1 = dim1 % ndim
    d2 = dim2 % ndim
    if d1 == d2:
        raise RuntimeError("diagonal dimensions cannot be identical")
    out = torch.empty_like(input)

    sizes = input.shape
    s1, s2 = sizes[d1], sizes[d2]
    if offset >= 0:
        diag_len = max(0, min(s1, s2 - offset))
    else:
        diag_len = max(0, min(s1 + offset, s2))

    if _try_fused(input, src, out, offset, d1, d2, ndim, sizes, diag_len):
        return out

    # ---- general fallback: full clone + diagonal scatter ----
    numel = input.numel()
    _clone_kernel[(triton.cdiv(numel, BLOCK_CLONE),)](
        input, out, numel, BLOCK=BLOCK_CLONE, num_warps=4
    )

    if diag_len > 0:
        cs = [1] * ndim
        p = 1
        for d in range(ndim - 1, -1, -1):
            cs[d] = p
            p *= sizes[d]
        st1, st2 = cs[d1], cs[d2]

        batch_dims = [d for d in range(ndim) if d != d1 and d != d2]
        m = len(batch_dims)
        assert m <= MAX_BATCH
        bsz = [sizes[d] for d in batch_dims]
        bp = [1] * m
        acc = 1
        for j in range(m - 1, -1, -1):
            bp[j] = acc
            acc *= bsz[j]
        batch_numel = acc
        src_strides = src.stride()
        src_st_t = src_strides[-1]

        args = []
        for j in range(MAX_BATCH):
            if j < m:
                args += [bp[j], bsz[j], cs[batch_dims[j]], src_strides[j]]
            else:
                args += [1, 1, 0, 0]

        n_diag = batch_numel * diag_len
        _scatter_kernel[(triton.cdiv(n_diag, BLOCK_SCATTER),)](
            src,
            out,
            n_diag,
            diag_len,
            offset,
            st1,
            st2,
            src_st_t,
            *args,
            N_BATCH=m,
            BLOCK=BLOCK_SCATTER,
            num_warps=4,
        )
    return out


# Alias for FlagGems import convention
diagonal_scatter = run

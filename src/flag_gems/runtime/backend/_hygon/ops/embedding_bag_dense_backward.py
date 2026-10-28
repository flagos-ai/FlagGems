import torch
import triton
import triton.language as tl


@triton.jit
def _emb_bag_dense_bwd_flat_kernel(
    grad_ptr,
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    per_sample_ptr,
    out_ptr,
    N,
    D,
    padding_idx,
    MEAN: tl.constexpr,
    SCALE_FREQ: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    PROMOTE: tl.constexpr,
    D_IS_POW2: tl.constexpr,
    D_SHIFT: tl.constexpr,
    D_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * D
    m = offs < total

    if D_IS_POW2:
        n = offs >> D_SHIFT
        d = offs & D_MASK
    else:
        n = offs // D
        d = offs % D

    idx = tl.load(indices_ptr + n, mask=m, other=padding_idx)
    valid = m & (idx != padding_idx)
    bag = tl.load(offset2bag_ptr + n, mask=m, other=0)

    g = tl.load(grad_ptr + bag * D + d, mask=valid, other=0.0)
    if PROMOTE:
        g = g.to(tl.float32)

    if MEAN or SCALE_FREQ:
        bs = tl.load(bag_size_ptr + bag, mask=m, other=1)
        if PROMOTE:
            bsf = bs.to(tl.float32)
        else:
            bsf = bs.to(g.dtype)
        if MEAN:
            g = g / bsf
        if SCALE_FREQ:
            g = g / bsf

    if HAS_WEIGHT:
        pw = tl.load(per_sample_ptr + n, mask=m, other=0.0)
        if PROMOTE:
            pw = pw.to(tl.float32)
        g = g * pw

    tl.atomic_add(out_ptr + idx * D + d, g, mask=valid, sem="relaxed")


@triton.jit
def _cast_kernel(src_ptr, dst_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_elements
    v = tl.load(src_ptr + offs, mask=m)
    tl.store(dst_ptr + offs, v.to(dst_ptr.dtype.element_ty), mask=m)


def run(
    grad,
    indices,
    offset2bag,
    bag_size,
    maximum_indices,
    num_weights,
    scale_grad_by_freq,
    mode,
    per_sample_weights=None,
    padding_idx=-1,
):
    if mode == 2:
        raise RuntimeError(
            "embedding_bag_dense_backward: can only use dense backward with sum or mean mode"
        )

    N = indices.shape[0]
    D = grad.shape[-1]
    dtype = grad.dtype
    device = grad.device
    is_lowp = dtype in (torch.float16, torch.bfloat16)

    if N == 0:
        return torch.zeros((num_weights, D), dtype=dtype, device=device)

    acc = torch.zeros(
        (num_weights, D), dtype=torch.float32 if is_lowp else dtype, device=device
    )
    if is_lowp:
        out = torch.empty((num_weights, D), dtype=dtype, device=device)
    else:
        out = acc

    d_pow2 = (D & (D - 1)) == 0
    d_shift = (D.bit_length() - 1) if d_pow2 else 0
    d_mask = (D - 1) if d_pow2 else 0
    BLOCK = 512
    grid = (triton.cdiv(N * D, BLOCK),)
    dummy = per_sample_weights if per_sample_weights is not None else acc

    _emb_bag_dense_bwd_flat_kernel[grid](
        grad,
        indices,
        offset2bag,
        bag_size,
        dummy,
        acc,
        N,
        D,
        padding_idx,
        mode == 1,
        scale_grad_by_freq,
        per_sample_weights is not None,
        is_lowp,
        d_pow2,
        d_shift,
        d_mask,
        BLOCK=BLOCK,
        num_warps=4,
    )

    if is_lowp:
        n_el = num_weights * D
        _cast_kernel[(triton.cdiv(n_el, 2048),)](
            acc,
            out,
            n_el,
            BLOCK=2048,
        )

    return out


# Alias for FlagGems import convention
embedding_bag_dense_backward = run

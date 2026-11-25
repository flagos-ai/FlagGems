import torch
import triton
import triton.language as tl

MAX_DEG = 10  # V_n^* computed for n in [0, MAX_DEG); out-of-range -> 0.0


@triton.jit
def _shifted_cheb_v_kernel(
    x_ptr,
    n_ptr,
    out_ptr,
    n_elements,
    n_scalar,
    BLOCK: tl.constexpr,
    MAX_DEG: tl.constexpr,
    SCALAR_N: tl.constexpr,
    BROADCAST_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        if SCALAR_N:
            nf = tl.full([BLOCK], n_scalar, tl.int32)
        elif BROADCAST_N:
            nf = tl.load(n_ptr + tl.zeros([BLOCK], tl.int32)).to(tl.int32)
        else:
            nf = tl.load(n_ptr + offs).to(tl.int32)
        # x is only needed for elements whose truncated degree is >= 1
        # (V_0 = 1 and negative-n results do not depend on x).
        xv = tl.load(x_ptr + offs, mask=nf >= 1, other=0.0)
    else:
        mask = offs < n_elements
        if SCALAR_N:
            nf = tl.full([BLOCK], n_scalar, tl.int32)
        elif BROADCAST_N:
            nf = tl.load(n_ptr + tl.zeros([BLOCK], tl.int32), mask=mask, other=0).to(
                tl.int32
            )
        else:
            nf = tl.load(n_ptr + offs, mask=mask, other=0).to(tl.int32)
        xv = tl.load(x_ptr + offs, mask=mask & (nf >= 1), other=0.0)

    xf = xv.to(tl.float32)

    # Shifted Chebyshev polynomial of the third kind V_n^*(x):
    #   V_0^*(x) = 1
    #   V_1^*(x) = 4x - 3
    #   V_k^*(x) = (4x - 2) * V_{k-1}^*(x) - V_{k-2}^*(x)
    c = 4.0 * xf - 2.0
    p0 = tl.full([BLOCK], 1.0, tl.float32)
    p1 = 4.0 * xf - 3.0
    res = tl.where(nf == 0, p0, tl.where(nf == 1, p1, 0.0))
    # Runtime-limited recurrence: only iterate up to this block's max degree
    # (capped at MAX_DEG-1 so out-of-range n still yields 0).
    n_max = tl.max(nf, axis=0)
    ub = tl.minimum(n_max, MAX_DEG - 1)
    for k in tl.range(2, ub + 1, loop_unroll_factor=2):
        p2 = c * p1 - p0
        res = tl.where(nf == k, p2, res)
        p0 = p1
        p1 = p2

    if EVEN:
        tl.store(out_ptr + offs, res.to(xv.dtype))
    else:
        tl.store(out_ptr + offs, res.to(xv.dtype), mask=mask)


def run(x, n):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK = 512
    grid = (triton.cdiv(n_elements, BLOCK),)
    EVEN = (n_elements % BLOCK == 0) and (n_elements > 0)

    if isinstance(n, (int,)) and not isinstance(n, bool):
        _shifted_cheb_v_kernel[grid](
            x,
            x,
            out,
            n_elements,
            n,
            BLOCK=BLOCK,
            MAX_DEG=MAX_DEG,
            SCALAR_N=True,
            BROADCAST_N=False,
            EVEN=EVEN,
            num_warps=4,
        )
    elif torch.is_tensor(n) and n.numel() == 1:
        _shifted_cheb_v_kernel[grid](
            x,
            n,
            out,
            n_elements,
            0,
            BLOCK=BLOCK,
            MAX_DEG=MAX_DEG,
            SCALAR_N=False,
            BROADCAST_N=True,
            EVEN=EVEN,
            num_warps=4,
        )
    else:
        _shifted_cheb_v_kernel[grid](
            x,
            n,
            out,
            n_elements,
            0,
            BLOCK=BLOCK,
            MAX_DEG=MAX_DEG,
            SCALAR_N=False,
            BROADCAST_N=False,
            EVEN=EVEN,
            num_warps=4,
        )
    return out


# Alias for FlagGems import convention
special_shifted_chebyshev_polynomial_v = run

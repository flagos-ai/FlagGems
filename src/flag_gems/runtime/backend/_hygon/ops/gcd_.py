import torch
import triton
import triton.language as tl


@triton.jit
def _gcd_core(a, b):
    # a, b: uint32 tensors of magnitudes.  One swapped-operand Euclid step
    # (gcd(a,b) = gcd(a, b%a)) finishes the steady-state a|b case; the
    # data-dependent while loop is a rarely-entered fallback.  Done lanes
    # freeze at (result, 0) so the while-entry reduction sees all-zero b.
    a_safe = tl.where(a != 0, a, 1)
    t = b % a_safe
    done = (a == 0) | (t == 0)
    res = tl.where(a == 0, b, a)
    a2 = tl.where(done, res, a)
    b2 = tl.where(done, 0, t)
    while tl.max(b2) > 0:
        b2_safe = tl.where(b2 != 0, b2, 1)
        t2 = a2 % b2_safe
        na = tl.where(b2 != 0, b2, a2)
        nb = tl.where(b2 != 0, t2, b2)
        a2 = na
        b2 = nb
    return a2


@triton.jit
def _gcd_kernel(A, B, n_elements, BLOCK: tl.constexpr, PROMOTE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(A + offs, mask=mask, other=0)
    b = tl.load(B + offs, mask=mask, other=0)
    out_dtype = a.dtype
    if PROMOTE:
        a = a.to(tl.int32)
        b = b.to(tl.int32)
    a = tl.where(a < 0, -a, a)
    b = tl.where(b < 0, -b, b)
    a = a.to(tl.uint32)
    b = b.to(tl.uint32)
    a2 = _gcd_core(a, b)
    a2 = a2.to(out_dtype)
    tl.store(A + offs, a2, mask=mask)


@triton.jit
def _gcd_pair_kernel(A4, B4, n4, BLOCK: tl.constexpr):
    # A4/B4 are int32 views of even-numel int16 tensors: each 32-bit word
    # packs two little-endian int16 lanes.  Moving 32-bit words (not 16-bit
    # elements) restores full memory-pipeline efficiency for int16 streams.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n4
    x = tl.load(A4 + offs, mask=mask, other=0)
    y = tl.load(B4 + offs, mask=mask, other=0)
    a_lo = (x << 16) >> 16  # sign-extended low int16
    a_hi = x >> 16
    b_lo = (y << 16) >> 16
    b_hi = y >> 16
    a_lo = tl.where(a_lo < 0, -a_lo, a_lo).to(tl.uint32)
    a_hi = tl.where(a_hi < 0, -a_hi, a_hi).to(tl.uint32)
    b_lo = tl.where(b_lo < 0, -b_lo, b_lo).to(tl.uint32)
    b_hi = tl.where(b_hi < 0, -b_hi, b_hi).to(tl.uint32)
    r_lo = _gcd_core(a_lo, b_lo)
    r_hi = _gcd_core(a_hi, b_hi)
    out = (r_hi << 16) | (r_lo & 0xFFFF)
    tl.store(A4 + offs, out, mask=mask)


def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    n = A.numel()
    if n == 0:
        return A
    BLOCK = 1024
    if A.dtype == torch.int16 and (n & 1) == 0 and B.dtype == torch.int16:
        n4 = n >> 1
        A4 = A.reshape(-1).view(torch.int32)
        B4 = B.reshape(-1).view(torch.int32)
        grid = (triton.cdiv(n4, BLOCK),)
        _gcd_pair_kernel[grid](A4, B4, n4, BLOCK=BLOCK)
        return A
    grid = (triton.cdiv(n, BLOCK),)
    _gcd_kernel[grid](A, B, n, BLOCK=BLOCK, PROMOTE=A.element_size() < 4)
    return A


# Alias for FlagGems import convention
gcd_ = run

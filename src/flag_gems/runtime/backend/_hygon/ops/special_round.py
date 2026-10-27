import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# special_round(A, decimals=0): elementwise round-half-to-even to `decimals`
# decimal places, replicating torch.round/torch.special.round arithmetic on the
# target ROCm build:
#   decimals == 0 : out = rint_T(x)
#   decimals  > 0 : p = T(10**decimals); out = T( rint_T(x*p) / p )   (T = input dtype)
#   decimals  < 0 : p = 10**decimals (fp64); out = T( rint_64(x*p) / p )
# ---------------------------------------------------------------------------


@triton.jit
def _round_half_even(vf, IS_FP32: tl.constexpr):
    # vf is fp32 or fp64; round to nearest, ties to even.
    if IS_FP32:
        a = tl.abs(vf)
        fl = tl.floor(a)
        frac = a - fl
        gt = frac > 0.5
        tie = frac == 0.5
        odd = (fl / 2.0) != tl.floor(fl / 2.0)
        ra = fl + (gt | (tie & odd)).to(tl.float32)
        return tl.where(vf < 0.0, -ra, ra)
    return libdevice.rint(vf)


@triton.jit
def _special_round_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    SCALE: tl.constexpr,
    IS_FP64: tl.constexpr,
    IS_FP32: tl.constexpr,
    IS_BF16: tl.constexpr,
    FP16_UNDERFLOW: tl.constexpr,
    MODE: tl.constexpr,  # 0: decimals==0, 1: decimals>0, 2: decimals<0
    EVEN: tl.constexpr,  # n_elements % BLOCK == 0 -> no masks
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)

    if MODE == 0:
        if IS_FP64:
            out = _round_half_even(x, False)
        elif IS_FP32:
            out = _round_half_even(x, True)
        else:
            out = libdevice.rint(x.to(tl.float32)).to(x.dtype)
    elif MODE == 1:
        p = tl.full((1,), SCALE, dtype=x.dtype)
        v = x * p
        if IS_FP64:
            r = _round_half_even(v, False)
        else:
            r = _round_half_even(v.to(tl.float32), True).to(x.dtype)
        out = r / p
    else:  # MODE == 2: negative decimals -> fp64 arithmetic
        if FP16_UNDERFLOW:
            # fp16(10**d) underflows to 0 in torch's arithmetic -> 0/0 = NaN
            out = tl.full((1,), float("nan"), dtype=x.dtype)
        else:
            p64 = tl.full((1,), SCALE, dtype=tl.float64)
            if IS_BF16:
                xw = x.to(tl.float32).to(tl.float64)
            else:
                xw = x.to(tl.float64)
            v = xw * p64
            r = _round_half_even(v, False)
            if IS_BF16:
                out = (r / p64).to(tl.float32).to(x.dtype)
            else:
                out = (r / p64).to(x.dtype)

    if EVEN:
        tl.store(y_ptr + offs, out)
    else:
        tl.store(y_ptr + offs, out, mask=mask)


def _launch_cfg(dtype, n):
    # fp32 saturates DRAM at BLOCK=1024/4 warps. 16-bit dtypes prefer wide
    # 4096/8 blocks for large streams; tiny tensors get several small blocks.
    if dtype == torch.float32:
        return 1024, 4
    if n <= 1 << 20:
        return 1024, 4
    return 4096, 8


def run(A, *, decimals=0):
    out = torch.empty_like(A)
    if A.numel() == 0:
        return out
    if not A.is_contiguous():
        A = A.contiguous()
        out = torch.empty_like(A)
    d = decimals
    scale = 10.0**d
    mode = 0 if d == 0 else (1 if d > 0 else 2)
    fp16_underflow = False
    if mode == 2 and A.dtype == torch.float16:
        p16 = torch.tensor(scale, dtype=torch.float16)
        fp16_underflow = p16.item() == 0.0
    n = A.numel()
    block, num_warps = _launch_cfg(A.dtype, n)
    grid = (triton.cdiv(n, block),)
    _special_round_kernel[grid](
        A,
        out,
        n,
        SCALE=scale,
        IS_FP64=(A.dtype == torch.float64),
        IS_FP32=(A.dtype == torch.float32),
        IS_BF16=(A.dtype == torch.bfloat16),
        FP16_UNDERFLOW=fp16_underflow,
        MODE=mode,
        EVEN=(n % block == 0),
        BLOCK=block,
        num_warps=num_warps,
    )
    return out


# Alias for FlagGems import convention
special_round = run

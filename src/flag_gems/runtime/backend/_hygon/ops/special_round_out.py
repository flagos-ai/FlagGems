import torch
import triton
import triton.language as tl


@triton.jit
def _rint32(x):
    # hardware round-to-nearest-even (half-to-even), immune to backend approx passes
    return tl.inline_asm_elementwise(
        "v_rndne_f32 $0, $1;", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _rint64(x):
    return tl.inline_asm_elementwise(
        "v_rndne_f64 $0, $1;", "=v,v", [x], dtype=tl.float64, is_pure=True, pack=1
    )


@triton.jit
def _special_round_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    factor,  # fp64 scalar: 10.0 ** abs(decimals)
    BLOCK: tl.constexpr,
    CTYPE: tl.constexpr,  # 0=f32 1=f64 2=f16 3=bf16
    DMODE: tl.constexpr,  # 0=d==0 1=d>0 (mul then div) 2=d<0 (div then mul)
    EVEN: tl.constexpr,  # n_elements % BLOCK == 0 -> maskless
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs)
    else:
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
    if CTYPE == 0:  # fp32
        f = factor.to(tl.float32)
        if DMODE == 0:
            r = _rint32(x)
        elif DMODE == 1:
            r = _rint32(x * f) / f
        else:
            r = _rint32(x / f) * f
    elif CTYPE == 1:  # fp64
        if DMODE == 0:
            r = _rint64(x)
        elif DMODE == 1:
            r = _rint64(x * factor) / factor
        else:
            r = _rint64(x / factor) * factor
    elif CTYPE == 2:  # fp16: native half-precision roundings
        f = factor.to(tl.float16)
        if DMODE == 0:
            r = _rint32(x.to(tl.float32)).to(tl.float16)
        elif DMODE == 1:
            r = (_rint32((x * f).to(tl.float32)) / f.to(tl.float32)).to(tl.float16)
        else:
            q = (x.to(tl.float32) / f.to(tl.float32)).to(tl.float16)
            r = _rint32(q.to(tl.float32)).to(tl.float16) * f
    else:  # bf16: native bf16-precision roundings
        f = factor.to(tl.bfloat16)
        if DMODE == 0:
            r = _rint32(x.to(tl.float32)).to(tl.bfloat16)
        elif DMODE == 1:
            r = (_rint32((x * f).to(tl.float32)) / f.to(tl.float32)).to(tl.bfloat16)
        else:
            q = (x.to(tl.float32) / f.to(tl.float32)).to(tl.bfloat16)
            r = _rint32(q.to(tl.float32)).to(tl.bfloat16) * f
    if EVEN:
        tl.store(y_ptr + offs, r)
    else:
        tl.store(y_ptr + offs, r, mask=mask)


_CTYPE = {
    torch.float32: 0,
    torch.float64: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
}


def _launch_config(ctype, dmode):
    if ctype == 0:  # fp32
        if dmode == 0:
            return 2048, 16
        return 4096, 16
    if ctype == 1:  # fp64
        return 1024, 4
    if dmode == 0:  # fp16/bf16, d==0
        return 2048, 8
    return 2048, 4  # fp16/bf16, d!=0


def run(self, out, *, decimals=0):
    dtype = self.dtype
    ctype = _CTYPE.get(dtype)
    if ctype is None:
        raise TypeError(f"special_round_out: unsupported dtype {dtype}")
    numel = self.numel()
    if numel == 0:
        return out
    decimals = int(decimals)
    if decimals == 0:
        dmode = 0
        factor = 1.0
    elif decimals > 0:
        dmode = 1
        try:
            factor = 10.0**decimals
        except OverflowError:
            factor = float("inf")
    else:
        dmode = 2
        try:
            factor = 10.0 ** (-decimals)
        except OverflowError:
            factor = float("inf")
    x = self.reshape(-1)
    y = out.reshape(-1)
    BLOCK, num_warps = _launch_config(ctype, dmode)
    even = (numel % BLOCK) == 0
    grid = (triton.cdiv(numel, BLOCK),)
    _special_round_kernel[grid](
        x,
        y,
        numel,
        factor,
        BLOCK=BLOCK,
        CTYPE=ctype,
        DMODE=dmode,
        EVEN=even,
        num_warps=num_warps,
    )
    return out


# Alias for FlagGems import convention
special_round_out = run

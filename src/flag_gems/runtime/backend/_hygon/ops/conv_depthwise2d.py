import torch
import triton
import triton.language as tl


def _pair(v, name="value"):
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError(f"{name} must be an int or a 2-tuple, got {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


# ---------------------------------------------------------------------------
# 1D flattened-spatial kernel (best for s1 planes: small/dilated/k5/large)
# ---------------------------------------------------------------------------
@triton.jit
def _dwconv2d_kernel(
    inp_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    H,
    W,
    OH,
    OW,
    C_IN,
    C_OUT,
    G,
    KH: tl.constexpr,
    KW: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    pH: tl.constexpr,
    pW: tl.constexpr,
    dH: tl.constexpr,
    dW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(OH * OW, BLOCK_SP)

    oc = pid // num_tiles  # output channel index (0 .. C_OUT-1) across batch
    tile = pid % num_tiles

    n = oc // C_OUT
    ocl = oc % C_OUT  # local output channel within one batch element
    g = ocl // G  # input channel feeding this output channel

    in_base = inp_ptr + (n * stride_n + g * stride_c).to(tl.int64)
    out_base = out_ptr + oc.to(tl.int64) * (OH * OW)
    w_base = w_ptr + ocl.to(tl.int64) * (KH * KW)

    pos = tile * BLOCK_SP + tl.arange(0, BLOCK_SP)  # (BLOCK_SP,)
    pmask = pos < OH * OW
    h = pos // OW
    w = pos % OW
    # output (h, w) reads input at (h*sH - pH + kh*dH, w*sW - pW + kw*dW);
    # the stride factors are folded into the base offsets
    off = in_base + (h * (sH * stride_h) + w * (sW * stride_w)).to(tl.int64)

    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    for kh in tl.static_range(KH):
        ih = h * sH - pH + kh * dH
        mh = pmask & (ih >= 0) & (ih < H)
        tap_h = (kh * dH - pH) * stride_h
        for kw in tl.static_range(KW):
            iw = w * sW - pW + kw * dW
            mw = (iw >= 0) & (iw < W)
            x = tl.load(
                off + tap_h + (kw * dW - pW) * stride_w,
                mask=mh & mw,
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(w_base + kh * KW + kw).to(tl.float32)
            acc += wv * x

    if HAS_BIAS:
        bv = tl.load(b_ptr + ocl.to(tl.int64)).to(tl.float32)
        acc += bv

    tl.store(out_base + pos.to(tl.int64), acc, mask=pmask)


# ---------------------------------------------------------------------------
# 2D tile kernel (best for stride>1 planes: measured 0.112/0.107/0.107ms for
# fp32/fp16/bf16 on the s2 workload vs 0.168/0.120/0.121ms for the 1D kernel)
# ---------------------------------------------------------------------------
@triton.jit
def _dwconv2d_kernel_2d(
    inp_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    H,
    W,
    OH,
    OW,
    C_IN,
    C_OUT,
    G,
    KH: tl.constexpr,
    KW: tl.constexpr,
    sH: tl.constexpr,
    sW: tl.constexpr,
    pH: tl.constexpr,
    pW: tl.constexpr,
    dH: tl.constexpr,
    dW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ow = tl.cdiv(OW, BLOCK_OW)
    num_oh = tl.cdiv(OH, BLOCK_OH)
    tiles = num_ow * num_oh

    oc = pid // tiles
    rem = pid % tiles
    oh0 = (rem // num_ow) * BLOCK_OH
    ow0 = (rem % num_ow) * BLOCK_OW

    n = oc // C_OUT
    ocl = oc % C_OUT
    g = ocl // G

    in_base = inp_ptr + (n * stride_n + g * stride_c).to(tl.int64)
    out_base = out_ptr + oc.to(tl.int64) * (OH * OW)
    w_base = w_ptr + ocl.to(tl.int64) * (KH * KW)

    offs_oh = oh0 + tl.arange(0, BLOCK_OH)
    offs_ow = ow0 + tl.arange(0, BLOCK_OW)

    acc = tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    for kh in tl.static_range(KH):
        ih = offs_oh * sH - pH + kh * dH
        mh = (ih >= 0) & (ih < H)
        for kw in tl.static_range(KW):
            iw = offs_ow * sW - pW + kw * dW
            mw = (iw >= 0) & (iw < W)
            x = tl.load(
                in_base + ih[:, None] * stride_h + iw[None, :] * stride_w,
                mask=mh[:, None] & mw[None, :],
                other=0.0,
            ).to(tl.float32)
            wv = tl.load(w_base + kh * KW + kw).to(tl.float32)
            acc += wv * x

    if HAS_BIAS:
        bv = tl.load(b_ptr + ocl.to(tl.int64)).to(tl.float32)
        acc += bv

    o_mask = (offs_oh[:, None] < OH) & (offs_ow[None, :] < OW)
    tl.store(out_base + offs_oh[:, None] * OW + offs_ow[None, :], acc, mask=o_mask)


def run(input, weight, kernel_size, bias, stride, padding, dilation):
    N, C_IN, H, W = input.shape
    C_OUT = weight.shape[0]
    KH, KW = weight.shape[-2], weight.shape[-1]
    sH, sW = _pair(stride, "stride")
    pH, pW = _pair(padding, "padding")
    dH, dW = _pair(dilation, "dilation")
    G = C_OUT // C_IN

    OH = (H + 2 * pH - dH * (KH - 1) - 1) // sH + 1
    OW = (W + 2 * pW - dW * (KW - 1) - 1) // sW + 1

    out = torch.empty((N, C_OUT, OH, OW), device=input.device, dtype=input.dtype)
    b_ptr = bias if bias is not None else weight

    if sH > 1 or sW > 1:
        # Strided planes: the 2D (4 x 64) tile is ~1.5x faster than the 1D
        # flattened kernel (debug sweep on the s2 workload class).  num_warps=1
        # is fastest for all dtypes: fp16/bf16 0.1011 vs 0.1066ms, and fp32
        # 0.1056 vs 0.1123ms (clean-sweep measurements).
        BLOCK_OH, BLOCK_OW = 4, 64
        nw2d = 1
        grid = (N * C_OUT * triton.cdiv(OH, BLOCK_OH) * triton.cdiv(OW, BLOCK_OW),)
        _dwconv2d_kernel_2d[grid](
            input,
            weight,
            b_ptr,
            out,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            H,
            W,
            OH,
            OW,
            C_IN,
            C_OUT,
            G,
            KH,
            KW,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            bias is not None,
            BLOCK_OH=BLOCK_OH,
            BLOCK_OW=BLOCK_OW,
            num_warps=nw2d,
        )
    else:
        # Unit-stride planes: BLOCK_SP=256 1D kernel.
        # num_warps=4 (2 elems/thread) only for fp32 large planes;
        # num_warps=1 for k5 planes (0.110 vs 0.120ms fp16, 0.0976 vs 0.1006
        # fp32) and for s2; num_warps=2 elsewhere (debug sweep optima).
        SP = OH * OW
        if input.dtype == torch.float32:
            num_warps = 4 if SP >= 4096 else (1 if KH * KW >= 16 else 2)
        elif KH * KW >= 16 and SP >= 784:
            num_warps = 1
        else:
            num_warps = 2
        BLOCK_SP = 256
        grid = (N * C_OUT * triton.cdiv(SP, BLOCK_SP),)
        _dwconv2d_kernel[grid](
            input,
            weight,
            b_ptr,
            out,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            H,
            W,
            OH,
            OW,
            C_IN,
            C_OUT,
            G,
            KH,
            KW,
            sH,
            sW,
            pH,
            pW,
            dH,
            dW,
            bias is not None,
            BLOCK_SP=BLOCK_SP,
            num_warps=num_warps,
        )
    return out


# Alias for FlagGems import convention
conv_depthwise2d = run

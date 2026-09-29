import torch
import triton
import triton.language as tl


@triton.jit
def _adaptive_avg_pool2d_bw_flat_div_kernel(
    go_ptr,
    out_ptr,
    s_n,
    s_c,
    s_h,
    s_w,
    total,
    BLOCK: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    H_out: tl.constexpr,
    W_in: tl.constexpr,
    W_out: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SHIFT8: tl.constexpr,
    I64: tl.constexpr,
    CONTIG: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # Divisible case: H_in % H_out == 0 and W_in % W_out == 0. Every input
    # element is covered by exactly one output element:
    #   grad_input[h, w] = grad_output[h//KH, w//KW] / (KH * KW)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    hw = H_in * W_in
    nc = offs // hw
    rem = offs % hw
    hh = rem // W_in
    w = rem % W_in
    if SHIFT8:
        oh = hh >> 3
        ow = w >> 3
    else:
        oh = hh // KH
        ow = w // KW
    inv = tl.full((), 1.0, ACC_DTYPE) / (KH * KW)
    if I64:
        if CONTIG:
            src = (
                nc.to(tl.int64) * (H_out * W_out)
                + oh.to(tl.int64) * W_out
                + ow.to(tl.int64)
            )
        else:
            n = nc // C
            c = nc % C
            src = (
                n.to(tl.int64) * s_n
                + c.to(tl.int64) * s_c
                + oh.to(tl.int64) * s_h
                + ow.to(tl.int64) * s_w
            )
        v = tl.load(go_ptr + src, mask=m, other=0.0)
        tl.store(out_ptr + offs.to(tl.int64), v.to(ACC_DTYPE) * inv, mask=m)
    else:
        if CONTIG:
            src = nc * (H_out * W_out) + oh * W_out + ow
        else:
            n = nc // C
            c = nc % C
            src = n * s_n + c * s_c + oh * s_h + ow * s_w
        v = tl.load(go_ptr + src, mask=m, other=0.0)
        tl.store(out_ptr + offs, v.to(ACC_DTYPE) * inv, mask=m)


@triton.jit
def _adaptive_avg_pool2d_bw_flat_kernel(
    go_ptr,
    out_ptr,
    s_n,
    s_c,
    s_h,
    s_w,
    total,
    BLOCK: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    H_out: tl.constexpr,
    W_in: tl.constexpr,
    W_out: tl.constexpr,
    MAX_H: tl.constexpr,
    MAX_W: tl.constexpr,
    I64: tl.constexpr,
    CONTIG: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # General adaptive backward: for each input element, sum over the covering
    # output window of grad_output[oh, ow] / (hspan(oh) * wspan(ow)).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    hw = H_in * W_in
    nc = offs // hw
    rem = offs % hw
    h = rem // W_in
    w = rem % W_in

    if I64:
        h64 = h.to(tl.int64)
        hs = ((h64 * H_out) // H_in).to(tl.int32)
        he = (((h64 + 1) * H_out + H_in - 1) // H_in).to(tl.int32)
    else:
        hs = (h * H_out) // H_in
        he = ((h + 1) * H_out + H_in - 1) // H_in
    hs = tl.minimum(hs, H_out - 1)
    he = tl.maximum(he, hs + 1)
    hspan = he - hs

    if I64:
        w64 = w.to(tl.int64)
        ws = ((w64 * W_out) // W_in).to(tl.int32)
        we = (((w64 + 1) * W_out + W_in - 1) // W_in).to(tl.int32)
    else:
        ws = (w * W_out) // W_in
        we = ((w + 1) * W_out + W_in - 1) // W_in
    wspan = we - ws

    if CONTIG:
        if I64:
            base = nc.to(tl.int64) * (H_out * W_out)
        else:
            base = nc * (H_out * W_out)
    else:
        n = nc // C
        c = nc % C
        if I64:
            base = n.to(tl.int64) * s_n + c.to(tl.int64) * s_c
        else:
            base = n * s_n + c * s_c

    acc = tl.zeros([BLOCK], dtype=ACC_DTYPE)
    for dh in tl.static_range(0, MAX_H):
        hm = dh < hspan
        oh = hs + dh
        if I64:
            oh64 = oh.to(tl.int64)
            fhs = ((oh64 * H_in) // H_out).to(tl.int32)
            fhe = (((oh64 + 1) * H_in + H_out - 1) // H_out).to(tl.int32)
        else:
            fhs = (oh * H_in) // H_out
            fhe = ((oh + 1) * H_in + H_out - 1) // H_out
        hn = 1.0 / (fhe - fhs).to(ACC_DTYPE)
        if CONTIG:
            if I64:
                row_off = base + oh.to(tl.int64) * W_out
            else:
                row_off = base + oh * W_out
        else:
            if I64:
                row_off = base + oh.to(tl.int64) * s_h
            else:
                row_off = base + oh * s_h
        for dw in tl.static_range(0, MAX_W):
            ow = ws + dw
            mm = m & hm & (dw < wspan)
            if CONTIG:
                if I64:
                    off = row_off + ow.to(tl.int64)
                else:
                    off = row_off + ow
            else:
                if I64:
                    off = row_off + ow.to(tl.int64) * s_w
                else:
                    off = row_off + ow * s_w
            v = tl.load(go_ptr + off, mask=mm, other=0.0)
            if I64:
                ow64 = ow.to(tl.int64)
                fws = ((ow64 * W_in) // W_out).to(tl.int32)
                fwe = (((ow64 + 1) * W_in + W_out - 1) // W_out).to(tl.int32)
            else:
                fws = (ow * W_in) // W_out
                fwe = ((ow + 1) * W_in + W_out - 1) // W_out
            wn = 1.0 / (fwe - fws).to(ACC_DTYPE)
            acc += v.to(ACC_DTYPE) * hn * wn

    if I64:
        tl.store(out_ptr + offs.to(tl.int64), acc, mask=m)
    else:
        tl.store(out_ptr + offs, acc, mask=m)


@triton.jit
def _adaptive_avg_pool2d_bw_flat_dyn_kernel(
    go_ptr,
    out_ptr,
    s_n,
    s_c,
    s_h,
    s_w,
    total,
    max_h,
    max_w,
    BLOCK: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    H_out: tl.constexpr,
    W_in: tl.constexpr,
    W_out: tl.constexpr,
    CONTIG: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    # Fallback for pathological upsampling ratios (large covering counts).
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    hw = H_in * W_in
    nc = offs // hw
    rem = offs % hw
    h = rem // W_in
    w = rem % W_in

    h64 = h.to(tl.int64)
    hs = ((h64 * H_out) // H_in).to(tl.int32)
    he = (((h64 + 1) * H_out + H_in - 1) // H_in).to(tl.int32)
    hs = tl.minimum(hs, H_out - 1)
    he = tl.maximum(he, hs + 1)
    hspan = he - hs

    w64 = w.to(tl.int64)
    ws = ((w64 * W_out) // W_in).to(tl.int32)
    we = (((w64 + 1) * W_out + W_in - 1) // W_in).to(tl.int32)
    wspan = we - ws

    if CONTIG:
        base = nc.to(tl.int64) * (H_out * W_out)
    else:
        n = nc // C
        c = nc % C
        base = n.to(tl.int64) * s_n + c.to(tl.int64) * s_c

    acc = tl.zeros([BLOCK], dtype=ACC_DTYPE)
    for dh in tl.range(0, max_h):
        hm = dh < hspan
        oh = hs + dh
        oh64 = oh.to(tl.int64)
        fhs = ((oh64 * H_in) // H_out).to(tl.int32)
        fhe = (((oh64 + 1) * H_in + H_out - 1) // H_out).to(tl.int32)
        hn = 1.0 / (fhe - fhs).to(ACC_DTYPE)
        if CONTIG:
            row_off = base + oh64 * W_out
        else:
            row_off = base + oh64 * s_h
        for dw in tl.range(0, max_w):
            ow = ws + dw
            mm = m & hm & (dw < wspan)
            if CONTIG:
                off = row_off + ow.to(tl.int64)
            else:
                off = row_off + ow.to(tl.int64) * s_w
            v = tl.load(go_ptr + off, mask=mm, other=0.0)
            ow64 = ow.to(tl.int64)
            fws = ((ow64 * W_in) // W_out).to(tl.int32)
            fwe = (((ow64 + 1) * W_in + W_out - 1) // W_out).to(tl.int32)
            wn = 1.0 / (fwe - fws).to(ACC_DTYPE)
            acc += v.to(ACC_DTYPE) * hn * wn

    tl.store(out_ptr + offs.to(tl.int64), acc, mask=m)


def run(grad_output, self):
    s = self
    g = grad_output
    squeeze = False
    if s.dim() == 3:
        s = s.unsqueeze(0)
        g = g.unsqueeze(0)
        squeeze = True
    if s.dim() != 4 or g.dim() != 4:
        raise RuntimeError("adaptive_avg_pool2d_backward expects 3D/4D tensors")

    N, C, H_in, W_in = s.shape
    H_out, W_out = g.shape[2], g.shape[3]

    out = torch.empty_like(s)
    if s.numel() == 0 or g.numel() == 0 or H_out == 0 or W_out == 0:
        out.zero_()
        return out.squeeze(0) if squeeze else out

    total = N * C * H_in * W_in
    # Safe upper bound on the number of covering output indices per input
    # index: for ratio r = out/in the count is <= 2 when r <= 1, otherwise
    # <= ceil(r) + 1; use the tight closed form below.
    max_h = (H_out + H_in - 1) // H_in + 1
    max_w = (W_out + W_in - 1) // W_in + 1

    acc_dtype = tl.float64 if s.dtype == torch.float64 else tl.float32
    contig = g.is_contiguous()
    i64_total = total >= (1 << 30) or any(stride >= (1 << 31) for stride in g.stride())

    if H_in % H_out == 0 and W_in % W_out == 0:
        BLOCK = 4096
        grid = (triton.cdiv(total, BLOCK),)
        kh = H_in // H_out
        kw = W_in // W_out
        _adaptive_avg_pool2d_bw_flat_div_kernel[grid](
            g,
            out,
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            total,
            BLOCK=BLOCK,
            C=C,
            H_in=H_in,
            H_out=H_out,
            W_in=W_in,
            W_out=W_out,
            KH=kh,
            KW=kw,
            SHIFT8=(kh == 8 and kw == 8),
            I64=i64_total,
            CONTIG=contig,
            ACC_DTYPE=acc_dtype,
            num_warps=8,
        )
    elif max_h * max_w > 64 or max_h > 16 or max_w > 16:
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        _adaptive_avg_pool2d_bw_flat_dyn_kernel[grid](
            g,
            out,
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            total,
            max_h,
            max_w,
            BLOCK=BLOCK,
            C=C,
            H_in=H_in,
            H_out=H_out,
            W_in=W_in,
            W_out=W_out,
            CONTIG=contig,
            ACC_DTYPE=acc_dtype,
            num_warps=8,
        )
    else:
        BLOCK = 1024
        grid = (triton.cdiv(total, BLOCK),)
        i64 = (max(H_in, H_out, W_in, W_out) > 16384) or i64_total
        _adaptive_avg_pool2d_bw_flat_kernel[grid](
            g,
            out,
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            total,
            BLOCK=BLOCK,
            C=C,
            H_in=H_in,
            H_out=H_out,
            W_in=W_in,
            W_out=W_out,
            MAX_H=max_h,
            MAX_W=max_w,
            I64=i64,
            CONTIG=contig,
            ACC_DTYPE=acc_dtype,
            num_warps=4,
        )

    return out.squeeze(0) if squeeze else out


# Alias for FlagGems import convention
adaptive_avg_pool2d_backward = run

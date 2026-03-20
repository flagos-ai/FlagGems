# col2im (fold) operator — Triton output-centric gather implementation for FlagGems
# Eliminates atomic contention by computing each output pixel via gather over KH*KW.
# Autotuned over BLOCK_HW to balance memory coalescing and occupancy.

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=2, num_stages=4),
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    ],
    key=["H_OUT", "W_OUT", "KH", "KW"],
)
@triton.jit
def _col2im_gather_kernel(
    input_ptr,  # (N, C*KH*KW, L)
    output_ptr,  # (N, C, H_OUT, W_OUT)
    N,
    C,
    H_OUT,
    W_OUT,
    OH_WIN,
    OW_WIN,
    STRIDE_H,
    STRIDE_W,
    PAD_H,
    PAD_W,
    DIL_H,
    DIL_W,
    L,  # total sliding positions = OH_WIN * OW_WIN
    in_strideN,
    in_strideCK,
    in_strideL,
    out_strideN,
    out_strideC,
    out_strideH,
    out_strideW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Grid: (N * C, cdiv(H_OUT * W_OUT, BLOCK_HW))
    pid_nc = tl.program_id(axis=0)
    pid_hw = tl.program_id(axis=1)

    # Compute n, c from flattened pid
    n = pid_nc // C
    c = pid_nc % C

    # Integer widths: use int64 for offsets/strides to avoid overflow
    n64 = n.to(tl.int64)
    c64 = c.to(tl.int64)

    W_OUT64 = tl.full((), W_OUT, tl.int64)
    OH64 = tl.full((), OH_WIN, tl.int64)
    OW64 = tl.full((), OW_WIN, tl.int64)

    stride_h64 = tl.full((), STRIDE_H, tl.int64)
    stride_w64 = tl.full((), STRIDE_W, tl.int64)
    pad_h64 = tl.full((), PAD_H, tl.int64)
    pad_w64 = tl.full((), PAD_W, tl.int64)
    dil_h64 = tl.full((), DIL_H, tl.int64)
    dil_w64 = tl.full((), DIL_W, tl.int64)

    in_strideN64 = tl.full((), in_strideN, tl.int64)
    in_strideCK64 = tl.full((), in_strideCK, tl.int64)
    in_strideL64 = tl.full((), in_strideL, tl.int64)

    out_strideN64 = tl.full((), out_strideN, tl.int64)
    out_strideC64 = tl.full((), out_strideC, tl.int64)
    out_strideH64 = tl.full((), out_strideH, tl.int64)
    out_strideW64 = tl.full((), out_strideW, tl.int64)

    # Vector of output pixel linear indices for this program
    hw_offsets = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    total_hw = H_OUT * W_OUT
    mask_hw = hw_offsets < total_hw
    hw_offsets64 = hw_offsets.to(tl.int64)

    # Map to (h, w)
    h = hw_offsets64 // W_OUT64
    w = hw_offsets64 % W_OUT64

    # Base pointers
    base_in_n = n64 * in_strideN64
    base_out_nc = n64 * out_strideN64 + c64 * out_strideC64

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Iterate over KH x KW kernel positions — gather from input
    for kh in range(KH):
        kh64 = tl.full((), kh, tl.int64)
        for kw in range(KW):
            kw64 = tl.full((), kw, tl.int64)

            # Compute source positions in the sliding window grid
            nh = h + pad_h64 - kh64 * dil_h64
            nw = w + pad_w64 - kw64 * dil_w64

            # Check divisibility by stride
            div_h_ok = (nh % stride_h64) == 0
            div_w_ok = (nw % stride_w64) == 0

            oh = nh // stride_h64
            ow = nw // stride_w64

            in_bounds = (
                mask_hw
                & div_h_ok
                & div_w_ok
                & (oh >= 0)
                & (oh < OH64)
                & (ow >= 0)
                & (ow < OW64)
            )

            l_idx = oh * OW64 + ow

            # Channel-kernel index
            ck_idx64 = (
                c64 * tl.full((), KH * KW, tl.int64)
                + kh64 * tl.full((), KW, tl.int64)
                + kw64
            )

            # Pointers for load
            in_ptrs = (
                input_ptr + base_in_n + ck_idx64 * in_strideCK64 + l_idx * in_strideL64
            )

            vals = tl.load(in_ptrs, mask=in_bounds, other=0.0).to(tl.float32)
            acc += vals

    # Store accumulated value — no atomics needed!
    out_ptrs = output_ptr + base_out_nc + h * out_strideH64 + w * out_strideW64
    tl.store(out_ptrs, acc, mask=mask_hw)


def _ensure_cuda_tensor(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    if x.device.type != "cuda":
        raise ValueError("Tensor must be on CUDA device")
    return x


def _to_2tuple(x):
    """Convert scalar / 1-element / 2-element sequence to a 2-tuple of ints."""
    if isinstance(x, torch.Size):
        x = tuple(x)
    if isinstance(x, torch.Tensor):
        vals = x.flatten().tolist()
        if len(vals) == 1:
            v = int(vals[0])
            return (v, v)
        elif len(vals) == 2:
            return (int(vals[0]), int(vals[1]))
        else:
            raise ValueError("Expected tensor with 1 or 2 elements for 2D parameter.")
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            v = int(x[0])
            return (v, v)
        if len(x) == 2:
            return (int(x[0]), int(x[1]))
        raise ValueError("Expected list/tuple of length 1 or 2 for 2D parameter.")
    v = int(x)
    return (v, v)


def col2im(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """col2im (fold): Rearrange columns back into a multidimensional image.

    Uses output-centric gather pattern: each output pixel gathers contributions
    from all relevant kernel positions, accumulating in registers. No atomic
    operations needed.

    Args:
        input: Tensor of shape (N, C*kH*kW, L) or (C*kH*kW, L).
        output_size: Desired spatial output size (H_out, W_out).
        kernel_size: Size of the sliding blocks (kH, kW).
        dilation: Dilation of sliding blocks. Default: 1.
        padding: Implicit zero padding. Default: 0.
        stride: Stride of sliding blocks. Default: 1.

    Returns:
        Reconstructed image tensor of shape (N, C, H_out, W_out) or (C, H_out, W_out).
    """
    input = _ensure_cuda_tensor(input)
    output_size = _to_2tuple(output_size)
    kernel_size = _to_2tuple(kernel_size)
    dilation = _to_2tuple(dilation)
    padding = _to_2tuple(padding)
    stride = _to_2tuple(stride)

    x = input
    squeeze_batch = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze_batch = True
    elif x.dim() != 3:
        raise ValueError("input must be of shape (N, C*kH*kW, L) or (C*kH*kW, L)")

    if not x.is_contiguous():
        x = x.contiguous()

    orig_dtype = x.dtype
    need_cast = orig_dtype not in (torch.float32, torch.float64)
    if need_cast:
        x = x.to(torch.float32)

    N_val = x.shape[0]
    CK = x.shape[1]
    L_val = x.shape[2]

    KH, KW = kernel_size
    if KH * KW == 0:
        raise ValueError("kernel_size elements must be > 0")
    if CK % (KH * KW) != 0:
        raise ValueError(
            "input second dimension must be divisible by kernel_size product (kH*kW)"
        )

    C_val = CK // (KH * KW)

    H_out, W_out = output_size
    dil_h, dil_w = dilation
    pad_h, pad_w = padding
    stride_h, stride_w = stride

    eff_kh = dil_h * (KH - 1) + 1
    eff_kw = dil_w * (KW - 1) + 1
    OH = (H_out + 2 * pad_h - eff_kh) // stride_h + 1
    OW = (W_out + 2 * pad_w - eff_kw) // stride_w + 1

    if OH <= 0 or OW <= 0:
        raise ValueError(
            "Calculated number of sliding blocks is non-positive; check parameters."
        )
    if OH * OW != L_val:
        raise ValueError(
            f"Input L ({L_val}) does not match computed number of "
            f"sliding positions ({OH * OW})"
        )

    device = x.device
    out_fp32 = torch.zeros(
        (N_val, C_val, H_out, W_out), dtype=torch.float32, device=device
    )

    in_strides = x.stride()
    out_strides = out_fp32.stride()

    def grid(meta):
        return (N_val * C_val, triton.cdiv(H_out * W_out, meta["BLOCK_HW"]))

    _col2im_gather_kernel[grid](
        x,
        out_fp32,
        N_val,
        C_val,
        H_out,
        W_out,
        OH,
        OW,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        L_val,
        in_strides[0],
        in_strides[1],
        in_strides[2],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        KH=KH,
        KW=KW,
    )

    out = out_fp32
    if squeeze_batch:
        out = out.squeeze(0)
    if need_cast:
        out = out.to(orig_dtype)
    return out

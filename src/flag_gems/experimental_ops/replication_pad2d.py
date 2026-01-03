import torch
import triton
import triton.language as tl


@triton.jit
def replication_pad2d_kernel(
    in_ptr,  # *Pointer* to input tensor
    out_ptr,  # *Pointer* to output tensor
    N,
    C,
    H,
    W,  # input dimensions
    OH,
    OW,  # output H and W
    PAD_LEFT,
    PAD_TOP,  # padding sizes
    TOTAL_ELEMS,  # total number of output elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL_ELEMS

    # Cast to int64 for safe indexing
    offs64 = offs.to(tl.int64)

    OW_i64 = tl.full([1], OW, dtype=tl.int64)
    OH_i64 = tl.full([1], OH, dtype=tl.int64)
    C_i64 = tl.full([1], C, dtype=tl.int64)
    W_i64 = tl.full([1], W, dtype=tl.int64)
    H_i64 = tl.full([1], H, dtype=tl.int64)
    PAD_LEFT_i64 = tl.full([1], PAD_LEFT, dtype=tl.int64)
    PAD_TOP_i64 = tl.full([1], PAD_TOP, dtype=tl.int64)

    ow = offs64 % OW_i64
    tmp = offs64 // OW_i64
    oh = tmp % OH_i64
    tmp = tmp // OH_i64
    c = tmp % C_i64
    n = tmp // C_i64

    ih = oh - PAD_TOP_i64
    iw = ow - PAD_LEFT_i64

    zero = tl.full([1], 0, dtype=tl.int64)
    Hm1 = H_i64 - 1
    Wm1 = W_i64 - 1

    ih = tl.maximum(zero, tl.minimum(Hm1, ih))
    iw = tl.maximum(zero, tl.minimum(Wm1, iw))

    in_index = ((n * C_i64 + c) * H_i64 + ih) * W_i64 + iw
    out_index = offs64

    x = tl.load(in_ptr + in_index, mask=mask)
    tl.store(out_ptr + out_index, x, mask=mask)


def _prepare_dims_and_out(input: torch.Tensor, padding, out: torch.Tensor | None):
    if not isinstance(padding, (tuple, list)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of 4 integers: (pad_left, pad_right, pad_top, pad_bottom)"
        )
    pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("replication_pad2d does not support negative padding")

    if input.dim() == 4:
        N, C, H, W = input.shape
        out_shape = (N, C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        kernel_N, kernel_C = N, C
    elif input.dim() == 3:
        C, H, W = input.shape
        out_shape = (C, H + pad_top + pad_bottom, W + pad_left + pad_right)
        kernel_N, kernel_C = 1, C
    else:
        raise ValueError(
            "replication_pad2d expects a 3D (C, H, W) or 4D (N, C, H, W) input"
        )

    if H <= 0 or W <= 0:
        raise ValueError(
            "Input height and width must be greater than 0 for replication padding"
        )

    if out is None:
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
    else:
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(
                f"Provided out tensor has shape {tuple(out.shape)}, expected {out_shape}"
            )
        if out.device != input.device:
            raise ValueError("Input and out must be on the same device")
        if out.dtype != input.dtype:
            raise ValueError("Input and out must have the same dtype")

    return (
        kernel_N,
        kernel_C,
        H,
        W,
        out.shape[-2],
        out.shape[-1],
        pad_left,
        pad_top,
    ), out


def _launch_replication_pad2d_kernel(
    input: torch.Tensor, out: torch.Tensor, kernel_params
):
    if not input.is_cuda or not out.is_cuda:
        raise ValueError("Tensors must be CUDA tensors")
    if not input.is_contiguous() or not out.is_contiguous():
        raise ValueError("Only contiguous tensors are supported")

    N, C, H, W, OH, OW, pad_left, pad_top = kernel_params
    total_elems = out.numel()
    if total_elems == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elems, BLOCK_SIZE),)

    replication_pad2d_kernel[grid](
        input,
        out,
        N,
        C,
        H,
        W,
        OH,
        OW,
        pad_left,
        pad_top,
        total_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def replication_pad2d(input: torch.Tensor, padding):
    kernel_params, out = _prepare_dims_and_out(input, padding, out=None)
    return _launch_replication_pad2d_kernel(input, out, kernel_params)


def replication_pad2d_out(input: torch.Tensor, padding, out: torch.Tensor):
    kernel_params, out = _prepare_dims_and_out(input, padding, out=out)
    return _launch_replication_pad2d_kernel(input, out, kernel_params)

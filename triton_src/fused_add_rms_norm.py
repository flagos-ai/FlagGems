import triton
import triton.language as tl


@triton.jit(do_not_specialize=["eps"])
def fused_add_rms_norm_kernel(
    in_ptr,  # pointer to the input
    re_ptr,  # pointer to the residual
    w_ptr,  # pointer to the weights
    in_stride_r,  # how much to increase the pointer when moving by 1 row
    in_stride_c,  # how much to increase the pointer when moving by 1 col
    r_stride_r,  # how much to increase the pointer when moving by 1 row
    r_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in in_ptr
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    in_ptr += pid * in_stride_r
    re_ptr += pid * r_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * in_stride_c, mask, other=0.0).to(cdtype)
    r = tl.load(re_ptr + cols * r_stride_c, mask, other=0.0).to(cdtype)

    x += r
    # write back to residual
    tl.store(re_ptr + cols * r_stride_c, x, mask=mask)

    var = tl.sum(x * x / N, axis=0)
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    # write back to input
    tl.store(in_ptr + cols * in_stride_c, y, mask=mask)

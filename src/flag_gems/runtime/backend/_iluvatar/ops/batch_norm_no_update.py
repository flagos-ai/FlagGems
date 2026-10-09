import torch
import triton
import triton.language as tl


@triton.jit
def _bn_infer_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    w_ptr,
    b_ptr,
    C,
    S,
    eps,
    BLOCK: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_B: tl.constexpr,
):
    pid_s = tl.program_id(0)  # block within spatial dim (contiguous, fastest)
    pid_n = tl.program_id(1)  # batch index
    pid_c = tl.program_id(2)  # channel index (0..C-1)

    mean = tl.load(mean_ptr + pid_c).to(tl.float32)
    var = tl.load(var_ptr + pid_c).to(tl.float32)
    inv_std = tl.rsqrt(var + eps)
    if HAS_W:
        w = tl.load(w_ptr + pid_c).to(tl.float32)
    else:
        w = 1.0
    if HAS_B:
        b = tl.load(b_ptr + pid_c).to(tl.float32)
    else:
        b = 0.0

    base = (pid_n * C + pid_c) * S
    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S
    x = tl.load(x_ptr + base + offs, mask=mask).to(tl.float32)
    y = w * (x - mean) * inv_std + b
    tl.store(out_ptr + base + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(
    input,
    weight=None,
    bias=None,
    running_mean=None,
    running_var=None,
    momentum=0.1,
    eps=1e-05,
):
    N = input.shape[0]
    C = input.shape[1]
    S = 1
    for i in range(2, input.dim()):
        S *= input.shape[i]
    out = torch.empty_like(input)

    # Launch-shape heuristic tuned on Iluvatar BI-V150 (do_bench sweeps):
    #   fp32: BLOCK=256, num_warps=4
    #   fp16/bf16: BLOCK=512 (S>=384) / 256 (128<=S<384), num_warps=4
    if input.element_size() == 4:
        BLOCK = 256 if S >= 128 else max(16, triton.next_power_of_2(S))
    else:
        if S >= 384:
            BLOCK = 512
        elif S >= 128:
            BLOCK = 256
        else:
            BLOCK = max(16, triton.next_power_of_2(S))
    nw = 1 if BLOCK <= 64 else 4

    grid = (triton.cdiv(S, BLOCK), N, C)
    dummy = input
    _bn_infer_kernel[grid](
        input,
        out,
        running_mean if running_mean is not None else dummy,
        running_var if running_var is not None else dummy,
        weight if weight is not None else dummy,
        bias if bias is not None else dummy,
        C,
        S,
        eps,
        BLOCK=BLOCK,
        HAS_W=weight is not None,
        HAS_B=bias is not None,
        num_warps=nw,
    )
    return out, None, None, None

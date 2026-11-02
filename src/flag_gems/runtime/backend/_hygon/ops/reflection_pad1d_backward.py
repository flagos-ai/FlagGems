import torch
import triton
import triton.language as tl


@triton.jit
def _reflection_pad1d_backward_kernel(
    grad_out_ptr,
    grad_in_ptr,
    W_in,
    W_out,
    pad_left,
    pad_right,
    BLOCK: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_r = tl.program_id(1)

    j = pid_w * BLOCK + tl.arange(0, BLOCK)
    mask = j < W_in

    base = pid_r * W_out

    # Direct contribution: output index j + pad_left
    idx0 = j + pad_left
    v0 = tl.load(grad_out_ptr + base + idx0, mask=mask, other=0.0)

    # Left reflected region: output index pad_left - j, for 1 <= j <= pad_left
    idx1 = pad_left - j
    mask1 = mask & (j >= 1) & (j <= pad_left)
    v1 = tl.load(grad_out_ptr + base + idx1, mask=mask1, other=0.0)

    # Right reflected region: output index 2*(W_in-1) + pad_left - j,
    # for W_in - pad_right - 1 <= j <= W_in - 2
    idx2 = 2 * (W_in - 1) + pad_left - j
    mask2 = mask & (j >= W_in - pad_right - 1) & (j <= W_in - 2)
    v2 = tl.load(grad_out_ptr + base + idx2, mask=mask2, other=0.0)

    acc = v0 + v1 + v2
    tl.store(grad_in_ptr + pid_r * W_in + j, acc, mask=mask)


def run(grad_output, input, padding):
    # ---- parse padding: int, 2-tuple/list, or 1/2-element tensor ----
    if torch.is_tensor(padding):
        p = padding.detach().reshape(-1).cpu().tolist()
    elif isinstance(padding, (tuple, list)):
        p = [int(x) for x in padding]
    else:
        p = [int(padding), int(padding)]
    if len(p) == 1:
        p = [p[0], p[0]]
    pad_left, pad_right = int(p[0]), int(p[1])

    grad_output = grad_output.contiguous()
    input = input.contiguous()

    W_in = input.shape[-1]
    W_out = grad_output.shape[-1]
    rows = input.numel() // W_in

    out = torch.empty_like(input)
    if rows == 0 or W_in == 0:
        return out

    grad_out_2d = grad_output.view(rows, W_out)
    grad_in_2d = out.view(rows, W_in)

    BLOCK = 1024
    grid = (triton.cdiv(W_in, BLOCK), rows)
    _reflection_pad1d_backward_kernel[grid](
        grad_out_2d,
        grad_in_2d,
        W_in,
        W_out,
        pad_left,
        pad_right,
        BLOCK=BLOCK,
    )
    return out


# Alias for FlagGems import convention
reflection_pad1d_backward = run

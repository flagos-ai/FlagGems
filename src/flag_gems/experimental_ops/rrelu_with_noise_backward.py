import torch
import triton
import triton.language as tl


@triton.jit
def rrelu_with_noise_backward_kernel(
    grad_out_ptr,  # *Pointer* to grad_output
    input_ptr,  # *Pointer* to input (or result if self_is_result, either works)
    noise_ptr,  # *Pointer* to noise
    grad_in_ptr,  # *Pointer* to output grad_input
    n_elements,  # Number of elements
    lower,  # float32
    upper,  # float32
    training,  # int32 (1 for training, 0 for eval)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    go = tl.load(grad_out_ptr + offsets, mask=mask, other=0)
    x = tl.load(input_ptr + offsets, mask=mask, other=0)
    nz = tl.load(noise_ptr + offsets, mask=mask, other=0)

    go_f32 = go.to(tl.float32)
    x_f32 = x.to(tl.float32)
    nz_f32 = nz.to(tl.float32)

    slope = (lower + upper) * 0.5

    grad_train = go_f32 * nz_f32
    grad_eval = go_f32 * tl.where(x_f32 > 0, 1.0, slope)

    cond = tl.full(go_f32.shape, training, tl.int1)
    grad_f32 = tl.where(cond, grad_train, grad_eval)

    grad_cast = grad_f32.to(go.dtype)
    tl.store(grad_in_ptr + offsets, grad_cast, mask=mask)


def _launch_rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    out: torch.Tensor,
):
    assert (
        grad_output.is_cuda and input.is_cuda and noise.is_cuda and out.is_cuda
    ), "All tensors must be CUDA"
    assert (
        grad_output.numel() == input.numel() == noise.numel() == out.numel()
    ), "All tensors must have the same number of elements"
    assert (
        grad_output.dtype == input.dtype == noise.dtype == out.dtype
    ), "All tensors must have the same dtype"

    go = grad_output.contiguous()
    x = input.contiguous()
    nz = noise.contiguous()
    out_t = out.contiguous()

    n_elements = out_t.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    rrelu_with_noise_backward_kernel[grid](
        go,
        x,
        nz,
        out_t,
        n_elements,
        float(lower),
        float(upper),
        1 if training else 0,
        BLOCK_SIZE=1024,
    )
    if out is not out_t:
        out.copy_(out_t)
    return out


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool = False,
):
    out = torch.empty_like(grad_output)
    return _launch_rrelu_with_noise_backward(
        grad_output, input, noise, lower, upper, training, out
    )


def rrelu_with_noise_backward_out(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool,
    out: torch.Tensor,
):
    return _launch_rrelu_with_noise_backward(
        grad_output, input, noise, lower, upper, training, out
    )

import torch
import triton
import triton.language as tl
from .libentry import libentry


def cfggen(all_args):
    x = all_args["input_ptr"]
    _, N, _ = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    return (BLOCK_SIZE,)


@libentry(cfggen=cfggen)
@triton.jit
def softmax_kernel(output_ptr, input_ptr, M, N, K, BLOCK_SIZE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offset = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_ptr + pid_m * N * K + n_offset * K + pid_k
    inp = tl.load(input_ptrs, mask=n_offset < N, other=-float("inf"))
    row_minus_max = inp - tl.max(inp, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_ptrs = output_ptr + pid_m * N * K + n_offset * K + pid_k
    tl.store(output_ptrs, softmax_output, mask=n_offset < N)


def softmax(x, dim=1, dtype=None, out=None):
    print("FLAG SOFTMAX")

    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]
    inp = x.contiguous()
    inp = inp.reshape(M, N, -1)
    K = inp.numel() // M // N

    if dtype is None:
        dtype = x.dtype
    if out is None:
        out = torch.empty_like(x, dtype=dtype)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    softmax_kernel[
        (
            M,
            K,
        )
    ](
        out,
        inp,
        M,
        N,
        K,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

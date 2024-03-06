import torch
import triton
import triton.language as tl
from .libentry import libentry


def cfggen(all_args):
    N = all_args["N"]
    BLOCK_N = triton.next_power_of_2(N)
    return (BLOCK_N,)


@libentry(cfggen=cfggen)
@triton.heuristics(
    values={"BLOCK_N": lambda args: triton.next_power_of_2(args["N"])},
)
@triton.jit
def cumsum_kernel(
    inp,
    out,
    N,
    stride_m,
    stride_n,
    stride_k,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    n_offset = tl.arange(0, BLOCK_N)
    inp_ptrs = inp + pid_m * stride_m + n_offset * stride_n + pid_k * stride_k
    inp_vals = tl.load(inp_ptrs, mask=n_offset < N)
    result = tl.cumsum(inp_vals)
    out_ptrs = out + pid_m * stride_m + n_offset * stride_n + pid_k * stride_k
    tl.store(out_ptrs, result, mask=n_offset < N)


def cumsum(inp, dim=1, *, dtype=None, out=None):
    print("FLAG CUMSUM")
    shape = inp.shape
    M = 1
    N = shape[dim]
    for i in range(dim):
        M *= shape[i]
    inp = inp.contiguous()
    inp = inp.reshape(M, N, -1)
    K = inp.numel() // M // N

    if dtype is None:
        dtype = inp.dtype
    if out is None:
        out = torch.empty_like(inp, dtype=dtype)

    grid = (
        M,
        K,
    )
    cumsum_kernel[grid](inp, out, N, inp.stride(0), inp.stride(1), inp.stride(2))
    out = out.reshape(shape)
    return out

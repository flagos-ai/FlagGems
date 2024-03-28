import torch
import triton
import triton.language as tl
from .__libentry__ import libentry
import math


@libentry()
@triton.heuristics(
    values={"BLOCK_M": lambda args: triton.next_power_of_2(math.ceil(math.sqrt(args["M"])))},
)
@triton.jit
def mean_kernel(
    inp,
    out,
    M,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=0.0)
    div_val = inp_val / M
    sum_val = tl.sum(div_val, axis=0).to(tl.float32)
    tl.atomic_add(out, sum_val)
    

def mean(inp, *, dtype=None):
    if __debug__:
        print("GEMS MEAN")
    M = inp.numel()
    if dtype is None:
        dtype = inp.dtype
    out = torch.zeros(1, dtype=torch.float32, device=inp.device)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        1, 1
    )
    mean_kernel[grid](inp, out, M)
    return out.to(dtype)

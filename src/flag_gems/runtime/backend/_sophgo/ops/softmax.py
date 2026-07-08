import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle

from flag_gems.ops.softmax import softmax as generic_softmax


def softmax_inner_tile_n(args):
    return triton.next_power_of_2(args["N"])


def softmax_inner_tile_m(args):
    return max(1, 512 // args["TILE_N"])


def softmax_inner_num_warps(args):
    return 4


@triton.heuristics(
    values={
        "TILE_N": softmax_inner_tile_n,
        "TILE_M": softmax_inner_tile_m,
        "num_warps": softmax_inner_num_warps,
    }
)
@triton.jit
def softmax_kernel_inner_multirow(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offsets = tl.arange(0, TILE_N)
    row_mask = m_offsets < M
    offsets = m_offsets[:, None] * N + n_offsets[None, :]
    mask = row_mask[:, None] & (n_offsets[None, :] < N)

    inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    m = tl.max(inp, axis=1)
    safe_m = tl.where(row_mask, m, 0.0)
    e = tl.exp(inp - safe_m[:, None])
    e = tl.where(mask, e, 0.0)
    z = tl.sum(e, axis=1)
    z = tl.where(row_mask, z, 1.0)
    out = e / z[:, None]
    tl.store(output_ptr + offsets, out, mask=mask)


def softmax(self, dim, half_to_float=False):
    logging.debug("GEMS SOFTMAX (SOPHGO)")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim

    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    K = self.numel() // M // N

    if K != 1 or N > 128:
        # The generic softmax path currently depends on CUDA-derived heuristics
        # in the new migration environment. Keep the validated narrow fast path
        # below, and use a device-side decomposition for broader cases.
        input_tensor = self.to(torch.float32) if half_to_float else self
        max_vals = torch.amax(input_tensor, dim=dim, keepdim=True)
        exp_vals = torch.exp(input_tensor - max_vals)
        denom = torch.sum(exp_vals, dim=dim, keepdim=True)
        return exp_vals / denom

    input_tensor = self.contiguous()
    out_dtype = torch.float32 if half_to_float else input_tensor.dtype
    out = torch.empty_like(input_tensor, dtype=out_dtype)

    grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
    with torch_device_fn.device(input_tensor.device):
        softmax_kernel_inner_multirow[grid](
            out,
            input_tensor,
            M,
            N,
        )
    return out

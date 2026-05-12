import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


# Filter (TILE_N, num_warps) pairs so each warp has at least 32 lanes.
# Drops gross over-subscription (num_warps * 32 > TILE_N) which leaves most
# lanes idle on tiny ONE_TILE_PER_CTA launches.
_INNER_CONFIGS = [
    triton.Config({"TILE_N": tile_n}, num_warps=num_warps)
    for tile_n in (64, 128, 256, 512, 1024)
    for num_warps in (1, 2, 4, 8, 16)
    if num_warps * 32 <= tile_n
]


def _one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@triton.jit
def _prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 256,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # TODO(chenfeiyu): consider float64 add add a utility function to get accumulator type
    m = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
    z = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(inp, m)
        all_neg_inf = m_new == float("-inf")
        z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
        m = m_new

    m_reduced = tl.max(m, 1)
    z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
    m = m_reduced

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        o = inp - m[:, None] - tl.log(z[:, None])
        tl.store(output_ptr + offset, o, mask=mask)


@libentry()
@triton.autotune(configs=_INNER_CONFIGS, key=["M", "N"])
@triton.heuristics({"ONE_TILE_PER_CTA": _one_tile_per_cta})
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        mask = n_offsets < N
        inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = inp - m - tl.log(z)
        tl.store(output_ptr + offset, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        # Pass 1: mask-free hot loop + masked tail
        previous_multiple = _prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offset = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offset)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        for start_n in range(previous_multiple, N, TILE_N):
            n_offset = start_n + tl.arange(0, TILE_N)
            mask = n_offset < N
            inp = tl.load(input_ptr + n_offset, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        log_z = tl.log(z)

        # Pass 2: reverse traversal with eviction hints
        previous_multiple = _prev_multiple_of(N, TILE_N)
        for start_n in range(0, TILE_N, TILE_N):
            n_offset = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offset < N
            inp = tl.load(
                input_ptr + n_offset,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = inp - m - log_z
            tl.store(output_ptr + n_offset, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offset = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offset, eviction_policy="evict_first")
            o = inp - m - log_z
            tl.store(output_ptr + n_offset, o)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    scale = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


@libentry()
@triton.autotune(configs=runtime.get_tuned_config("log_softmax"), key=["M", "N"])
@triton.jit
def log_softmax_backward_kernel_opt(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    scale = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N + n_offset[None, :]
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N + n_offset[None, :]
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(inp, dtype=dtype)
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K == 1:
            grid = (M, 1, 1)
            log_softmax_kernel_inner[grid](out, inp, M, N)
        else:
            grid = lambda meta: (
                triton.cdiv(M, meta["BLOCK_M"]),
                K,
            )
            log_softmax_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
                num_warps=16,
            )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS LOG_SOFTMAX VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    in_grad = torch.empty_like(output, dtype=input_dtype)
    K = output.numel() // M // N
    if K == 1:
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        with torch_device_fn.device(in_grad.device):
            log_softmax_backward_kernel_opt[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
            )
    else:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            K,
        )
        with torch_device_fn.device(in_grad.device):
            log_softmax_backward_kernel[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                K,
            )
    return in_grad

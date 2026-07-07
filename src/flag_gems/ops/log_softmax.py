import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# Self-contained heuristics for the inner log_softmax kernel. These intentionally
# do NOT reuse runtime.get_heuristic_config("softmax_inner"): that config name is
# redefined per-backend with different signatures (e.g. Cambricon exposes only
# TILE_MODE, not TILE_N/ONE_TILE_PER_CTA), so sharing it would couple this kernel
# to whatever a backend happens to export. Defining them here keeps the kernel
# self-contained and correct on every backend that falls through to this generic
# implementation.
def _log_softmax_inner_tile_n(args):
    if args["N"] <= (32 * 1024):
        return triton.next_power_of_2(args["N"])
    return 4096


def _log_softmax_inner_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def _log_softmax_inner_num_warps(args):
    tile_size = args["TILE_N"]
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    return 16


_LOG_SOFTMAX_INNER_HEURISTICS = {
    "TILE_N": _log_softmax_inner_tile_n,
    "ONE_TILE_PER_CTA": _log_softmax_inner_one_tile_per_cta,
    "num_warps": _log_softmax_inner_num_warps,
}


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def log_softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
        mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
            tl.float32
        )
        m = tl.max(inp, 0)
        z = tl.sum(tl.exp(inp - m[None, :]), 0)
        out = inp - m[None, :] - tl.log(z)[None, :]
        tl.store(output_ptr + offsets, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(inp, m)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
        m = m_reduced
        log_z = tl.log(z)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets[None, :]
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            out = inp - m[None, :] - log_z[None, :]
            tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@triton.heuristics(_LOG_SOFTMAX_INNER_HEURISTICS)
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
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = inp - m - tl.log(z)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = tl.cdiv(N, TILE_N) * TILE_N - TILE_N
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets).to(tl.float32)
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced
        log_z = tl.log(z)

        previous_multiple = tl.cdiv(N, TILE_N) * TILE_N - TILE_N
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            ).to(tl.float32)
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first").to(
                tl.float32
            )
            o = inp - m - log_z
            tl.store(output_ptr + n_offsets, o)


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
        mask = m_offset[:, None] < M and n_offset[None, :] < N
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
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        o = inp - m[:, None] - tl.log(z[:, None])
        tl.store(output_ptr + offset, o, mask=mask)


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
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        scale += out_grad
    scale = tl.sum(scale, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS LOG_SOFTMAX_OUT")

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
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    K = inp.numel() // M // N

    with torch_device_fn.device(inp.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            log_softmax_kernel_non_inner[grid](
                out,
                inp,
                M,
                N,
                K,
            )
        else:
            grid = (M, 1, 1)
            log_softmax_kernel_inner[grid](
                out,
                inp,
                M,
                N,
            )
    return out


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS LOG_SOFTMAX")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self.contiguous(), dtype=dtype)
    return log_softmax_out(self, dim, half_to_float, out=out)


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS LOG_SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    if out.dtype != input_dtype:
        raise RuntimeError(
            f"_log_softmax_backward_data.out: expected out dtype {input_dtype}, got {out.dtype}"
        )
    K = output.numel() // M // N

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    with torch_device_fn.device(out.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            out,
            M,
            N,
            K,
        )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS LOG_SOFTMAX_BACKWARD")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return log_softmax_backward_out(grad_output, output, dim, input_dtype, out=in_grad)

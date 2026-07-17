import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


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
    pid_k = ext.program_id(1)
    pid_m = ext.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N * K + n_offsets[:, None] * K + k_offsets
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = inp - m[None, :] - tl.log(z)[None, :]
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * N * K + n_offsets[:, None] * K + k_offsets
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            o = inp - m[None, :] - tl.log(z)[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


def log_softmax_heur_tile_m(args):
    """Heuristic for TILE_M in inner kernel."""
    M = args["M"]
    N = args["N"]
    if N <= 256:
        # For small N, process multiple rows
        if M >= 4096:
            return 8
        elif M >= 1024:
            return 4
        else:
            return 1
    elif N <= 1024:
        # For medium N
        if M >= 4096:
            return 4
        elif M >= 1024:
            return 2
        else:
            return 1
    else:
        return 1


def log_softmax_heur_tile_n_inner(args):
    """Heuristic for TILE_N in inner kernel."""
    N = args["N"]
    M = args["M"]
    if N <= (32 * 1024):
        tile_n = triton.next_power_of_2(N)
        # For very small N, we might want larger TILE_N
        if N <= 32 and M > 1000:
            return 32
        # For medium-large N where we process 1 row per CTA,
        # use smaller TILE_N to enable loop for better register usage
        if N > 1024 and N <= 8192:
            return min(tile_n, 2048)
        return tile_n
    else:
        return 4096


def log_softmax_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def log_softmax_heur_num_warps_inner(args):
    tile_m = args["TILE_M"]
    tile_n = args["TILE_N"]
    tile_size = tile_m * tile_n
    if tile_size < 2048:
        return 4
    elif tile_size < 4096:
        return 8
    else:
        return 16


@libentry()
@triton.heuristics(
    {
        "TILE_M": log_softmax_heur_tile_m,
        "TILE_N": log_softmax_heur_tile_n_inner,
        "ONE_TILE_PER_CTA": log_softmax_heur_one_tile_per_cta,
        "num_warps": log_softmax_heur_num_warps_inner,
    }
)
@triton.jit
def log_softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offset = pid_m * TILE_M + tl.arange(0, TILE_M)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = m_offset[:, None] * N + n_offsets[None, :]
        mask = (m_offset[:, None] < M) & (n_offsets[None, :] < N)
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 1)
        e = tl.exp(inp - m[:, None])
        z = tl.sum(e, 1)
        out = inp - m[:, None] - tl.log(z)[:, None]
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_M, TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_M, TILE_N], value=0.0, dtype=tl.float32)

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offset = m_offset[:, None] * N + n_offsets[None, :]
            mask = (m_offset[:, None] < M) & (n_offsets[None, :] < N)
            inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 1)
        z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
        m = m_reduced

        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offset = m_offset[:, None] * N + n_offsets[None, :]
            mask = (m_offset[:, None] < M) & (n_offsets[None, :] < N)
            inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(
                tl.float32
            )
            out = inp - m[:, None] - tl.log(z)[:, None]
            tl.store(output_ptr + offset, out, mask=mask)


# ------------------------  backward -------------------------------
@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_non_inner"),
    key=[
        "M",
        "N",
        "K",
    ],
)
@triton.heuristics(runtime.get_heuristic_config("softmax_backward_non_inner"))
@triton.jit
def log_softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_grad_tile, axis=0)
        in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * K
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * N * K + offsets_n[:, None] * K + offsets_k
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[None, :]
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * K


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_inner"),
    key=["M", "N"],
)
@triton.heuristics(
    values=runtime.get_heuristic_config("softmax_backward_inner"),
)
@triton.jit
def log_softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_grad_tile, 1)
        in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            scale += out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            ).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_grad_tile - tl.exp(out_tile) * scale[:, None]
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


def log_softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_MTHREADS LOG_SOFTMAX_OUT")

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
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            log_softmax_kernel_inner[grid](
                out,
                inp,
                M,
                N,
            )
    return out


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_MTHREADS LOG_SOFTMAX")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self.contiguous(), dtype=dtype)
    return log_softmax_out(self, dim, half_to_float, out=out)


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_MTHREADS LOG_SOFTMAX_BACKWARD_OUT")

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

    with torch_device_fn.device(out.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            log_softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                out,
                M,
                N,
                K,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            log_softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                out,
                M,
                N,
            )
    return out


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_MTHREADS LOG_SOFTMAX_BACKWARD")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return log_softmax_backward_out(grad_output, output, dim, input_dtype, out=in_grad)

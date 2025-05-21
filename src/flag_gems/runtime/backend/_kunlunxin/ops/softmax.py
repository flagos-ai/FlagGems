import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cdiv(a, b) * b


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_inner"))
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * N
        output_ptr += pid_m * N

        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            m_new = tl.maximum(m, inp)
            # it is possible that there are -inf's in the input
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new
        # specialize the last iteration
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)
        z = tl.sum(z * tl.exp(m - m_reduced), 0)
        m = m_reduced

        previous_multiple = prev_multiple_of(N, TILE_N)
        # specialize the first iteration
        for start_n in range(0, TILE_N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(
                input_ptr + n_offsets,
                mask=mask,
                other=-float("inf"),
                eviction_policy="evict_first",
            )
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)
        for start_n in range(TILE_N, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets, eviction_policy="evict_first")
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)


# ------------------------  backward -------------------------------


def softmax_backward_kernel_inner_heur_tile_m(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def softmax_backward_kernel_inner_heru_tile_n(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


def softmax_backward_kernel_inner_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@libentry()
# @triton.autotune(
#     configs=runtime.get_tuned_config("softmax_inner"),
#     key=["M", "N"],
# )
# @triton.heuristics(
#     values=runtime.get_heuristic_config("softmax_backward_inner"),
# )
@triton.heuristics(
    values={
        "TILE_M": softmax_backward_kernel_inner_heur_tile_m,
        "TILE_N": softmax_backward_kernel_inner_heru_tile_n,
        "ONE_TILE_PER_CTA": softmax_backward_kernel_inner_heur_one_tile_per_cta,
    },
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = tle.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            )
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        logger.debug("GEMS SOFTMAX")

        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        dim = dim % x.ndim
        M = 1
        N = x.shape[dim]
        for i in range(dim):
            M *= x.shape[i]  # pre_dim
        inp = x.contiguous()
        if dtype is None:
            dtype = x.dtype
        out = torch.empty_like(inp, dtype=dtype)
        K = inp.numel() // M // N  # post_dim

        with torch_device_fn.device(inp.device):
            if K > 1:
                # 重新排列输入数据为 [M, K, N]
                inp_view = inp.view(M, N, K).transpose(1, 2).contiguous()
                # 合并 M 和 K 维为 M' = M * K
                inp_reshaped = inp_view.view(M * K, N)
                # 分配输出的视图
                out_view = out.view(M, N, K).transpose(1, 2).contiguous()
                out_reshaped = out_view.view(M * K, N)

                grid = lambda meta: (M * K, 1, 1)

                # 调用 Triton 前向内核
                softmax_kernel_inner[grid](out_reshaped, inp_reshaped, M * K, N)

                # 将输出恢复到原始布局
                out_view.copy_(out_reshaped.view(M, K, N).transpose(1, 2))
            else:
                grid = (M, 1, 1)
                softmax_kernel_inner[grid](
                    out,
                    inp,
                    M,
                    N,
                )
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS SOFTMAX VJP")
        dim = ctx.dim
        (out,) = ctx.saved_tensors

        assert dim >= -out.ndim and dim < out.ndim, "Invalid dim"
        dim = dim % out.ndim
        M = 1
        N = out.shape[dim]
        for i in range(dim):
            M *= out.shape[i]

        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        K = out.numel() // M // N

        with torch_device_fn.device(in_grad.device):
            if K > 1:
                # how to use softmax_backward_kernel_inner?
                # some transpose and continuous
                out_grad_view = out_grad.view(M, N, K).transpose(1, 2).contiguous()
                out_view = out.view(M, N, K).transpose(1, 2).contiguous()
                # 合并 M 和 K 维为 M' = M * K
                out_grad_reshaped = out_grad_view.view(M * K, N)
                out_reshaped = out_view.view(M * K, N)
                # 分配输入梯度的视图
                in_grad_view = in_grad.view(M, N, K).transpose(1, 2).contiguous()
                in_grad_reshaped = in_grad_view.view(M * K, N)

                grid = lambda meta: (triton.cdiv(M * K, meta["TILE_M"]), 1, 1)

                # 调用 Triton 反向内核
                softmax_backward_kernel_inner[grid](
                    out_reshaped, out_grad_reshaped, in_grad_reshaped, M * K, N
                )

                # 将输入梯度恢复到原始布局
                in_grad_view.copy_(in_grad_reshaped.view(M, K, N).transpose(1, 2))
            else:
                grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
                softmax_backward_kernel_inner[grid](
                    out,
                    out_grad,
                    in_grad,
                    M,
                    N,
                )
        return in_grad, None, None


def softmax(x, dim=-1, dtype=None):
    return Softmax.apply(x, dim, dtype)

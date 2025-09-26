import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


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
    return triton.cdiv(args["M"], 12)  # cluster_num
    # return triton.next_power_of_2(triton.cdiv(args["M"], 12))


def softmax_backward_kernel_inner_heru_tile_n(args):
    import builtins

    return builtins.min(args["N"], 4096)
    # return builtins.min(triton.next_power_of_2(args["N"]), 8192)


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
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float64)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float64)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float64)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            ).to(tl.float64)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float64)
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
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask).to(tl.float64)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None]).to(tl.float64)
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]  # pre_dim
    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype
    out = torch.empty_like(self, dtype=dtype)
    K = self.numel() // M // N  # post_dim

    with torch_device_fn.device(self.device):
        if K > 1:
            # grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            # 重新排列输入数据为 [M, K, N]
            inp_view = self.view(M, N, K).transpose(1, 2).contiguous()
            # 合并 M 和 K 维为 M' = M * K
            inp_reshaped = inp_view.view(M * K, N)
            if out.ndim == 3:
                m, n, k = out.shape
            elif out.ndim == 2:
                m, n = out.shape
            origin_dim = out.ndim

            # 分配输出的视图
            out_view = out.view(M, N, K).transpose(1, 2).contiguous()
            out_reshaped = out_view.view(M * K, N)

            grid = lambda meta: (M * K, 1, 1)

            # 调用 Triton 前向内核
            softmax_kernel_inner[grid](
                out_reshaped,
                inp_reshaped,
                M * K,
                N,
                buffer_size_limit=2048,
            )

            # 将输出恢复到原始布局
            # out_view.copy_(out_reshaped.view(M, K, N).transpose(1, 2))
            if M == 1 and origin_dim == 2:
                out = out_reshaped.view(K, N).transpose(0, 1)
            elif M == 1 and origin_dim == 3:
                out = out_reshaped.transpose(0, 1).view(m, n, k)
            else:
                out = out_reshaped.view(m, k, n).transpose(1, 2)
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
                buffer_size_limit=2048,
                isCloseVectorization=True,
            )
    return out


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS SOFTMAX VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=torch.float64)
    K = output.numel() // M // N

    with torch_device_fn.device(in_grad.device):
        if K > 1:
            # how to use softmax_backward_kernel_inner?
            # some transpose and continuous
            out_grad_view = grad_output.view(M, N, K).transpose(1, 2).contiguous()
            out_view = output.view(M, N, K).transpose(1, 2).contiguous()
            # # 合并 M 和 K 维为 M' = M * K
            out_grad_reshaped = out_grad_view.view(M * K, N)
            out_reshaped = out_view.view(M * K, N)
            # 分配输入梯度的视图
            in_grad_view = in_grad.view(M, N, K).transpose(1, 2).contiguous()
            in_grad_reshaped = in_grad_view.view(M * K, N)

            grid = lambda meta: (12, 1, 1)

            # 调用 Triton 反向内核
            softmax_backward_kernel_inner[grid](
                out_reshaped,
                out_grad_reshaped,
                in_grad_reshaped,
                M * K,
                N,
                buffer_size_limit=2048,
                isCloseUnrollControl=True,
            )
            # 将输入梯度恢复到原始布局
            # in_grad_view.copy_(in_grad_reshaped.view(M, K, N).transpose(1, 2))
            origin_dim = output.ndim
            if output.ndim == 3:
                m, n, k = output.shape
            elif output.ndim == 2:
                m, n = output.shape
            if M == 1 and origin_dim == 2:
                in_grad = in_grad_reshaped.view(K, N).transpose(0, 1)
            elif M == 1 and origin_dim == 3:
                in_grad = in_grad_reshaped.transpose(0, 1).view(m, n, k)
            else:
                in_grad = in_grad_reshaped.view(m, k, n).transpose(1, 2)
        else:
            grid = lambda meta: (12, 1, 1)

            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                buffer_size_limit=2048,
                isCloseUnrollControl=True,
            )
    return in_grad.to(input_dtype)

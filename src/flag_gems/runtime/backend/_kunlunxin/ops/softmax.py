# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

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
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        # Pre-offset the base pointers so the inner `ptr + n_offsets` access is a
        # scalar-base + stride-1 arange that OffsetAnalysis proves contiguous
        # (block DMA). The old inline `pid_m * N + n_offsets` addressing blocked
        # the analysis -> discrete scalar gather (~1-3 GB/s, e.g. [4096,4096] took
        # ~37ms). Pre-offsetting drops it to ~1.1ms (~35x).
        input_ptr += pid_m * N
        output_ptr += pid_m * N
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf")).to(
            output_ptr.dtype.element_ty
        )
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = e / z
        tl.store(output_ptr + n_offsets, out, mask=mask)
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

        # Normalize pass. Iterate ASCENDING so each `input_ptr + n_offsets` load
        # and `output_ptr + n_offsets` store is a scalar-base + stride-1 arange
        # (block DMA). The old code walked the tiles DESCENDING
        # (`previous_multiple - start_n`) as a cache-locality trick, but on this
        # XPU the backward walk defeats OffsetAnalysis/prefetch -> discrete access
        # (~1-3 GB/s: [1024,65536] took ~154ms). Ascending drops it to ~4ms (~35x).
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            inp = tl.load(input_ptr + n_offsets)
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            inp = tl.load(input_ptr + n_offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m) / z
            tl.store(output_ptr + n_offsets, o, mask=mask)


# ------------------------  backward -------------------------------


def softmax_backward_kernel_inner_heru_tile_n(args):
    N = args["N"]
    if N <= 32768:
        return triton.next_power_of_2(N)
    return 4096


def softmax_backward_kernel_inner_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


@libentry()
@triton.heuristics(
    values={
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
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # One program per row (grid=(M,)), mirroring the forward. Pre-offset the base
    # pointers so the inner `ptr + n_offsets` access is a scalar-base + stride-1
    # arange that OffsetAnalysis proves contiguous (block DMA). The old impl used a
    # fixed grid=(12,) with a [TILE_M, TILE_N] tile whose `m_offsets[:,None]*N +
    # n_offsets` addressing blocked the analysis -> discrete scalar gather
    # (~1-3 GB/s: [4096,4096] took ~38ms). It also computed in float64 (2x traffic,
    # unnecessary). float32 accumulation matches the forward and the generic backend.
    pid_m = ext.program_id(0)
    out_ptr += pid_m * N
    out_grad_ptr += pid_m * N
    in_grad_ptr += pid_m * N
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.sum(out_tile * out_grad_tile, 0)
        in_grad_tile = out_tile * (out_grad_tile - scale)
        tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)
    else:
        # Pass 1: accumulate scale = sum(out * out_grad) over the row. Iterate
        # ASCENDING so each load is a scalar-base + stride-1 arange (block DMA).
        scale = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            scale += out_tile * out_grad_tile
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale += out_tile * out_grad_tile
        scale = tl.sum(scale, 0)  # scalar

        # Pass 2: write in_grad = out * (out_grad - scale), ASCENDING.
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            out_tile = tl.load(out_ptr + n_offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + n_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + n_offsets, in_grad_tile, mask=mask)


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    # special handling for dim = 0 and empty tensor
    if self.numel() == 0:
        out_shape = list(self.shape)
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        zero_(out)
        return out

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
    K = self.numel() // M // N  # post_dim

    with torch_device_fn.device(self.device):
        if K > 1:
            origin_dim = self.ndim
            if origin_dim == 3:
                m, n, k = self.shape
            else:  # 2D, dim == 0 -> M == 1
                n, k = self.shape
                m = 1
            # Rearrange [M, N, K] -> [M, K, N] so the reduced dim N is innermost
            # (the only fast axis on this XPU). Allocate the output tile directly
            # instead of `empty_like(self).view(...).transpose(...).contiguous()`,
            # which used to copy an uninitialized [M,K,N] buffer (a wasted
            # transpose-copy on top of the input transpose).
            inp_reshaped = (
                self.view(M, N, K).transpose(1, 2).contiguous().view(M * K, N)
            )
            out_reshaped = torch.empty((M * K, N), dtype=dtype, device=self.device)

            grid = lambda meta: (M * K, 1, 1)  # noqa: E731

            softmax_kernel_inner[grid](
                out_reshaped,
                inp_reshaped,
                M * K,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )

            # Restore original layout (returns a transposed view, no copy).
            if M == 1 and origin_dim == 2:
                out = out_reshaped.view(K, N).transpose(0, 1)
            elif M == 1 and origin_dim == 3:
                out = out_reshaped.transpose(0, 1).view(m, n, k)
            else:
                out = out_reshaped.view(m, k, n).transpose(1, 2)
        else:
            out = torch.empty_like(self, dtype=dtype)
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
                buffer_size_limit=2048,
                is_use_mask_zero=True,
            )
    return out


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    in_grad = torch.empty_like(output, dtype=torch.float32)
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

            grid = lambda meta: (M * K, 1, 1)  # noqa: E731

            # 调用 Triton 反向内核
            softmax_backward_kernel_inner[grid](
                out_reshaped,
                out_grad_reshaped,
                in_grad_reshaped,
                M * K,
                N,
                buffer_size_limit=2048,
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
            grid = lambda meta: (M, 1, 1)  # noqa: E731

            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                in_grad,
                M,
                N,
                buffer_size_limit=2048,
            )
    return in_grad.to(input_dtype)

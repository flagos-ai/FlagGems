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

# The P800 runtime cannot reliably execute the generic non-inner softmax
# preparation path: materializing the transposed view can dispatch to a broken
# ``copy_`` kernel, and the replacement inner kernel can still fail with an XPU
# launch exception. The same kernel has zero-length-DMA failures for small or
# masked-tail reductions, so those are covered by the fallback as well. Capture
# the native CUDA implementation before FlagGems installs its registration so
# the affected layouts can redispatch safely.
_NATIVE_SOFTMAX = torch.library.get_kernel("aten::_softmax", "CUDA")
_NATIVE_SOFTMAX_OUT = torch.library.get_kernel("aten::_softmax.out", "CUDA")
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
_SOFTMAX_NATIVE_MIN_N = 64


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
    K: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    # One program owns one logical [N] row at (m, k). For K == 1 the reduction
    # dimension is contiguous; for K > 1 the same one-dimensional tile walks N
    # with stride K and avoids materializing [M, N, K] -> [M, K, N].
    pid_mk = ext.program_id(0)
    pid_m = pid_mk // K
    pid_k = pid_mk % K
    row_offset = pid_m * N * K + pid_k
    out_ptr += row_offset
    out_grad_ptr += row_offset
    in_grad_ptr += row_offset
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        mask = n_offsets < N
        offsets = n_offsets * K
        out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        scale = tl.sum(out_tile * out_grad_tile, 0)
        in_grad_tile = out_tile * (out_grad_tile - scale)
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        # Pass 1: accumulate scale = sum(out * out_grad) over the row. Iterate
        # in ascending N order.
        scale = tl.zeros([TILE_N], dtype=tl.float32)
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = n_offsets * K
            out_tile = tl.load(out_ptr + offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets).to(tl.float32)
            scale += out_tile * out_grad_tile
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = n_offsets * K
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale += out_tile * out_grad_tile
        scale = tl.sum(scale, 0)  # scalar

        # Pass 2: write in_grad = out * (out_grad - scale), ASCENDING.
        for start_n in range(0, previous_multiple, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = n_offsets * K
            out_tile = tl.load(out_ptr + offsets).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + offsets, in_grad_tile)
        for start_n in range(previous_multiple, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = n_offsets * K
            mask = n_offsets < N
            out_tile = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            in_grad_tile = out_tile * (out_grad_tile - scale)
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)


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
    K = self.numel() // M // N  # post_dim
    if K > 1 or N < _SOFTMAX_NATIVE_MIN_N or (N & (N - 1)) != 0:
        return _NATIVE_SOFTMAX.call_boxed(_CUDA_KEYSET, self, dim, half_to_float)

    self = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    with torch_device_fn.device(self.device):
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


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_OUT")
    return _NATIVE_SOFTMAX_OUT.call_boxed(
        _CUDA_KEYSET, self, dim, half_to_float, out=out
    )


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            "_softmax_backward_data.out: expected grad_input dtype "
            f"{input_dtype}, got {grad_input.dtype}"
        )
    if output.numel() == 0:
        zero_(grad_input)
        return grad_input

    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    grad_output = grad_output.contiguous()
    output = output.contiguous()
    K = output.numel() // M // N
    kernel_grad_input = grad_input
    if not grad_input.is_contiguous():
        kernel_grad_input = torch.empty(
            output.shape, dtype=input_dtype, device=grad_input.device
        )

    with torch_device_fn.device(kernel_grad_input.device):
        grid = (M * K, 1, 1)
        softmax_backward_kernel_inner[grid](
            output,
            grad_output,
            kernel_grad_input,
            M,
            N,
            K,
            buffer_size_limit=2048,
        )
    if kernel_grad_input is not grad_input:
        grad_input.copy_(kernel_grad_input)
    return grad_input


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_KUNLUNXIN SOFTMAX_VJP")
    in_grad = torch.empty(output.shape, dtype=input_dtype, device=output.device)
    return softmax_backward_out(
        grad_output, output, dim, input_dtype, grad_input=in_grad
    )

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


@libentry()
@triton.heuristics(runtime.get_heuristic_config("softmax_non_inner"))
@triton.jit
def softmax_kernel_non_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    row_stride,
    n_stride,
    k_stride,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_k = ext.program_id(1)
    pid_m = ext.program_id(0)

    k_offsets = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * row_stride + n_offsets[:, None] * n_stride + k_offsets * k_stride
        mask = (n_offsets[:, None] < N) & (k_offsets < K)
        input_ptrs = input_ptr + offset
        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject
        # fp16/bf16 in tl.exp, and fp32 accumulation is more accurate anyway.
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m[None, :])
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N, TILE_K], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N, TILE_K], value=0.0, dtype=tl.float32)

        # specialization does not improve performance inn this example, as tested
        for start_n in range(0, N, TILE_N):
            n_offsets = start_n + tl.arange(0, TILE_N)
            offsets = pid_m * row_stride + n_offsets[:, None] * n_stride + k_offsets * k_stride
            mask = (n_offsets[:, None] < N) & (k_offsets < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, inp)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 0)  # (TILE_K,)
        z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)  # (TILE_K, )
        m = m_reduced

        # specialization does not improve performance inn this example, as tested
        previous_multiple = prev_multiple_of(N, TILE_N)
        for start_n in range(0, N, TILE_N):
            n_offsets = (previous_multiple - start_n) + tl.arange(0, TILE_N)
            offsets = pid_m * row_stride + n_offsets[:, None] * n_stride + k_offsets * k_stride
            mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
            inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[None, :]) / z[None, :]
            tl.store(output_ptr + offsets, o, mask=mask)


@triton.jit
def next_multiple_of(a, b):
    # the smallest x>=a that x%b ==0
    return tl.cidv(a, b) * b


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
    row_stride,
    col_stride,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = pid_m * row_stride + n_offsets * col_stride
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        # Reduce in fp32: some triton backends (e.g. Cambricon MLU) reject
        # fp16/bf16 in tl.exp, and fp32 accumulation is more accurate anyway.
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(inp, 0)
        e = tl.exp(inp - m)
        z = tl.sum(e, 0)
        out = (e / z).to(output_ptr.dtype.element_ty)
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_N], value=0.0, dtype=tl.float32)
        input_ptr += pid_m * row_stride
        output_ptr += pid_m * row_stride

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
def softmax_backward_kernel_non_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    row_stride,
    n_stride,
    k_stride,
    grad_row_stride,
    grad_n_stride,
    grad_k_stride,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    offsets_k = pid_k * TILE_K + tl.arange(0, TILE_K)

    if ONE_TILE_PER_CTA:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * row_stride + offsets_n[:, None] * n_stride + offsets_k * k_stride
        mask = (offsets_n < N)[:, None] & (offsets_k < K)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        grad_offsets = pid_m * grad_row_stride + offsets_n[:, None] * grad_n_stride + offsets_k * grad_k_stride
        out_grad_tile = tl.load(out_grad_ptr + grad_offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, axis=0)
        in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * row_stride + offsets_n[:, None] * n_stride + offsets_k * k_stride
        scale = tl.zeros([TILE_N, TILE_K], dtype=tl.float32)
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            grad_offsets = pid_m * grad_row_stride + offsets_n[:, None] * grad_n_stride + offsets_k * grad_k_stride
            out_grad_tile = tl.load(out_grad_ptr + grad_offsets, mask=mask).to(tl.float32)
            scale += out_tile * out_grad_tile
            offsets_n += TILE_N
            offsets += TILE_N * n_stride
        scale = tl.sum(scale, axis=0)  # (TILE_K)

        offsets_n = tl.arange(0, TILE_N)
        offsets = pid_m * row_stride + offsets_n[:, None] * n_stride + offsets_k * k_stride
        for _ in range(0, N, TILE_N):
            mask = (offsets_n < N)[:, None] & (offsets_k < K)
            out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
            grad_offsets = pid_m * grad_row_stride + offsets_n[:, None] * grad_n_stride + offsets_k * grad_k_stride
            out_grad_tile = tl.load(out_grad_ptr + grad_offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            offsets_n += TILE_N
            offsets += TILE_N * n_stride


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("softmax_inner"),
    key=["M", "N"],
)
@triton.heuristics(
    values=runtime.get_heuristic_config("softmax_backward_inner"),
)
@triton.jit
def softmax_backward_kernel_inner(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    row_stride,
    col_stride,
    grad_row_stride,
    grad_col_stride,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * row_stride + n_offsets * col_stride
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        out_tile = tl.load(out_ptr + offsets, mask=mask).to(tl.float32)
        grad_offsets = m_offsets[:, None] * grad_row_stride + n_offsets * grad_col_stride
        out_grad_tile = tl.load(out_grad_ptr + grad_offsets, mask=mask).to(tl.float32)
        scale = tl.sum(out_tile * out_grad_tile, 1)
        in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
        tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
    else:
        scale = tl.zeros([TILE_M, TILE_N], dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * row_stride + n_offsets * col_stride
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_last"
            ).to(tl.float32)
            grad_offsets = m_offsets[:, None] * grad_row_stride + n_offsets * grad_col_stride
            out_grad_tile = tl.load(out_grad_ptr + grad_offsets, mask=mask).to(tl.float32)
            scale += out_tile * out_grad_tile
            n_offsets += TILE_N
            offsets += TILE_N * col_stride
        scale = tl.sum(scale, 1)  # (TILE_M,)

        n_offsets = tl.arange(0, TILE_N)
        offsets = m_offsets[:, None] * row_stride + n_offsets * col_stride
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            out_tile = tl.load(
                out_ptr + offsets, mask=mask, eviction_policy="evict_first"
            ).to(tl.float32)
            grad_offsets = m_offsets[:, None] * grad_row_stride + n_offsets * grad_col_stride
            out_grad_tile = tl.load(out_grad_ptr + grad_offsets, mask=mask).to(tl.float32)
            in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
            tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
            n_offsets += TILE_N
            offsets += TILE_N


def softmax_out(self, dim, half_to_float=False, *, out):
    logger.debug("GEMS SOFTMAX_OUT")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        if tuple(out.shape) != tuple(self.shape):
            out.resize_(self.shape)
        zero_(out)
        return out

    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(self.shape):
        out.resize_(self.shape)
    if out.dtype != dtype:
        raise RuntimeError(f"_softmax.out: expected out dtype {dtype}, got {out.dtype}")
    K = self.numel() // M // N

    row_stride = self.stride(dim - 1) if dim > 0 else 1
    with torch_device_fn.device(self.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_kernel_non_inner[grid](
                out,
                self,
                M,
                N,
                K,
                row_stride,
                self.stride(dim),
                self.stride(dim + 1) if dim + 1 < self.ndim else 1,
            )
        else:
            grid = (M, 1, 1)
            softmax_kernel_inner[grid](
                out,
                self,
                M,
                N,
                row_stride,
                self.stride(dim),
            )
    return out


def softmax(self, dim, half_to_float=False):
    logger.debug("GEMS SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"

    if self.numel() == 0:
        out_shape = list(self.shape)
        out = torch.empty(out_shape, dtype=self.dtype, device=self.device)
        zero_(out)
        return out

    dtype = torch.float32 if half_to_float else self.dtype
    out = torch.empty_like(self, dtype=dtype)
    return softmax_out(self, dim, half_to_float, out=out)


def softmax_backward_out(grad_output, output, dim, input_dtype, *, grad_input):
    logger.debug("GEMS SOFTMAX_BACKWARD_OUT")

    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    if tuple(grad_input.shape) != tuple(output.shape):
        grad_input.resize_(output.shape)
    if grad_input.dtype != input_dtype:
        raise RuntimeError(
            f"_softmax_backward_data.out: expected grad_input dtype {input_dtype}, got {grad_input.dtype}"
        )
    K = output.numel() // M // N
    row_stride = output.stride(dim - 1) if dim > 0 else 1

    with torch_device_fn.device(grad_input.device):
        if K > 1:
            grid = lambda meta: (M, triton.cdiv(K, meta["TILE_K"]), 1)
            softmax_backward_kernel_non_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
                K,
                row_stride,
                output.stride(dim),
                output.stride(dim + 1) if dim + 1 < output.ndim else 1,
                grad_output.stride(dim - 1) if dim > 0 else 1,
                grad_output.stride(dim),
                grad_output.stride(dim + 1) if dim + 1 < grad_output.ndim else 1,
            )
        else:
            grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
            softmax_backward_kernel_inner[grid](
                output,
                grad_output,
                grad_input,
                M,
                N,
                row_stride,
                output.stride(dim),
                grad_output.stride(dim - 1) if dim > 0 else 1,
                grad_output.stride(dim),
            )
    return grad_input


def softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS SOFTMAX_BACKWARD")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return softmax_backward_out(
        grad_output, output, dim, input_dtype, grad_input=in_grad
    )

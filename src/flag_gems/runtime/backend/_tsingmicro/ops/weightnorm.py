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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

_ROW_BLOCK = 1024
_ELEM_BLOCK = 1024


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_first_kernel(
    output,
    norm,
    v,
    g,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for base in range(0, N, BLOCK_N):
        cols = base + offsets
        mask = cols < N
        vals = tl.load(v + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        acc += vals * vals

    norm_val = tl.sqrt(tl.sum(acc, axis=0) + eps)
    tl.store(norm + row, norm_val)
    g_val = tl.load(g + row).to(tl.float32)

    for base in range(0, N, BLOCK_N):
        cols = base + offsets
        mask = cols < N
        vals = tl.load(v + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(output + row * N + cols, vals / norm_val * g_val, mask=mask)


@libentry()
@triton.jit
def weight_norm_first_bwd_kernel(
    v_grad,
    g_grad,
    w_grad,
    v,
    g,
    norm,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for base in range(0, N, BLOCK_N):
        cols = base + offsets
        mask = cols < N
        v_val = tl.load(v + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(w_grad + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        acc += v_val * w_val

    vw_sum = tl.sum(acc, axis=0)
    norm_val = tl.load(norm + row).to(tl.float32)
    g_val = tl.load(g + row).to(tl.float32)
    inv_norm = 1.0 / norm_val
    tl.store(g_grad + row, vw_sum * inv_norm)

    for base in range(0, N, BLOCK_N):
        cols = base + offsets
        mask = cols < N
        v_val = tl.load(v + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(w_grad + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        out = g_val * (
            w_val * inv_norm - v_val * inv_norm * inv_norm * inv_norm * vw_sum
        )
        tl.store(v_grad + row * N + cols, out, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_last_partial_sum_kernel(
    v,
    partial,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M
    vals = tl.load(v + rows * N + pid_n, mask=mask, other=0.0).to(tl.float32)
    sq_sum = tl.sum(vals * vals, axis=0)
    tl.store(partial + pid_n * tl.num_programs(0) + pid_m, sq_sum)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def weight_norm_last_norm_kernel(
    partial,
    norm,
    PARTIAL_M: tl.constexpr,
    BLOCK_PARTIAL: tl.constexpr,
    eps,
):
    pid_n = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_PARTIAL)
    mask = offsets < PARTIAL_M
    vals = tl.load(partial + pid_n * PARTIAL_M + offsets, mask=mask, other=0.0)
    sum_sq = tl.sum(vals, axis=0)
    tl.store(norm + pid_n, tl.sqrt(sum_sq + eps))


@libentry()
@triton.jit
def weight_norm_last_output_kernel(
    output,
    v,
    g,
    norm,
    total,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    col = offsets % N
    v_val = tl.load(v + offsets, mask=mask, other=0.0).to(tl.float32)
    g_val = tl.load(g + col, mask=mask, other=0.0).to(tl.float32)
    norm_val = tl.load(norm + col, mask=mask, other=1.0).to(tl.float32)
    tl.store(output + offsets, v_val / norm_val * g_val, mask=mask)


@libentry()
@triton.jit
def weight_norm_last_bwd_partial_sum_kernel(
    w_grad,
    v,
    partial,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M
    offsets = rows * N + pid_n
    v_val = tl.load(v + offsets, mask=mask, other=0.0).to(tl.float32)
    w_val = tl.load(w_grad + offsets, mask=mask, other=0.0).to(tl.float32)
    vw_sum = tl.sum(v_val * w_val, axis=0)
    tl.store(partial + pid_n * tl.num_programs(0) + pid_m, vw_sum)


@libentry()
@triton.jit
def weight_norm_last_bwd_reduce_kernel(
    partial,
    g_grad,
    norm,
    PARTIAL_M: tl.constexpr,
    BLOCK_PARTIAL: tl.constexpr,
):
    pid_n = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_PARTIAL)
    mask = offsets < PARTIAL_M
    vals = tl.load(partial + pid_n * PARTIAL_M + offsets, mask=mask, other=0.0)
    vw_sum = tl.sum(vals, axis=0)
    norm_val = tl.load(norm + pid_n).to(tl.float32)
    tl.store(g_grad + pid_n, vw_sum / norm_val)


@libentry()
@triton.jit
def weight_norm_last_bwd_v_kernel(
    v_grad,
    w_grad,
    v,
    g,
    norm,
    g_grad,
    total,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    col = offsets % N
    w_val = tl.load(w_grad + offsets, mask=mask, other=0.0).to(tl.float32)
    v_val = tl.load(v + offsets, mask=mask, other=0.0).to(tl.float32)
    g_val = tl.load(g + col, mask=mask, other=0.0).to(tl.float32)
    norm_val = tl.load(norm + col, mask=mask, other=1.0).to(tl.float32)
    vw_sum = tl.load(g_grad + col, mask=mask, other=0.0).to(tl.float32) * norm_val
    inv_norm = 1.0 / norm_val
    out = g_val * (w_val * inv_norm - v_val * inv_norm * inv_norm * inv_norm * vw_sum)
    tl.store(v_grad + offsets, out, mask=mask)


def _next_power_of_2(x):
    return 1 if x <= 1 else triton.next_power_of_2(x)


def _block_n(n):
    return min(_ELEM_BLOCK, _next_power_of_2(n))


def _weight_norm_interface_first(v, g, output, norm, m, n):
    eps = torch.finfo(torch.float32).tiny
    weight_norm_first_kernel[(m,)](
        output, norm, v, g, m, n, eps=eps, BLOCK_N=_block_n(n)
    )


def _weight_norm_interface_last(v, g, output, norm, m, n):
    partial_m = triton.cdiv(m, _ROW_BLOCK)
    partial_block = _next_power_of_2(partial_m)
    partial = torch.empty((n, partial_m), dtype=torch.float32, device=v.device)
    eps = torch.finfo(torch.float32).tiny

    weight_norm_last_partial_sum_kernel[(partial_m, n)](
        v, partial, m, n, eps=eps, BLOCK_M=_ROW_BLOCK
    )
    weight_norm_last_norm_kernel[(n,)](partial, norm, partial_m, partial_block, eps=eps)
    total = v.numel()
    grid = (triton.cdiv(total, _ELEM_BLOCK),)
    weight_norm_last_output_kernel[grid](
        output, v, g, norm, total, n, BLOCK_SIZE=_ELEM_BLOCK
    )


def _weight_norm_interface_backward_last(w_grad, v, g, norm, v_grad, g_grad, m, n):
    partial_m = triton.cdiv(m, _ROW_BLOCK)
    partial_block = _next_power_of_2(partial_m)
    partial = torch.empty((n, partial_m), dtype=torch.float32, device=v.device)

    weight_norm_last_bwd_partial_sum_kernel[(partial_m, n)](
        w_grad, v, partial, m, n, BLOCK_M=_ROW_BLOCK
    )
    weight_norm_last_bwd_reduce_kernel[(n,)](
        partial, g_grad, norm, partial_m, partial_block
    )
    total = v.numel()
    grid = (triton.cdiv(total, _ELEM_BLOCK),)
    weight_norm_last_bwd_v_kernel[grid](
        v_grad,
        w_grad,
        v,
        g,
        norm,
        g_grad,
        total,
        n,
        BLOCK_SIZE=_ELEM_BLOCK,
    )


def _weight_norm_interface_backward_first(w_grad, v, g, norm, v_grad, g_grad, m, n):
    weight_norm_first_bwd_kernel[(m,)](
        v_grad, g_grad, w_grad, v, g, norm, m, n, BLOCK_N=_block_n(n)
    )


def weight_norm_interface(v, g, dim=0):
    logger.debug("GEMS_TSINGMICRO WEIGHT_NORM_INTERFACE")
    dim = dim % v.ndim
    assert (
        dim == 0 or dim == v.ndim - 1
    ), "weight_norm_interface only supports first or last dim"

    v = v.contiguous()
    g = g.contiguous()
    output = torch.empty_like(v)
    norm = torch.empty_like(g, dtype=torch.float32)
    m = math.prod(v.shape[:-1])
    n = v.shape[-1]

    with torch_device_fn.device(v.device):
        if dim == 0:
            m = v.shape[0]
            n = math.prod(v.shape[1:])
            _weight_norm_interface_first(v, g, output, norm, m, n)
        else:
            _weight_norm_interface_last(v, g, output, norm, m, n)
    return output, norm


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_TSINGMICRO WEIGHT_NORM_INTERFACE_BACKWARD")
    dim = dim % saved_v.ndim
    assert (
        dim == 0 or dim == saved_v.ndim - 1
    ), "weight_norm_interface_backward only supports first or last dim"

    w_grad = w_grad.contiguous()
    saved_v = saved_v.contiguous()
    saved_g = saved_g.contiguous()
    saved_norms = saved_norms.contiguous()
    v_grad = torch.empty_like(saved_v)
    g_grad = torch.empty_like(saved_g)
    m = math.prod(saved_v.shape[:-1])
    n = saved_v.shape[-1]

    with torch_device_fn.device(saved_v.device):
        if dim == 0:
            m = saved_v.shape[0]
            n = math.prod(saved_v.shape[1:])
            _weight_norm_interface_backward_first(
                w_grad, saved_v, saved_g, saved_norms, v_grad, g_grad, m, n
            )
        else:
            _weight_norm_interface_backward_last(
                w_grad, saved_v, saved_g, saved_norms, v_grad, g_grad, m, n
            )
    return v_grad, g_grad

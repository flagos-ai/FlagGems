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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _weight_norm_backward_first_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    M,
    N,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    groups = ext.program_id(0) * BLOCK_GROUP + tl.arange(0, BLOCK_GROUP)[:, None]
    group_mask = groups < M
    cols = tl.arange(0, BLOCK_REDUCE)[None, :]

    products = tl.zeros((BLOCK_GROUP, BLOCK_REDUCE), dtype=tl.float32)
    for start in range(0, N, BLOCK_REDUCE):
        offsets = groups * N + start + cols
        mask = group_mask & (start + cols < N)
        v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
        products += v * w
    dot = tl.sum(products, axis=1)[:, None]

    g = tl.load(saved_g + groups, mask=group_mask, other=0.0).to(tl.float32)
    norm = tl.load(saved_norms + groups, mask=group_mask, other=1.0).to(tl.float32)
    scale = g / norm
    projection = dot / (norm * norm)

    for start in range(0, N, BLOCK_REDUCE):
        offsets = groups * N + start + cols
        mask = group_mask & (start + cols < N)
        v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_v + offsets, scale * (w - v * projection), mask=mask)
    tl.store(grad_g + groups, dot / norm, mask=group_mask)


@libentry()
@triton.jit
def _weight_norm_backward_last_kernel(
    grad_v,
    grad_g,
    grad_w,
    saved_v,
    saved_g,
    saved_norms,
    M,
    N,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    groups = ext.program_id(0) * BLOCK_GROUP + tl.arange(0, BLOCK_GROUP)[:, None]
    group_mask = groups < N
    rows = tl.arange(0, BLOCK_REDUCE)[None, :]

    products = tl.zeros((BLOCK_GROUP, BLOCK_REDUCE), dtype=tl.float32)
    for start in range(0, M, BLOCK_REDUCE):
        offsets = (start + rows) * N + groups
        mask = group_mask & (start + rows < M)
        v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
        products += v * w
    dot = tl.sum(products, axis=1)[:, None]

    g = tl.load(saved_g + groups, mask=group_mask, other=0.0).to(tl.float32)
    norm = tl.load(saved_norms + groups, mask=group_mask, other=1.0).to(tl.float32)
    scale = g / norm
    projection = dot / (norm * norm)

    for start in range(0, M, BLOCK_REDUCE):
        offsets = (start + rows) * N + groups
        mask = group_mask & (start + rows < M)
        v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(grad_v + offsets, scale * (w - v * projection), mask=mask)
    tl.store(grad_g + groups, dot / norm, mask=group_mask)


def _composite_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    """PyTorch's composite formula, retained for double backward and rare dtypes."""
    broadcast_shape = [1] * saved_v.ndim
    broadcast_shape[dim] = saved_v.shape[dim]
    norms = saved_norms.to(saved_g.dtype)
    products = grad_w * saved_v
    if dim == 0:
        per_dim_sums = products.reshape(saved_v.shape[0], -1).sum(1)
    else:
        per_dim_sums = products.reshape(-1, saved_v.shape[-1]).sum(0)
    per_dim_sums = per_dim_sums.reshape(broadcast_shape)
    grad_v = (saved_g / norms) * (grad_w - saved_v * (per_dim_sums / (norms * norms)))
    grad_g = per_dim_sums / norms
    return grad_v, grad_g


def _block_sizes(reduce_size):
    block_reduce = min(1024, triton.next_power_of_2(max(1, reduce_size)))
    if block_reduce <= 64:
        block_group = 16
    elif block_reduce <= 512:
        block_group = 8
    else:
        block_group = 2
    return block_group, block_reduce


def _weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS _WEIGHT_NORM_DIFFERENTIABLE_BACKWARD")

    for name, tensor in (
        ("grad_w", grad_w),
        ("saved_v", saved_v),
        ("saved_g", saved_g),
        ("saved_norms", saved_norms),
    ):
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
    if dim != 0 and dim != saved_v.ndim - 1:
        raise RuntimeError("Expected dim to be the first or last dimension")
    if saved_v.ndim == 0:
        raise IndexError("Dimension specified as -1 but tensor has no dimensions")

    # A differentiable decomposition is required while constructing a
    # higher-order graph. Float64 and unusual metadata combinations also use
    # it so the fused kernel never narrows values or performs unsafe loads.
    broadcast_shape = [1] * saved_v.ndim
    broadcast_shape[dim] = saved_v.shape[dim]
    expected_norm_dtype = (
        torch.float32
        if saved_v.dtype in (torch.float16, torch.bfloat16)
        else saved_v.dtype
    )
    can_fuse = (
        not (
            torch.is_grad_enabled()
            and any(x.requires_grad for x in (grad_w, saved_v, saved_g, saved_norms))
        )
        and saved_v.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and grad_w.dtype == saved_v.dtype
        and saved_g.dtype == saved_v.dtype
        and saved_norms.dtype == expected_norm_dtype
        and grad_w.shape == saved_v.shape
        and list(saved_g.shape) == broadcast_shape
        and list(saved_norms.shape) == broadcast_shape
        and grad_w.device == saved_v.device == saved_g.device == saved_norms.device
        and saved_v.numel() > 0
    )
    if not can_fuse:
        return _composite_backward(grad_w, saved_v, saved_g, saved_norms, dim)

    grad_v = torch.empty_like(saved_v)
    grad_g = torch.empty_like(saved_g)
    if dim == 0:
        groups = saved_v.shape[0]
        reduce_size = math.prod(saved_v.shape[1:])
        kernel = _weight_norm_backward_first_kernel
    else:
        groups = saved_v.shape[-1]
        reduce_size = math.prod(saved_v.shape[:-1])
        kernel = _weight_norm_backward_last_kernel
    block_group, block_reduce = _block_sizes(reduce_size)
    grid = (triton.cdiv(groups, block_group),)
    num_warps = 8 if block_group * block_reduce >= 4096 else 4
    with torch_device_fn.device(saved_v.device):
        kernel[grid](
            grad_v,
            grad_g,
            grad_w,
            saved_v,
            saved_g,
            saved_norms,
            math.prod(saved_v.shape[:-1]) if dim != 0 else saved_v.shape[0],
            saved_v.shape[-1] if dim != 0 else math.prod(saved_v.shape[1:]),
            BLOCK_GROUP=block_group,
            BLOCK_REDUCE=block_reduce,
            num_warps=num_warps,
        )
    return grad_v, grad_g

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

# GCU400 has 24 SIPs. Both kernels below run a persistent grid of at most one
# program per SIP and walk the remaining group blocks with a grid-stride loop,
# which is markedly faster than one program per block for the large shapes.
_NUM_SIPS = 24
# Target number of tensor elements per program tile, split between the group
# and reduce axes. The reduce axis is capped so the accumulator stays in
# registers.
_TILE_ELEMENTS_FIRST = 16384
_TILE_ELEMENTS_LAST = 32768
_MAX_BLOCK_REDUCE_FIRST = 512
_MAX_BLOCK_REDUCE_LAST = 256
# When the whole reduction fits in one tile the kernel keeps ``v`` and
# ``grad_w`` live across the projection and writes ``grad_v`` from registers,
# which removes the second read pass (half the traffic). The cap keeps the
# live tile small enough to stay in the data scratchpad. For a trailing-``dim``
# reduction the tile is oriented as (reduce, group), so a long reduce axis
# narrows the memory-contiguous group axis and the two-pass kernel wins.
_MAX_ONEPASS_ELEMENTS = 32768
_MAX_ONEPASS_REDUCE = 1024
_MAX_NONCONTIG_ONEPASS_REDUCE = 128


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
    ONEPASS: tl.constexpr,
):
    # dim == 0: one group per leading index, reduction over the contiguous
    # trailing elements. Tile is (GROUP rows, REDUCE cols), so the innermost
    # tile axis is the memory-contiguous axis.
    pid = ext.program_id(0)
    npids = tl.num_programs(0)
    group_offsets = tl.arange(0, BLOCK_GROUP)
    cols = tl.arange(0, BLOCK_REDUCE)[None, :]
    num_group_blocks = (M + BLOCK_GROUP - 1) // BLOCK_GROUP

    for block_id in tl.range(pid, num_group_blocks, npids):
        group_index = block_id * BLOCK_GROUP + group_offsets
        group_mask = group_index < M
        groups = group_index[:, None]

        if ONEPASS:
            # The whole reduction is resident, so the projection and the
            # `grad_v` write reuse the already loaded tiles.
            offsets = groups * N + cols
            mask = group_mask[:, None] & (cols < N)
            v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
            dot = tl.sum(v * w, axis=1)
            g = tl.load(saved_g + group_index, mask=group_mask, other=0.0).to(
                tl.float32
            )
            norm = tl.load(saved_norms + group_index, mask=group_mask, other=1.0).to(
                tl.float32
            )
            scale = (g / norm)[:, None]
            projection = (dot / (norm * norm))[:, None]
            tl.store(grad_v + offsets, scale * (w - v * projection), mask=mask)
            tl.store(grad_g + group_index, dot / norm, mask=group_mask)
        else:
            products = tl.zeros((BLOCK_GROUP, BLOCK_REDUCE), dtype=tl.float32)
            for start in range(0, N, BLOCK_REDUCE):
                offsets = groups * N + start + cols
                mask = group_mask[:, None] & (start + cols < N)
                v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
                products += v * w
            dot = tl.sum(products, axis=1)

            g = tl.load(saved_g + group_index, mask=group_mask, other=0.0).to(
                tl.float32
            )
            norm = tl.load(saved_norms + group_index, mask=group_mask, other=1.0).to(
                tl.float32
            )
            scale = (g / norm)[:, None]
            projection = (dot / (norm * norm))[:, None]

            for start in range(0, N, BLOCK_REDUCE):
                offsets = groups * N + start + cols
                mask = group_mask[:, None] & (start + cols < N)
                v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
                tl.store(grad_v + offsets, scale * (w - v * projection), mask=mask)
            tl.store(grad_g + group_index, dot / norm, mask=group_mask)


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
    ONEPASS: tl.constexpr,
):
    # dim == last: one group per trailing index, reduction over the leading
    # rows. Tile is (REDUCE rows, GROUP cols) so the innermost tile axis is the
    # memory-contiguous group axis. A masked tile store whose innermost axis is
    # not the contiguous axis is miscompiled on GCU (it silently writes wrong
    # offsets), so this orientation is mandatory.
    pid = ext.program_id(0)
    npids = tl.num_programs(0)
    group_offsets = tl.arange(0, BLOCK_GROUP)
    rows = tl.arange(0, BLOCK_REDUCE)[:, None]
    num_group_blocks = (N + BLOCK_GROUP - 1) // BLOCK_GROUP

    for block_id in tl.range(pid, num_group_blocks, npids):
        group_index = block_id * BLOCK_GROUP + group_offsets
        group_mask = group_index < N
        groups = group_index[None, :]

        if ONEPASS:
            offsets = rows * N + groups
            mask = (rows < M) & group_mask[None, :]
            v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
            dot = tl.sum(v * w, axis=0)
            g = tl.load(saved_g + group_index, mask=group_mask, other=0.0).to(
                tl.float32
            )
            norm = tl.load(saved_norms + group_index, mask=group_mask, other=1.0).to(
                tl.float32
            )
            scale = (g / norm)[None, :]
            projection = (dot / (norm * norm))[None, :]
            tl.store(grad_v + offsets, scale * (w - v * projection), mask=mask)
            tl.store(grad_g + group_index, dot / norm, mask=group_mask)
        else:
            products = tl.zeros((BLOCK_REDUCE, BLOCK_GROUP), dtype=tl.float32)
            for start in range(0, M, BLOCK_REDUCE):
                offsets = (start + rows) * N + groups
                mask = (start + rows < M) & group_mask[None, :]
                v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
                products += v * w
            dot = tl.sum(products, axis=0)

            g = tl.load(saved_g + group_index, mask=group_mask, other=0.0).to(
                tl.float32
            )
            norm = tl.load(saved_norms + group_index, mask=group_mask, other=1.0).to(
                tl.float32
            )
            scale = (g / norm)[None, :]
            projection = (dot / (norm * norm))[None, :]

            for start in range(0, M, BLOCK_REDUCE):
                offsets = (start + rows) * N + groups
                mask = (start + rows < M) & group_mask[None, :]
                v = tl.load(saved_v + offsets, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(grad_w + offsets, mask=mask, other=0.0).to(tl.float32)
                tl.store(grad_v + offsets, scale * (w - v * projection), mask=mask)
            tl.store(grad_g + group_index, dot / norm, mask=group_mask)


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


def _launch_plan(groups, reduce_size, reduce_is_contiguous):
    """Pick tile shape, persistent grid and warp count for one launch.

    The tile is sized so a program holds roughly ``_TILE_ELEMENTS_*`` elements,
    then shrunk when the group count is small so that the grid still spreads
    work over the SIPs (capped at one program per SIP). When the whole reduce
    axis fits in one tile the plan switches to the single-pass kernel, which
    touches ``saved_v`` and ``grad_w`` once instead of twice.
    """
    if reduce_is_contiguous:
        max_block_reduce = _MAX_BLOCK_REDUCE_FIRST
        tile_elements = _TILE_ELEMENTS_FIRST
    else:
        max_block_reduce = _MAX_BLOCK_REDUCE_LAST
        tile_elements = _TILE_ELEMENTS_LAST
    pow2_reduce = triton.next_power_of_2(max(1, reduce_size))
    onepass_groups = _MAX_ONEPASS_ELEMENTS // pow2_reduce
    onepass_reduce_cap = (
        _MAX_ONEPASS_REDUCE if reduce_is_contiguous else _MAX_NONCONTIG_ONEPASS_REDUCE
    )
    onepass = pow2_reduce <= onepass_reduce_cap and onepass_groups >= 1
    if onepass:
        block_reduce = pow2_reduce
        tile_block_group = onepass_groups
    else:
        block_reduce = min(max_block_reduce, pow2_reduce)
        tile_block_group = max(1, tile_elements // block_reduce)
    # Never use more groups than exist; keep at least ~2 groups per SIP so the
    # grid can fill the machine when the group count is small.
    parallel_block_group = 2 * max(
        1, triton.next_power_of_2(triton.cdiv(groups, _NUM_SIPS))
    )
    block_group = max(
        1,
        min(tile_block_group, triton.next_power_of_2(groups), parallel_block_group),
    )
    grid = min(triton.cdiv(groups, block_group), _NUM_SIPS)
    num_warps = max(1, min(32, (block_group * block_reduce) // 512))
    return block_group, block_reduce, grid, num_warps, onepass


def weight_norm_differentiable_backward(grad_w, saved_v, saved_g, saved_norms, dim):
    logger.debug("GEMS_ENFLAME _WEIGHT_NORM_DIFFERENTIABLE_BACKWARD")

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
        block_group, block_reduce, grid, num_warps, onepass = _launch_plan(
            groups, reduce_size, True
        )
    else:
        groups = saved_v.shape[-1]
        reduce_size = math.prod(saved_v.shape[:-1])
        kernel = _weight_norm_backward_last_kernel
        block_group, block_reduce, grid, num_warps, onepass = _launch_plan(
            groups, reduce_size, False
        )
    with torch_device_fn.device(saved_v.device):
        kernel[(grid,)](
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
            ONEPASS=onepass,
            num_warps=num_warps,
        )
    return grad_v, grad_g

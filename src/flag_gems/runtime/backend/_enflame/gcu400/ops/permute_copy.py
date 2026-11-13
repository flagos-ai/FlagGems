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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _permute_copy_flat_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)


@triton.jit
def _permute_copy_gather_2d(
    src_ptr,
    dst_ptr,
    n_elements,
    dst_shape1,
    src_stride0,
    src_stride1,
    dst_stride0,
    dst_stride1,
    perm0,
    perm1,
    rank,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    o1 = offsets % dst_shape1
    o0 = offsets // dst_shape1

    perm_inv0 = tl.where(perm1 == 0, 1, 0)
    perm_inv1 = tl.where(perm0 == 1, 0, 1)

    src_offset = (
        tl.where(perm_inv0 == 0, o0, o1) * src_stride0
        + tl.where(perm_inv1 == 0, o0, o1) * src_stride1
    )

    dst_offset = o0 * dst_stride0 + o1 * dst_stride1

    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


@triton.jit
def _permute_copy_gather_3d(
    src_ptr,
    dst_ptr,
    n_elements,
    dst_shape2,
    dst_shape1,
    src_stride0,
    src_stride1,
    src_stride2,
    dst_stride0,
    dst_stride1,
    dst_stride2,
    perm0,
    perm1,
    perm2,
    rank,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    temp = offsets
    o2 = temp % dst_shape2
    temp = temp // dst_shape2
    o1 = temp % dst_shape1
    o0 = temp // dst_shape1

    perm_inv0 = tl.where(perm1 == 0, 1, tl.where(perm2 == 0, 2, 0))
    perm_inv1 = tl.where(perm0 == 1, 0, tl.where(perm2 == 1, 2, 1))
    perm_inv2 = tl.where(perm0 == 2, 0, tl.where(perm1 == 2, 1, 2))

    src_offset = (
        tl.where(perm_inv0 == 0, o0, tl.where(perm_inv0 == 1, o1, o2)) * src_stride0
        + tl.where(perm_inv1 == 0, o0, tl.where(perm_inv1 == 1, o1, o2)) * src_stride1
        + tl.where(perm_inv2 == 0, o0, tl.where(perm_inv2 == 1, o1, o2)) * src_stride2
    )

    dst_offset = o0 * dst_stride0 + o1 * dst_stride1 + o2 * dst_stride2

    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


@triton.jit
def _permute_copy_tile_kernel(
    src_ptr,
    dst_ptr,
    M,
    N,
    batch,
    src_row_stride,
    src_batch_stride,
    dst_row_stride,
    dst_batch_stride,
    TRANSPOSE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Batch dim is folded into grid axis 0 (GCU caps grid.z at 255).
    pid0 = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_b = pid0 % batch
    pid_m = pid0 // batch
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    src_offs = pid_b * src_batch_stride + rows[:, None] * src_row_stride + cols[None, :]
    vals = tl.load(src_ptr + src_offs)
    if TRANSPOSE:
        dst_offs = (
            pid_b * dst_batch_stride + cols[None, :] * dst_row_stride + rows[:, None]
        )
    else:
        dst_offs = (
            pid_b * dst_batch_stride + rows[:, None] * dst_row_stride + cols[None, :]
        )
    tl.store(dst_ptr + dst_offs, vals)


def permute_copy(x: torch.Tensor, dims):
    """
    Wrapper for aten::permute_copy
    Creates and returns a copy of x with permuted dimensions.
    """
    logger.debug("GEMS PERMUTE_COPY")
    ndim = x.ndim
    if ndim == 0:
        # Scalar tensor - just return a copy
        return x.clone()

    # Normalize dims (handle negative indices)
    dims = [d if d >= 0 else d + ndim for d in dims]

    # Compute output shape
    out_shape = [x.shape[d] for d in dims]

    # Check for empty tensor
    if x.numel() == 0:
        return torch.empty(out_shape, dtype=x.dtype, device=x.device)

    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    n_elements = out.numel()

    if list(dims) == list(range(ndim)):
        # Identity permutation: plain linear copy, no index arithmetic.
        src = x if x.is_contiguous() else x.contiguous()
        grid = (triton.cdiv(n_elements, 4096),)
        with torch_device_fn.device(x.device):
            _permute_copy_flat_kernel[grid](
                src,
                out,
                n_elements,
                BLOCK_SIZE=4096,
                num_warps=8,
            )
        return out

    if ndim == 1 or ndim > 3:
        # Invalid 1D perm falls through to torch; ranks > 3 use permute view
        # + contiguous copy (same as the generic implementation).
        return x.permute(dims).contiguous()

    # Tile-based path: requires the innermost dim of x to be contiguous in
    # memory so that loads run along stride-1 segments.
    if x.stride(ndim - 1) != 1:
        x = x.contiguous()

    # Axis of `out` fed by the innermost (stride-1) dim of x.
    axis_a = dims.index(ndim - 1)
    transpose = axis_a != ndim - 1
    cols_axis = axis_a
    if transpose:
        rows_axis = ndim - 1
    else:
        rows_axis = 0 if axis_a != 0 else 1
    used = {rows_axis, cols_axis}
    batch_axis = next(a for a in range(ndim) if a not in used) if ndim == 3 else -1

    M = out_shape[rows_axis]
    N = out_shape[cols_axis]
    batch = out_shape[batch_axis] if ndim == 3 else 1
    src_row_stride = x.stride(dims[rows_axis])
    src_batch_stride = x.stride(dims[batch_axis]) if ndim == 3 else 0
    dst_row_stride = out.stride(cols_axis) if transpose else out.stride(rows_axis)
    dst_batch_stride = out.stride(batch_axis) if ndim == 3 else 0

    # If the batch dim sits directly outside the rows dim in both x and out
    # memory order, fold it into rows: a single flat 2D transpose covers the
    # whole tensor and larger tiles become usable.
    if (
        ndim == 3
        and src_batch_stride == M * src_row_stride
        and dst_batch_stride == M * out.stride(rows_axis)
    ):
        M = M * batch
        batch = 1
        src_batch_stride = 0
        dst_batch_stride = 0

    # Tile sizing: copy tiles scale up until DSM/occupancy limits. Small
    # tensors favor small tiles for program count; large ones favor 256x256.
    if M * N * batch >= 65536:
        block_m_cap, block_n_cap = 256, 256
        num_warps = 4 if x.dtype == torch.float32 else 2
    elif ndim == 3:
        block_m_cap = 128 if batch == 1 else 64
        block_n_cap = 128
        num_warps = 4
    else:
        block_m_cap, block_n_cap, num_warps = 64, 64, 2
    block_m = min(block_m_cap, triton.next_power_of_2(M))
    block_n = min(block_n_cap, triton.next_power_of_2(N))

    # Masked 2D tiles are only correct on this backend for trivially-aligned
    # masks, so the tile kernel runs unmasked and only for exact divisions;
    # everything else falls back to the flat gather kernels. Grid hard
    # limits (P19): grid.x <= 65535 (edge value itself unsafe), grid.y/z <= 255.
    num_m = triton.cdiv(M, block_m)
    num_n = triton.cdiv(N, block_n)
    if (
        M % block_m == 0
        and N % block_n == 0
        and num_m * batch <= 65534
        and num_n <= 255
    ):
        grid = (num_m * batch, num_n, 1)
        with torch_device_fn.device(x.device):
            _permute_copy_tile_kernel[grid](
                x,
                out,
                M,
                N,
                batch,
                src_row_stride,
                src_batch_stride,
                dst_row_stride,
                dst_batch_stride,
                TRANSPOSE=transpose,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        return out

    # Fallback: original flat gather kernels (correct for any shape).
    grid = (triton.cdiv(n_elements, 1024),)
    with torch_device_fn.device(x.device):
        if ndim == 2:
            _permute_copy_gather_2d[grid](
                x,
                out,
                n_elements,
                out.shape[1],
                x.stride(0),
                x.stride(1),
                out.stride(0),
                out.stride(1),
                dims[0],
                dims[1],
                ndim,
                BLOCK_SIZE=1024,
            )
        else:
            _permute_copy_gather_3d[grid](
                x,
                out,
                n_elements,
                out.shape[2],
                out.shape[1],
                x.stride(0),
                x.stride(1),
                x.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                dims[0],
                dims[1],
                dims[2],
                ndim,
                BLOCK_SIZE=1024,
            )
    return out

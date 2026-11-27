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

# GCU400 specialization of the ``diagonal_scatter`` backward.
#
# Two things differ from the generic implementation:
#
# 1. Grid folding.  The generic version issues one program per 1024 elements,
#    so a 64 MiB fp32 tensor needs 65536 programs and trips the GCU400
#    ``grid.x`` hard limit (65535) with ``out of resource: grid.x``.  Here the
#    copy block is sized up to 64K elements per program, which keeps the grid
#    below the limit for any int32-addressable tensor.
#
# 2. A plane-major diagonal pass.  The generic version re-derives (plane, k)
#    from the flat index for every element, so the div/mod chain runs over the
#    whole tensor.  Here the diagonal work is one program per plane, which
#    hoists that chain out of the element loop, and the same pass both clears
#    the scattered diagonal in ``grad_input`` and gathers it into ``grad_src``.
#
# The copy and the diagonal pass stay separate launches on purpose: they write
# overlapping addresses, so fusing them into one launch (branching on
# ``program_id``) races -- the copy programs can land after the diagonal
# programs and restore the values that were just cleared.  A launch here costs
# ~3 us while the measured delta of the fused version was 3.25x -> 1.8x, i.e.
# correctness first.
#
# 64K elements per program is deliberate: on GCU400 a flat copy only saturates
# the load/store pipes with large per-program tiles (measured 1.36 / 2.64 TB/s
# at BLOCK=65536 versus 0.13 / 0.13 TB/s at BLOCK=1024), because the scalar
# pipe cannot issue enough work per program otherwise.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Elements moved per copy program; the plain copy wins with large tiles.
_COPY_BLOCK = 65536
_COPY_WARPS = 16
# Upper bound for the per-plane diagonal tile (diagonal length is usually small).
_DIAG_BLOCK_MAX = 1024
_GRID_MAX = 65535


@triton.jit
def _copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """Flat contiguous copy for the ``grad_input`` clone."""
    idx = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    tl.store(dst_ptr + idx, tl.load(src_ptr + idx, mask=mask), mask=mask)


@triton.jit
def _diag_kernel(
    grad_ptr,
    grad_input_ptr,
    grad_src_ptr,
    n_planes,
    n_m,
    n_r,
    diag_len,
    stride_l,
    stride_m,
    stride_row,
    stride_col,
    row_off,
    col_off,
    BLOCK: tl.constexpr,
):
    """Clear the scattered diagonal in ``grad_input`` and gather it to ``grad_src``.

    Each program owns one plane (strided by the program count for very large
    plane counts), so the (lo, m, r) decomposition runs once per plane instead
    of once per element.
    """
    plane_pid = tl.program_id(axis=0)
    num_plane_progs = tl.num_programs(axis=0)
    k = tl.arange(0, BLOCK)
    for k0 in range(0, diag_len, BLOCK):
        diag_mask = k0 + k < diag_len
        for plane in range(plane_pid, n_planes, num_plane_progs):
            r = plane % n_r
            t = plane // n_r
            m = t % n_m
            lo = t // n_m
            plane_base = (
                lo * stride_l
                + m * stride_m
                + r
                + row_off * stride_row
                + col_off * stride_col
            )
            # The row and column advance together along the diagonal, so a
            # single stride covers both axes.
            offsets = plane_base + (k0 + k) * (stride_row + stride_col)
            value = tl.load(grad_ptr + offsets, mask=diag_mask)
            # ``grad_input`` is a full copy of ``grad``; only the scattered
            # diagonal has to be cleared.
            tl.store(grad_input_ptr + offsets, tl.zeros_like(value), mask=diag_mask)
            tl.store(grad_src_ptr + plane * diag_len + k0 + k, value, mask=diag_mask)


def _diagonal_geometry(shape, dim1, dim2):
    """Split ``shape`` around the (dim1, dim2) plane of a contiguous tensor.

    Returns the nested plane factorization together with the strides and plane
    spans needed by the kernel.
    """
    ndim = len(shape)
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]

    n_l = 1
    for d in range(dim1):
        n_l *= shape[d]
    n_m = 1
    for d in range(dim1 + 1, dim2):
        n_m *= shape[d]
    n_r = 1
    for d in range(dim2 + 1, ndim):
        n_r *= shape[d]

    return (
        n_l,
        n_m,
        n_r,
        strides[dim1] * shape[dim1],
        strides[dim2] * shape[dim2],
        strides[dim1],
        strides[dim2],
    )


def diagonal_scatter_backward(
    grad_output: torch.Tensor,
    src: torch.Tensor,
    offset: int = 0,
    dim1: int = 0,
    dim2: int = 1,
):
    """Gradients of ``torch.diagonal_scatter`` with respect to ``self`` and ``src``."""
    logger.debug("GEMS_ENFLAME DIAGONAL_SCATTER_BACKWARD")

    ndim = grad_output.ndim
    a = dim1 % ndim
    b = dim2 % ndim
    if a == b:
        raise ValueError("dim1 and dim2 must be different")

    # ``torch.diagonal`` swaps the plane axes and negates the offset when
    # dim1 > dim2; use the same normalisation so the kernels only see a < b.
    if a > b:
        a, b = b, a
        offset = -offset

    grad = grad_output if grad_output.is_contiguous() else grad_output.contiguous()

    diag_len = max(
        0, min(grad.shape[a] + min(offset, 0), grad.shape[b] - max(offset, 0))
    )
    grad_src_shape = [grad.shape[d] for d in range(ndim) if d != a and d != b]
    grad_src_shape.append(diag_len)

    grad_input = torch.empty_like(grad)
    grad_src = torch.empty(grad_src_shape, dtype=grad.dtype, device=grad.device)

    n_elements = grad.numel()
    if n_elements == 0:
        return grad_input, grad_src

    block = min(_COPY_BLOCK, max(2048, triton.next_power_of_2(n_elements)))
    _copy_kernel[(triton.cdiv(n_elements, block),)](
        grad, grad_input, n_elements, BLOCK=block, num_warps=_COPY_WARPS
    )

    if diag_len > 0:
        n_l, n_m, n_r, stride_l, stride_m, stride_row, stride_col = _diagonal_geometry(
            grad.shape, a, b
        )
        n_planes = n_l * n_m * n_r
        diag_block = min(_DIAG_BLOCK_MAX, max(64, triton.next_power_of_2(diag_len)))
        grid = (min(n_planes, _GRID_MAX),)
        _diag_kernel[grid](
            grad,
            grad_input,
            grad_src,
            n_planes,
            n_m,
            n_r,
            diag_len,
            stride_l,
            stride_m,
            stride_row,
            stride_col,
            max(-offset, 0),
            max(offset, 0),
            BLOCK=diag_block,
            num_warps=4,
        )

    return grad_input, grad_src

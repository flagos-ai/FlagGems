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
from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.ops.permute_copy import permute_copy as default_permute_copy
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Moore Threads hardware does not support fp64; int64 is supported for
# indexing, but we keep the supported dtype set aligned with the other
# mthreads specialized operators and fall back to the generic impl
# otherwise.
_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.bool,
}

# Highest tensor rank handled directly by the specialized kernel. Tensors
# with a larger rank fall back to the generic implementation.
_MAX_KERNEL_RANK = 5


@libentry()
@triton.autotune(
    configs=[
        # BLOCK_M = 1: one output row per program, large inner block. Best when
        # the last output dimension (N) is large, giving a long coalesced run.
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=8, num_stages=2),
        # BLOCK_M = 1, small inner block, 2 warps: best for rank-3 permutes
        # where the last output dim is small (N ~= 64) — a short coalesced run
        # with minimal thread overhead.
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=2, num_stages=2),
        # BLOCK_M = 2 / 4: a few rows per program, mid-sized inner block.
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4, num_stages=2),
        # BLOCK_M = 8 / 16: many rows per program, small inner block. Best when
        # the last output dimension is small (e.g. rank-3 permutes with N=64),
        # so each program still has enough work without wasting threads on
        # masked-out elements.
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ],
    key=["n_elements", "N", "dtype_size"],
)
@triton.jit
def _permute_copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    M,
    N,
    dtype_size,
    # Outer (all-but-last) output shapes, padded to 4 with leading 1s.
    out_shape0,
    out_shape1,
    out_shape2,
    out_shape3,
    # Reordered source strides (``src_stride_j = input.stride(perm[j])``),
    # padded to 5 with leading 0s; index 4 is the inner (last) dim.
    src_stride0,
    src_stride1,
    src_stride2,
    src_stride3,
    src_stride4,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # outer flat index
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # last-dim index
    mask_m = rm < M
    mask_n = rn < N
    mask = mask_m[:, None] & mask_n[None, :]

    # Decompose the outer flat index into per-dimension output indices. This is
    # done once per program (scalar work over BLOCK_M rows) rather than per
    # element, which is the key difference from a flat 1-D kernel and keeps the
    # inner-dim load/store loop free of integer divisions. Leading size-1
    # dimensions (padding for ranks < 5) simply yield index 0.
    rem = rm
    o3 = rem % out_shape3
    rem = rem // out_shape3
    o2 = rem % out_shape2
    rem = rem // out_shape2
    o1 = rem % out_shape1
    rem = rem // out_shape1
    o0 = rem % out_shape0

    # Source offset: dot product of the (reordered) source strides with the
    # output indices. The outer part is constant across the inner dim, so it
    # forms a per-row base; the inner part is the strided last-dim access.
    src_base = o0 * src_stride0 + o1 * src_stride1 + o2 * src_stride2 + o3 * src_stride3
    src_offset = src_base[:, None] + rn[None, :] * src_stride4

    # Destination offset: the output is row-major contiguous, so the flat index
    # is simply ``outer_index * N + inner_index`` (inner stride == 1).
    dst_offset = rm[:, None] * N + rn[None, :]

    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


@libentry()
@triton.autotune(
    configs=[
        # For rank-3 permutes the output is small and launch-bound, so a tight
        # 3-D grid (one program per (d0, d1) cell, BLOCK_N covering the last
        # dim) with few warps minimizes per-program overhead and avoids the
        # integer divisions of the flat-index decomposition above.
        # num_warps=2 gives the best warm latency on Moore Threads for the
        # small per-program workloads (one row of the last dim).
        triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=2, num_stages=2),
    ],
    key=["n_elements", "d2", "dtype_size"],
)
@triton.jit
def _permute_copy_kernel_3d(
    src_ptr,
    dst_ptr,
    n_elements,
    d0,
    d1,
    d2,
    dtype_size,
    # Reordered source strides (``src_stride_j = input.stride(perm[j])``).
    src_stride0,
    src_stride1,
    src_stride2,
    # Output strides (row-major contiguous).
    out_stride0,
    out_stride1,
    out_stride2,
    BLOCK_N: tl.constexpr,
):
    p0 = tl.program_id(0)  # output dim 0
    p1 = tl.program_id(1)  # output dim 1
    p2 = tl.program_id(2)  # output dim 2 (blocked)

    rn = p2 * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = rn < d2

    # Source offset: dot product of the reordered source strides with the
    # output indices. p0/p1 are scalar per program; rn is the inner run.
    src_offset = p0 * src_stride0 + p1 * src_stride1 + rn * src_stride2
    # Destination offset: the output is row-major contiguous.
    dst_offset = p0 * out_stride0 + p1 * out_stride1 + rn * out_stride2

    vals = tl.load(src_ptr + src_offset, mask=mask)
    tl.store(dst_ptr + dst_offset, vals, mask=mask)


def _use_triton_kernel(x: torch.Tensor, dims) -> Tuple[bool, int]:
    if not isinstance(x, torch.Tensor):
        return False, 0
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False, 0
    if x.ndim == 0 or x.ndim > _MAX_KERNEL_RANK:
        return False, 0
    if x.numel() == 0 or not x.is_contiguous():
        return False, 0
    # ``dims`` must be a genuine permutation of range(ndim); otherwise the
    # copy semantics differ from ``aten::permute_copy``.
    ndim = x.ndim
    try:
        normalized = [d if d >= 0 else d + ndim for d in dims]
    except TypeError:
        return False, 0
    if sorted(normalized) != list(range(ndim)) or len(normalized) != ndim:
        return False, 0
    return True, x.element_size()


def _launch_permute_copy(
    x: torch.Tensor, out: torch.Tensor, dims, dtype_size: int
) -> torch.Tensor:
    ndim = x.ndim

    # Reorder the input strides by the permutation so the kernel can compute
    # the source offset as a plain dot product of the output indices.
    in_strides = list(x.stride())
    src_strides = [in_strides[dims[j]] for j in range(ndim)]
    out_strides = list(out.stride())
    out_shapes = list(out.shape)
    n_elements = out.numel()

    with torch_device_fn.device(out.device):
        if ndim == 3:
            # Rank-3 fast path: a 3-D grid with no flat-index decomposition.
            grid = lambda meta: (
                out_shapes[0],
                out_shapes[1],
                triton.cdiv(out_shapes[2], meta["BLOCK_N"]),
            )
            _permute_copy_kernel_3d[grid](
                x,
                out,
                n_elements,
                out_shapes[0],
                out_shapes[1],
                out_shapes[2],
                dtype_size,
                src_strides[0],
                src_strides[1],
                src_strides[2],
                out_strides[0],
                out_strides[1],
                out_strides[2],
            )
            return out

        # General path for ranks 1, 2, 4, 5: pad shapes/strides up to the fixed
        # 5-D kernel signature. The inner (last) dimension is kept real; leading
        # dimensions are padded with shape 1 (index always 0) and stride 0 (no
        # offset contribution).
        pad_n = _MAX_KERNEL_RANK - ndim
        out_shapes_pad = [1] * pad_n + out_shapes
        src_strides_pad = [0] * pad_n + src_strides

        # Outer = all but the last dim; inner = last dim.
        M = 1
        for s in out_shapes_pad[:-1]:
            M = M * s
        N = out_shapes_pad[-1]

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _permute_copy_kernel[grid](
            x,
            out,
            n_elements,
            M,
            N,
            dtype_size,
            out_shapes_pad[0],
            out_shapes_pad[1],
            out_shapes_pad[2],
            out_shapes_pad[3],
            src_strides_pad[0],
            src_strides_pad[1],
            src_strides_pad[2],
            src_strides_pad[3],
            src_strides_pad[4],
        )
    return out


def permute_copy(x: torch.Tensor, dims):
    logger.debug("GEMS_MTHREADS PERMUTE_COPY")
    use_triton, dtype_size = _use_triton_kernel(x, dims)
    if not use_triton:
        return default_permute_copy(x, dims)

    ndim = x.ndim
    normalized = [d if d >= 0 else d + ndim for d in dims]
    out_shape = [x.shape[d] for d in normalized]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    return _launch_permute_copy(x, out, normalized, dtype_size)

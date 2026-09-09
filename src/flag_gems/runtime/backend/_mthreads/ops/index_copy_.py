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

from flag_gems.ops.index_copy_ import index_copy as default_index_copy
from flag_gems.ops.index_copy_ import index_copy_ as default_index_copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# mthreads hardware does not support fp64/int64 arithmetic inside kernels.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
# All linear offsets must fit into int32 (int64 not supported on mthreads).
_INT32_MAX = 2**31 - 1


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def index_copy_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    N,
    inp_numel,
    inp_stride_dim,
    inp_shape_dim,
    src_shape_dim,
    delta,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flat scatter kernel for index_copy.

    `src` is contiguous, so its linear offset equals `offsets`. `out` is
    contiguous with the same layout as `src` for all dims != dim, hence
    `inp_stride_dim == out.stride(dim) == src.stride(dim)`.

    For each src linear offset:
        pre_idx    -> flattened index over dims before `dim`
        dim_idx    -> position along `dim` inside src
        src_dim_idx = index[dim_idx] -> target position along `dim` in out
        out_idx    = src_offset + (delta*pre_idx + src_dim_idx - dim_idx) * inp_stride_dim

    All arithmetic stays in int32; callers guarantee offsets fit in int32.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    src_offset = offsets
    pre_cal = inp_stride_dim * src_shape_dim
    pre_idx = src_offset // pre_cal
    dim_idx = (src_offset % pre_cal) // inp_stride_dim

    src_dim_idx = tl.load(index_ptr + dim_idx, mask=mask, other=0)

    out_idx = src_offset + (delta * pre_idx + src_dim_idx - dim_idx) * inp_stride_dim
    out_mask = mask & (out_idx >= 0) & (out_idx < inp_numel)

    src_val = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, src_val, mask=out_mask)


def _use_triton_kernel(inp, index, src):
    if not (
        isinstance(inp, torch.Tensor)
        and isinstance(index, torch.Tensor)
        and isinstance(src, torch.Tensor)
    ):
        return False
    if inp.device.type != "musa":
        return False
    if inp.dtype not in _SUPPORTED_DTYPES:
        return False
    if inp.numel() == 0 or src.numel() == 0:
        return False
    if inp.ndim != src.ndim:
        return False
    # Keep all offsets inside int32 range (no int64 on mthreads).
    if inp.numel() > _INT32_MAX or src.numel() > _INT32_MAX:
        return False
    return True


def _launch_index_copy(out, dim, index, src):
    """Scatter `src` into the contiguous tensor `out` in place along `dim`."""
    dim = dim % out.ndim
    inp_stride_dim = out.stride(dim)
    src_shape_dim = src.size(dim)
    inp_shape_dim = out.size(dim)
    delta = inp_shape_dim - src_shape_dim
    N = src.numel()

    # index must be int32 (int64 not supported on mthreads hardware)
    index = index.contiguous().to(torch.int32)
    src = src.contiguous()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(out.device):
        index_copy_kernel[grid](
            index,
            src,
            out,
            N,
            out.numel(),
            inp_stride_dim,
            inp_shape_dim,
            src_shape_dim,
            delta,
        )
    return out


def index_copy(inp, dim, index, src):
    logger.debug("GEMS_MTHREADS INDEX_COPY")
    if not _use_triton_kernel(inp, index, src):
        return default_index_copy(inp, dim, index, src)

    out = inp.clone().contiguous()
    return _launch_index_copy(out, dim, index, src)


def index_copy_(inp, dim, index, src):
    logger.debug("GEMS_MTHREADS INDEX_COPY_")
    if not _use_triton_kernel(inp, index, src):
        return default_index_copy_(inp, dim, index, src)

    if inp.is_contiguous():
        _launch_index_copy(inp, dim, index, src)
    else:
        work = inp.contiguous()
        _launch_index_copy(work, dim, index, src)
        inp.copy_(work)
    return inp

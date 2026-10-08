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

from flag_gems.ops.diagonal_scatter import diagonal_scatter as default_diagonal_scatter
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

MAX_BATCH_DIMS = 4


@libentry()
@triton.jit
def _diag_scatter_kernel(
    in_ptr,
    src_ptr,
    out_ptr,
    offset: tl.constexpr,
    off_min: tl.constexpr,
    n1: tl.constexpr,
    n2: tl.constexpr,
    s1: tl.constexpr,
    s2: tl.constexpr,
    diag_size: tl.constexpr,
    br0: tl.constexpr,
    br1: tl.constexpr,
    br2: tl.constexpr,
    br3: tl.constexpr,
    bs0: tl.constexpr,
    bs1: tl.constexpr,
    bs2: tl.constexpr,
    bs3: tl.constexpr,
    n_i_tiles: tl.constexpr,
    BI: tl.constexpr,
    BJ: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # i-tiles vary fastest so consecutive programs touch adjacent memory.
    i_tile = pid0 % n_i_tiles
    batch_flat = pid0 // n_i_tiles

    # Decompose batch_flat (mixed-radix counter over the non-diagonal dims in
    # ascending dim order) into per-dim digits, innermost batch dim first, and
    # accumulate the corresponding flat base offset. Unused slots use radix=1,
    # stride=0, which makes them no-ops. All divisors are constexpr so the
    # div/mod lower to fast magic-number sequences.
    bf = batch_flat
    base = 0
    dig = bf % br0
    base += dig * bs0
    bf = bf // br0
    dig = bf % br1
    base += dig * bs1
    bf = bf // br1
    dig = bf % br2
    base += dig * bs2
    bf = bf // br2
    dig = bf % br3
    base += dig * bs3

    i = i_tile * BI + tl.arange(0, BI)
    j = pid1 * BJ + tl.arange(0, BJ)
    mi = i < n1
    mj = j < n2

    d = i + off_min
    src_idx = batch_flat * diag_size + d
    src_val = tl.load(
        src_ptr + src_idx, mask=mi & (d >= 0) & (d < diag_size), other=0.0
    )

    on_diag = (
        ((j[None, :] - i[:, None]) == offset)
        & mj[None, :]
        & mi[:, None]
        & (d[:, None] >= 0)
        & (d[:, None] < diag_size)
    )
    f = base + i[:, None] * s1 + j[None, :] * s2
    m = mi[:, None] & mj[None, :]
    in_val = tl.load(in_ptr + f, mask=m, other=0.0)
    out_val = tl.where(on_diag, src_val[:, None], in_val)
    tl.store(out_ptr + f, out_val, mask=m)


def _use_triton_kernel(input, src, offset, dim1, dim2) -> bool:
    if not isinstance(input, torch.Tensor) or not isinstance(src, torch.Tensor):
        return False
    if input.device.type != "musa" or input.dtype not in _SUPPORTED_DTYPES:
        return False
    if not input.is_contiguous() or input.numel() == 0:
        return False
    ndim = input.ndim
    d1 = dim1 % ndim
    d2 = dim2 % ndim
    batch_dims = [d for d in range(ndim) if d != d1 and d != d2]
    if len(batch_dims) > MAX_BATCH_DIMS:
        return False
    return True


def diagonal_scatter(input, src, offset=0, dim1=0, dim2=1):
    logger.debug("GEMS_MTHREADS DIAGONAL_SCATTER")
    if not _use_triton_kernel(input, src, offset, dim1, dim2):
        return default_diagonal_scatter(input, src, offset=offset, dim1=dim1, dim2=dim2)

    ndim = input.ndim
    d1 = dim1 % ndim
    d2 = dim2 % ndim

    shape = input.shape
    strides = input.stride()
    n1 = shape[d1]
    n2 = shape[d2]
    s1 = strides[d1]
    s2 = strides[d2]

    batch_dims = [d for d in range(ndim) if d != d1 and d != d2]
    if len(batch_dims) > MAX_BATCH_DIMS:
        raise ValueError("tensor too high-dimensional for this kernel")

    batch_count = 1
    for d in batch_dims:
        batch_count *= shape[d]

    off_min = min(offset, 0)
    off_max = max(offset, 0)
    diag_size = max(0, min(n1 + off_min, n2 - off_max))

    rad = [1] * MAX_BATCH_DIMS
    bst = [0] * MAX_BATCH_DIMS
    for k, d in enumerate(reversed(batch_dims)):
        rad[k] = shape[d]
        bst[k] = strides[d]

    # Tile the (dim1, dim2) plane: BJ along dim2, BI along dim1, targeting
    # ~1024 elements per program block and ~4 elements per thread. Rows of
    # width n2=128 (BJ==128) uniquely prefer 4 warps (8 elems/thread) on the
    # MTT S5000; all other widths run 8 warps (4 elems/thread).
    BJ = min(256, 1 << (n2 - 1).bit_length()) if n2 > 0 else 256
    BI = min(1 << (n1 - 1).bit_length(), max(1, 1024 // BJ)) if n1 > 0 else 1
    n_i_tiles = (n1 + BI - 1) // BI
    n_j_tiles = (n2 + BJ - 1) // BJ
    grid = (batch_count * n_i_tiles, n_j_tiles)
    if BJ == 128:
        num_warps = max(1, min(8, (BI * BJ) // 256))
    else:
        num_warps = max(1, min(8, (BI * BJ) // 128))

    with torch_device_fn.device(input.device):
        output = torch.empty_like(input)
        _diag_scatter_kernel[grid](
            input,
            src,
            output,
            offset,
            off_min,
            n1,
            n2,
            s1,
            s2,
            diag_size,
            rad[0],
            rad[1],
            rad[2],
            rad[3],
            bst[0],
            bst[1],
            bst[2],
            bst[3],
            n_i_tiles,
            BI,
            BJ,
            num_warps=num_warps,
        )
    return output


__all__ = ["diagonal_scatter"]

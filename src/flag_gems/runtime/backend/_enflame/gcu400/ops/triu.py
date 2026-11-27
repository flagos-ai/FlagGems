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
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU400 sweet spot: 48 resident programs (2 per SP). Larger grids pay the
# per-CTA dispatch tax (~55ns/CTA) which dominates triu on medium shapes.
_ROW_GRID = 48
# Row-stream tile width: covers whole rows up to 4096 columns in one block.
_ROW_BLOCK_CAP = 4096
# Flat 3D tile size: bounded by the 917504B DSM budget (~491KB at fp32).
_FLAT_BLOCK = 32768
# Tensors at or beyond this many elements need widened (int64) index math.
_INT32_LIMIT = 2**31


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def _is_pow2(n):
    return n > 0 and (n & (n - 1)) == 0


@libentry()
@triton.jit(do_not_specialize=["diagonal"])
def triu_kernel(
    X,
    Y,
    M,
    N,
    diagonal,
    N_BLOCK_SIZE: tl.constexpr,
    SAFE64: tl.constexpr,
):
    # Row-stream tri-segment triu: each program walks rows with stride 48 and
    # writes every row as a disjoint zero prefix [0, lo) followed by a pure
    # copy suffix [lo, N). No tl.where couples the store, the lower triangle
    # is never loaded, and all accesses are 1-D dense stores with prefix
    # masks - window masks fused into stores are a known GCU degradation.
    # Index math stays int32 (ext.program_id's int64 widening taxes every
    # element on GCU); the row base is widened explicitly only when the
    # tensor can exceed the int32 element range.
    pid = tl.program_id(0)
    cols0 = tl.arange(0, N_BLOCK_SIZE)
    zero = tl.zeros([N_BLOCK_SIZE], dtype=Y.dtype.element_ty)
    for row in range(pid, M, 48):
        lo = row + diagonal
        zlen = tl.minimum(tl.maximum(lo, 0), N)
        cstart = tl.maximum(lo, 0)
        if SAFE64:
            ybase = Y + row.to(tl.int64) * N
            xbase = X + row.to(tl.int64) * N
        else:
            ybase = Y + row * N
            xbase = X + row * N
        for off in range(0, zlen, N_BLOCK_SIZE):
            cols = off + cols0
            tl.store(ybase + cols, zero, mask=cols < zlen)
        for off in range(cstart, N, N_BLOCK_SIZE):
            cols = off + cols0
            cols_mask = cols < N
            tl.store(
                ybase + cols,
                tl.load(xbase + cols, mask=cols_mask, other=0.0),
                mask=cols_mask,
            )


@libentry()
@triton.jit(do_not_specialize=["diagonal"])
def triu_batch_kernel(
    X,
    Y,
    batch,
    MN,
    N,
    diagonal,
    MN_BLOCK_SIZE: tl.constexpr,
    LOG2N: tl.constexpr,
    KEEP: tl.constexpr,
    SAFE64: tl.constexpr,
):
    # Flat-tile triu over (batch, M*N) with one row band per program: the
    # within-matrix row index is cols // N. When N is a power of two the
    # div/mod pair collapses into shift/and (the per-element integer div is
    # the dominant cost of the legacy kernel); otherwise div/mod is kept as
    # the odd-N fallback. fp32 tiles are load-bandwidth bound so the keep
    # predicate skips lower-triangle loads (-13% on large tiles); 16-bit
    # dtypes are issue bound and the extra per-element predicate costs +80%
    # - dispatch by dtype on the host side. All index math stays int32
    # (the int64 pid widening of ext.program_id taxes every element
    # address ~2.2x on GCU); only the per-program row base is widened.
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    if SAFE64:
        rowoff = pid0.to(tl.int64) * MN
        cols = pid1.to(tl.int64) * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    else:
        rowoff = pid0 * MN
        cols = pid1 * MN_BLOCK_SIZE + tl.arange(0, MN_BLOCK_SIZE)[None, :]
    Xb = X + rowoff
    Yb = Y + rowoff
    mn_mask = cols < MN
    if LOG2N > 0:
        m = cols >> LOG2N
        n = cols & ((1 << LOG2N) - 1)
    else:
        m = cols // N
        n = cols % N
    if KEEP:
        keep = (m + diagonal) <= n
        x = tl.load(Xb + cols, mn_mask & keep, other=0.0)
        tl.store(Yb + cols, x, mask=mn_mask)
    else:
        x = tl.load(Xb + cols, mn_mask, other=0.0)
        y = tl.where(m + diagonal <= n, x, 0.0)
        tl.store(Yb + cols, y, mask=mn_mask)


def _check_batch_contiguous(tensor, allow_zero_stride=True):
    if tensor.is_contiguous():
        return True, tensor

    dims = tensor.dim()

    if dims >= 2:
        n = tensor.size(-1)
        stride_row, stride_col = tensor.stride(-2), tensor.stride(-1)

        if not (stride_col == 1 and stride_row == n):
            return False, tensor.contiguous()

    if allow_zero_stride and dims <= 3:
        return True, tensor

    expected_stride = tensor.size(-1) * tensor.size(-2)
    for i in range(dims - 3, -1, -1):
        if (
            allow_zero_stride
            and i == 0
            and (tensor.stride(i) == 0 or tensor.size(i) == 1)
        ):
            continue

        if tensor.stride(i) != expected_stride:
            return False, tensor.contiguous()

        expected_stride *= tensor.size(i)

    return True, tensor


def _launch_triu_2d(x, y, m, n, diagonal):
    n_block = _next_pow2(min(n, _ROW_BLOCK_CAP))
    grid = (min(_ROW_GRID, m),)
    safe64 = 1 if m * n >= _INT32_LIMIT else 0
    triu_kernel[grid](x, y, m, n, diagonal, n_block, safe64, num_warps=8)


def _launch_triu_3d(x, y, batch, mn, n, diagonal, dtype):
    grid = (batch, triton.cdiv(mn, _FLAT_BLOCK))
    log2n = n.bit_length() - 1 if _is_pow2(n) else 0
    keep = dtype == torch.float32
    safe64 = 1 if mn >= _INT32_LIMIT else 0
    triu_batch_kernel[grid](
        x,
        y,
        batch,
        mn,
        n,
        diagonal,
        _FLAT_BLOCK,
        log2n,
        keep,
        safe64,
        num_warps=4,
    )


def triu(A, diagonal=0):
    logger.debug("GEMS_ENFLAME TRIU")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"

    can_use_directly, A_input = _check_batch_contiguous(A, allow_zero_stride=False)

    out = torch.empty(
        A.shape, dtype=A.dtype, device=A.device, memory_format=torch.contiguous_format
    )

    M, N = A_input.shape[-2:]

    with torch_device_fn.device(A_input.device):
        if len(A_input.shape) == 2:
            _launch_triu_2d(A_input, out, M, N, diagonal)
        else:
            batch = int(torch.numel(A_input) / M / N)
            B = A_input.view(batch, -1)
            _launch_triu_3d(B, out, batch, M * N, N, diagonal, A_input.dtype)
            out = out.view(A.shape)

    return out


def triu_(A, diagonal=0):
    logger.debug("GEMS_ENFLAME TRIU_ (inplace)")

    assert len(A.shape) > 1, "Input tensor must have at least 2 dimensions"
    diagonal = int(diagonal)
    M, N = A.shape[-2:]

    can_use_directly, A_to_use = _check_batch_contiguous(A, allow_zero_stride=True)

    if not can_use_directly:
        logger.debug(
            "Input tensor does not satisfy contiguity requirements, "
            "using temporary tensor for computation"
        )

        result_temp = torch.empty_like(A_to_use, memory_format=torch.contiguous_format)

        with torch_device_fn.device(A.device):
            if len(A.shape) == 2:
                _launch_triu_2d(A_to_use, result_temp, M, N, diagonal)
            else:
                batch = int(torch.numel(A) / M / N)
                B = A_to_use.view(batch, -1)
                result_temp_flat = result_temp.view(batch, -1)
                _launch_triu_3d(B, result_temp_flat, batch, M * N, N, diagonal, A.dtype)

        A.copy_(result_temp)
    else:
        with torch_device_fn.device(A.device):
            if len(A.shape) == 2:
                _launch_triu_2d(A, A, M, N, diagonal)
            else:
                batch = int(torch.numel(A) / M / N)
                B = A.view(batch, -1)
                _launch_triu_3d(B, B, batch, M * N, N, diagonal, A.dtype)

    return A

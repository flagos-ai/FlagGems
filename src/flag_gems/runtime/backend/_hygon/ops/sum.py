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

from flag_gems.ops.sum import sum_dim as general_sum_dim
from flag_gems.ops.sum import sum_kernel_1, sum_kernel_2
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def sum_short_rows_kernel(
    inp,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    values = tl.load(
        inp + rows[:, None] * ROW_STRIDE + cols[None, :] * COL_STRIDE,
        (rows[:, None] < M) & (cols[None, :] < N),
        other=0,
    )
    totals = tl.sum(values, axis=1)
    tl.store(out + rows, totals, rows < M)


@libentry()
@triton.jit
def sum_hygon_tail_kernel(inp, out, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    values = tl.load(inp + row * N + cols, cols < N, other=0).to(tl.float64)
    tl.store(out + row, tl.sum(values, axis=0))


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    """Specialize short rows, trailing pairs and one long contiguous row."""
    dims = [dim] if isinstance(dim, int) else dim
    if (
        inp.ndim >= 3
        and isinstance(dims, (list, tuple))
        and len(dims) == 2
        and all(d in (-2, -1, inp.ndim - 2, inp.ndim - 1) for d in dims)
        and len({d % inp.ndim for d in dims}) == 2
        and inp.is_contiguous()
        and inp.dtype == torch.float32
        and (dtype is None or dtype == inp.dtype)
        and 4096 <= inp.shape[-2] * inp.shape[-1] <= 16384
        and inp.numel() > 0
    ):
        logger.debug("GEMS_HYGON SUM_DIM TRAILING_PAIR")
        n = inp.shape[-2] * inp.shape[-1]
        m = inp.numel() // n
        shape = (*inp.shape[:-2], 1, 1) if keepdim else inp.shape[:-2]
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_hygon_tail_kernel[(m,)](inp, out, n, triton.next_power_of_2(n))
        return out
    if (
        inp.ndim > 0
        and isinstance(dims, (list, tuple))
        and len(dims) == 1
        and dims[0] in (-1, inp.ndim - 1)
        and inp.is_contiguous()
        and inp.dtype in (torch.float32, torch.float64)
        and (dtype is None or dtype == inp.dtype)
        # Bound the second pass to at most 4096 partial sums, and keep
        # the shared first-pass kernel's 32-bit indexing in range.
        and 16384 <= inp.shape[-1] <= 2**24
        and inp.numel() == inp.shape[-1]
    ):
        logger.debug("GEMS_HYGON SUM_DIM SPLIT_ROW")
        n = inp.numel()
        block_n = 4096
        parts = triton.cdiv(n, block_n)
        mid = torch.empty((parts,), dtype=inp.dtype, device=inp.device)
        shape = (*inp.shape[:-1], 1) if keepdim else inp.shape[:-1]
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            sum_kernel_1[(parts,)](inp, mid, n, block_n, num_warps=8)
            sum_kernel_2[(1,)](mid, out, parts, triton.next_power_of_2(parts))
        return out
    if (
        inp.ndim > 0
        and isinstance(dims, (list, tuple))
        and len(dims) == 1
        and dims[0] in (-1, inp.ndim - 1)
        and (inp.is_contiguous() or inp.ndim == 2)
        and inp.dtype in (torch.float32, torch.float64)
        and (dtype is None or dtype == inp.dtype)
        and inp.shape[-1] in (2, 3, 5, 7, 10, 13, 15, 32)
        and inp.numel() // inp.shape[-1] >= (1024 if inp.shape[-1] == 2 else 8192)
    ):
        logger.debug("GEMS_HYGON SUM_DIM SHORT_ROWS")
        n = inp.shape[-1]
        m = inp.numel() // n
        row_stride = inp.stride(0) if inp.ndim == 2 else n
        col_stride = inp.stride(-1)
        shape = (*inp.shape[:-1], 1) if keepdim else inp.shape[:-1]
        out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
        # Small row counts need more workgroups; larger inputs amortize launch
        # and scheduling overhead by reducing multiple rows in each workgroup.
        if n == 3:
            block_m = 256
        elif n == 32 and m < 65536:
            block_m = 16
        else:
            block_m = 64
        with torch_device_fn.device(inp.device):
            sum_short_rows_kernel[(triton.cdiv(m, block_m),)](
                inp,
                out,
                m,
                n,
                row_stride,
                col_stride,
                block_m,
                triton.next_power_of_2(n),
                num_warps=4,
            )
        return out
    return general_sum_dim(inp, dim, keepdim, dtype=dtype)

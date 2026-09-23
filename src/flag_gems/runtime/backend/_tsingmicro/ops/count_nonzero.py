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

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def count_nonzero_kernel_1(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tle.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    numel = tl.cast(numel, tl.int64)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero, axis=0)
    tl.atomic_add(out_ptr, nonzero_count.to(tl.int32))


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("count_nonzero"),
    key=["numel"],
    strategy=["align32"],
    warmup=1,
    rep=2,
)
@triton.jit
def count_nonzero_kernel(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_0 = tle.program_id(0).to(tl.int64)
    num_p = tle.num_programs(0).to(tl.int64)
    rows = (numel + N - 1) // N
    rows_per_p = rows // num_p
    numel = tl.cast(numel, tl.int64)

    for pid_n in range(0, rows_per_p):
        pid_x = pid_0 * rows_per_p + pid_n

        nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
        for start_n in range(0, N, BLOCK_SIZE):
            cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
            offset = pid_x * N + cols_offsets
            mask = (offset < numel) & (cols_offsets < N)
            x = tl.load(x_ptr + offset, mask=mask, other=0)
            is_nonzero = (x != 0).to(tl.int64)
            nonzero_count += tl.sum(is_nonzero)

        tl.store(out_ptr + pid_x, nonzero_count)

    remain = rows % num_p
    if pid_0 < remain:
        pid_x = rows // num_p * num_p + pid_0
        nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
        for start_n in range(0, N, BLOCK_SIZE):
            cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
            offset = pid_x * N + cols_offsets
            mask = (offset < numel) & (cols_offsets < N)
            x = tl.load(x_ptr + offset, mask=mask, other=0)
            is_nonzero = (x != 0).to(tl.int64)
            nonzero_count += tl.sum(is_nonzero)

        tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("count_nonzero"),
    key=["numel"],
    strategy=["align32"],
    warmup=1,
    rep=2,
)
@triton.jit
def count_nonzero_combin_kernel_1(x_ptr, out_ptr, N, numel, BLOCK_SIZE: tl.constexpr):
    pid_x = tle.program_id(0).to(tl.int64)
    numel = tl.cast(numel, tl.int64)
    nonzero_count = tl.full((), value=0, dtype=out_ptr.dtype.element_ty)
    for start_n in range(0, N, BLOCK_SIZE):
        cols_offsets = start_n + tl.arange(0, BLOCK_SIZE)
        offset = pid_x * N + cols_offsets
        mask = (offset < numel) & (cols_offsets < N)
        x = tl.load(x_ptr + offset, mask=mask, other=0)
        nonzero_count += tl.sum((x != 0).to(tl.int64))
    tl.store(out_ptr + pid_x, nonzero_count)


@libentry()
@triton.jit
def count_nonzero_combin_kernel(
    x_ptr, combin_ptr, N, combin_N, numel, BLOCK_SIZE: tl.constexpr
):
    pid_x = tle.program_id(0).to(tl.int64)
    pid_y = tle.program_id(1).to(tl.int64)
    numel = tl.cast(numel, tl.int64)
    cols_offsets = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset = pid_x * N + cols_offsets
    mask = (offset < numel) & (cols_offsets < N)
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    is_nonzero = (x != 0).to(tl.int64)
    nonzero_count = tl.sum(is_nonzero)
    tl.store(combin_ptr + pid_x * combin_N + pid_y, nonzero_count)


@libentry()
@triton.heuristics(runtime.get_heuristic_config("count_nonzero_reduce"))
@triton.jit
def count_nonzero_reduce_rows_kernel(
    x_ptr,
    out_ptr,
    out_numel,
    out_dim1,
    reduce_size,
    stride0,
    stride1,
    reduce_stride,
    OUT_NDIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0).to(tl.int64)
    row_start = pid * BLOCK_ROWS
    for row_offset in range(0, BLOCK_ROWS):
        row = row_start + row_offset
        if row < out_numel:
            if OUT_NDIM == 1:
                base = row * stride0
            else:
                row0 = row // out_dim1
                row1 = row % out_dim1
                base = row0 * stride0 + row1 * stride1

            count = tl.zeros((), dtype=tl.int64)
            for start_n in range(0, reduce_size, BLOCK_SIZE):
                offsets = start_n + tl.arange(0, BLOCK_SIZE)
                mask = offsets < reduce_size
                x = tl.load(
                    x_ptr + base + offsets * reduce_stride,
                    mask=mask,
                    other=0,
                )
                count += tl.sum((x != 0).to(tl.int64))

            tl.store(out_ptr + row, count)


def count_nonzero(x, dim=None):
    logger.debug("GEMS_TSINGMICRO COUNT NONZERO")
    if dim is not None:
        assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
        shape = x.shape
        dim = dim % x.ndim
        x = x.movedim(dim, -1)

        out_shape = list(shape)
        reduce_size = out_shape.pop(dim)
        out = torch.zeros(out_shape, dtype=torch.int64, device=x.device)

        if x.ndim == 1:
            x = x.contiguous().flatten().view(1, -1)
            out = torch.zeros((), dtype=torch.int64, device=x.device)
            grid = lambda meta: (1,)
            count_nonzero_reduce_rows_kernel[grid](
                x,
                out.view(1),
                1,
                0,
                reduce_size,
                x.stride(0),
                0,
                x.stride(-1),
                OUT_NDIM=1,
            )
            return out

        if x.ndim in (2, 3):
            stride0 = x.stride(0)
            stride1 = x.stride(1) if x.ndim == 3 else 0
            reduce_stride = x.stride(-1)
            out_numel = out.numel()
            out_dim1 = out.shape[1] if x.ndim == 3 else 0
            grid = lambda meta: (triton.cdiv(out_numel, meta["BLOCK_ROWS"]),)
            count_nonzero_reduce_rows_kernel[grid](
                x,
                out,
                out_numel,
                out_dim1,
                reduce_size,
                stride0,
                stride1,
                reduce_stride,
                OUT_NDIM=2 if x.ndim == 3 else 1,
            )
            return out

        x = x.contiguous().flatten()
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        count_nonzero_kernel_1[grid](x, out.reshape(1), x.numel(), BLOCK_SIZE=1024 * 8)
        return out
    else:
        x = x.contiguous().flatten().view(1, -1)
        numel = x.numel()

        out = torch.zeros(1, dtype=torch.int64, device=x.device)

        grid = lambda meta: (1,)
        count_nonzero_reduce_rows_kernel[grid](
            x,
            out,
            1,
            0,
            numel,
            x.stride(0),
            0,
            x.stride(-1),
            OUT_NDIM=1,
        )

        return out[0]

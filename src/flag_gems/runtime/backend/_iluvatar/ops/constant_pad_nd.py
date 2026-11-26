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

import torch
import triton
import triton.language as tl

MAX_DIM = (
    8  # max dims handled by the fast int32 kernel (row loop over dims 0..MAX_DIM-1)
)


@triton.jit
def _constant_pad_nd_i32(
    in_ptr,
    out_ptr,
    ndim,  # number of input dims
    out_rows,  # product of output dims except the last one
    out_cols,  # size of the last output dim
    value,  # fill value
    # per-dim metadata for dims 0..MAX_DIM-1 (used only while d < ndim-1):
    #   isz_d = input size, istr_d = input stride, rstr_d = row-space stride, pb_d = pad before
    isz0,
    istr0,
    rstr0,
    pb0,
    isz1,
    istr1,
    rstr1,
    pb1,
    isz2,
    istr2,
    rstr2,
    pb2,
    isz3,
    istr3,
    rstr3,
    pb3,
    isz4,
    istr4,
    rstr4,
    pb4,
    isz5,
    istr5,
    rstr5,
    pb5,
    isz6,
    istr6,
    rstr6,
    pb6,
    isz7,
    istr7,
    rstr7,
    pb7,
    # last-dim metadata:
    pb_last,  # pad before on the innermost dim
    istr_last,  # input stride of the innermost dim
    in_cols,  # input size of the innermost dim
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CONTIG: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = (out_cols + COLS - 1) // COLS
    pid_c = pid % num_col_blocks
    pid_r = pid // num_col_blocks

    row = pid_r * ROWS + tl.arange(0, ROWS)
    col = pid_c * COLS + tl.arange(0, COLS)
    row_in = row < out_rows
    col_in = col < out_cols
    out_mask = row_in[:, None] & col_in[None, :]

    # Decompose the flattened row index into dims 0 .. ndim-2 and compute the
    # matching input base offset for each row (amortized over COLS columns).
    idx = row
    in_base = tl.zeros((ROWS,), dtype=tl.int32)
    row_valid = tl.full((ROWS,), True, dtype=tl.int1)
    for d in tl.static_range(8):
        if d < ndim - 1:
            if d == 0:
                ostride = rstr0
                isize = isz0
                istride = istr0
                pb = pb0
            elif d == 1:
                ostride = rstr1
                isize = isz1
                istride = istr1
                pb = pb1
            elif d == 2:
                ostride = rstr2
                isize = isz2
                istride = istr2
                pb = pb2
            elif d == 3:
                ostride = rstr3
                isize = isz3
                istride = istr3
                pb = pb3
            elif d == 4:
                ostride = rstr4
                isize = isz4
                istride = istr4
                pb = pb4
            elif d == 5:
                ostride = rstr5
                isize = isz5
                istride = istr5
                pb = pb5
            elif d == 6:
                ostride = rstr6
                isize = isz6
                istride = istr6
                pb = pb6
            else:
                ostride = rstr7
                isize = isz7
                istride = istr7
                pb = pb7
            coord = idx // ostride
            idx = idx - coord * ostride
            in_coord = coord - pb
            ok = (in_coord >= 0) & (in_coord < isize)
            row_valid = row_valid & ok
            in_coord = tl.where(ok, in_coord, 0)
            in_base = in_base + in_coord * istride

    # Last (innermost) dim: pure column logic, no division needed.
    in_col = col - pb_last
    col_valid = (in_col >= 0) & (in_col < in_cols)
    in_col = tl.where(col_valid, in_col, 0)
    valid = row_valid[:, None] & col_valid[None, :]
    if CONTIG:
        in_flat = in_base[:, None] + in_col[None, :]
    else:
        in_flat = in_base[:, None] + in_col[None, :] * istr_last

    value_t = value.to(in_ptr.dtype.element_ty)
    loaded = tl.load(in_ptr + in_flat, mask=valid, other=value_t)
    out_flat = row[:, None] * out_cols + col[None, :]
    tl.store(out_ptr + out_flat, loaded, mask=out_mask)


@triton.jit
def _constant_pad_nd_i64(
    in_ptr,
    out_ptr,
    meta_ptr,  # int64 tensor, shape (ndim, 4): [in_size, in_stride, row_stride, pad_before]
    ndim,  # number of input dims (>= 1)
    out_rows,  # product of output dims except the last one
    out_cols,  # size of the last output dim
    value,  # fill value
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_col_blocks = (out_cols + COLS - 1) // COLS
    pid_c = pid % num_col_blocks
    pid_r = pid // num_col_blocks

    row = pid_r.to(tl.int64) * ROWS + tl.arange(0, ROWS).to(tl.int64)
    col = pid_c.to(tl.int64) * COLS + tl.arange(0, COLS).to(tl.int64)

    row_in = row < out_rows
    col_in = col < out_cols
    out_mask = row_in[:, None] & col_in[None, :]

    idx = row
    in_base = tl.zeros((ROWS,), dtype=tl.int64)
    row_valid = tl.full((ROWS,), True, dtype=tl.int1)
    for d in tl.range(0, ndim - 1):
        ostride = tl.load(meta_ptr + d * 4 + 2)
        coord = idx // ostride
        idx = idx - coord * ostride
        isize = tl.load(meta_ptr + d * 4 + 0)
        istride = tl.load(meta_ptr + d * 4 + 1)
        pad_before = tl.load(meta_ptr + d * 4 + 3)
        in_coord = coord - pad_before
        ok = (in_coord >= 0) & (in_coord < isize)
        row_valid = row_valid & ok
        in_coord = tl.where(ok, in_coord, 0)
        in_base = in_base + in_coord * istride

    d_last = ndim - 1
    pb_last = tl.load(meta_ptr + d_last * 4 + 3)
    istride_last = tl.load(meta_ptr + d_last * 4 + 1)
    in_cols = tl.load(meta_ptr + d_last * 4 + 0)
    in_col = col - pb_last
    col_valid = (in_col >= 0) & (in_col < in_cols)
    in_col = tl.where(col_valid, in_col, 0)

    valid = row_valid[:, None] & col_valid[None, :]
    in_flat = in_base[:, None] + in_col[None, :] * istride_last

    value_t = value.to(in_ptr.dtype.element_ty)
    loaded = tl.load(in_ptr + in_flat, mask=valid, other=value_t)
    out_flat = row[:, None] * out_cols + col[None, :]
    tl.store(out_ptr + out_flat, loaded, mask=out_mask)


@triton.jit
def _constant_pad_nd_2d(
    in_ptr,
    out_ptr,
    out_rows,  # product of output dims except the last one (= out_shape[0])
    out_cols,  # size of the last output dim
    value,  # fill value
    # dim 0 metadata:
    pb0,  # pad before on dim 0
    istr0,  # input stride of dim 0
    isz0,  # input size of dim 0
    # last-dim metadata:
    pb_last,  # pad before on the innermost dim
    in_cols,  # input size of the innermost dim
    istr_last,  # input stride of the innermost dim
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CONTIG: tl.constexpr,
):
    # 2D-specialized path: no row-decomposition loop and a minimal argument
    # list, which measurably reduces per-call launch overhead (important for
    # small tensors) and removes all per-row division work.
    pid = tl.program_id(0)
    num_col_blocks = (out_cols + COLS - 1) // COLS
    pid_c = pid % num_col_blocks
    pid_r = pid // num_col_blocks

    row = pid_r * ROWS + tl.arange(0, ROWS)
    col = pid_c * COLS + tl.arange(0, COLS)
    row_in = row < out_rows
    col_in = col < out_cols
    out_mask = row_in[:, None] & col_in[None, :]

    in_coord0 = row - pb0
    ok0 = (in_coord0 >= 0) & (in_coord0 < isz0)
    in_coord0 = tl.where(ok0, in_coord0, 0)
    in_base = in_coord0 * istr0

    in_col = col - pb_last
    col_valid = (in_col >= 0) & (in_col < in_cols)
    in_col = tl.where(col_valid, in_col, 0)

    valid = ok0[:, None] & col_valid[None, :]
    if CONTIG:
        in_flat = in_base[:, None] + in_col[None, :]
    else:
        in_flat = in_base[:, None] + in_col[None, :] * istr_last

    value_t = value.to(in_ptr.dtype.element_ty)
    loaded = tl.load(in_ptr + in_flat, mask=valid, other=value_t)
    out_flat = row[:, None] * out_cols + col[None, :]
    tl.store(out_ptr + out_flat, loaded, mask=out_mask)


@triton.jit
def _constant_pad_nd_3d(
    in_ptr,
    out_ptr,
    out_rows,  # product of output dims except the last one (= out_shape[0]*out_shape[1])
    out_cols,  # size of the last output dim
    value,  # fill value
    # dim 0 metadata:
    pb0,
    istr0,
    isz0,
    rstr0,
    # dim 1 metadata (row-space stride is 1 by construction, no division):
    pb1,
    istr1,
    isz1,
    # last-dim metadata:
    pb_last,  # pad before on the innermost dim
    in_cols,  # input size of the innermost dim
    istr_last,  # input stride of the innermost dim
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    CONTIG: tl.constexpr,
):
    # 3D-specialized path: one division (dim 0), no division for dim 1 or the
    # innermost dim, and a lean argument list.
    pid = tl.program_id(0)
    num_col_blocks = (out_cols + COLS - 1) // COLS
    pid_c = pid % num_col_blocks
    pid_r = pid // num_col_blocks

    row = pid_r * ROWS + tl.arange(0, ROWS)
    col = pid_c * COLS + tl.arange(0, COLS)
    row_in = row < out_rows
    col_in = col < out_cols
    out_mask = row_in[:, None] & col_in[None, :]

    coord0 = row // rstr0
    idx1 = row - coord0 * rstr0
    in_coord0 = coord0 - pb0
    ok0 = (in_coord0 >= 0) & (in_coord0 < isz0)
    in_coord0 = tl.where(ok0, in_coord0, 0)
    in_base = in_coord0 * istr0

    in_coord1 = idx1 - pb1
    ok1 = (in_coord1 >= 0) & (in_coord1 < isz1)
    row_valid = ok0 & ok1
    in_coord1 = tl.where(ok1, in_coord1, 0)
    in_base = in_base + in_coord1 * istr1

    in_col = col - pb_last
    col_valid = (in_col >= 0) & (in_col < in_cols)
    in_col = tl.where(col_valid, in_col, 0)

    valid = row_valid[:, None] & col_valid[None, :]
    if CONTIG:
        in_flat = in_base[:, None] + in_col[None, :]
    else:
        in_flat = in_base[:, None] + in_col[None, :] * istr_last

    value_t = value.to(in_ptr.dtype.element_ty)
    loaded = tl.load(in_ptr + in_flat, mask=valid, other=value_t)
    out_flat = row[:, None] * out_cols + col[None, :]
    tl.store(out_ptr + out_flat, loaded, mask=out_mask)


def _next_pow2(x):
    return 1 << (max(int(x), 1) - 1).bit_length()


@triton.jit
def _constant_pad_nd_1d(
    in_ptr,
    out_ptr,
    out_total,  # size of the output (1D)
    value,  # fill value
    pb_last,  # pad before
    in_cols,  # input size
    BLOCK: tl.constexpr,
):
    # 1D-specialized path: no row loop, no dead dim branches, minimal args.
    pid = tl.program_id(0)
    col = pid * BLOCK + tl.arange(0, BLOCK)
    omask = col < out_total
    in_col = col - pb_last
    col_valid = (in_col >= 0) & (in_col < in_cols)
    in_col = tl.where(col_valid, in_col, 0)
    value_t = value.to(in_ptr.dtype.element_ty)
    loaded = tl.load(in_ptr + in_col, mask=col_valid & omask, other=value_t)
    tl.store(out_ptr + col, loaded, mask=omask)


def constant_pad_nd(self, pad_list, value=0):
    if isinstance(pad_list, torch.Tensor):
        pad_list = pad_list.tolist()
    pad = [int(p) for p in pad_list]
    if isinstance(value, torch.Tensor):
        value = value.item()

    ndim = self.dim()
    k = len(pad) // 2
    if ndim == 0:
        if k == 0:
            return self.clone()
        raise ValueError("constant_pad_nd: cannot pad a 0-dim tensor")

    # Output shape: the last k dims grow by (before, after) each.
    out_shape = list(self.shape)
    for i in range(k):
        d = ndim - 1 - i
        out_shape[d] = self.shape[d] + pad[2 * i] + pad[2 * i + 1]

    if any(s <= 0 for s in out_shape):
        out_shape = [max(s, 0) for s in out_shape]
        return torch.empty(out_shape, dtype=self.dtype, device=self.device)

    out = torch.empty(out_shape, dtype=self.dtype, device=self.device)

    out_rows = 1
    for s in out_shape[:-1]:
        out_rows *= s
    out_cols = out_shape[-1]

    in_sizes = list(self.shape)
    in_strides = list(self.stride())

    # Strides of dims 0..ndim-2 within the flattened "row" space
    # (out_shape[0]*...*out_shape[ndim-2]); last dim entry is a dummy.
    row_strides = [1] * ndim
    acc = 1
    for d in range(ndim - 2, -1, -1):
        row_strides[d] = acc
        acc *= out_shape[d]

    pad_before = [0] * ndim
    for i in range(k):
        pad_before[ndim - 1 - i] = pad[2 * i]

    in_total = self.numel()
    out_total = out.numel()

    use_i32 = (
        ndim <= MAX_DIM + 1
        and in_total < 2**31
        and out_total < 2**31
        and all(0 <= s < 2**31 for s in in_strides)
        and all(0 <= r < 2**31 for r in row_strides)
        and all(0 <= s < 2**31 for s in in_sizes)
    )

    # Tile selection (empirically tuned on Iluvatar BI-V150 via interleaved A/B):
    # - 1D: 512-wide tiles stream best (~563GB/s vs ~550 at 1024).
    # - 2D with a wide innermost dim: one row per program, 1024 columns
    #   (fp16/bf16 ~538-541GB/s vs ~525-530 at 2048; fp32 576GB/s).
    # - other multi-dim shapes: ~2048 elements per program.
    if ndim == 1:
        ROWS = 1
        COLS = min(_next_pow2(out_cols), 512)
    elif ndim == 2 and out_cols >= 2048:
        ROWS = 1
        COLS = 1024
    else:
        COLS = min(max(_next_pow2(out_cols) // 4, 64), 1024)
        ROWS = max(1, min(2048 // COLS, out_rows, 32))

    grid = (triton.cdiv(out_rows, ROWS) * triton.cdiv(out_cols, COLS),)

    if use_i32 and ndim == 1:
        BLOCK = min(_next_pow2(out_cols), 512)
        grid1 = (triton.cdiv(out_cols, BLOCK),)
        _constant_pad_nd_1d[grid1](
            self,
            out,
            out_cols,
            value,
            pad_before[0],
            in_sizes[0],
            BLOCK=BLOCK,
            num_warps=4,
        )
    elif use_i32 and ndim == 2:
        contig = in_strides[-1] == 1
        _constant_pad_nd_2d[grid](
            self,
            out,
            out_rows,
            out_cols,
            value,
            pad_before[0],
            in_strides[0],
            in_sizes[0],
            pad_before[1],
            in_sizes[1],
            in_strides[1],
            ROWS=ROWS,
            COLS=COLS,
            CONTIG=contig,
            num_warps=4,
        )
    elif use_i32 and ndim == 3:
        contig = in_strides[-1] == 1
        _constant_pad_nd_3d[grid](
            self,
            out,
            out_rows,
            out_cols,
            value,
            pad_before[0],
            in_strides[0],
            in_sizes[0],
            row_strides[0],
            pad_before[1],
            in_strides[1],
            in_sizes[1],
            pad_before[2],
            in_sizes[2],
            in_strides[2],
            ROWS=ROWS,
            COLS=COLS,
            CONTIG=contig,
            num_warps=4,
        )
    elif use_i32:
        dim_args = []
        for d in range(MAX_DIM):
            if d < ndim:
                dim_args += [in_sizes[d], in_strides[d], row_strides[d], pad_before[d]]
            else:
                dim_args += [1, 1, 1, 0]
        contig = in_strides[-1] == 1
        _constant_pad_nd_i32[grid](
            self,
            out,
            ndim,
            out_rows,
            out_cols,
            value,
            *dim_args,
            pad_before[ndim - 1],
            in_strides[ndim - 1],
            in_sizes[ndim - 1],
            ROWS=ROWS,
            COLS=COLS,
            CONTIG=contig,
            num_warps=4,
        )
    else:
        meta = torch.tensor(
            [
                [in_sizes[d], in_strides[d], row_strides[d], pad_before[d]]
                for d in range(ndim)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        _constant_pad_nd_i64[grid](
            self,
            out,
            meta,
            ndim,
            out_rows,
            out_cols,
            value,
            ROWS=ROWS,
            COLS=COLS,
            num_warps=4,
        )
    return out

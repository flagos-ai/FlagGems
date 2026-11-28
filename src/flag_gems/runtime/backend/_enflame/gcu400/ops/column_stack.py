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
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Per-dtype tile caps (elements) for the one-block-per-program arm: a load
# tile plus a store tile stays inside the 448KB dual-program DSM window.
_AFFINE_BLOCK = {2: 65536, 4: 32768, 8: 16384}
_AFFINE_BLOCK_DEFAULT = 32768
_AFFINE_GRID_MAX = 1024
_AFFINE_DIVISOR_MIN = 8192
# Row-loop arm: at most 16384 elements per tile (per-lane side tensors such
# as offsets/masks scale with the tile width and blew the DSM budget at 32K).
_SPANLOOP_TILE_MAX = 16384
_SPANLOOP_INFLIGHT_BYTES = 131072
# Single-launch arm is only worth it while the op is launch-bound.
_ROWS_MAX = 192
_SPAN_SMALL_MAX = 32768
# Fixed grid width of the GCU400 device (SP count); the loop arms below keep
# every program inside one wave.
_NUM_SPC = 48
# Per-program DSM window (bytes) shared by the live tiles of one program.
_DSM_WINDOW = 448 * 1024
# Conservative tile budget for the small row-loop arm (matches the historic
# multi-pointer arm so equal shapes cannot regress past a known-good bound).
_ROWS_ARM_TILE_BUDGET = 256 * 1024


@triton.jit
def column_stack_copy_kernel(
    out_ptr,
    in_ptr,
    dim_size_in,
    dim_size_out,
    dim_prod_post,
    dim_offset,
    total_elements,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0).to(tl.int64)
    block_start = pid_x * BLOCK_X
    offsets = tl.arange(0, BLOCK_X).to(tl.int64)
    idx = block_start + offsets
    mask = idx < total_elements

    # Concatenation happens along `dim` (== 1 after reshaping to 2D-plus).
    pre_idx = idx // (dim_size_in * dim_prod_post)
    dim_idx = (idx // dim_prod_post) % dim_size_in
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post
        + (dim_idx + dim_offset) * dim_prod_post
        + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)


@triton.jit
def _column_stack_affine_kernel(
    out_ptr,
    in_ptr,
    row_stride,
    offset,
    span,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One output-aligned block per program; the block never crosses a row
    # boundary (span % BLOCK == 0), so the row is decoded from the program
    # id with scalar arithmetic and both accesses stay purely affine.
    pid = tl.program_id(0)
    row = pid // K
    col0 = (pid - row * K) * BLOCK
    offs = tl.arange(0, BLOCK)
    data = tl.load(in_ptr + row * span + col0 + offs)
    tl.store(out_ptr + row * row_stride + offset + col0 + offs, data)


@triton.jit
def _column_stack_spanloop_kernel(
    out_ptr,
    in_ptr,
    n_rows,
    span,
    row_stride,
    offset,
    NPROG: tl.constexpr,
    TILE: tl.constexpr,
    NCHUNKS: tl.constexpr,
    U: tl.constexpr,
):
    # Row-strided arm for many-row shapes: each program copies U rows per
    # iteration with all U loads issued before the first store so the
    # independent DMA segments overlap. Row bases are clamped so masked-out
    # lanes never form out-of-range addresses.
    pid = tl.program_id(0)
    n_iter = (n_rows + NPROG * U - 1) // (NPROG * U)
    for it in tl.range(0, n_iter):
        r0 = pid + it * NPROG * U
        r0s = tl.minimum(r0, n_rows - 1)
        ok0 = r0 < n_rows
        if U >= 2:
            r1 = r0 + NPROG
            r1s = tl.minimum(r1, n_rows - 1)
            ok1 = r1 < n_rows
        if U >= 4:
            r2 = r0 + 2 * NPROG
            r2s = tl.minimum(r2, n_rows - 1)
            ok2 = r2 < n_rows
            r3 = r0 + 3 * NPROG
            r3s = tl.minimum(r3, n_rows - 1)
            ok3 = r3 < n_rows
        if U >= 8:
            r4 = r0 + 4 * NPROG
            r4s = tl.minimum(r4, n_rows - 1)
            ok4 = r4 < n_rows
            r5 = r0 + 5 * NPROG
            r5s = tl.minimum(r5, n_rows - 1)
            ok5 = r5 < n_rows
            r6 = r0 + 6 * NPROG
            r6s = tl.minimum(r6, n_rows - 1)
            ok6 = r6 < n_rows
            r7 = r0 + 7 * NPROG
            r7s = tl.minimum(r7, n_rows - 1)
            ok7 = r7 < n_rows
        for j in tl.range(NCHUNKS):
            offs = j * TILE + tl.arange(0, TILE)
            m0 = (offs < span) & ok0
            d0 = tl.load(in_ptr + r0s * span + offs, mask=m0)
            if U >= 2:
                m1 = (offs < span) & ok1
                d1 = tl.load(in_ptr + r1s * span + offs, mask=m1)
            if U >= 4:
                m2 = (offs < span) & ok2
                d2 = tl.load(in_ptr + r2s * span + offs, mask=m2)
                m3 = (offs < span) & ok3
                d3 = tl.load(in_ptr + r3s * span + offs, mask=m3)
            if U >= 8:
                m4 = (offs < span) & ok4
                d4 = tl.load(in_ptr + r4s * span + offs, mask=m4)
                m5 = (offs < span) & ok5
                d5 = tl.load(in_ptr + r5s * span + offs, mask=m5)
                m6 = (offs < span) & ok6
                d6 = tl.load(in_ptr + r6s * span + offs, mask=m6)
                m7 = (offs < span) & ok7
                d7 = tl.load(in_ptr + r7s * span + offs, mask=m7)
            tl.store(out_ptr + r0s * row_stride + offset + offs, d0, mask=m0)
            if U >= 2:
                tl.store(out_ptr + r1s * row_stride + offset + offs, d1, mask=m1)
            if U >= 4:
                tl.store(out_ptr + r2s * row_stride + offset + offs, d2, mask=m2)
                tl.store(out_ptr + r3s * row_stride + offset + offs, d3, mask=m3)
            if U >= 8:
                tl.store(out_ptr + r4s * row_stride + offset + offs, d4, mask=m4)
                tl.store(out_ptr + r5s * row_stride + offset + offs, d5, mask=m5)
                tl.store(out_ptr + r6s * row_stride + offset + offs, d6, mask=m6)
                tl.store(out_ptr + r7s * row_stride + offset + offs, d7, mask=m7)


@triton.jit
def _column_stack_rows_kernel(
    out_ptr,
    p0,
    span0,
    off0,
    B0: tl.constexpr,
    p1,
    span1,
    off1,
    B1: tl.constexpr,
    p2,
    span2,
    off2,
    B2: tl.constexpr,
    p3,
    span3,
    off3,
    B3: tl.constexpr,
    row_stride,
    NT: tl.constexpr,
):
    # Single-launch path for small row counts: program `row` copies every
    # tensor's segment for one output row. All loads are issued before the
    # first store so the independent DMA segments overlap.
    row = tl.program_id(0)
    o0 = tl.arange(0, B0)
    m0 = o0 < span0
    d0 = tl.load(p0 + row * span0 + o0, mask=m0)
    if NT > 1:
        o1 = tl.arange(0, B1)
        m1 = o1 < span1
        d1 = tl.load(p1 + row * span1 + o1, mask=m1)
    if NT > 2:
        o2 = tl.arange(0, B2)
        m2 = o2 < span2
        d2 = tl.load(p2 + row * span2 + o2, mask=m2)
    if NT > 3:
        o3 = tl.arange(0, B3)
        m3 = o3 < span3
        d3 = tl.load(p3 + row * span3 + o3, mask=m3)
    tl.store(out_ptr + row * row_stride + off0 + o0, d0, mask=m0)
    if NT > 1:
        tl.store(out_ptr + row * row_stride + off1 + o1, d1, mask=m1)
    if NT > 2:
        tl.store(out_ptr + row * row_stride + off2 + o2, d2, mask=m2)
    if NT > 3:
        tl.store(out_ptr + row * row_stride + off3 + o3, d3, mask=m3)


@triton.jit
def _column_stack_rowloop_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    n_rows,
    span,
    o0,
    o1,
    o2,
    o3,
    row_stride,
    NPROG: tl.constexpr,
    NT: tl.constexpr,
    B: tl.constexpr,
):
    # Equal-span single-launch arm for small row counts: `NPROG` programs
    # (one wave) walk the rows with a stride loop, issuing every tensor's
    # load before the first store of that row. Spans are powers of two so
    # the loads and stores stay unmasked and purely affine.
    pid = tl.program_id(0)
    offs = tl.arange(0, B)
    for row in tl.range(pid, n_rows, NPROG):
        base_in = row * span
        base_out = row * row_stride
        d0 = tl.load(p0 + base_in + offs)
        if NT > 1:
            d1 = tl.load(p1 + base_in + offs)
        if NT > 2:
            d2 = tl.load(p2 + base_in + offs)
        if NT > 3:
            d3 = tl.load(p3 + base_in + offs)
        tl.store(out_ptr + base_out + o0 + offs, d0)
        if NT > 1:
            tl.store(out_ptr + base_out + o1 + offs, d1)
        if NT > 2:
            tl.store(out_ptr + base_out + o2 + offs, d2)
        if NT > 3:
            tl.store(out_ptr + base_out + o3 + offs, d3)


@triton.jit
def _column_stack_pair_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    n_rows,
    span,
    row_stride,
    o0,
    o1,
    o2,
    o3,
    NPROG: tl.constexpr,
    NT: tl.constexpr,
    TILE: tl.constexpr,
):
    # Many-rows medium-span arm: each program copies two consecutive output
    # rows per iteration (all parts loaded first so the independent DMA
    # segments overlap), then a masked strided tail loop drains the rows the
    # full groups did not cover.
    pid = tl.program_id(0)
    offs = tl.arange(0, TILE)
    n_full = n_rows // (NPROG * 2)
    for it in tl.range(0, n_full):
        r0 = it * NPROG * 2 + pid * 2
        r1 = r0 + 1
        in0 = r0 * span
        in1 = r1 * span
        out0 = r0 * row_stride
        out1 = r1 * row_stride
        a0 = tl.load(p0 + in0 + offs)
        if NT > 1:
            b0 = tl.load(p1 + in0 + offs)
        if NT > 2:
            c0 = tl.load(p2 + in0 + offs)
        if NT > 3:
            d0 = tl.load(p3 + in0 + offs)
        a1 = tl.load(p0 + in1 + offs)
        if NT > 1:
            b1 = tl.load(p1 + in1 + offs)
        if NT > 2:
            c1 = tl.load(p2 + in1 + offs)
        if NT > 3:
            d1 = tl.load(p3 + in1 + offs)
        tl.store(out_ptr + out0 + o0 + offs, a0)
        if NT > 1:
            tl.store(out_ptr + out0 + o1 + offs, b0)
        if NT > 2:
            tl.store(out_ptr + out0 + o2 + offs, c0)
        if NT > 3:
            tl.store(out_ptr + out0 + o3 + offs, d0)
        tl.store(out_ptr + out1 + o0 + offs, a1)
        if NT > 1:
            tl.store(out_ptr + out1 + o1 + offs, b1)
        if NT > 2:
            tl.store(out_ptr + out1 + o2 + offs, c1)
        if NT > 3:
            tl.store(out_ptr + out1 + o3 + offs, d1)
    base = n_full * NPROG * 2
    for row in tl.range(pid + base, n_rows, NPROG):
        base_in = row * span
        base_out = row * row_stride
        a = tl.load(p0 + base_in + offs)
        if NT > 1:
            b = tl.load(p1 + base_in + offs)
        if NT > 2:
            c = tl.load(p2 + base_in + offs)
        if NT > 3:
            d = tl.load(p3 + base_in + offs)
        tl.store(out_ptr + base_out + o0 + offs, a)
        if NT > 1:
            tl.store(out_ptr + base_out + o1 + offs, b)
        if NT > 2:
            tl.store(out_ptr + base_out + o2 + offs, c)
        if NT > 3:
            tl.store(out_ptr + base_out + o3 + offs, d)


@triton.jit
def _column_stack_gs_affine_kernel(
    out_ptr,
    p0,
    p1,
    p2,
    p3,
    span,
    row_stride,
    o0,
    o1,
    o2,
    o3,
    n_work,
    LOGK: tl.constexpr,
    NPROG: tl.constexpr,
    NT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Large-span arm for power-of-two spans: a fixed grid of programs walks
    # the (row, block) work items with a stride loop; the work id decodes
    # into (row, block) with shifts, and every part's block is loaded before
    # the first store so the parts overlap in flight.
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    k = 1 << LOGK
    for w in tl.range(pid, n_work, NPROG):
        row = w >> LOGK
        col0 = (w & (k - 1)) * BLOCK
        base_in = row * span
        base_out = row * row_stride
        d0 = tl.load(p0 + base_in + col0 + offs)
        if NT > 1:
            d1 = tl.load(p1 + base_in + col0 + offs)
        if NT > 2:
            d2 = tl.load(p2 + base_in + col0 + offs)
        if NT > 3:
            d3 = tl.load(p3 + base_in + col0 + offs)
        tl.store(out_ptr + base_out + o0 + col0 + offs, d0)
        if NT > 1:
            tl.store(out_ptr + base_out + o1 + col0 + offs, d1)
        if NT > 2:
            tl.store(out_ptr + base_out + o2 + col0 + offs, d2)
        if NT > 3:
            tl.store(out_ptr + base_out + o3 + col0 + offs, d3)


def _reshape_input(t: torch.Tensor) -> torch.Tensor:
    # Zero or one dimensional tensors are reshaped into a (numel, 1) column.
    if t.ndim <= 1:
        return t.reshape(t.numel(), 1)
    return t


def _next_pow2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _pow2_divisor(value: int) -> int:
    # Largest power of two dividing `value`.
    return value & (-value)


def _launch_generic(
    out: torch.Tensor,
    tensor: torch.Tensor,
    dim_size_out: int,
    dim_prod_post: int,
    dim_offset: int,
) -> None:
    total_elements = tensor.numel()
    if total_elements <= 0:
        return
    want = _next_pow2(-(-total_elements // 48))
    block = min(32768, max(1024, want >> 1))
    grid = (-(-total_elements // block),)
    column_stack_copy_kernel[grid](
        out,
        tensor,
        int(tensor.shape[1]),
        dim_size_out,
        dim_prod_post,
        dim_offset,
        total_elements,
        BLOCK_X=block,
    )


def _launch_affine(
    out: torch.Tensor,
    tensor: torch.Tensor,
    span: int,
    row_stride: int,
    offset: int,
    itemsize: int,
) -> None:
    block = min(_AFFINE_BLOCK.get(itemsize, _AFFINE_BLOCK_DEFAULT), _pow2_divisor(span))
    k = span // block
    _column_stack_affine_kernel[(tensor.shape[0] * k,)](
        out,
        tensor,
        row_stride,
        offset,
        span,
        K=k,
        BLOCK=block,
        num_warps=4,
    )


def _launch_spanloop(
    out: torch.Tensor,
    tensor: torch.Tensor,
    rows: int,
    span: int,
    row_stride: int,
    offset: int,
    itemsize: int,
) -> None:
    tile = min(_SPANLOOP_TILE_MAX, _next_pow2(span))
    nchunks = -(-span // tile)
    u_pref = 8 if itemsize <= 2 else (4 if itemsize == 4 else 2)
    u = min(u_pref, max(1, _SPANLOOP_INFLIGHT_BYTES // (tile * itemsize)))
    u = 1 << max(0, u.bit_length() - 1)
    num_warps = 2 if itemsize <= 2 else (4 if itemsize == 4 else 2)
    _column_stack_spanloop_kernel[(min(48, rows),)](
        out,
        tensor,
        rows,
        span,
        row_stride,
        offset,
        NPROG=min(48, rows),
        TILE=tile,
        NCHUNKS=nchunks,
        U=u,
        num_warps=num_warps,
    )


def _launch_rowloop(
    out: torch.Tensor,
    parts: List,
    rows: int,
    span: int,
    row_stride: int,
    itemsize: int,
) -> None:
    nt = len(parts)
    pointers = [p[0] for p in parts] + [out] * (4 - nt)
    offs = [p[2] for p in parts] + [0] * (4 - nt)
    nprog = min(_NUM_SPC, rows)
    num_warps = max(1, min(4, span >> 8))
    _column_stack_rowloop_kernel[(nprog,)](
        out,
        pointers[0],
        pointers[1],
        pointers[2],
        pointers[3],
        rows,
        span,
        offs[0],
        offs[1],
        offs[2],
        offs[3],
        row_stride,
        NPROG=nprog,
        NT=nt,
        B=span,
        num_warps=num_warps,
    )


def _launch_pair(
    out: torch.Tensor,
    parts: List,
    rows: int,
    span: int,
    row_stride: int,
) -> None:
    nt = len(parts)
    pointers = [p[0] for p in parts] + [out] * (4 - nt)
    offs = [p[2] for p in parts] + [0] * (4 - nt)
    _column_stack_pair_kernel[(_NUM_SPC,)](
        out,
        pointers[0],
        pointers[1],
        pointers[2],
        pointers[3],
        rows,
        span,
        row_stride,
        offs[0],
        offs[1],
        offs[2],
        offs[3],
        NPROG=_NUM_SPC,
        NT=nt,
        TILE=span,
        num_warps=max(1, min(4, span >> 8)),
    )


def _launch_gs_affine(
    out: torch.Tensor,
    parts: List,
    rows: int,
    span: int,
    row_stride: int,
    itemsize: int,
) -> None:
    nt = len(parts)
    pointers = [p[0] for p in parts] + [out] * (4 - nt)
    offs = [p[2] for p in parts] + [0] * (4 - nt)
    block = 32768 if itemsize <= 4 else 16384
    while nt * block * itemsize > _DSM_WINDOW and block > 4096:
        block //= 2
    k = span // block
    n_work = rows * k
    nprog = min(_NUM_SPC, n_work)
    _column_stack_gs_affine_kernel[(nprog,)](
        out,
        pointers[0],
        pointers[1],
        pointers[2],
        pointers[3],
        span,
        row_stride,
        offs[0],
        offs[1],
        offs[2],
        offs[3],
        n_work,
        LOGK=k.bit_length() - 1,
        NPROG=nprog,
        NT=nt,
        BLOCK=block,
        num_warps=2,
    )


def _try_launch_equal_span_arms(
    out: torch.Tensor,
    parts: List,
    rows: int,
    row_stride: int,
    itemsize: int,
) -> bool:
    # Fast paths for the common equal-span layouts: a single launch covers
    # every part with unmasked affine addressing. Returns True when one of
    # the fast arms handled the copy.
    nt = len(parts)
    if nt > 4:
        return False
    spans = {span for _, span, _ in parts}
    if len(spans) != 1:
        return False
    span = parts[0][1]
    if span & (span - 1):
        return False
    if rows * row_stride >= 2**31:
        return False
    if rows <= _ROWS_MAX and span < _SPAN_SMALL_MAX:
        if nt * span * itemsize <= _ROWS_ARM_TILE_BUDGET:
            _launch_rowloop(out, parts, rows, span, row_stride, itemsize)
            return True
        return False
    if span >= _SPAN_SMALL_MAX:
        _launch_gs_affine(out, parts, rows, span, row_stride, itemsize)
        return True
    if (
        rows > _ROWS_MAX
        and 4 <= span <= _SPANLOOP_TILE_MAX
        and 2 * nt * span * itemsize <= _DSM_WINDOW
    ):
        _launch_pair(out, parts, rows, span, row_stride)
        return True
    return False


def column_stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME COLUMN_STACK")

    if len(tensors) == 0:
        raise RuntimeError("column_stack expected a non-empty TensorList")

    reshaped = [_reshape_input(t) for t in tensors]

    inp0_shape = list(reshaped[0].shape)
    dim = 1

    for tensor_num, tensor in enumerate(reshaped[1:]):
        if tensor.ndim != reshaped[0].ndim:
            raise RuntimeError(
                f"Tensors must have same number of dimensions: got "
                f"{reshaped[0].ndim} and {tensor.ndim}"
            )
        inp_shape = list(tensor.shape)
        for i in range(len(inp_shape)):
            if i != dim and inp_shape[i] != inp0_shape[i]:
                raise RuntimeError(
                    f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected size {inp0_shape[i]} but got size {inp_shape[i]} "
                    f"for tensor number {tensor_num + 1} in the list."
                )

    # Type promotion: find the common dtype for all tensors.
    dtype = reshaped[0].dtype
    for t in reshaped[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    reshaped = [t.to(dtype) if t.dtype != dtype else t for t in reshaped]
    device = reshaped[0].device

    out_shape = list(inp0_shape)
    out_shape[dim] = sum(int(t.shape[dim]) for t in reshaped)

    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=device)
    elif tuple(out.shape) != tuple(out_shape):
        out = out.view(out_shape)

    dim_prod_post = 1
    for s in inp0_shape[dim + 1 :]:
        dim_prod_post *= s

    dim_size_out = out_shape[dim]
    row_stride = dim_size_out * dim_prod_post
    rows = int(reshaped[0].shape[0])

    # Collect the non-empty parts: (tensor, span, element offset). A span is
    # the contiguous per-row chunk each tensor contributes to the output.
    parts = []
    offset = 0
    for tensor in reshaped:
        tensor = tensor.contiguous()
        span = int(tensor.shape[dim]) * dim_prod_post
        if tensor.numel() > 0:
            parts.append((tensor, span, offset))
        offset += span
    if not parts:
        return out

    itemsize = out.element_size()
    out_numel = out.numel()
    fast_global = out.is_contiguous() and out_numel < 2**31

    if fast_global and _try_launch_equal_span_arms(
        out, parts, rows, row_stride, itemsize
    ):
        return out

    if (
        fast_global
        and rows <= _ROWS_MAX
        and len(parts) <= 4
        and rows * row_stride < 2**31
        and all(span < _SPAN_SMALL_MAX for _, span, _ in parts)
    ):
        # Single launch covering every tensor; the live tile budget keeps the
        # DSM footprint inside the dual-program window.
        tiles = [_next_pow2(max(2, span)) for _, span, _ in parts]
        if sum(tiles) * itemsize <= 256 * 1024:
            tiles = tiles + [2] * (4 - len(tiles))
            pointers = [p[0] for p in parts] + [out] * (4 - len(parts))
            spans = [p[1] for p in parts] + [2] * (4 - len(parts))
            offs = [p[2] for p in parts] + [0] * (4 - len(parts))
            num_warps = max(1, min(4, max(tiles) >> 8))
            _column_stack_rows_kernel[(rows,)](
                out,
                pointers[0],
                spans[0],
                offs[0],
                tiles[0],
                pointers[1],
                spans[1],
                offs[1],
                tiles[1],
                pointers[2],
                spans[2],
                offs[2],
                tiles[2],
                pointers[3],
                spans[3],
                offs[3],
                tiles[3],
                row_stride,
                NT=len(parts),
                num_warps=num_warps,
            )
            return out

    for tensor, span, part_offset in parts:
        if not fast_global:
            _launch_generic(
                out, tensor, dim_size_out, dim_prod_post, part_offset // dim_prod_post
            )
            continue
        affine_ok = False
        if span >= _SPAN_SMALL_MAX:
            block = min(
                _AFFINE_BLOCK.get(itemsize, _AFFINE_BLOCK_DEFAULT), _pow2_divisor(span)
            )
            k = span // block
            affine_ok = (
                block >= _AFFINE_DIVISOR_MIN
                and tensor.shape[0] * k <= _AFFINE_GRID_MAX
                and tensor.shape[0] * k <= 65535
            )
        if affine_ok:
            _launch_affine(out, tensor, span, row_stride, part_offset, itemsize)
        else:
            _launch_spanloop(out, tensor, rows, span, row_stride, part_offset, itemsize)

    return out


def column_stack_out(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME COLUMN_STACK_OUT")
    return column_stack(tensors, out=out)

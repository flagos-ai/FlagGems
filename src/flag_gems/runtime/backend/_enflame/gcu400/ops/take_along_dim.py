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

# Row-persistent kernel fills the device once the row count reaches this bound;
# below it the col-block kernel parallelizes over rows on grid axis 1 instead.
_ROW_KERNEL_MIN_ROWS = 48
_GRID_CAP = 48
_MAX_ROWS_PER_TILE = 16
_FLAT_KERNEL_MAX_N = 32768
_FLAT_BLOCK = 1024
_FLAT_MIN_PROGRAMS = 256
_FLAT_ITERS = 16


@libentry()
@triton.jit(do_not_specialize=["R", "D_in", "D_out", "N"])
def take_along_dim_flat_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    R,
    D_in,
    D_out,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    nprog = tl.num_programs(axis=0)
    lane = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    n_full = (N // BLOCK_SIZE) * BLOCK_SIZE

    for base in range(pid * BLOCK_SIZE, n_full, nprog * BLOCK_SIZE):
        offs = base + lane
        r = offs // D_out
        idx = tl.load(idx_ptr + offs).to(tl.int64)
        idx = tl.where(idx < 0, idx + D_in, idx)
        valid = (idx >= 0) & (idx < D_in)
        tl.device_assert(valid, "take_along_dim(): index out of range")
        in_off = r * D_in + idx
        vals = tl.load(inp_ptr + in_off, mask=valid, other=0)
        tl.store(out_ptr + offs, vals)

    # Trailing partial tile: handled once by program 0, with a mask.
    if n_full < N:
        if pid == 0:
            offs = n_full + lane
            mask = offs < N
            r = offs // D_out
            idx = tl.load(idx_ptr + offs, mask=mask, other=0).to(tl.int64)
            idx = tl.where(idx < 0, idx + D_in, idx)
            valid = (idx >= 0) & (idx < D_in)
            tl.device_assert(valid | (~mask), "take_along_dim(): index out of range")
            in_off = r * D_in + idx
            vals = tl.load(inp_ptr + in_off, mask=mask & valid, other=0)
            tl.store(out_ptr + offs, vals, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["R", "D_in", "D_out"])
def take_along_dim_row_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    R,
    D_in,
    D_out,
    PF: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT_ROWS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather along the last axis of a contiguous (R, D_in) view.

    The flat decomposition `r = offs // D_out` used by the generic kernel puts a
    per-element vector division and a vector-row subscript inside the load
    address; on gcu400 both lower to the slow per-lane gather path and cost
    1.7-2.5x versus scalar-row addressing (16.7M-elem shapes: 1151 -> 464us).
    Here the row index is a scalar induction variable and only the gathered
    column offset stays a vector, so the idx load / output store are plain
    contiguous tiles and the input load is the only gather.

    ROWS rows are processed per loop iteration so narrow rows still fill a
    whole lane tile (fp16 512-wide rows want 8 rows: 1099 -> 461us).  PF keeps
    a software-pipelined idx prefetch for the next row-group in flight across
    iterations, overlapping the gather latency of the current group (1024^3
    fp16: 35.6 -> 28.4ms).
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lane = tl.arange(0, BLOCK_D * ROWS).to(tl.int64)
    r_in_lane = lane // BLOCK_D
    c = lane % BLOCK_D
    idx_off = r_in_lane * D_out + c
    in_off_row = r_in_lane * D_in
    nrowg = (R + ROWS - 1) // ROWS
    if PF:
        idx_next = tl.load(
            idx_ptr + pid * ROWS * D_out + idx_off,
            mask=((pid * ROWS + r_in_lane) < R) & (c < D_out),
            other=0,
        )
    for rowg in range(pid, nrowg, nprog):
        row_top = rowg * ROWS
        if PF:
            idx_cur = idx_next
            nxt = rowg + nprog
            if EVEN_COLS:
                pfm = (nxt * ROWS + r_in_lane) < R
            else:
                pfm = ((nxt * ROWS + r_in_lane) < R) & (c < D_out)
            idx_next = tl.load(
                idx_ptr + nxt * ROWS * D_out + idx_off, mask=pfm, other=0
            )
        else:
            if EXACT_ROWS and EVEN_COLS:
                idx_cur = tl.load(idx_ptr + row_top * D_out + idx_off)
            else:
                lm = (row_top + r_in_lane) < R
                if not EVEN_COLS:
                    lm = lm & (c < D_out)
                idx_cur = tl.load(idx_ptr + row_top * D_out + idx_off, mask=lm, other=0)
        idx = tl.where(idx_cur < 0, idx_cur + D_in, idx_cur)
        if EXACT_ROWS and EVEN_COLS:
            vm = (idx >= 0) & (idx < D_in)
        else:
            vm = ((row_top + r_in_lane) < R) & (idx >= 0) & (idx < D_in)
            if not EVEN_COLS:
                vm = vm & (c < D_out)
        vals = tl.load(inp_ptr + row_top * D_in + in_off_row + idx, mask=vm, other=0)
        tl.device_assert(vm, "take_along_dim(): index out of range")
        if EXACT_ROWS and EVEN_COLS:
            tl.store(out_ptr + row_top * D_out + idx_off, vals)
        else:
            sm = (row_top + r_in_lane) < R
            if not EVEN_COLS:
                sm = sm & (c < D_out)
            tl.store(out_ptr + row_top * D_out + idx_off, vals, mask=sm)


@libentry()
@triton.jit(do_not_specialize=["R", "D_in", "D_out", "NCB"])
def take_along_dim_row_multi_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    R,
    D_in,
    D_out,
    NCB,
    EVEN_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Wide rows (D_out > BLOCK_D): each program strides over rows and loops
    over the column blocks of one row per iteration (4096-wide fp16 rows run
    2x2048 blocks: 476us vs 510us for a single 4096-lane tile)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    for row in range(pid, R, nprog):
        in_base = row * D_in
        idx_base = row * D_out
        for cb in range(0, NCB):
            base_c = cb * BLOCK_D
            if EVEN_COLS:
                idx_cur = tl.load(idx_ptr + idx_base + base_c + cols)
            else:
                idx_cur = tl.load(
                    idx_ptr + idx_base + base_c + cols,
                    mask=(base_c + cols) < D_out,
                    other=0,
                )
            idx = tl.where(idx_cur < 0, idx_cur + D_in, idx_cur)
            if EVEN_COLS:
                vm = (idx >= 0) & (idx < D_in)
            else:
                vm = ((base_c + cols) < D_out) & (idx >= 0) & (idx < D_in)
            vals = tl.load(inp_ptr + in_base + idx, mask=vm, other=0)
            tl.device_assert(vm, "take_along_dim(): index out of range")
            if EVEN_COLS:
                tl.store(out_ptr + idx_base + base_c + cols, vals)
            else:
                tl.store(
                    out_ptr + idx_base + base_c + cols,
                    vals,
                    mask=(base_c + cols) < D_out,
                )


@libentry()
@triton.jit(do_not_specialize=["R", "D_in", "D_out"])
def take_along_dim_col_kernel(
    inp_ptr,
    idx_ptr,
    out_ptr,
    R,
    D_in,
    D_out,
    PF: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Small-R companion: rows map to grid axis 1 (<= 255), col blocks stride
    over axis 0.  Keeps the single-row case (dim=None flattening, 1-D tensors)
    parallel -- a row-stride loop would serialize 2^30 elements on one program.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    row = tl.program_id(1)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    ncb = tl.cdiv(D_out, BLOCK_D)
    if PF:
        idx_next = tl.load(
            idx_ptr + row * D_out + pid * BLOCK_D + cols,
            mask=(pid * BLOCK_D + cols) < D_out,
            other=0,
        )
    for cb in range(pid, ncb, nprog):
        base_c = cb * BLOCK_D
        if PF:
            idx_cur = idx_next
            nxt = cb + nprog
            idx_next = tl.load(
                idx_ptr + row * D_out + nxt * BLOCK_D + cols,
                mask=(nxt * BLOCK_D + cols) < D_out,
                other=0,
            )
        else:
            if EVEN_COLS:
                idx_cur = tl.load(idx_ptr + row * D_out + base_c + cols)
            else:
                idx_cur = tl.load(
                    idx_ptr + row * D_out + base_c + cols,
                    mask=(base_c + cols) < D_out,
                    other=0,
                )
        idx = tl.where(idx_cur < 0, idx_cur + D_in, idx_cur)
        if EVEN_COLS:
            vm = (idx >= 0) & (idx < D_in)
        else:
            vm = ((base_c + cols) < D_out) & (idx >= 0) & (idx < D_in)
        vals = tl.load(inp_ptr + row * D_in + idx, mask=vm, other=0)
        tl.device_assert(vm, "take_along_dim(): index out of range")
        if EVEN_COLS:
            tl.store(out_ptr + row * D_out + base_c + cols, vals)
        else:
            tl.store(
                out_ptr + row * D_out + base_c + cols,
                vals,
                mask=(base_c + cols) < D_out,
            )


def _launch_gather(inp, idx, out, R, D_in, D_out):
    N = R * D_out
    if N == 0:
        return
    with torch_device_fn.device(inp.device):
        if N <= _FLAT_KERNEL_MAX_N:
            nblocks = triton.cdiv(N, _FLAT_BLOCK)
            nprog = max(
                min(nblocks, _FLAT_MIN_PROGRAMS), triton.cdiv(nblocks, _FLAT_ITERS)
            )
            take_along_dim_flat_kernel[(nprog,)](
                inp,
                idx,
                out,
                R,
                D_in,
                D_out,
                N,
                BLOCK_SIZE=_FLAT_BLOCK,
                num_warps=4,
            )
            return
        if R >= _ROW_KERNEL_MIN_ROWS:
            itemsize = inp.element_size()
            BLOCK_D = min(triton.next_power_of_2(D_out), 2048)
            ncol_blocks = triton.cdiv(D_out, BLOCK_D)
            if ncol_blocks == 1:
                # 2048 lanes per iteration measured best on this backend once
                # the idx prefetch is off (1024-wide rows: fp16 130 -> 93us,
                # fp32 101 -> 65us), so keep the tile at 2 rows of 1024 lanes
                # for wide rows and fall back to the long-row widening only
                # when the row loop is long enough to amortize it (R >= 2**18).
                lanes = 2048
                if R >= (1 << 18):
                    lanes = 4096 if itemsize < 4 else 2048
                ROWS = lanes // BLOCK_D
                ROWS = min(ROWS, _MAX_ROWS_PER_TILE)
                ROWS = max(ROWS, 1)
                # A carried idx prefetch only pays off for single-row tiles of
                # 16-bit data (latency bound); on multi-row fp32 tiles it
                # measured 13% slower (1024^3).
                PF = (itemsize < 4) and (ROWS == 1)
                grid = (min(triton.cdiv(R, ROWS), _GRID_CAP),)
                take_along_dim_row_kernel[grid](
                    inp,
                    idx,
                    out,
                    R,
                    D_in,
                    D_out,
                    PF=PF,
                    ROWS=ROWS,
                    EXACT_ROWS=(R % ROWS == 0),
                    EVEN_COLS=(D_out % BLOCK_D == 0),
                    BLOCK_D=BLOCK_D,
                    num_warps=4,
                )
            else:
                take_along_dim_row_multi_kernel[(min(R, _GRID_CAP),)](
                    inp,
                    idx,
                    out,
                    R,
                    D_in,
                    D_out,
                    ncol_blocks,
                    EVEN_COLS=(D_out % BLOCK_D == 0),
                    BLOCK_D=BLOCK_D,
                    num_warps=4,
                )
        else:
            cap = 16384 if inp.element_size() < 4 else 8192
            BLOCK_D = min(triton.next_power_of_2(D_out), cap)
            grid = (min(triton.cdiv(D_out, BLOCK_D), _GRID_CAP), R)
            take_along_dim_col_kernel[grid](
                inp,
                idx,
                out,
                R,
                D_in,
                D_out,
                PF=True,
                EVEN_COLS=(D_out % BLOCK_D == 0),
                BLOCK_D=BLOCK_D,
                num_warps=4,
            )


def _fast_layout(input, indices, dim):
    """Return (R, D_in, D_out, out_shape) when the gather reduces to a
    contiguous 2D (R, D) problem with no broadcasting, i.e. exactly the
    layout the kernels above assume.  Otherwise return None and let the
    general path run."""
    if indices.dtype not in (torch.int64, torch.int32):
        return None
    if dim is None:
        # dim=None gathers over the flattened input and returns a 1-D result
        # of indices.numel() elements (same contract as the generic path).
        if input.is_contiguous() and indices.is_contiguous():
            return (1, input.numel(), indices.numel(), (indices.numel(),))
        return None
    ndim = input.ndim
    if ndim == 0 or indices.ndim != ndim:
        return None
    dim = dim % ndim
    if dim != ndim - 1:
        return None
    if not (input.is_contiguous() and indices.is_contiguous()):
        return None
    for d in range(ndim):
        if d != dim and input.shape[d] != indices.shape[d]:
            return None
    R = 1
    for d in range(ndim):
        if d != dim:
            R *= indices.shape[d]
    return (R, input.shape[dim], indices.shape[dim], tuple(indices.shape))


def _prepare(input, indices, dim):
    """Reduce take_along_dim to a 2D (R, D) gather.

    Returns (inp2d, idx2d, out_shape, restore) where ``restore`` maps the flat
    (R, D_out) output back to the final tensor layout.
    """
    if dim is None:
        # Flatten the input; the result has the shape of ``indices``.
        inp2d = input.reshape(1, -1).contiguous()
        idx2d = indices.reshape(1, -1).contiguous()
        # dim=None flattens both operands; the result is 1-D.
        out_shape = (indices.numel(),)

        def restore_flat(out2d):
            return out2d.reshape(out_shape)

        return inp2d, idx2d, out_shape, restore_flat

    if input.ndim != indices.ndim:
        raise RuntimeError(
            "take_along_dim: input and indices should have the same "
            f"number of dimensions, but got {input.ndim} dimensions for input, "
            f"and {indices.ndim} dimensions for indices"
        )

    dim = dim % input.ndim

    # Broadcast every non-``dim`` axis; the ``dim`` axis keeps each tensor's own
    # size (input provides the value space, indices the output extent).
    batch_shape = []
    for d in range(input.ndim):
        if d == dim:
            batch_shape.append(1)
        else:
            batch_shape.append(max(input.shape[d], indices.shape[d]))

    inp_shape = list(batch_shape)
    inp_shape[dim] = input.shape[dim]
    idx_shape = list(batch_shape)
    idx_shape[dim] = indices.shape[dim]

    inp_b = torch.broadcast_to(input, inp_shape)
    idx_b = torch.broadcast_to(indices, idx_shape)

    # Move ``dim`` to the last axis and collapse the batch dims into rows.
    inp_p = inp_b.movedim(dim, -1).contiguous()
    idx_p = idx_b.movedim(dim, -1).contiguous()

    D_in = input.shape[dim]
    D_out = indices.shape[dim]
    inp2d = inp_p.reshape(-1, D_in)
    idx2d = idx_p.reshape(-1, D_out)

    # Final output shape: broadcasted batch dims with the ``dim`` axis == D_out.
    out_shape = list(batch_shape)
    out_shape[dim] = D_out
    out_shape = tuple(out_shape)
    permuted_shape = tuple(idx_p.shape)

    def restore_nd(out2d):
        return out2d.reshape(permuted_shape).movedim(-1, dim).contiguous()

    return inp2d, idx2d, out_shape, restore_nd


def take_along_dim(input, indices, dim=None):
    logger.debug("GEMS_ENFLAME TAKE_ALONG_DIM")
    if input.device != indices.device:
        raise RuntimeError("input and indices must be on the same device")
    fast = _fast_layout(input, indices, dim)
    if fast is not None:
        R, D_in, D_out, out_shape = fast
        out = torch.empty(out_shape, device=input.device, dtype=input.dtype)
        _launch_gather(input, indices, out, R, D_in, D_out)
        return out
    inp2d, idx2d, out_shape, restore = _prepare(input, indices, dim)
    out2d = torch.empty(idx2d.shape, device=input.device, dtype=input.dtype)
    _launch_gather(inp2d, idx2d, out2d, inp2d.shape[0], inp2d.shape[1], idx2d.shape[1])
    return restore(out2d)


def take_along_dim_out(input, indices, dim=None, *, out):
    logger.debug("GEMS_ENFLAME TAKE_ALONG_DIM_OUT")
    if not (input.device == indices.device == out.device):
        raise RuntimeError("input, indices and out must be on the same device")
    if out.dtype != input.dtype:
        raise RuntimeError(f"out must have dtype {input.dtype}, but got {out.dtype}")
    fast = _fast_layout(input, indices, dim)
    if fast is not None and out.is_contiguous() and tuple(out.shape) == fast[3]:
        # The gather output already lands in the caller's buffer in the common
        # contiguous last-dim case: skipping the intermediate buffer + copy_
        # halves the extra memory traffic and removes the second kernel.
        R, D_in, D_out, _ = fast
        _launch_gather(input, indices, out, R, D_in, D_out)
        return out
    inp2d, idx2d, out_shape, restore = _prepare(input, indices, dim)
    out2d = torch.empty(idx2d.shape, device=input.device, dtype=input.dtype)
    _launch_gather(inp2d, idx2d, out2d, inp2d.shape[0], inp2d.shape[1], idx2d.shape[1])
    result = restore(out2d)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out

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
import math

import torch
import triton
import triton.language as tl

from flag_gems import runtime  # noqa: F401
from flag_gems.ops.sum import sum_dim
from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.copy import copy_
from flag_gems.utils import dim_compress, libentry, pointwise_dynamic

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def nanmean_kernel_1(
    inp,
    mid_sum,
    mid_cnt,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    # Persistent partial pass (2026-11-17 probe, gcu400): grid-stride tiles,
    # per-tile immediate dual 1D trees.  Count tree input is 16-bit (2B/elem)
    # with fp32 accumulation -- the i32 count tree measured ~15x slower on
    # (1024,1024,512) fp16 (9.84ms vs 0.96ms row analogue).  fp32 partials.
    if tl.constexpr(out.dtype.element_ty == tl.float32):
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        cnt_in_dtype = tl.float16
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        cnt_in_dtype = tl.float16
    else:
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32

    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    num_tile = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    for tile_id in tl.range(pid, num_tile, num_prog):
        offset = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        # masked load without other: on gcu400, fp32 masked load + other on large tiles
        # silently returns wrong values (compiler_issues bucket); masked lanes are backed up by valid = mask & (x==x).
        x = tl.load(inp + offset, mask=mask).to(value_dtype)
        valid = mask & (x == x)
        x0 = tl.where(valid, x, tl.zeros([BLOCK_SIZE], dtype=x.dtype))
        vh = valid.to(cnt_in_dtype)
        tl.store(mid_sum + tile_id, tl.sum(x0, axis=0, dtype=tl.float32))
        tl.store(mid_cnt + tile_id, tl.sum(vh, axis=0, dtype=tl.float32))


@libentry()
@triton.jit
def nanmean_kernel_2(
    mid_sum,
    mid_cnt,
    out,
    mid_size,
    BLOCK_SIZE: tl.constexpr,
):
    acc_dtype = (
        tl.float32
        if tl.constexpr(mid_sum.dtype.element_ty == tl.float32)
        else tl.float32
    )

    _sum = tl.zeros((), dtype=acc_dtype)
    _cnt = tl.zeros((), dtype=acc_dtype)

    for start in range(0, mid_size, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < mid_size
        sv = tl.load(mid_sum + idx, mask=mask, other=0.0).to(acc_dtype)
        cv = tl.load(mid_cnt + idx, mask=mask, other=0.0).to(acc_dtype)
        _sum += tl.sum(sv, axis=0, dtype=acc_dtype)
        _cnt += tl.sum(cv, axis=0)

    tl.store(out, _sum / _cnt)


@libentry()
@triton.jit
def nanmean_global_single_kernel(
    inp,
    out,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(out.dtype.element_ty == tl.float32):
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        cnt_in_dtype = tl.float16
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        cnt_in_dtype = tl.float16
    else:
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32

    _sum = tl.zeros((), dtype=tl.float32)
    _cnt = tl.zeros((), dtype=tl.float32)

    for start in range(0, M, BLOCK_SIZE):
        idx = start + tl.arange(0, BLOCK_SIZE)
        mask = idx < M
        # masked load without other: masked lanes are backed up by valid = mask & (x==x) (see
        # the nanmean_kernel_1 comment, avoiding the gcu400 masked-load+other large-tile wrong-value issue).
        x = tl.load(inp + idx, mask=mask).to(value_dtype)
        valid = mask & (x == x)
        x0 = tl.where(valid, x, tl.zeros([BLOCK_SIZE], dtype=x.dtype))
        vh = valid.to(cnt_in_dtype)
        _sum += tl.sum(x0, axis=0, dtype=tl.float32)
        _cnt += tl.sum(vh, axis=0, dtype=tl.float32)

    tl.store(out, _sum / _cnt)


def _nanmean_global(inp, *, dtype=None, out_flat=None):
    if dtype is None:
        dtype = inp.dtype

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()

    if out_flat is not None:
        out = out_flat
    else:
        out = torch.empty([], dtype=dtype, device=inp.device)

    if M == 0:
        out.fill_(float("nan"))
        return out

    if M <= 32768:
        with torch_device_fn.device(inp.device):
            nanmean_global_single_kernel[(1,)](
                inp, out, M, BLOCK_SIZE=max(16, triton.next_power_of_2(M))
            )
        return out

    # Partial + merge (2026-11-17 probe, gcu400): block 64K elem (2B) / 32K (4B),
    # grid capped at 48 persistent programs, nw4; merge single-program nw1.
    # Old sqrt-split (grid = cdiv(M, 4096) one-tile programs + 32K-wide merge)
    # measured 0.254ms on (4096,4096) fp16 vs 0.065ms here (-74%).
    element_size = inp.element_size()
    block_size = 65536 if element_size == 2 else 32768
    block_size = min(triton.next_power_of_2(M), block_size)
    mid_size = triton.cdiv(M, block_size)
    mid_sum = torch.empty(mid_size, dtype=torch.float32, device=inp.device)
    mid_cnt = torch.empty(mid_size, dtype=torch.float32, device=inp.device)

    with torch_device_fn.device(inp.device):
        nanmean_kernel_1[(min(mid_size, NANMEAN_MAX_GRID), 1, 1)](
            inp, mid_sum, mid_cnt, out, M, block_size, num_warps=4
        )
        block_mid = triton.next_power_of_2(mid_size)
        nanmean_kernel_2[(1,)](
            mid_sum, mid_cnt, out, mid_size, BLOCK_SIZE=block_mid, num_warps=1
        )

    return out


# Tile/grid parameters of the same origin as gcu400 sum/mean.py row reduction (calibrated by probes on 2026-11-17).
# grid cap 48 (GCU400 has 48 SPs): persistent strided loop, always <= 48.
NANMEAN_MAX_GRID = 48
# Row-reduction tile byte budget: fp16 64KB / fp32 128KB (probes measured 64x512 fp16 nw4
# (1024,1024,512) optimal at 0.96ms; a 128KB tile is worse at 1.20ms). Per tile, the nan mechanism
# additionally keeps two 2B copies x0/vh plus where intermediates, with peak DSM ~5.2x the tile bytes,
# so a 64KB tile => ~333KB < the 448KB threshold for two concurrent programs per SP.
NANMEAN_ROW_TILE_BYTES_2B = 65536
NANMEAN_ROW_TILE_BYTES_4B = 131072
NANMEAN_ROW_MAX_WIDTH = 4096
# Task kernel (non-last-dimension reduction) column-width cap 512 (same as mean.py: the axis reduction tree has a known
# slow path for contiguous dimension <=256); when B*cdiv(N,bn) is below 32, shrink bn down to 128 to
# increase parallel tasks (measured on (4096,4096) dim=0: bn128/bm256/nw1 is optimal at 0.073ms).
NANMEAN_TASK_MAX_WIDTH = 512
NANMEAN_TASK_MIN_WIDTH = 128
NANMEAN_TASK_MIN_TASKS = 32


def _row_params(element_size, M, N):
    """Row-reduction hyperparameters: bn = min(np2(N), 4096), bm from the byte budget and 48-way spreading."""
    block_n = min(triton.next_power_of_2(N), NANMEAN_ROW_MAX_WIDTH)
    tile_budget = (
        NANMEAN_ROW_TILE_BYTES_4B if element_size >= 4 else NANMEAN_ROW_TILE_BYTES_2B
    )
    tile_rows = max(1024, tile_budget // element_size) // block_n
    batch_rows = triton.next_power_of_2(triton.cdiv(M, NANMEAN_MAX_GRID))
    block_m = max(1, min(tile_rows, batch_rows))
    grid = min(NANMEAN_MAX_GRID, triton.cdiv(M, block_m))
    lanes = block_m * block_n
    if lanes * element_size >= 16384:
        num_warps = 4
    elif lanes >= 2048:
        num_warps = 2
    else:
        num_warps = 1
    return block_m, block_n, grid, num_warps


def _task_params(element_size, B, M, N):
    """Non-last-dim reduction hyperparameters: a task is (batch, column tile), the grid strides over tasks."""
    block_n = min(triton.next_power_of_2(N), NANMEAN_TASK_MAX_WIDTH)
    while (
        block_n > NANMEAN_TASK_MIN_WIDTH
        and B * triton.cdiv(N, block_n) < NANMEAN_TASK_MIN_TASKS
    ):
        block_n //= 2
    if block_n >= 512:
        block_m_cap = 64
    else:
        block_m_cap = 256 if element_size <= 2 else 128
    block_m = max(1, min(triton.next_power_of_2(M), block_m_cap))
    if block_n >= 512:
        num_warps = 4
    elif block_n >= 256:
        num_warps = 2
    else:
        num_warps = 1
    grid = min(NANMEAN_MAX_GRID, B * triton.cdiv(N, block_n))
    return block_m, block_n, grid, num_warps


@libentry()
@triton.jit
def nanmean_dim_inner_kernel(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Row reduction with the reduced dim innermost: persistent row-tile walk, grid <= 48.

    2026-11-17 probe calibration (gcu400, official do_bench):
    - the count tree takes a 16-bit (fp16) input with dtype=fp32 accumulation -- an i32 count tree
      on (1024,1024,512) fp16 measured 9.84ms vs 0.96ms for this recipe (~10x) and was the main
      pathology of the old implementation;
    - tl.sum(x, axis, dtype=tl.float32) keeps the tree input at 2B per element, avoiding a 4B
      copy materialized by .to(f32);
    - EVEN (M%bm==0 and N%bn==0) takes the mask/other-free hot path (merged loads) and every bench
      shape hits it; non-divisible shapes take the masked tail path.
    """
    if tl.constexpr(out.dtype.element_ty == tl.float32):
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        cnt_in_dtype = tl.float16
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        cnt_in_dtype = tl.float16
    else:
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tile, step):
        rows = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        sum_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        cnt_acc = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in tl.range(0, N, BLOCK_N):
            cols = start_n + tl.arange(0, BLOCK_N)
            if EVEN:
                val = tl.load(inp + rows[:, None] * N + cols[None, :]).to(value_dtype)
                valid = val == val
            else:
                mask = row_mask[:, None] & (cols[None, :] < N)
                val = tl.load(inp + rows[:, None] * N + cols[None, :], mask=mask).to(
                    value_dtype
                )
                valid = mask & (val == val)
            x0 = tl.where(valid, val, tl.zeros([BLOCK_M, BLOCK_N], dtype=val.dtype))
            vh = valid.to(cnt_in_dtype)
            sum_acc += tl.sum(x0, axis=1, dtype=tl.float32)
            cnt_acc += tl.sum(vh, axis=1, dtype=tl.float32)

        result = sum_acc / cnt_acc
        if EVEN:
            tl.store(out + rows, result.to(out.dtype.element_ty))
        else:
            tl.store(out + rows, result.to(out.dtype.element_ty), row_mask)


@libentry()
@triton.jit
def nanmean_dim_task_kernel(
    inp,
    out,
    B,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Non-last-dim single-dim reduction: a task is (batch, column tile), reducing axis=0 per M chunk.

    Replaces the old non_inner kernel ((BLOCK_N, BLOCK_K) kept 2-D resident with grid=(M, .);
    on (1024,1024,512) dim=0 was 2.81ms / dim=1 6.51ms -> 0.80/0.79ms with this recipe).
    B == 1 is the leading-dim reduction; a middle dim goes through the (B, M, N) three-way split.
    """
    if tl.constexpr(out.dtype.element_ty == tl.float32):
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32
    elif tl.constexpr(out.dtype.element_ty == tl.float16):
        value_dtype = tl.float16
        cnt_in_dtype = tl.float16
    elif tl.constexpr(out.dtype.element_ty == tl.bfloat16):
        value_dtype = tl.bfloat16
        cnt_in_dtype = tl.float16
    else:
        value_dtype = tl.float32
        cnt_in_dtype = tl.float32

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (N + BLOCK_N - 1) // BLOCK_N
    num_task = B * num_k_tile
    for task_id in tl.range(pid, num_task, step):
        b_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        n_offset = k_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offset < N
        base = b_index * M * N
        sum_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        cnt_acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        for i_start in tl.range(0, M, BLOCK_M):
            m_offset = i_start + tl.arange(0, BLOCK_M)
            if EVEN:
                val = tl.load(
                    inp + base + m_offset[:, None] * N + n_offset[None, :]
                ).to(value_dtype)
                valid = val == val
            else:
                mask = (m_offset[:, None] < M) & n_mask[None, :]
                val = tl.load(
                    inp + base + m_offset[:, None] * N + n_offset[None, :],
                    mask=mask,
                ).to(value_dtype)
                valid = mask & (val == val)
            x0 = tl.where(valid, val, tl.zeros([BLOCK_M, BLOCK_N], dtype=val.dtype))
            vh = valid.to(cnt_in_dtype)
            sum_acc += tl.sum(x0, axis=0, dtype=tl.float32)
            cnt_acc += tl.sum(vh, axis=0, dtype=tl.float32)

        out_ptrs = out + b_index * N + n_offset
        if EVEN:
            tl.store(out_ptrs, (sum_acc / cnt_acc).to(out.dtype.element_ty))
        else:
            tl.store(
                out_ptrs,
                (sum_acc / cnt_acc).to(out.dtype.element_ty),
                n_mask,
            )


def _normalize_dims(dim, ndim):
    if isinstance(dim, (list, tuple)) and len(dim) == 0:
        return []
    if dim is None:
        return list(range(ndim))
    if isinstance(dim, int):
        dim = [dim]
    dims = []
    for d in dim:
        if ndim == 0:
            if d not in (-1, 0):
                raise IndexError(
                    f"Dimension out of range (expected to be in range of [-1, 0], but got {d})"
                )
            wrapped = 0
        else:
            if d < -ndim or d >= ndim:
                raise IndexError(
                    "Dimension out of range (expected to be in range of "
                    f"[-{ndim}, {ndim - 1}], but got {d})"
                )
            wrapped = d % ndim
        if wrapped in dims:
            raise RuntimeError(
                f"dim {wrapped} appears multiple times in the list of dims"
            )
        dims.append(wrapped)
    return sorted(dims, reverse=True)


def _squeeze_dims(result, dims):
    for d in sorted(dims, reverse=True):
        result = result.squeeze(dim=d)
    return result


def _launch_row(inp, out_flat, M, N, dtype):
    element_size = inp.element_size()
    block_m, block_n, grid, num_warps = _row_params(element_size, M, N)
    even = (M % block_m == 0) and (N % block_n == 0)
    with torch_device_fn.device(inp.device):
        nanmean_dim_inner_kernel[(grid,)](
            inp,
            out_flat,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN=even,
            num_warps=num_warps,
        )


def _launch_task(inp, out_flat, B, M, N, dtype):
    element_size = inp.element_size()
    block_m, block_n, grid, num_warps = _task_params(element_size, B, M, N)
    even = (M % block_m == 0) and (N % block_n == 0)
    if not even:
        # gcu400 backend bug: on the task kernel's masked path, when tile lanes >= 16384
        # it silently returns wrong values (reproduces with old/new kernel bodies, with or without other; triggered by
        # both
        # m-tail and n-tail masks; the EVEN path with 32768 lanes is fine). Cap tile lanes on the masked path
        # at 8192 to avoid this (verified correct on (129,257) with bm64/bn128). All bench shapes
        # divide evenly and take the EVEN path, so they are unaffected.
        while block_m > 1 and block_m * block_n > 8192:
            block_m //= 2
        grid = min(NANMEAN_MAX_GRID, B * triton.cdiv(N, block_n))
    with torch_device_fn.device(inp.device):
        nanmean_dim_task_kernel[(grid,)](
            inp,
            out_flat,
            B,
            M,
            N,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            EVEN=even,
            num_warps=num_warps,
        )


def nanmean_dim(inp, dim=None, keepdim=False, *, dtype=None, out_flat=None):
    logger.debug("GEMS_ENFLAME NANMEAN_DIM")
    if dtype is None:
        dtype = inp.dtype

    dims = _normalize_dims(dim, inp.ndim)

    if inp.ndim == 0:
        return _nanmean_global(inp, dtype=dtype, out_flat=out_flat)

    # dim=[] -> reduce all
    if len(dims) == 0:
        result = _nanmean_global(inp, dtype=dtype, out_flat=out_flat)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    # full-dimensional reduction -> delegate to global
    if len(dims) == inp.ndim:
        result = _nanmean_global(inp, dtype=dtype, out_flat=out_flat)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result

    shape = list(inp.shape)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1

    if N == 0:
        out = torch.full(shape, float("nan"), dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    if math.prod(shape) == 0:
        out = torch.empty(shape, dtype=dtype, device=inp.device)
        return out if keepdim else _squeeze_dims(out, dims)

    if len(dims) == 1:
        dim = dims[0]
        if not inp.is_contiguous():
            inp = inp.contiguous()
        if dim == inp.ndim - 1:
            # last-dim reduction: row kernel over (M, N)
            M = inp.numel() // N
            if out_flat is None:
                out_flat = torch.empty(M, dtype=dtype, device=inp.device)
            _launch_row(inp, out_flat, M, N, dtype)
        elif dim == 0:
            # first-dim reduction: task kernel with B == 1
            M = inp.shape[0]
            N_cols = inp.numel() // M
            if out_flat is None:
                out_flat = torch.empty(N_cols, dtype=dtype, device=inp.device)
            _launch_task(inp, out_flat, 1, M, N_cols, dtype)
        else:
            # middle-dim reduction: task kernel over (B, M, N)
            B = 1
            for i in range(dim):
                B *= inp.shape[i]
            M = inp.shape[dim]
            N_cols = 1
            for i in range(dim + 1, inp.ndim):
                N_cols *= inp.shape[i]
            if B <= N_cols * 128:
                if out_flat is None:
                    out_flat = torch.empty(B * N_cols, dtype=dtype, device=inp.device)
                _launch_task(inp, out_flat, B, M, N_cols, dtype)
            else:
                in_reshape = inp.reshape(B, M, N_cols)
                inp_new = dim_compress(in_reshape, {0, 2})
                M_new = inp_new.shape[0]
                N_new = inp_new.numel() // M_new
                if out_flat is None:
                    out_flat = torch.empty(N_new, dtype=dtype, device=inp.device)
                _launch_task(inp_new, out_flat, 1, M_new, N_new, dtype)
    else:
        inp = dim_compress(inp, dims)
        M = inp.numel() // N
        if out_flat is None:
            out_flat = torch.empty(M, dtype=dtype, device=inp.device)
        _launch_row(inp, out_flat, M, N, dtype)

    out = out_flat.reshape(shape)
    return out if keepdim else _squeeze_dims(out, dims)


@pointwise_dynamic(
    is_tensor=[True, True, False, False, False],
    num_outputs=3,
    promotion_methods=[(0, 1, "DEFAULT")] * 3,
)
@triton.jit
def _masked_parts(
    real,
    imag,
    real_output: tl.constexpr,
    conjugated: tl.constexpr,
    complex_input: tl.constexpr,
):
    valid = real == real
    real_valid = valid
    if complex_input:
        valid = valid & (imag == imag)
    if not real_output:
        real_valid = valid
    if conjugated:
        imag = -imag
    if not complex_input:
        imag = tl.full(real.shape, 0, real.dtype)
    return tl.where(real_valid, real, 0), tl.where(valid, imag, 0), valid.to(real.dtype)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _mean_divide(total, count):
    return total / count


@pointwise_dynamic(promotion_methods=[(0, 1, 2, "DEFAULT")])
@triton.jit
def _mean_backward(grad, valid, count):
    # Multiplication follows division: an all-NaN slice must yield NaN gradients.
    return (grad / count) * valid


def _assemble_complex(real, imag, dtype):
    result = torch.empty(real.shape, device=real.device, dtype=dtype)
    parts = torch.view_as_real(result)
    copy_(parts[..., 0], real)
    copy_(parts[..., 1], imag)
    return result


def _masked_input_parts(inp, dtype):
    real_dtype = (
        torch.float32 if dtype in (torch.float32, torch.complex128) else torch.float32
    )
    real = torch.empty(inp.shape, dtype=real_dtype, device=inp.device)
    imag = torch.empty_like(real)
    valid = torch.empty_like(real)
    if inp.is_complex():
        physical = inp.conj() if inp.is_conj() else inp
        parts = torch.view_as_real(physical)
        xr, xi = parts[..., 0], parts[..., 1]
    else:
        xr, xi = inp, inp
    _masked_parts(
        xr,
        xi,
        not dtype.is_complex,
        inp.is_conj(),
        inp.is_complex(),
        out0=real,
        out1=imag,
        out2=valid,
    )
    return real, imag, valid


def _complex_mean(inp, dim, keepdim, dtype):
    dim = _normalize_dims(dim, inp.ndim)
    real, imag, valid = _masked_input_parts(inp, dtype)
    count = sum_dim(valid, dim=dim, keepdim=keepdim)
    real = _mean_divide(sum_dim(real, dim=dim, keepdim=keepdim), count)
    if not dtype.is_complex:
        result = torch.empty_like(real, dtype=dtype)
        copy_(result, real)
        return result
    imag = _mean_divide(sum_dim(imag, dim=dim, keepdim=keepdim), count)
    return _assemble_complex(real, imag, dtype)


class _NanmeanAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dim, keepdim, dtype):
        dtype = dtype or inp.dtype
        dims = _normalize_dims(dim, inp.ndim)
        if not dims:
            dims = list(range(inp.ndim))
        ctx.dims = dims
        ctx.keepdim = keepdim
        ctx.input_dtype = inp.dtype
        _, _, valid = _masked_input_parts(inp, dtype)
        count = sum_dim(valid, dim=dims, keepdim=True)
        ctx.save_for_backward(valid, count)
        if inp.is_complex() or dtype.is_complex:
            return _complex_mean(inp, dim, keepdim, dtype)
        return nanmean(inp.detach(), dim, keepdim, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        valid, count = ctx.saved_tensors
        if not ctx.keepdim:
            for dim in sorted(ctx.dims):
                grad = grad.unsqueeze(dim)
        if grad.is_complex():
            physical = grad.conj() if grad.is_conj() else grad
            parts = torch.view_as_real(physical)
            real = _mean_backward(parts[..., 0], valid, count)
            if not ctx.input_dtype.is_complex:
                result = torch.empty_like(valid, dtype=ctx.input_dtype)
                copy_(result, real)
                return result, None, None, None
            imag = _mean_backward(parts[..., 1], valid, count)
            if grad.is_conj():
                from .neg import neg

                imag = neg(imag)
            result = _assemble_complex(real, imag, ctx.input_dtype)
        else:
            result = torch.empty_like(valid, dtype=ctx.input_dtype)
            if ctx.input_dtype.is_complex:
                real = _mean_backward(grad, valid, count)
                from flag_gems.ops.zeros_like import zeros_like

                result = _assemble_complex(real, zeros_like(real), ctx.input_dtype)
            else:
                _mean_backward(grad, valid, count, out0=result)
        return result, None, None, None


def _nanmean_output_shape(inp, dim, keepdim):
    """Mirrors nanmean()'s output-shape decision (pure host metadata, never touches device data)."""
    ndim = inp.ndim
    dims = _normalize_dims(dim, ndim)
    if ndim == 0 or len(dims) == 0 or len(dims) == ndim:
        return [1] * ndim if keepdim else []
    shape = list(inp.shape)
    for d in dims:
        shape[d] = 1
    if not keepdim:
        shape = [s for i, s in enumerate(shape) if i not in dims]
    return shape


def nanmean(inp, dim=None, keepdim=False, *, dtype=None, out_flat=None):
    logger.debug("GEMS_ENFLAME NANMEAN")
    if not (inp.is_floating_point() or inp.is_complex()):
        raise NotImplementedError(
            "nanmean(): expected input to have floating point or complex dtype but got "
            f"{inp.dtype}"
        )
    if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
        raise RuntimeError(
            "nanmean(): could not infer output dtype. Optional dtype must be either "
            f"a floating point or complex dtype. Got: {dtype}"
        )
    complex_output = dtype is not None and dtype.is_complex
    if inp.requires_grad and torch.is_grad_enabled():
        return _NanmeanAutograd.apply(inp, dim, keepdim, dtype)
    if inp.is_complex() or complex_output:
        return _complex_mean(inp, dim, keepdim, dtype or inp.dtype)
    if dim is None:
        result = _nanmean_global(inp, dtype=dtype, out_flat=out_flat)
        if keepdim:
            result = result.reshape([1] * inp.ndim)
        return result
    return nanmean_dim(inp, dim=dim, keepdim=keepdim, dtype=dtype, out_flat=out_flat)


def nanmean_out(inp, dim=None, keepdim=False, *, dtype=None, out=None):
    logger.debug("GEMS_ENFLAME NANMEAN_OUT")
    if out is None:
        raise RuntimeError("nanmean(): missing required out tensor")
    if torch.is_grad_enabled() and (inp.requires_grad or out.requires_grad):
        raise RuntimeError(
            "nanmean(): functions with out= arguments don't support automatic differentiation"
        )
    if out.device != inp.device:
        raise RuntimeError(
            "nanmean: expected result tensor to be on the same device as input"
        )
    if dtype is not None and dtype != out.dtype:
        raise RuntimeError(
            "nanmean: provided dtype must match dtype of result. Got "
            f"{out.dtype} and {dtype}."
        )
    # Fast path: use a real, contiguous out directly as the kernel target, saving the result buffer + one copy_
    # extra kernel launch ((64,64) tiny shape ~8.5us -> ~6us). Complex/non-contiguous/empty inputs
    # use the original copy path.
    if (
        out.is_contiguous()
        and not inp.is_complex()
        and not (dtype is not None and dtype.is_complex)
        and not inp.requires_grad
        and inp.is_floating_point()
    ):
        # Validate arguments in the same order as nanmean() to keep exception behavior consistent
        expected = _nanmean_output_shape(inp, dim, keepdim)
        if list(out.shape) != expected:
            out.resize_(expected)
        result = nanmean(
            inp, dim=dim, keepdim=keepdim, dtype=dtype or out.dtype, out_flat=out
        )
        if result is not out:
            if out.shape != result.shape:
                out.resize_(result.shape)
            copy_(out, result)
        return out
    result = nanmean(inp, dim=dim, keepdim=keepdim, dtype=dtype or out.dtype)
    if out.shape != result.shape:
        out.resize_(result.shape)
    if out.is_complex():
        out_parts = torch.view_as_real(out)
        result_parts = torch.view_as_real(result)
        copy_(out_parts, result_parts)
    else:
        copy_(out, result)
    return out


# Module-level handle keeps the Library (and its kernels) alive for the
# process; the registration itself must never break the flag_gems import.

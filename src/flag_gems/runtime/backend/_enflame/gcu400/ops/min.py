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

import builtins
import logging
import math
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime  # noqa: F401
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

_min = builtins.min
_max = builtins.max

# Creating a namedtuple class costs ~87us in Python.  Building it on every
# min_dim call dominated the total latency for small shapes (the Triton kernel
# itself is only ~5us), so the class is defined once at module scope.
MinOut = namedtuple("min", ["values", "indices"])


@libentry()
@triton.jit
def min_kernel_1_simple(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    max_value = get_dtype_max(inp.type.element_ty)
    inp_val = tl.load(inp + offset, mask=mask, other=max_value)
    tl.store(mid + pid, tl.min(inp_val))


@libentry()
@triton.jit(do_not_specialize=["M"])
def min_kernel_1_grid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    dtype = inp.type.element_ty
    max_val = get_dtype_max(dtype)

    acc = tl.full([BLOCK_SIZE], value=max_val, dtype=dtype)

    block_id = pid
    while block_id * BLOCK_SIZE < M:
        offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        vals = tl.load(inp + offset, mask=mask, other=max_val)
        acc = tl.where(vals < acc, vals, acc)
        block_id += num_progs

    tl.store(mid + pid, tl.min(acc))


@libentry()
@triton.jit
def min_kernel_2(mid, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    max_value = get_dtype_max(mid.type.element_ty)
    mid_val = tl.load(mid_ptrs, mask=mask, other=max_value)
    min_val = tl.min(mid_val)
    tl.store(out, min_val)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_kernel_inner_1d(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)

    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        # min only compares and selects values, so keeping f16/bf16 instead
        # of widening to f32 is bit-exact.  It also avoids a predicate that is
        # computed through an f16 -> f32 widening and then drives the i32 index
        # select, which GCU400 miscompiles: on (4096, 256) dim=1 the value is
        # right but the index points at a non-maximum element for ~46% of the
        # rows.  fp32 input, which is not widened, is unaffected.
        acc_dtype = dtype
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    for row_id in tl.range(pid, M, step, num_stages=num_stages):
        min_vals = tl.full([BLOCK_N], value=get_dtype_max(acc_dtype), dtype=acc_dtype)
        min_idxs = tl.zeros([BLOCK_N], dtype=tl.int32)

        base = row_id * N
        for col_off in tl.range(0, N, BLOCK_N):
            n_offset = col_off + tl.arange(0, BLOCK_N)
            mask = n_offset < N
            x = tl.load(inp + base + n_offset, mask=mask, other=max_value).to(acc_dtype)

            update = x < min_vals
            min_vals = tl.where(update, x, min_vals)
            min_idxs = tl.where(update, n_offset, min_idxs)

        final_min = tl.min(min_vals, axis=0)
        eq_mask = min_vals == final_min
        min_idxs_masked = tl.where(eq_mask, min_idxs, 2147483647)
        final_idx = tl.min(min_idxs_masked, axis=0)

        tl.store(out_value + row_id, final_min.to(dtype))
        tl.store(out_index + row_id, final_idx)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K", "num_k_tiles"])
def min_kernel_non_inner(
    inp,
    out_value,
    out_index,
    M,
    N,
    K,
    num_k_tiles,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    num_prog = tl.num_programs(0)
    total_work = M * num_k_tiles

    dtype = inp.type.element_ty
    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        # See min_kernel_inner_1d: bit-exact for min, and avoids the predicate
        # computed through an f16 -> f32 widening that GCU400 miscompiles.
        acc_dtype = dtype
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype
    max_value = get_dtype_max(acc_dtype)
    orig_max = get_dtype_max(dtype)

    for work_id in tl.range(pid, total_work, num_prog, num_stages=num_stages):
        m = work_id // num_k_tiles
        k_tile = work_id % num_k_tiles
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        min_vals = tl.full([BLOCK_K], dtype=acc_dtype, value=max_value)
        min_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

        base = m * N * K
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            offset = base + n_idx[:, None] * K + k_offset[None, :]
            mask = (n_idx[:, None] < N) & k_mask[None, :]
            vals = tl.load(inp + offset, mask=mask, other=orig_max).to(acc_dtype)

            # tl.min promotes f16/bf16 to f32; cast back so the index select
            # below still sees an f16 predicate.  The result is one of the
            # inputs, so this is exact.
            local_min = tl.min(vals, axis=0).to(acc_dtype)
            eq = vals == local_min[None, :]
            idx_matrix = tl.where(eq, n_idx[:, None], 2147483647)
            local_argmin = tl.min(idx_matrix, axis=0)

            update = local_min < min_vals
            min_vals = tl.where(update, local_min, min_vals)
            min_idxs = tl.where(update, local_argmin, min_idxs)

        out_offset = m * K + k_offset
        tl.store(out_value + out_offset, min_vals.to(dtype), k_mask)
        tl.store(out_index + out_offset, min_idxs, k_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_kernel_inner_tile(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """K==1 (inner dim reduction) 2D tile implementation: a single load covers
    BLOCK_N contiguous columns of BLOCK_M rows (N is the innermost dim, so the
    access is contiguous) and the reduction runs along axis=1. Taking the row
    min first and then the smallest column index among the entries equal to it
    reproduces the torch.min "first minimal element" semantics.

    This replaces the per-row grid-stride version, which did not bound BLOCK_N
    by the 512B OACC limit and produced wrong values or indices for f16/bf16 at
    N=8192 and for int64 at N=1024. The tile form keeps every loop-carried
    quantity within BLOCK_M (<=64 lanes).
    """
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)
    # f16/bf16 reduce in their own dtype (min is pure selection, bit-exact);
    # int8/int16 are widened to int32
    if tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype
    acc_max = get_dtype_max(acc_dtype)

    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step, num_stages=num_stages):
        m_off = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_off < M

        min_vals = tl.full([BLOCK_M], value=acc_max, dtype=acc_dtype)
        min_idxs = tl.zeros([BLOCK_M], dtype=tl.int32)

        # The reduction axis is fixed to axis=0 on a [BLOCK_N, BLOCK_M] layout:
        # on GCU400 an axis=1 reduction over a wide BLOCK_N silently
        # miscomputes or crashes the process; axis=0 is the form verified
        # to work on this backend.
        base = m_off[None, :] * N
        for n_off in tl.range(0, N, BLOCK_N):
            n_idx = n_off + tl.arange(0, BLOCK_N)
            mask = m_mask[None, :] & (n_idx[:, None] < N)
            vals = tl.load(inp + base + n_idx[:, None], mask=mask, other=max_value).to(
                acc_dtype
            )

            # Cast back to acc_dtype before comparing: tl.min promotes f16 to f32,
            # and building the select predicate from the promoted value is
            # miscompiled on GCU400 (values right, indices wrong).
            local_min = tl.min(vals, axis=0).to(acc_dtype)
            eq = vals == local_min[None, :]
            idx_matrix = tl.where(eq, n_idx[:, None], 2147483647)
            local_argmin = tl.min(idx_matrix, axis=0)

            update = local_min < min_vals
            min_vals = tl.where(update, local_min, min_vals)
            min_idxs = tl.where(update, local_argmin, min_idxs)

        tl.store(out_value + m_off, min_vals.to(dtype), m_mask)
        tl.store(out_index + m_off, min_idxs, m_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def min_kernel_inner_batch(
    inp,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)

    dtype = inp.type.element_ty
    max_value = get_dtype_max(dtype)

    if tl.constexpr(dtype == tl.float16) | tl.constexpr(dtype == tl.bfloat16):
        acc_dtype = tl.float32
    elif tl.constexpr(dtype == tl.int16) | tl.constexpr(dtype == tl.int8):
        acc_dtype = tl.int32
    else:
        acc_dtype = dtype

    num_tiles = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tiles, step):
        m_off = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_off < M

        min_vals = tl.full([BLOCK_M], value=get_dtype_max(acc_dtype), dtype=acc_dtype)
        min_idxs = tl.zeros([BLOCK_M], dtype=tl.int32)

        for n in tl.range(0, N, 1):
            offsets = m_off * N + n
            vals = tl.load(inp + offsets, mask=m_mask, other=max_value).to(acc_dtype)
            update = vals < min_vals
            min_vals = tl.where(update, vals, min_vals)
            min_idxs = tl.where(update, n, min_idxs)

        tl.store(out_value + m_off, min_vals.to(dtype), m_mask)
        tl.store(out_index + m_off, min_idxs, m_mask)


def min(inp):
    logger.debug("GEMS_ENFLAME MIN")

    if not inp.is_contiguous():
        inp = inp.contiguous()
    M = inp.numel()
    dtype = inp.dtype

    if M == 0:
        # Mirror the aten contract (CPU): a full reduction over an empty
        # tensor is an error, not a value.
        raise RuntimeError(
            "min(): Expected reduction dim to be specified for input.numel()"
            " == 0. Specify the reduction dim with the 'dim' argument."
        )

    if M <= 10 * 1024 * 1024:
        bsize = 16384
        num_programs = _min(triton.cdiv(M, bsize), 48)
        block_mid = triton.next_power_of_2(num_programs)
        mid = torch.empty((num_programs,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            min_kernel_1_grid[(num_programs,)](
                inp, mid, M, BLOCK_SIZE=bsize, num_warps=1
            )
            min_kernel_2[(1,)](mid, out, num_programs, block_mid, num_warps=1)
    else:
        block_size = triton.next_power_of_2(math.ceil(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        mid = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        out = torch.empty([], dtype=dtype, device=inp.device)
        with torch_device_fn.device(inp.device):
            min_kernel_1_simple[(mid_size, 1, 1)](inp, mid, M, block_size, num_warps=1)
            min_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid, num_warps=1)
    return out


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_ENFLAME MIN_DIM")

    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim

    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    inp = inp.contiguous()

    shape_list = list(shape)
    if keepdim:
        shape_list[dim] = 1
    else:
        # Kernels already write a flat m*K + k layout, so allocate the final
        # squeezed shape directly instead of two squeeze calls later.
        shape_list.pop(dim)
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    # GCU400 Triton writes only 4 bytes per int32 lane even when the tensor is
    # int64, so an int64 output ends up with every other index zeroed.  Emit
    # into an int32 buffer (all three kernels store int32) and cast once on the
    # host; the op contract still returns int64 indices like torch.min.
    out_index = torch.empty(shape_list, dtype=torch.int32, device=inp.device)

    if K == 1:
        BLOCK_N = _min(triton.next_power_of_2(N), 4096)
        # Bytes per element: decides how many reduction lanes fit in the 512B
        # OACC budget
        _bpe = (
            8
            if inp.dtype in (torch.int64, torch.float64)
            else (2 if inp.dtype in (torch.float16, torch.bfloat16) else 4)
        )
        if N <= 64 and M > 1024:
            BLOCK_M = 1024 if inp.dtype in (torch.float16, torch.bfloat16) else 512
            grid_m = _min(triton.cdiv(M, BLOCK_M), 48)
            with torch_device_fn.device(inp.device):
                min_kernel_inner_batch[(grid_m,)](
                    inp,
                    out_value,
                    out_index,
                    M,
                    N,
                    BLOCK_M=BLOCK_M,
                    num_warps=1,
                )
        elif triton.next_power_of_2(N) * _bpe < 512:
            # Narrow reduction (BLOCK_N*bpe strictly below the 512B OACC budget;
            # a full 512B already miscomputes, e.g. 128 lanes of (17,33,65) dim2
            # f32): the old grid-stride version is both correct and faster here,
            # so it is kept as a fast path -- but only with a single pipeline
            # stage (3 stages miscompute on multi-row shapes like (1024,1024)).
            with torch_device_fn.device(inp.device):
                min_kernel_inner_1d[(_min(M, 48),)](
                    inp,
                    out_value,
                    out_index,
                    M,
                    N,
                    BLOCK_N=_min(triton.next_power_of_2(N), 4096),
                    num_stages=1,
                    num_warps=1,
                )
        else:
            # The per-row grid-stride version (min_kernel_inner_1d) keeps a
            # BLOCK_N-wide vector in the loop-carried state, which silently
            # miscomputes once it exceeds the GCU400 512B OACC limit: besides
            # (4096,4096), the shapes (1024,1024)/(1536,640) f16/f32/uint8,
            # (144,8192) f16/bf16 and int64 with N=1024/640/65 all fail in
            # measurement, and pipeline depth 3 adds another layer of error.
            # Per-row reductions therefore use a 2D tile: only BLOCK_M is
            # loop-carried (<=512B/bpe), while the BLOCK_N reduction width only
            # affects a single reduction and never the cross-iteration state.
            # The loop-carried vector (BLOCK_M lanes) uses half of the 512B OACC:
            # a full 512B already yields ~10% wrong indices on (200,2560,3) dim0
            # int16/int32.
            lane_budget = _max(1, 256 // _bpe)
            row_cap = triton.next_power_of_2(M)
            # The tile area stays at ~32K elements: BLOCK_N=1024 is optimal or
            # near-optimal in every measured case
            tile_n = _min(triton.next_power_of_2(N), 1024)
            tile_m = _min(row_cap, _min(lane_budget, _max(1, 32768 // tile_n)))
            if N > 1024 and N <= 8192 and triton.cdiv(M, tile_m) < 48:
                # With few rows (e.g. M=144) the CTAs cannot fill all 48 slots,
                # so a wider BLOCK_N buys more row tiles: (144,8192) drops from
                # 66us to 47us in measurement
                tile_n = _min(triton.next_power_of_2(N), 4096)
                tile_m = _min(row_cap, _min(lane_budget, _max(1, 32768 // tile_n)))
            grid_m = _min(triton.cdiv(M, tile_m), 48)
            num_stages = 2 if triton.cdiv(M, tile_m) > grid_m else 1
            with torch_device_fn.device(inp.device):
                min_kernel_inner_tile[(grid_m,)](
                    inp,
                    out_value,
                    out_index,
                    M,
                    N,
                    BLOCK_M=tile_m,
                    BLOCK_N=tile_n,
                    num_stages=num_stages,
                    num_warps=1,
                )

    else:
        # BLOCK_K is the width of the loop-carried accumulator (output
        # columns; the reduction walks BLOCK_N).  GCU400 keeps such a vector
        # in at most 512B of OACC: 128 int64 lanes need 1024B and the
        # reduction then silently covers only part of the axis -- the value is
        # the min over a subset and the index matches that wrong value.
        # int64 dim=0 on (4096, 256) gets 219/256 columns wrong, while int32
        # at the same 128 lanes (512B) is exact.  The wide tile is also ~12x
        # slower.
        _acc_bpes = {
            torch.float64: 8,
            torch.int64: 8,
            torch.float32: 4,
            torch.int32: 4,
            torch.int16: 4,
            torch.int8: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
        }
        # With few output columns (M <= 64) the parallelism already comes from
        # the k blocks, and the loop-carried vector must shrink to 256B to stay
        # safe: (200,2560,3) dim0 int16/int32 gives ~10% wrong indices at
        # 512B/128 lanes, while large-M shapes like (64,512,512) are always exact
        # at 512B -- dropping to 256B halves BLOCK_K and makes the non-inner form
        # 3x slower.
        _k_budget_bytes = 512 if M > 64 else 256
        BLOCK_K = _min(
            triton.next_power_of_2(K),
            _min(_k_budget_bytes // _acc_bpes.get(inp.dtype, 8), 128),
        )
        # Also cap the dsm footprint by byte count: an int64 [32,1024] tile
        # measures 917632B, above the 917504B hardware limit ((4096,256) dim0
        # fails to compile outright). BLOCK_N must stay a power of two, so it is
        # halved step by step instead of using the exact quotient.
        _acc_bpe = _acc_bpes.get(inp.dtype, 8)
        max_bn = _min(2048, 32768 // BLOCK_K)
        while max_bn > 32 and max_bn * BLOCK_K * _acc_bpe > 200000:
            max_bn //= 2
        BLOCK_N = _min(triton.next_power_of_2(N), max_bn)

        num_k_tiles = triton.cdiv(K, BLOCK_K)
        total_work = M * num_k_tiles
        grid_size = _min(total_work, 48)

        num_stages = 1

        with torch_device_fn.device(inp.device):
            min_kernel_non_inner[(grid_size,)](
                inp,
                out_value,
                out_index,
                M,
                N,
                K,
                num_k_tiles,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
                num_stages=num_stages,
                num_warps=1,
            )

    out_index = out_index.to(torch.int64)

    return MinOut(values=out_value, indices=out_index)

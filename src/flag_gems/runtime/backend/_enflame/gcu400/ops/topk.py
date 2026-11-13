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
import triton.language.core as core

try:
    # TODO: Triton 2.1 does not implement _log2.
    # Remove the try-catch block once all vendors upgrade to a newer version of Triton.
    from triton.language.standard import _log2, zeros_like
except ImportError:
    pass

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max, get_dtype_min
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)

try:
    # HW MRADIX level-1 TopK (TLE-Struct + TLE-Raw). Importing it reads the
    # .tops source, so a checkout/install without that file, or a toolchain
    # without TLE-Raw, must degrade to the generic path instead of failing.
    from . import topk_mradix as _mradix
except Exception as err:  # noqa: BLE001
    _mradix = None
    logger.warning("MRADIX topk unavailable, falling back: %s", err)

if HAS_TLE:
    import triton.experimental.tle.language as tle_gpu
else:
    tle_gpu = None

_MIN_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).min)
_MAX_FLOAT32_VAL = tl.constexpr(torch.finfo(torch.float32).max)
_MIN_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).min)
_MAX_FLOAT16_VAL = tl.constexpr(torch.finfo(torch.float16).max)
_MIN_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).min)
_MAX_BFLOAT16_VAL = tl.constexpr(torch.finfo(torch.bfloat16).max)
_MIN_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).min)
_MAX_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).max)
_MIN_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).min)
_MAX_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).max)
_MIN_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).min)
_MAX_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).max)
_MIN_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).min)
_MAX_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).max)


@triton.jit
def _get_finfo_val(
    dtype,
    return_max,
):
    if dtype is tl.float32:
        if return_max:
            return _MAX_FLOAT32_VAL
        else:
            return _MIN_FLOAT32_VAL
    elif dtype is tl.float16:
        if return_max:
            return _MAX_FLOAT16_VAL
        else:
            return _MIN_FLOAT16_VAL
    elif dtype is tl.bfloat16:
        if return_max:
            return _MAX_BFLOAT16_VAL
        else:
            return _MIN_BFLOAT16_VAL


@triton.jit
def _get_iinfo_val(
    dtype,
    return_max,
):
    if return_max:
        return get_dtype_max(dtype)
    else:
        return get_dtype_min(dtype)


@libentry()
@triton.jit
def topk_stage1_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
            chunk_select_idx = tl.argmax(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val)
            chunk_select_idx = tl.argmin(x_val, axis=0)

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, chunk_select_idx + chunk_offset)

        if DESCENDING:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=False),
                x_val,
            )
        else:
            x_val = tl.where(
                cols == chunk_select_idx,
                _get_finfo_val(tl.float32, return_max=True),
                x_val,
            )


@libentry()
@triton.jit
def topk_stage1_with_index_kernel(
    y_ptr,
    index_ptr,
    x_ptr,
    x_index_ptr,
    k,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    """Like topk_stage1_kernel but propagates pre-existing indices
    instead of computing positional ones. Used for intermediate
    reduction rounds when stage2 BLOCK_SIZE would be too large."""
    cur_batch = tle.program_id(0)
    cur_chunk_idx = tle.program_id(1)
    chunk_num = tle.num_programs(1)

    y_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k
    index_ptr += cur_batch * chunk_num * k + cur_chunk_idx * k

    chunk_offset = cur_chunk_idx * CHUNK_SIZE
    x_ptr += cur_batch * N + chunk_offset
    x_index_ptr += cur_batch * N + chunk_offset

    cols = tl.arange(0, CHUNK_SIZE)
    mask = (chunk_offset + cols) < N

    mask_val = _get_finfo_val(x_ptr.dtype.element_ty, return_max=not DESCENDING)
    x_val = tl.load(x_ptr + cols, mask=mask, other=mask_val).to(tl.float32)
    x_idx = tl.load(x_index_ptr + cols, mask=mask, other=0).to(tl.int64)

    for k_idx in range(k):
        if DESCENDING:
            chunk_select_val = tl.max(x_val)
            chunk_select_idx = tl.argmax(x_val, axis=0)
        else:
            chunk_select_val = tl.min(x_val)
            chunk_select_idx = tl.argmin(x_val, axis=0)

        selected_mask = cols == chunk_select_idx
        original_idx = tl.sum(
            tl.where(selected_mask, x_idx, tl.full([CHUNK_SIZE], 0, dtype=tl.int64))
        )

        tl.store(y_ptr + k_idx, chunk_select_val)
        tl.store(index_ptr + k_idx, original_idx)

        if DESCENDING:
            x_val = tl.where(
                selected_mask,
                _get_finfo_val(tl.float32, return_max=False),
                x_val,
            )
        else:
            x_val = tl.where(
                selected_mask,
                _get_finfo_val(tl.float32, return_max=True),
                x_val,
            )


"""
Note(Zhengzekang):
Refer from triton2.2 official `sort` implementation:
https://github.com/triton-lang/triton/blob/release/2.2.x/python/triton/language/standard.py#L392-L404
Just add indices to sort with values.
"""


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]

    # tl.device_print("shape is: ", shape)
    y = core.reshape(x, shape)
    y_idx = core.reshape(ids, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(x.dtype)
    right = core.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(x.dtype)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    left_idx = core.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape).to(
        ids.dtype
    )
    right_idx = core.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape).to(
        ids.dtype
    )
    left_idx = core.reshape(left_idx, ids.shape)
    right_idx = core.reshape(right_idx, ids.shape)

    # actual compare-and-swap
    if core.constexpr(x.dtype.primitive_bitwidth) == 8:
        idtype = core.int8
    elif core.constexpr(x.dtype.primitive_bitwidth) == 16:
        idtype = core.int16
    elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
        idtype = core.int32
    elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
        idtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))

    if core.constexpr(ids.dtype.primitive_bitwidth) == 8:
        idx_dtype = core.int8
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 16:
        idx_dtype = core.int16
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 32:
        idx_dtype = core.int32
    elif core.constexpr(ids.dtype.primitive_bitwidth) == 64:
        idx_dtype = core.int64
    else:
        raise ValueError("Unsupported dtype")

    ileft_idx = left_idx.to(idx_dtype, bitcast=True)
    iright_idx = right_idx.to(idx_dtype, bitcast=True)
    ix_idx = ids.to(idx_dtype, bitcast=True)
    ret_idx = ix_idx ^ core.where(cond, ileft_idx ^ iright_idx, zeros_like(ix_idx))

    return ret.to(x.dtype, bitcast=True), ret_idx.to(ids.dtype, bitcast=True)


@triton.jit
def _bitonic_merge(
    x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: core.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, dim: tl.constexpr, descending: core.constexpr):
    # handle default dimension or check that it is the most minor dim
    _dim: core.constexpr = dim
    n_dims: core.constexpr = _log2(x.shape[_dim])
    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


@libentry()
@triton.jit
def topk_stage2_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    sort_dim: tl.constexpr,
    k: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tle.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    y_ptr += cur_batch * k
    index_ptr += cur_batch * k

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    mask_val = _get_finfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)
    mask_index_val = _MIN_INT32_VAL if DESCENDING else _MAX_INT32_VAL

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=mask_val).to(tl.float32)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask, other=mask_index_val).to(
        tl.int32
    )

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < k)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < k)


if HAS_TLE:

    @triton.jit
    def _get_topmask_and_fullmask(x):
        tl.static_assert(
            x.dtype.is_int_unsigned(),
            "floating-point value must be passed as bits",
        )
        tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
        fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
        tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
        fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
        return tm_arr, fm_arr

    @triton.jit
    def _fpval_to_key_with_nan(x, x_bits):
        tm, fm = _get_topmask_and_fullmask(x_bits)
        mask = tl.where((x_bits & tm) != 0, fm, tm)
        key = x_bits ^ mask
        return tl.where(x == x, key, fm)

    @triton.jit
    def _key_to_fpval(x):
        tm, fm = _get_topmask_and_fullmask(x)
        mask = tl.where((x & tm) != 0, tm, fm)
        return x ^ mask

    @libentry()
    @triton.jit
    def topk_kernel_radix_tle(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_ym,
        n_cols,
        K: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        x_dtype = X.dtype.element_ty
        x_nbits: tl.constexpr = x_dtype.primitive_bitwidth
        if x_nbits < 16:
            y_nbits: tl.constexpr = 32
        else:
            y_nbits: tl.constexpr = x_nbits * 2
        x_utype = tl.dtype(f"uint{x_nbits}")
        x_ultype = tl.dtype(f"uint{y_nbits}")

        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_N], 1, tl.int32)

        desired = tl.full((), 0, dtype=x_utype)
        desired_mask = tl.full((), 0, dtype=x_utype)
        k_to_find = tl.full((), K, dtype=tl.int32)
        n_tiles = tl.cdiv(n_cols, BLOCK_N)

        smem_counts = tle_gpu.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle_gpu.gpu.local_ptr(smem_counts, (bins,))

        for digit_pos in tl.static_range(x_nbits - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(smem_count_ptrs, tl.zeros([RADIX_SIZE], dtype=tl.int32))
            for t in tl.range(0, n_tiles):
                offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
                mask_n = offs_n < n_cols
                x_ptrs = X + pid * stride_xm + offs_n
                x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
                x_bits = x.to(x_utype, bitcast=True)
                x_key = _fpval_to_key_with_nan(x, x_bits)
                matches = (x_key & desired_mask) == desired
                digit = ((x_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                valid = mask_n & matches
                count_addrs = tle_gpu.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(count_addrs, one, mask=valid, sem="relaxed", scope="cta")

            counts = tl.load(smem_count_ptrs)

            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            tl.store(smem_count_ptrs, cumsum_desc)

            selected_scalar = 0
            counts_gt_scalar = 0
            found = 0
            for rev in tl.static_range(RADIX_SIZE):
                d = RADIX_SIZE - 1 - rev
                cum_d = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d,)))
                if d + 1 < RADIX_SIZE:
                    cum_next = tl.load(tle_gpu.gpu.local_ptr(smem_counts, (d + 1,)))
                else:
                    cum_next = 0
                take = (found == 0) & (cum_d >= k_to_find) & (cum_next < k_to_find)
                selected_scalar = tl.where(take, d, selected_scalar)
                counts_gt_scalar = tl.where(take, cum_next, counts_gt_scalar)
                found = tl.where(take, 1, found)

            selected_u = selected_scalar.to(x_utype)
            desired = desired | (selected_u << digit_pos)
            desired_mask = desired_mask | (
                tl.full((), RADIX_MASK, dtype=x_utype) << digit_pos
            )
            k_to_find = k_to_find - counts_gt_scalar

        thr_key = desired

        min_val = tl.full((), float("-inf"), tl.float32).to(x_dtype)
        min_bits = min_val.to(x_utype, bitcast=True)
        min_key = _fpval_to_key_with_nan(min_val, min_bits)
        min_packed = min_key.to(x_ultype) << 16
        offs_k = tl.arange(0, K_PAD)

        smem_selected = tle_gpu.gpu.alloc(
            [K_PAD],
            dtype=x_ultype,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_selected_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (offs_k,))
        tl.store(smem_selected_ptrs, tl.full([K_PAD], min_packed, dtype=x_ultype))

        smem_write_count = tle_gpu.gpu.alloc(
            [1],
            dtype=tl.int32,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        tl.store(tle_gpu.gpu.local_ptr(smem_write_count, (0,)), 0)
        write_count_ptrs = tle_gpu.gpu.local_ptr(
            smem_write_count, (tl.zeros([BLOCK_N], dtype=tl.int32),)
        )

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = _fpval_to_key_with_nan(x, x_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_gt = mask_n & (x_key > thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_gt, sem="relaxed", scope="cta"
            )
            write_mask = take_gt & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        for t in tl.range(0, n_tiles):
            offs_n = t * BLOCK_N + tl.arange(0, BLOCK_N)
            mask_n = offs_n < n_cols
            x_ptrs = X + pid * stride_xm + offs_n
            x = tl.load(x_ptrs, mask=mask_n, other=float("-inf"))
            x_bits = x.to(x_utype, bitcast=True)
            x_key = _fpval_to_key_with_nan(x, x_bits)
            idx_key = (n_cols - offs_n).to(x_ultype)
            packed = (x_key.to(x_ultype) << 16) | idx_key
            take_eq = mask_n & (x_key == thr_key)
            pos = tl.atomic_add(
                write_count_ptrs, one, mask=take_eq, sem="relaxed", scope="cta"
            )
            write_mask = take_eq & (pos < K_PAD)
            dst_ptrs = tle_gpu.gpu.local_ptr(smem_selected, (pos.to(tl.int32),))
            tl.store(dst_ptrs, packed, mask=write_mask)

        selected_packed = tl.load(smem_selected_ptrs)

        topk = tl.sort(selected_packed, dim=0, descending=True)
        idx_mask = tl.full(topk.shape, (1 << 16) - 1, dtype=topk.dtype)
        idx_raw = (topk & idx_mask).to(tl.uint32)
        y_indices = (n_cols - idx_raw.to(tl.int32)).to(tl.int32)
        y_values_raw = (topk >> 16).to(x_utype)
        y_values = _key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

        mask_k = offs_k < K
        yv_ptrs = Yv + pid * stride_ym + offs_k
        yi_ptrs = Yi + pid * stride_ym + offs_k
        tl.store(yv_ptrs, y_values, mask=mask_k)
        tl.store(yi_ptrs, y_indices, mask=mask_k)


# ---------------------------------------------------------------------------
# Small-K fast path (k <= 10)
# ---------------------------------------------------------------------------
# For a small k the MRADIX select is a poor fit: it pays all seven radix passes
# over the row no matter how small k is, so its cost is almost independent of k.
# A direct k-pass max over the row is much cheaper: it is O(k) instead of
# O(32 bits), `tl.max` collapses the row to one scalar, the smallest matching
# column reproduces torch's ascending-index tie break, and the selected lane is
# masked out for the next pass.  `num_warps=1` keeps every reduction inside one
# warp -- a multi-warp block needs shared memory and a barrier per reduction,
# which is what makes the generic chunked path slow here.
#
# Measured on gcu400 (fp32, sorted=True, speedup vs torch.topk):
#   (10000, 256)  k=5   MRADIX 0.24x -> 0.94x
#   (4096, 4096)  k=5   MRADIX 0.25x -> 0.45x
#   (4, 128)      k=5   MRADIX 0.90x -> 2.92x
#   (8, 256)      k=5   MRADIX 1.07x -> 3.67x
#   (64, 64)      k=5   MRADIX 0.56x -> 1.69x
# Beyond ~4K columns a whole row no longer fits the register file (the row is
# read into vector registers once), so wider rows stay on MRADIX, which wins
# there anyway ((10000, 65536) k=5 is 0.78x fp32 / 1.56x fp16 on MRADIX).
_SMALL_K_MAX = 10
_SMALL_K_MAX_COLS = 4096
_SMALL_K_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit
def topk_small_k_kernel(
    X,
    Yv,
    Yi,
    stride_xm,
    stride_ym,
    n_cols,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    live = cols < n_cols
    xv = tl.load(X + row * stride_xm + cols, mask=live, other=float("-inf"))
    # Compare on the monotone signed-int key of the float bits rather than on
    # the floats themselves. IEEE makes -0.0 == +0.0, but torch.topk orders
    # +0.0 above -0.0 (it sorts bit patterns), so a plain float compare picks
    # the wrong index whenever a -0.0 and a +0.0 compete for the last slot.
    # The key is bijective, so the winner's bits -- and therefore its exact
    # -0.0 / NaN payload -- are recovered by inverting it, with no gather.
    bits = xv.to(tl.float32).to(tl.int32, bitcast=True)
    key = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    taken = tl.zeros([BLOCK_N], tl.int1)
    for j in tl.static_range(K):
        vk = tl.max(tl.where(taken, -2147483648, key), axis=0)
        # Leftmost column among the maximum keys: torch's ascending-index tie
        # break. (`tl.max(..., return_indices=True)` would fold this into one
        # reduction, but on GCU it is ~2x slower in throughput at large N, so
        # the two plain reductions win wherever the row set is big.)
        i = tl.min(tl.where((key == vk) & live & (~taken), cols, BLOCK_N), axis=0)
        vb = tl.where(vk >= 0, vk, vk ^ 0x7FFFFFFF)
        tl.store(Yv + row * stride_ym + j, vb.to(tl.float32, bitcast=True))
        tl.store(Yi + row * stride_ym + j, i)
        taken = taken | (cols == i)


def _topk_small_k(x_2d, k, out_shape):
    n_rows, n_cols = x_2d.shape
    y_vals = torch.empty((n_rows, k), device=x_2d.device, dtype=x_2d.dtype)
    y_idx32 = torch.empty((n_rows, k), device=x_2d.device, dtype=torch.int32)
    block_n = max(128, triton.next_power_of_2(n_cols))
    with torch_device_fn.device(x_2d.device):
        topk_small_k_kernel[(n_rows,)](
            x_2d,
            y_vals,
            y_idx32,
            x_2d.stride(0),
            y_vals.stride(0),
            n_cols,
            K=k,
            BLOCK_N=block_n,
            num_warps=1,
            num_stages=1,
        )
    return y_vals.reshape(out_shape), y_idx32.to(torch.int64).reshape(out_shape)


def topk(x, k, dim=-1, largest=True, sorted=True):
    # Small-K fast path first. It is the common small-k case (attention and
    # sampling kernels) and its checks are cheap, so it must not sit behind the
    # generic prologue (logging, shape product, the MRADIX envelope probe):
    # every Python step costs ~1-2us and dominates these tiny launches.
    if x.ndim >= 1 and (dim == -1 or dim == x.ndim - 1):
        n_cols = x.shape[-1]
        if (
            largest
            and 0 < k <= _SMALL_K_MAX
            and k <= n_cols <= _SMALL_K_MAX_COLS
            and x.dtype in _SMALL_K_DTYPES
            and x.is_contiguous()
        ):
            n_rows = x.numel() // n_cols
            return _topk_small_k(x.view(n_rows, n_cols), k, x.shape[:-1] + (k,))

    logger.debug("GEMS_ENFLAME TOPK")
    # If dim equals to last dim, we set it to -1.
    if dim < 0:
        dim = dim + x.ndim

    assert dim == x.ndim - 1, "Currently only support topk in last dimension"
    # assert sorted, "Currently only support sorted == True"

    # Early return for k=0 to avoid Triton kernel compilation error.
    # Triton's tl.arange(0, BLOCK_SIZE) requires BLOCK_SIZE > 0.
    # When k=0, stage2_elem_cnt becomes 0, leading to BLOCK_SIZE=0.
    if k == 0:
        out_shape = list(x.shape[:-1]) + [0]
        return (
            torch.empty(out_shape, device=x.device, dtype=x.dtype),
            torch.empty(out_shape, device=x.device, dtype=torch.int64),
        )

    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    # HW MRADIX level-1 path: one hw thread per SIP walks a strided row set,
    # stages each row into VDMEM (tle.gpu.copy) and radix-selects it on the
    # MRADIX unit, so a row costs one HBM read plus a few VDMEM passes instead
    # of k sequential max/argmax sweeps. Its envelope (dtype, dim, N, K) is
    # checked inside; everything else falls through to the generic path.
    if _mradix is not None and _mradix.topk_mradix_supported(
        x, k, dim == x.ndim - 1, largest, sorted
    ):
        out_shape = x.shape[:-1] + (k,)
        x_2d = x.reshape(batch_size, topk_elem_cnt)
        return _mradix.topk_mradix(x_2d, k, out_shape, sorted)

    if (
        HAS_TLE
        and sorted
        and descending
        and x.is_cuda
        and x.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and k >= 8
        and topk_elem_cnt <= 65535
        and triton.next_power_of_2(k) <= 1024
    ):
        k_pad = triton.next_power_of_2(k)
        out_shape = x.shape[:-1] + (k,)
        y_vals = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        y_idx = torch.empty(out_shape, device=x.device, dtype=torch.int32)
        block_n_radix = max(k_pad, min(512, triton.next_power_of_2(topk_elem_cnt)))
        block_n_radix = min(block_n_radix, 1024)

        x_2d = x.reshape(batch_size, topk_elem_cnt)
        y_vals_2d = y_vals.reshape(batch_size, k)
        y_idx_2d = y_idx.reshape(batch_size, k)
        with torch_device_fn.device(x.device):
            topk_kernel_radix_tle[(batch_size,)](
                x_2d,
                y_vals_2d,
                y_idx_2d,
                x_2d.stride(0),
                y_vals_2d.stride(0),
                topk_elem_cnt,
                K=k,
                K_PAD=k_pad,
                BLOCK_N=block_n_radix,
                RADIX_BITS=4,
                num_warps=4,
                num_stages=1,
            )
        return (y_vals, y_idx.to(torch.int64))

    # Note(Zhengzekang): Maybe we should add a heuristic search in selecting a proper chunk size.
    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    # Note(Zhengzekang): We should promise chunk_size is larger than k.
    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int64
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    with torch_device_fn.device(x.device):
        topk_stage1_kernel[
            batch_size,
            chunk_num,
        ](
            stage1_out,  # pointer to the output
            stage1_out_idx,  # pointer to the output
            x,  # pointer to the input
            k,
            topk_elem_cnt,
            chunk_size,
            descending,
        )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    MAX_STAGE2_BLOCK_SIZE = 4096
    while BLOCK_SIZE > MAX_STAGE2_BLOCK_SIZE:
        reduction_chunk_size = triton.next_power_of_2(max(1024, k * 2))
        new_chunk_num = triton.cdiv(stage2_elem_cnt, reduction_chunk_size)
        new_stage1_out = torch.empty(
            batch_size * new_chunk_num * k, device=x.device, dtype=x.dtype
        )
        new_stage1_out_idx = torch.empty(
            batch_size * new_chunk_num * k, device=x.device, dtype=torch.int64
        )

        with torch_device_fn.device(x.device):
            topk_stage1_with_index_kernel[batch_size, new_chunk_num](
                new_stage1_out,
                new_stage1_out_idx,
                stage1_out,
                stage1_out_idx,
                k,
                stage2_elem_cnt,
                reduction_chunk_size,
                descending,
            )

        stage1_out = new_stage1_out
        stage1_out_idx = new_stage1_out_idx
        stage2_elem_cnt = new_chunk_num * k
        BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    with torch_device_fn.device(x.device):
        topk_stage2_kernel[batch_size,](
            stage2_out,
            stage2_out_idx,
            stage1_out,
            stage1_out_idx,
            dim,
            k,
            stage2_elem_cnt,
            BLOCK_SIZE,
            descending,
        )

    return (stage2_out, stage2_out_idx)

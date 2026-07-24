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
from flag_gems.utils import dim_compress, libentry

from ..utils import TOTAL_CORE_NUM

logger = logging.getLogger(__name__)

REDUCE_PROD = 0
REDUCE_MEAN = 1
REDUCE_AMAX = 2
REDUCE_AMIN = 3


def _heur_block_m(args):
    M = args["M"]
    return 1 if M < 4 else 4


def _heur_block_n(args):
    N = args["N"]
    return max(1, min(256, triton.next_power_of_2(N)))


def _heur_flat_block(args):
    total = args["TOTAL"] if "TOTAL" in args else args["N"]
    return max(1, min(256, triton.next_power_of_2(total)))


@libentry()
@triton.heuristics({"BLOCK_M": _heur_block_m, "BLOCK_N": _heur_block_n})
@triton.jit(do_not_specialize=["M", "N", "OUT_N"])
def _index_reduce_kernel(
    out,
    index,
    src,
    count,
    touched,
    M,
    N,
    OUT_N,
    REDUCE: tl.constexpr,
    USE_COUNT: tl.constexpr,
    USE_TOUCHED: tl.constexpr,
    USE_CAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)

    dst_cols = tl.load(index + cols, mask=cols < N, other=0).to(tl.int64)
    src_offsets = rows * N + cols
    out_offsets = rows * OUT_N + dst_cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)

    if REDUCE == 1:
        tl.atomic_add(out + out_offsets, values, mask=mask, sem="relaxed")
        ones_i = tl.full((BLOCK_M, BLOCK_N), 1, dtype=tl.int32)
        tl.atomic_add(count + out_offsets, ones_i, mask=mask, sem="relaxed")
    elif REDUCE == 0:
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur = tl.load(out + out_offsets, mask=mask, other=0.0)
            new = tl.where(stop, cur, cur * values)
            is_nan = new != new
            new = tl.where(is_nan, values, new)
            cas = tl.atomic_cas(out + out_offsets, cur, new, mask=mask, sem="relaxed")
            stop |= (cur == cas) | is_nan
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK_M * BLOCK_N
    else:
        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur = tl.load(out + out_offsets, mask=mask, other=0.0)
                if REDUCE == 2:
                    new = tl.maximum(cur, values)
                else:
                    new = tl.minimum(cur, values)
                cas = tl.atomic_cas(
                    out + out_offsets, cur, new, mask=mask, sem="relaxed"
                )
                stop |= cur == cas
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK_M * BLOCK_N
        else:
            if REDUCE == 2:
                tl.atomic_max(out + out_offsets, values, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out + out_offsets, values, mask=mask, sem="relaxed")

    if USE_TOUCHED:
        ones_i = tl.full((BLOCK_M, BLOCK_N), 1, dtype=tl.int32)
        tl.atomic_add(touched + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": _heur_flat_block})
@triton.jit(do_not_specialize=["TOTAL", "M", "N", "OUT_N"])
def _index_reduce_flat_kernel(
    out,
    index,
    src,
    count,
    touched,
    TOTAL,
    M,
    N,
    OUT_N,
    REDUCE: tl.constexpr,
    USE_COUNT: tl.constexpr,
    USE_TOUCHED: tl.constexpr,
    USE_CAS: tl.constexpr,
    INDEX_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    if INDEX_MAJOR:
        cols = offsets // M
        rows = offsets - cols * M
    else:
        rows = offsets // N
        cols = offsets - rows * N

    dst_cols = tl.load(index + cols, mask=mask, other=0).to(tl.int64)
    src_offsets = rows * N + cols
    out_offsets = rows * OUT_N + dst_cols
    values = tl.load(src + src_offsets, mask=mask, other=0.0)

    if REDUCE == 1:
        tl.atomic_add(out + out_offsets, values, mask=mask, sem="relaxed")
        ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(count + out_offsets, ones_i, mask=mask, sem="relaxed")
    elif REDUCE == 0:
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur = tl.load(out + out_offsets, mask=mask, other=0.0)
            new = tl.where(stop, cur, cur * values)
            is_nan = new != new
            new = tl.where(is_nan, values, new)
            cas = tl.atomic_cas(out + out_offsets, cur, new, mask=mask, sem="relaxed")
            stop |= (cur == cas) | is_nan
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
    else:
        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur = tl.load(out + out_offsets, mask=mask, other=0.0)
                if REDUCE == 2:
                    new = tl.maximum(cur, values)
                else:
                    new = tl.minimum(cur, values)
                new = new.to(cur.dtype)
                cas = tl.atomic_cas(
                    out + out_offsets, cur, new, mask=mask, sem="relaxed"
                )
                stop |= cur == cas
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if REDUCE == 2:
                tl.atomic_max(out + out_offsets, values, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out + out_offsets, values, mask=mask, sem="relaxed")

    if USE_TOUCHED:
        ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(touched + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": _heur_flat_block})
@triton.jit(do_not_specialize=["TOTAL", "PRE", "POST", "N", "OUT_N"])
def _index_reduce_contiguous_flat_kernel(
    out,
    index,
    src,
    count,
    touched,
    TOTAL,
    PRE,
    POST,
    N,
    OUT_N,
    REDUCE: tl.constexpr,
    USE_COUNT: tl.constexpr,
    USE_TOUCHED: tl.constexpr,
    USE_CAS: tl.constexpr,
    INDEX_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    slice_size = PRE * POST

    if INDEX_MAJOR:
        cols = offsets // slice_size
        element = offsets - cols * slice_size
    else:
        element = offsets // N
        cols = offsets - element * N

    pre = element // POST
    post = element - pre * POST
    dst_cols = tl.load(index + cols, mask=mask, other=0).to(tl.int64)

    src_offsets = pre * N * POST + cols * POST + post
    out_offsets = pre * OUT_N * POST + dst_cols * POST + post
    values = tl.load(src + src_offsets, mask=mask, other=0.0)

    if REDUCE == 1:
        tl.atomic_add(out + out_offsets, values, mask=mask, sem="relaxed")
        ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(count + out_offsets, ones_i, mask=mask, sem="relaxed")
    elif REDUCE == 0:
        stop = tl.where(mask, 0, 1).to(tl.int1)
        block_stop = False
        while not block_stop:
            cur = tl.load(out + out_offsets, mask=mask, other=0.0)
            new = tl.where(stop, cur, cur * values)
            is_nan = new != new
            new = tl.where(is_nan, values, new)
            cas = tl.atomic_cas(out + out_offsets, cur, new, mask=mask, sem="relaxed")
            stop |= (cur == cas) | is_nan
            block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
    else:
        if USE_CAS:
            stop = tl.where(mask, 0, 1).to(tl.int1)
            block_stop = False
            while not block_stop:
                cur = tl.load(out + out_offsets, mask=mask, other=0.0)
                if REDUCE == 2:
                    new = tl.maximum(cur, values)
                else:
                    new = tl.minimum(cur, values)
                new = new.to(cur.dtype)
                cas = tl.atomic_cas(
                    out + out_offsets, cur, new, mask=mask, sem="relaxed"
                )
                stop |= cur == cas
                block_stop = tl.sum(stop.to(tl.int32)) == BLOCK
        else:
            if REDUCE == 2:
                tl.atomic_max(out + out_offsets, values, mask=mask, sem="relaxed")
            else:
                tl.atomic_min(out + out_offsets, values, mask=mask, sem="relaxed")

    if USE_TOUCHED:
        ones_i = tl.full((BLOCK,), 1, dtype=tl.int32)
        tl.atomic_add(touched + out_offsets, ones_i, mask=mask, sem="relaxed")


@libentry()
@triton.heuristics({"BLOCK": _heur_flat_block})
@triton.jit(do_not_specialize=["TOTAL"])
def _index_reduce_mean_finalize_kernel(
    result,
    acc,
    original,
    count,
    TOTAL,
    INCLUDE_SELF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    cnt = tl.load(count + offsets, mask=mask, other=0)
    acc_val = tl.load(acc + offsets, mask=mask, other=0.0).to(tl.float32)
    if INCLUDE_SELF:
        denom = cnt + 1
        result_val = acc_val / denom.to(tl.float32)
    else:
        denom = tl.maximum(cnt, 1)
        mean_val = acc_val / denom.to(tl.float32)
        original_val = tl.load(original + offsets, mask=mask, other=0.0)
        result_val = tl.where(cnt > 0, mean_val, original_val)
    tl.store(result + offsets, result_val, mask=mask)


@libentry()
@triton.heuristics({"BLOCK_M": _heur_block_m, "BLOCK_N": _heur_block_n})
@triton.jit(do_not_specialize=["M", "N", "OUT_N"])
def _index_reduce_unique_kernel(
    out,
    index,
    src,
    M,
    N,
    OUT_N,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    mask = (rows < M) & (cols < N)

    dst_cols = tl.load(index + cols, mask=cols < N, other=0).to(tl.int64)
    src_offsets = rows * N + cols
    out_offsets = rows * OUT_N + dst_cols
    src_values = tl.load(src + src_offsets, mask=mask, other=0.0)

    if INCLUDE_SELF:
        inp_values = tl.load(out + out_offsets, mask=mask, other=0.0)
        if REDUCE == 0:
            result = inp_values * src_values
        elif REDUCE == 1:
            result = (inp_values + src_values) * 0.5
        elif REDUCE == 2:
            result = tl.maximum(inp_values, src_values)
        else:
            result = tl.minimum(inp_values, src_values)
    else:
        result = src_values

    tl.store(out + out_offsets, result, mask=mask)


@libentry()
@triton.heuristics({"BLOCK_N": _heur_block_n})
@triton.jit(do_not_specialize=["TOTAL", "N", "OUT_N"])
def _index_reduce_scan_kernel(
    out,
    index,
    src,
    inp,
    TOTAL,
    N,
    OUT_N,
    REDUCE: tl.constexpr,
    INCLUDE_SELF: tl.constexpr,
    USE_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_jobs = tl.num_programs(axis=0)
    out_pos = pid
    while out_pos < TOTAL:
        row = out_pos // OUT_N
        dst_col = (out_pos - row * OUT_N).to(tl.int64)
        inp_val = tl.load(inp + out_pos).to(tl.float32)
        if USE_FP64:
            inp_val = inp_val.to(tl.float64)

        if REDUCE == 0:
            if USE_FP64:
                acc = inp_val if INCLUDE_SELF else tl.full((), 1.0, dtype=tl.float64)
            else:
                acc = inp_val if INCLUDE_SELF else tl.full((), 1.0, dtype=tl.float32)
        elif REDUCE == 1:
            if USE_FP64:
                acc = inp_val if INCLUDE_SELF else tl.full((), 0.0, dtype=tl.float64)
            else:
                acc = inp_val if INCLUDE_SELF else tl.full((), 0.0, dtype=tl.float32)
        elif REDUCE == 2:
            if USE_FP64:
                acc = (
                    inp_val
                    if INCLUDE_SELF
                    else tl.full((), float("-inf"), dtype=tl.float64)
                )
            else:
                acc = (
                    inp_val
                    if INCLUDE_SELF
                    else tl.full((), float("-inf"), dtype=tl.float32)
                )
        else:
            if USE_FP64:
                acc = (
                    inp_val
                    if INCLUDE_SELF
                    else tl.full((), float("inf"), dtype=tl.float64)
                )
            else:
                acc = (
                    inp_val
                    if INCLUDE_SELF
                    else tl.full((), float("inf"), dtype=tl.float32)
                )

        hit_count = tl.full((), 1 if INCLUDE_SELF else 0, dtype=tl.int32)
        if REDUCE == 0:
            col = 0
            while col < N:
                current_col = tl.load(index + col).to(tl.int64)
                matched = current_col == dst_col
                value = tl.load(src + row * N + col, mask=matched, other=1.0)
                if USE_FP64:
                    value = value.to(tl.float64)
                else:
                    value = value.to(tl.float32)
                acc *= tl.where(matched, value, 1.0)
                hit_count += matched.to(tl.int32)
                col += 1
        else:
            offsets = tl.arange(0, BLOCK_N)
            start = 0
            while start < N:
                cols = start + offsets
                mask = cols < N
                dst_cols = tl.load(index + cols, mask=mask, other=-1).to(tl.int64)
                matched = mask & (dst_cols == dst_col)
                values = tl.load(src + row * N + cols, mask=mask, other=0.0)
                if USE_FP64:
                    values = values.to(tl.float64)
                else:
                    values = values.to(tl.float32)

                matched_count = tl.sum(matched.to(tl.int32), axis=0)
                hit_count += matched_count
                if REDUCE == 1:
                    acc += tl.sum(tl.where(matched, values, 0.0), axis=0)
                elif REDUCE == 2:
                    acc = tl.maximum(
                        acc, tl.max(tl.where(matched, values, float("-inf")), axis=0)
                    )
                else:
                    acc = tl.minimum(
                        acc, tl.min(tl.where(matched, values, float("inf")), axis=0)
                    )
                start += BLOCK_N

        if REDUCE == 1:
            acc = acc / tl.maximum(hit_count, 1).to(tl.float32)
        result = tl.where(hit_count > 0, acc, inp_val)
        tl.store(out + out_pos, result)
        out_pos += num_jobs


def _reduce_id(reduce):
    if reduce == "prod":
        return REDUCE_PROD
    if reduce == "mean":
        return REDUCE_MEAN
    if reduce == "amax":
        return REDUCE_AMAX
    if reduce == "amin":
        return REDUCE_AMIN
    raise RuntimeError(f"Unsupported reduce: {reduce}")


def _identity_like(inp, reduce):
    if reduce == "prod":
        return torch.ones_like(inp)
    if reduce == "mean":
        return torch.zeros_like(inp)
    if reduce == "amax":
        return torch.full_like(inp, float("-inf"))
    if reduce == "amin":
        return torch.full_like(inp, float("inf"))
    raise RuntimeError(f"Unsupported reduce: {reduce}")


def _index_is_unique(index, out_dim):
    if index.numel() > out_dim:
        return False
    if index.numel() <= 1:
        return True
    return index.unique().numel() == index.numel()


def _prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def _validate_args(inp, dim, index, source, reduce):
    assert reduce in ("prod", "mean", "amax", "amin"), f"Unsupported reduce: {reduce}"
    assert inp.ndim > 0, "Expected self to have at least one dimension"
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim == 1, "Index is supposed to be a vector"
    assert index.dtype in (
        torch.int32,
        torch.int64,
    ), "Expected dtype int32/int64 for index"
    assert (
        inp.is_floating_point()
    ), "index_reduce_(): Expected self to be floating point"
    assert (
        source.dtype == inp.dtype
    ), "index_reduce_(): Expected self and source to have same dtype"
    assert (
        inp.ndim == source.ndim
    ), "Self and source should have the same number of dimensions"
    assert index.numel() == source.size(
        dim
    ), "The dimth dimension of source must have the same size as the length of index"
    assert all(
        inp.size(i) == source.size(i) or i == dim for i in range(inp.ndim)
    ), "source.size(d) == self.size(d) for all dimensions d != dim"


def _restore_dim(out, inp, dim):
    if (
        out.data_ptr() == inp.data_ptr()
        and out.shape == inp.shape
        and out.stride() == inp.stride()
    ):
        return inp
    final_dim = inp.ndim - 1
    if dim != final_dim:
        order = list(range(out.ndim - 1))
        order.insert(dim, final_dim)
        out = out.permute(order).contiguous()
    inp.copy_(out)
    return inp


def index_reduce_(inp, dim, index, source, reduce, *, include_self=True):
    logger.debug("GEMS_CAMBRICON INDEX_REDUCE_")
    _validate_args(inp, dim, index, source, reduce)

    if index.numel() == 0:
        return inp

    dim = dim % inp.ndim
    index = index.contiguous()
    reduce_id = _reduce_id(reduce)

    inp_work = dim_compress(inp, dim)
    source_work = dim_compress(source, dim)

    N = index.numel()
    out_n = inp_work.size(-1)

    if _index_is_unique(index, out_n):
        M = source_work.numel() // N
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        with torch_device_fn.device(inp.device):
            _index_reduce_unique_kernel[grid](
                inp_work,
                index,
                source_work,
                M,
                N,
                out_n,
                reduce_id,
                include_self,
            )
        return _restore_dim(inp_work, inp, dim)

    compute_dtype = torch.float64 if inp_work.dtype == torch.float64 else torch.float32
    inp_compute = inp_work.to(compute_dtype)
    source_compute = source_work.to(compute_dtype)
    out = torch.empty_like(inp_compute)
    total = inp_compute.numel()
    grid = lambda meta: (min(total, TOTAL_CORE_NUM),)
    with torch_device_fn.device(inp.device):
        _index_reduce_scan_kernel[grid](
            out,
            index,
            source_compute,
            inp_compute,
            total,
            N,
            out_n,
            reduce_id,
            include_self,
            compute_dtype == torch.float64,
        )
    return _restore_dim(out.to(inp.dtype), inp, dim)

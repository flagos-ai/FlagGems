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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device as runtime_device  # noqa: F401
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.limits import get_dtype_max, get_dtype_min
from flag_gems.utils.shape_utils import c_contiguous_stride

from .topk import _get_finfo_val

logger = logging.getLogger(__name__)

NanMedian = namedtuple("nanmedian", ["values", "indices"])
MAX_BLOCK_N = 512
RADIX_BLOCK_N = 4096
LARGE_FLOAT_REDUCTION_N = 4096
LONG_RADIX_REDUCTION_N = 131072
# GCU400 radix-select tuning: 4 bit digits keep the per-row kernel at
# nbits/4 row scans (fp32: 10 scans incl. count+find) while the one-hot
# bin counting stays in the proven tl.sum(axis=0) form (tl.histogram is
# untrusted on this backend: silent mask drop + nondeterminism).
GCU_RADIX_BITS = 4
GCU_RADIX_BLOCK_N = 8192
GCU_RADIX_BLOCK_C = 64
# Hardware grid.x limit on this backend. Row-parallel launches chunk M in
# host-side batches of GCU_GRID_X_SAFE, staying clear of the raw 65535
# boundary (P19: boundary values themselves are hazardous).
GCU_GRID_X_LIMIT = 65535
GCU_GRID_X_SAFE = 32768
GCU_MULTIK_N_MIN = 32768
GCU_MULTIK_M_MAX = 32
# Grid-parallel count capacity: one 1D grid.x. Beyond this the giant 2D
# fan-out measured device-unstable (async Sip faults at 600M-element scale
# even with a safe grid.x margin), so flat inputs above it defer to the
# catchable aten kthvalue fallback instead of risking a device fault.
GCU_MULTIK_MAX_ELEMS = GCU_GRID_X_LIMIT * GCU_RADIX_BLOCK_N
RADIX_SELECT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@triton.jit
def _is_not_nan(vals, USE_ISNAN: tl.constexpr):
    vals_fp32 = vals.to(tl.float32)
    return vals_fp32 == vals_fp32


@triton.jit
def _to_order_key(vals, valid):
    dtype = vals.dtype
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    top_mask: tl.constexpr = 1 << (nbits - 1)
    full_mask: tl.constexpr = (1 << nbits) - 1
    full = tl.full(vals.shape, full_mask, dtype=utype)

    if dtype.is_floating():
        bits = vals.to(utype, bitcast=True)
        sign_mask = tl.where((bits & top_mask) != 0, full_mask, top_mask)
        key = bits ^ sign_mask
    elif dtype.is_int_signed():
        bits = vals.to(utype, bitcast=True)
        key = bits ^ top_mask
    else:
        key = vals.to(utype)
    return tl.where(valid, key, full)


@libentry()
@triton.jit
def count_valid_kernel(
    inp,
    valid_counts,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    count = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + pid * N + cols, mask=mask, other=float("nan"))
        valid = mask & _is_not_nan(vals, USE_ISNAN)
        count += tl.sum(valid.to(tl.int32), axis=0)
    tl.store(valid_counts + pid, count)


@libentry()
@triton.jit
def nanmedian_select_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N
    dtype = inp.dtype.element_ty
    if dtype.is_floating():
        max_value = _get_finfo_val(dtype, return_max=True)
        fallback_value = _get_finfo_val(dtype, return_max=False)
    else:
        max_value = get_dtype_max(dtype)
        fallback_value = get_dtype_min(dtype)
    vals = tl.load(inp + pid * N + offsets, mask=mask, other=max_value)

    if dtype.is_floating():
        valid = mask & _is_not_nan(vals, USE_ISNAN)
    else:
        valid = mask
    valid_count = tl.sum(valid.to(tl.int32), axis=0)
    median_rank = (valid_count - 1) // 2

    active = valid
    median_val = tl.full((), fallback_value, dtype=vals.dtype)
    median_idx = tl.full((), 0, dtype=tl.int32)
    for select_iter in tl.static_range(0, BLOCK_N):
        select_vals = tl.where(active, vals, max_value)
        cur_val = tl.min(select_vals, axis=0)
        cur_idx = tl.min(tl.where(active & (vals == cur_val), offsets, BLOCK_N), axis=0)
        take = select_iter == median_rank
        median_val = tl.where(take, cur_val, median_val)
        median_idx = tl.where(take, cur_idx, median_idx)
        active = active & (offsets != cur_idx)

    if dtype.is_floating():
        all_nan = valid_count == 0
        median_val = tl.where(all_nan, float("nan"), median_val)
        median_idx = tl.where(all_nan, 0, median_idx)

    tl.store(out_values + pid, median_val)
    tl.store(out_indices + pid, median_idx)


@libentry()
@triton.jit
def gcu_nanmedian_row_radix_kernel(
    inp,
    out_values,
    out_indices,
    M,
    N,
    BLOCK_N: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    row = tle.program_id(0)
    dtype = inp.dtype.element_ty
    nbits: tl.constexpr = dtype.primitive_bitwidth
    utype = tl.dtype(f"uint{nbits}")
    radix_size: tl.constexpr = 1 << RADIX_BITS_
    radix_mask: tl.constexpr = radix_size - 1
    bins = tl.arange(0, radix_size)
    radix_mask_val = tl.full((), radix_mask, dtype=utype)
    offsets = tl.arange(0, BLOCK_N)

    valid_count = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + row * N + cols, mask=mask, other=0.0)
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask
        valid_count += tl.sum(valid.to(tl.int32), axis=0)
    k = tl.where(valid_count > 0, (valid_count + 1) // 2, 0)

    desired = tl.full((), 0, dtype=utype)
    desired_mask = tl.full((), 0, dtype=utype)
    for i in tl.range(0, nbits, RADIX_BITS_):
        digit_pos = nbits - RADIX_BITS_ - i
        counts = tl.zeros((radix_size,), dtype=tl.int32)
        for start in tl.range(0, N, BLOCK_N):
            cols = start + offsets
            mask = cols < N
            vals = tl.load(inp + row * N + cols, mask=mask, other=0.0)
            if dtype.is_floating():
                valid = mask & _is_not_nan(vals, USE_ISNAN)
            else:
                valid = mask
            keys = _to_order_key(vals, valid)
            active = valid & ((keys & desired_mask) == desired)
            digit = ((keys >> digit_pos) & radix_mask_val).to(tl.int32)
            for radix_bin in tl.static_range(0, radix_size):
                bin_count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
                counts += tl.where(bins == radix_bin, bin_count, 0)
        cumsum = tl.cumsum(counts, axis=0)
        prev = cumsum - counts
        take = (cumsum >= k) & (prev < k)
        selected_bin = tl.min(tl.where(take, bins, radix_size - 1), axis=0)
        counts_before = tl.max(tl.where(take, prev, 0), axis=0)
        # Runtime digit_pos promotes the shifted value past utype; truncating
        # back keeps the loop-carried key/mask types stable.
        desired = desired | ((selected_bin.to(utype) << digit_pos).to(utype))
        desired_mask = desired_mask | ((radix_mask_val << digit_pos).to(utype))
        k = k - counts_before

    result_idx = tl.full((), N, dtype=tl.int32)
    for start in tl.range(0, N, BLOCK_N):
        cols = start + offsets
        mask = cols < N
        vals = tl.load(inp + row * N + cols, mask=mask, other=0.0)
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask
        keys = _to_order_key(vals, valid)
        local_idx = tl.min(tl.where(valid & (keys == desired), cols, N), axis=0)
        result_idx = tl.where(local_idx < result_idx, local_idx, result_idx)

    if dtype.is_floating():
        result_val = tl.load(
            inp + row * N + result_idx, mask=valid_count > 0, other=float("nan")
        )
        result_val = tl.where(valid_count > 0, result_val, float("nan"))
        result_idx = tl.where(valid_count > 0, result_idx, 0)
    else:
        result_val = tl.load(
            inp + row * N + result_idx, mask=valid_count > 0, other=get_dtype_min(dtype)
        )

    tl.store(out_values + row, result_val)
    tl.store(out_indices + row, result_idx.to(tl.int64))


def _check_supported_dtype(inp):
    if inp.dtype is torch.bool:
        raise NotImplementedError("\"median_out_impl\" not implemented for 'Bool'")


def _normalize_dim(dim, ndim):
    if ndim == 0:
        if dim in (0, -1):
            return 0
    elif -ndim <= dim < ndim:
        return dim % ndim
    raise IndexError(
        f"Dimension out of range (expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})"
    )


def _empty_flat_value(inp):
    out = torch.empty((), dtype=inp.dtype, device=inp.device)
    if torch.is_floating_point(inp):
        out.fill_(float("nan"))
    elif inp.dtype in (torch.int8, torch.uint8, torch.int16):
        # PyTorch converts the empty reduction sentinel to zero for narrow integers.
        out.zero_()
    else:
        out.fill_(torch.iinfo(inp.dtype).min)
    return out


def _full_nan_result(shape, dtype, device):
    values = torch.full(shape, float("nan"), dtype=dtype, device=device)
    indices = torch.zeros(shape, dtype=torch.long, device=device)
    return NanMedian(values=values, indices=indices)


def _count_block_n(inp, n):
    block_n = triton.next_power_of_2(n)
    if inp.is_cuda and n >= LONG_RADIX_REDUCTION_N:
        return min(block_n, 16384)
    if n >= LONG_RADIX_REDUCTION_N:
        return min(block_n, 4096)
    if n >= LARGE_FLOAT_REDUCTION_N:
        return min(block_n, 2048)
    return min(block_n, RADIX_BLOCK_N)


def _nanmedian_kthvalue_fallback(inp, M, N):
    inp = inp.reshape(M, N)
    if torch.is_floating_point(inp):
        valid_count = torch.empty((M,), dtype=torch.long, device=inp.device)
        block_n = _count_block_n(inp, N)
        with torch_device_fn.device(inp.device):
            count_valid_kernel[(M,)](inp, valid_count, M, N, block_n, inp.is_cuda)
        # Replace NaN with +inf so kthvalue sorts them to the end deterministically
        kth_inp = torch.where(
            torch.isnan(inp),
            torch.tensor(float("inf"), dtype=inp.dtype, device=inp.device),
            inp,
        )
        min_count = int(torch.min(valid_count).item())
        max_count = int(torch.max(valid_count).item())
        if min_count == max_count:
            if max_count == 0:
                return _full_nan_result((M,), inp.dtype, inp.device)
            values, indices = torch.kthvalue(kth_inp, (max_count + 1) // 2, dim=1)
            return NanMedian(values=values, indices=indices)

        if max_count - min_count <= 1:
            min_k = (min_count + 1) // 2 if min_count > 0 else 0
            max_k = (max_count + 1) // 2

            if min_k == max_k:
                values, indices = torch.kthvalue(kth_inp, max_k, dim=1)
                if min_count > 0:
                    return NanMedian(values=values, indices=indices)
                fallback = _full_nan_result((M,), inp.dtype, inp.device)
                positive = valid_count > 0
                return NanMedian(
                    values=torch.where(positive, values, fallback.values),
                    indices=torch.where(positive, indices, fallback.indices),
                )

            result = _full_nan_result((M,), inp.dtype, inp.device)

            if min_count > 0:
                values, indices = torch.kthvalue(kth_inp, min_k, dim=1)
                mask = valid_count == min_count
                result = NanMedian(
                    values=torch.where(mask, values, result.values),
                    indices=torch.where(mask, indices, result.indices),
                )

            values, indices = torch.kthvalue(kth_inp, max_k, dim=1)
            mask = valid_count == max_count
            return NanMedian(
                values=torch.where(mask, values, result.values),
                indices=torch.where(mask, indices, result.indices),
            )

        result = _full_nan_result((M,), inp.dtype, inp.device)
        for count in torch.unique(valid_count).tolist():
            count = int(count)
            if count == 0:
                continue
            row_indices = torch.nonzero(valid_count == count).flatten()
            rows = torch.index_select(kth_inp, 0, row_indices)
            values, indices = torch.kthvalue(rows, (count + 1) // 2, dim=1)
            result.values[row_indices] = values
            result.indices[row_indices] = indices
        return result
    else:
        values, indices = torch.kthvalue(inp, (N + 1) // 2, dim=1)
        return NanMedian(values=values, indices=indices)


def _gcu_radix_multik_impl(inp, M, N, out_values, out_indices):
    num_chunks = triton.cdiv(N, GCU_RADIX_BLOCK_N)
    radix_size = 1 << GCU_RADIX_BITS
    nbits = inp.element_size() * 8
    device = inp.device
    partial = torch.empty(
        (M * num_chunks, radix_size), dtype=torch.int32, device=device
    )
    # desired / desired_mask / k per row; zero-init selects all elements in pass 0.
    # 64-bit order keys (int64) need int64 state slots so the accumulating
    # desired / desired_mask survive the round trip through global memory.
    state_dtype = torch.int64 if nbits > 32 else torch.int32
    state = torch.zeros((M, 3), dtype=state_dtype, device=device)
    valid_counts = torch.empty((M,), dtype=torch.int32, device=device)
    cand = torch.empty((M * num_chunks,), dtype=torch.int32, device=device)
    # The dispatch gate keeps the count grid within one 1D grid.x; the 2D
    # linearized form remains only as a defensive cap.
    total = M * num_chunks
    grid_x = min(total, GCU_GRID_X_LIMIT)
    count_grid = (grid_x, triton.cdiv(total, grid_x))
    state_is_64bit = nbits > 32
    with torch_device_fn.device(device):
        for pass_index, digit_pos in enumerate(
            range(nbits - GCU_RADIX_BITS, -1, -GCU_RADIX_BITS)
        ):
            gcu_radix_partial_count_kernel[count_grid](
                inp,
                partial,
                state,
                N,
                num_chunks,
                total,
                grid_x,
                GCU_RADIX_BLOCK_N,
                digit_pos,
                radix_size,
                False,
                state_is_64bit,
                num_warps=4,
                num_stages=1,
            )
            gcu_radix_row_update_kernel[(M,)](
                partial,
                state,
                valid_counts,
                num_chunks,
                digit_pos,
                radix_size,
                GCU_RADIX_BITS,
                nbits,
                GCU_RADIX_BLOCK_C,
                pass_index == 0,
                state_is_64bit,
                num_warps=4,
                num_stages=1,
            )
        gcu_radix_find_index_kernel[count_grid](
            inp,
            state,
            cand,
            N,
            num_chunks,
            total,
            grid_x,
            GCU_RADIX_BLOCK_N,
            False,
            num_warps=4,
            num_stages=1,
        )
        gcu_radix_finalize_kernel[(M,)](
            inp,
            state,
            valid_counts,
            cand,
            out_values,
            out_indices,
            N,
            num_chunks,
            GCU_RADIX_BLOCK_C,
            num_warps=2,
            num_stages=1,
        )


def _gcu_dim_compute(inp, M, N, out_values, out_indices):
    """Launch a GCU kernel path for the (M, N) contiguous input.

    Returns True when one of the Triton paths handled the reduction, False
    when the caller must fall back to the aten-based kthvalue path.
    """
    if out_values.dim() == 0:
        # Flat callers hand in the 0-dim result tensor (M == 1).
        out_values = out_values.reshape(1)
        out_indices = out_indices.reshape(1)

    if N <= MAX_BLOCK_N and inp.dtype is not torch.float64:
        block_n = triton.next_power_of_2(N)
        with torch_device_fn.device(inp.device):
            for row_start in range(0, M, GCU_GRID_X_SAFE):
                rows = min(GCU_GRID_X_SAFE, M - row_start)
                nanmedian_select_kernel[(rows,)](
                    inp[row_start : row_start + rows],
                    out_values[row_start : row_start + rows],
                    out_indices[row_start : row_start + rows],
                    rows,
                    N,
                    block_n,
                    False,
                )
        return True

    if inp.dtype not in RADIX_SELECT_DTYPES:
        return False

    if (
        N > GCU_MULTIK_N_MIN
        and M <= GCU_MULTIK_M_MAX
        and M * triton.cdiv(N, GCU_RADIX_BLOCK_N) <= GCU_GRID_X_LIMIT
    ):
        _gcu_radix_multik_impl(inp, M, N, out_values, out_indices)
        return True

    # One program per row: right for row-parallel dim inputs and for flat
    # input below the grid-parallel threshold; beyond grid capacity or for a
    # single-program row too long to be practical, defer to the fallback.
    if M > 1 or N <= GCU_MULTIK_MAX_ELEMS:
        block_n = min(triton.next_power_of_2(N), GCU_RADIX_BLOCK_N)
        with torch_device_fn.device(inp.device):
            for row_start in range(0, M, GCU_GRID_X_SAFE):
                rows = min(GCU_GRID_X_SAFE, M - row_start)
                gcu_nanmedian_row_radix_kernel[(rows,)](
                    inp[row_start : row_start + rows],
                    out_values[row_start : row_start + rows],
                    out_indices[row_start : row_start + rows],
                    rows,
                    N,
                    block_n,
                    GCU_RADIX_BITS,
                    False,
                    num_warps=4,
                    num_stages=1,
                )
        return True

    return False


def _nanmedian_dim_impl(inp, dim, keepdim, out=None):
    dim = _normalize_dim(dim, inp.ndim)

    if inp.ndim == 0:
        if out is None:
            values = inp.clone()
            indices = torch.zeros((), dtype=torch.long, device=inp.device)
        else:
            values, indices = out
            values.copy_(inp)
            indices.zero_()
        return NanMedian(values=values, indices=indices)

    shape = list(inp.shape)
    N = shape[dim]
    out_shape = shape[:dim] + shape[dim + 1 :]
    M = math.prod(out_shape)

    keepdim_shape = shape.copy()
    keepdim_shape[dim] = 1
    output_shape = keepdim_shape if keepdim else out_shape
    compute_shape = output_shape if out is not None else keepdim_shape

    if N == 0 and M != 0:
        raise IndexError(
            f"median(): Expected reduction dim {dim} to have non-zero size."
        )

    if M == 0:
        if out is None:
            strides = c_contiguous_stride(compute_shape)
            values = torch.empty_strided(
                compute_shape, strides, dtype=inp.dtype, device=inp.device
            )
            indices = torch.empty_strided(
                compute_shape, strides, dtype=torch.long, device=inp.device
            )
            if not keepdim:
                values = torch.squeeze(values, dim)
                indices = torch.squeeze(indices, dim)
        else:
            values, indices = out
        return NanMedian(values=values, indices=indices)

    if out is None:
        values = torch.empty(compute_shape, dtype=inp.dtype, device=inp.device)
        indices = torch.empty(compute_shape, dtype=torch.long, device=inp.device)
    else:
        values, indices = out

    inp = dim_compress(inp, dim)
    values_contiguous = values.is_contiguous()
    indices_contiguous = indices.is_contiguous()
    flat_values = values.reshape(M) if values_contiguous else values.new_empty((M,))
    flat_indices = indices.reshape(M) if indices_contiguous else indices.new_empty((M,))

    if _gcu_dim_compute(inp, M, N, flat_values, flat_indices):
        if not values_contiguous:
            values.copy_(flat_values.reshape(values.shape))
        if not indices_contiguous:
            indices.copy_(flat_indices.reshape(indices.shape))
    else:
        result = _nanmedian_kthvalue_fallback(inp, M, N)
        computed_values = result.values.reshape(compute_shape)
        computed_indices = result.indices.reshape(compute_shape)
        if out is None:
            values = computed_values
            indices = computed_indices
        else:
            values.copy_(computed_values)
            indices.copy_(computed_indices)

    if out is None and not keepdim:
        values = torch.squeeze(values, dim)
        indices = torch.squeeze(indices, dim)

    return NanMedian(values=values, indices=indices)


def _nanmedian_flat_impl(inp, out=None):
    n = inp.numel()
    if n == 0:
        result = _empty_flat_value(inp)
        if out is not None:
            out.copy_(result)
            return out
        return result

    flat = inp.reshape(-1)
    if out is None:
        out = torch.empty((), dtype=flat.dtype, device=flat.device)
    # Flat nanmedian reports no indices; the kernels still need the pointer.
    indices = torch.empty((1,), dtype=torch.long, device=flat.device)
    if _gcu_dim_compute(flat, 1, n, out, indices):
        return out
    _nanmedian_dim_impl(flat, 0, False, out=(out, indices))
    return out


def nanmedian(inp):
    logger.debug("GEMS_ENFLAME NANMEDIAN")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp)


def nanmedian_out(inp, *, out):
    logger.debug("GEMS_ENFLAME NANMEDIAN_OUT")
    _check_supported_dtype(inp)
    return _nanmedian_flat_impl(inp, out=out)


def nanmedian_dim(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_ENFLAME NANMEDIAN_DIM")
    _check_supported_dtype(inp)
    return _nanmedian_dim_impl(inp, dim, keepdim)


def nanmedian_dim_values(inp, dim=-1, keepdim=False, *, values, indices):
    logger.debug("GEMS_ENFLAME NANMEDIAN_DIM_VALUES")
    return _nanmedian_dim_impl(inp, dim, keepdim, out=(values, indices))


def _aten_nanmedian(self):
    return nanmedian(self)


def _aten_nanmedian_out(self, *, out):
    return nanmedian_out(self, out=out)


def _aten_nanmedian_dim(self, dim=-1, keepdim=False):
    result = nanmedian_dim(self, dim=dim, keepdim=keepdim)
    return (result.values, result.indices)


def _aten_nanmedian_dim_values(self, dim=-1, keepdim=False, *, values, indices):
    # The torch.nanmedian(..., out=(v, i)) Python binding unpacks the kernel's
    # return value as a (values, indices) sequence, so hand back the tuple.
    return nanmedian_dim_values(
        self, dim=dim, keepdim=keepdim, values=values, indices=indices
    )


# Module-level handle keeps the Library (and its kernels) alive for the
# process; the registration itself must never break the flag_gems import.


@libentry()
@triton.jit
def gcu_radix_partial_count_kernel(
    inp,
    partial,
    state,
    N,
    NUM_CHUNKS,
    TOTAL,
    GRID_X,
    BLOCK_N: tl.constexpr,
    DIGIT_POS,
    RADIX_SIZE: tl.constexpr,
    USE_ISNAN: tl.constexpr,
    STATE_64: tl.constexpr,
):
    pid = tle.program_id(1) * GRID_X + tle.program_id(0)
    if pid < TOTAL:
        row = pid // NUM_CHUNKS
        chunk = pid % NUM_CHUNKS
        offsets = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        vals = tl.load(inp + row * N + offsets, mask=mask, other=0.0)
        dtype = inp.dtype.element_ty
        nbits: tl.constexpr = dtype.primitive_bitwidth
        utype = tl.dtype(f"uint{nbits}")
        radix_mask: tl.constexpr = RADIX_SIZE - 1
        radix_mask_val = tl.full((), radix_mask, dtype=utype)

        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask

        desired = tl.load(state + row * 3 + 0).to(utype)
        desired_mask = tl.load(state + row * 3 + 1).to(utype)
        keys = _to_order_key(vals, valid)
        active = valid & ((keys & desired_mask) == desired)
        digit = ((keys >> DIGIT_POS) & radix_mask_val).to(tl.int32)
        bins = tl.arange(0, RADIX_SIZE)
        counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
        for radix_bin in tl.static_range(0, RADIX_SIZE):
            bin_count = tl.sum((active & (digit == radix_bin)).to(tl.int32), axis=0)
            counts += tl.where(bins == radix_bin, bin_count, 0)
        tl.store(partial + pid * RADIX_SIZE + bins, counts)


@libentry()
@triton.jit
def gcu_radix_row_update_kernel(
    partial,
    state,
    valid_counts,
    NUM_CHUNKS,
    DIGIT_POS,
    RADIX_SIZE: tl.constexpr,
    RADIX_BITS_: tl.constexpr,
    NBITS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    IS_FIRST_PASS,
    STATE_64: tl.constexpr,
):
    row = tle.program_id(0)
    bins = tl.arange(0, RADIX_SIZE)
    counts = tl.zeros((RADIX_SIZE,), dtype=tl.int32)
    # GCU400: a [BLOCK_C, RADIX_SIZE] 2D tile load + tl.sum(axis=0) lowers to
    # a silently-zero reduction in this kernel (2D accumulate-tile reduction
    # hazard family; the trigger boundary does not extrapolate across
    # kernels). Reduce per bin with 1D loads/sums -- the same proven form the
    # count kernel uses.
    for start in tl.range(0, NUM_CHUNKS, BLOCK_C):
        chunk_rows = start + tl.arange(0, BLOCK_C)
        cmask = chunk_rows < NUM_CHUNKS
        base = (row * NUM_CHUNKS + chunk_rows) * RADIX_SIZE
        for radix_bin in tl.static_range(0, RADIX_SIZE):
            lane_counts = tl.load(partial + base + radix_bin, mask=cmask, other=0)
            counts += tl.where(bins == radix_bin, tl.sum(lane_counts, axis=0), 0)

    if IS_FIRST_PASS:
        valid_count = tl.sum(counts, axis=0)
        tl.store(valid_counts + row, valid_count)
        # Keep k in a register: re-loading the slot this kernel just stored
        # is not reliably visible on this backend.
        k = tl.where(valid_count > 0, (valid_count + 1) // 2, 0)
    else:
        # Narrow back to int32 so both branches of the runtime IS_FIRST_PASS
        # test carry the same type (64-bit state slots return int64).
        k = tl.load(state + row * 3 + 2).to(tl.int32)
    cumsum = tl.cumsum(counts, axis=0)
    prev = cumsum - counts
    take = (cumsum >= k) & (prev < k)
    selected_bin = tl.min(tl.where(take, bins, RADIX_SIZE - 1), axis=0)
    counts_before = tl.max(tl.where(take, prev, 0), axis=0)

    utype = tl.dtype(f"uint{NBITS}")
    radix_mask: tl.constexpr = (1 << RADIX_BITS_) - 1
    radix_mask_val = tl.full((), radix_mask, dtype=utype)
    desired = tl.load(state + row * 3 + 0).to(utype)
    desired_mask = tl.load(state + row * 3 + 1).to(utype)
    # Runtime DIGIT_POS promotes the shifted value past utype; truncating
    # back keeps the stored key/mask types stable.
    desired = desired | ((selected_bin.to(utype) << DIGIT_POS).to(utype))
    desired_mask = desired_mask | ((radix_mask_val << DIGIT_POS).to(utype))
    # int64 state slots keep 64-bit keys intact; the uint64 -> int64 cast is a
    # modulo-2**64 reinterpretation, so the key bits are preserved.
    if STATE_64:
        tl.store(state + row * 3 + 0, desired.to(tl.int64))
        tl.store(state + row * 3 + 1, desired_mask.to(tl.int64))
    else:
        tl.store(state + row * 3 + 0, desired.to(tl.int32))
        tl.store(state + row * 3 + 1, desired_mask.to(tl.int32))
    tl.store(state + row * 3 + 2, (k - counts_before).to(tl.int32))


@libentry()
@triton.jit
def gcu_radix_find_index_kernel(
    inp,
    state,
    cand,
    N,
    NUM_CHUNKS,
    TOTAL,
    GRID_X,
    BLOCK_N: tl.constexpr,
    USE_ISNAN: tl.constexpr,
):
    pid = tle.program_id(1) * GRID_X + tle.program_id(0)
    if pid < TOTAL:
        row = pid // NUM_CHUNKS
        chunk = pid % NUM_CHUNKS
        offsets = chunk * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        vals = tl.load(inp + row * N + offsets, mask=mask, other=0.0)
        dtype = inp.dtype.element_ty
        if dtype.is_floating():
            valid = mask & _is_not_nan(vals, USE_ISNAN)
        else:
            valid = mask
        nbits: tl.constexpr = dtype.primitive_bitwidth
        utype = tl.dtype(f"uint{nbits}")
        desired = tl.load(state + row * 3 + 0).to(utype)
        keys = _to_order_key(vals, valid)
        local_idx = tl.min(tl.where(valid & (keys == desired), offsets, N), axis=0)
        tl.store(cand + pid, local_idx.to(tl.int32))


@libentry()
@triton.jit
def gcu_radix_finalize_kernel(
    inp,
    state,
    valid_counts,
    cand,
    out_values,
    out_indices,
    N,
    NUM_CHUNKS,
    BLOCK_C: tl.constexpr,
):
    row = tle.program_id(0)
    valid_count = tl.load(valid_counts + row)
    result_idx = tl.full((), N, dtype=tl.int32)
    if valid_count > 0:
        for start in tl.range(0, NUM_CHUNKS, BLOCK_C):
            chunk_rows = start + tl.arange(0, BLOCK_C)
            cmask = chunk_rows < NUM_CHUNKS
            tile = tl.load(cand + row * NUM_CHUNKS + chunk_rows, mask=cmask, other=N)
            result_idx = tl.minimum(result_idx, tl.min(tile, axis=0))

    dtype = inp.dtype.element_ty
    if dtype.is_floating():
        result_val = tl.load(
            inp + row * N + result_idx, mask=valid_count > 0, other=float("nan")
        )
        result_val = tl.where(valid_count > 0, result_val, float("nan"))
        result_idx = tl.where(valid_count > 0, result_idx, 0)
    else:
        result_val = tl.load(
            inp + row * N + result_idx, mask=valid_count > 0, other=get_dtype_min(dtype)
        )

    tl.store(out_values + row, result_val)
    tl.store(out_indices + row, result_idx.to(tl.int64))

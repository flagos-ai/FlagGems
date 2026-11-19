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
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.sort import convert_to_uint_preverse_order
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as tle

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

# Largest BLOCK_M the mode scan can be tuned to.  The int8/uint8 path stores
# its values without a mask, so the caller must pad the value buffer to a
# multiple of this.
_MODE_VALUE_MAX_BLOCK_M = max(
    conf.kwargs["BLOCK_M"] for conf in runtime.get_tuned_config("naive_reduction")
)
# gcu400: for int64 the run scan miscompiles once BLOCK_M reaches 128 (wide
# int64 value select / predicate layout).  The counters still advance, but
# best_value/best_index stop updating and mode collapses to the sorted
# minimum.  BLOCK_M 16 and 64 are verified correct, so restrict the int64
# kernel's tuning space to <= 64.
_MODE_I64_MAX_BLOCK_M = 64


@triton.jit
def _mode_sort_histogram(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    # arr_ptr: (m, n)
    # out_ptr: (m, n_passes, r), where r = 2 ** k_bits is the number of bins
    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1  # a.k.a. 2 ** k_bits - 1
    # Runtime value (tiles_n_per_cta is a kernel argument), so not constexpr.
    CTA_TILE_N = TILE_N * tiles_n_per_cta

    # gcu400: the launch is capped at MAX_GRID_DIM, so a grid-stride loop
    # walks the (pid_n, pid_m) tiles instead of one program per tile.
    n_tiles = m * tl.cdiv(n, CTA_TILE_N)
    for pid in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        pid_n = pid // m
        pid_m = pid % m
        cta_n_start = CTA_TILE_N * pid_n
        cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

        for p in range(0, num_passes):  # parallel
            bit_offset = p * num_bits_per_pass
            for r_start in range(0, r, TILE_R):  # parallel
                bin_indices = r_start + tl.arange(0, TILE_R)
                acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
                for n_start in range(cta_n_start, cta_n_end, TILE_N):  # sequantial
                    n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                    mask = n_offsets < cta_n_end
                    arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask, other=0)
                    arr = convert_to_uint_preverse_order(arr, descending)
                    key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                    matches = (n_offsets[None, :] < n) & (
                        bin_indices[:, None] == key[None, :]
                    )  # (TILE_R, TILE_N)
                    acc += matches
                local_sum = tl.sum(acc, axis=1)
                tl.atomic_add(
                    out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                    local_sum,
                    sem="relaxed",
                )


@triton.jit
def _mode_sort_sweep(
    arr_ptr,
    associate_arr_ptr,  # inputs: (key & value)
    out_ptr,
    associate_out_ptr,  # outputs: (key & value)
    excumsum_bins_ptr,
    status_ptr,  # aux input and status
    n_passes,
    pass_id,
    bit_offset,
    m,
    N,
    OUT_N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
):
    # r: num_bins = 2 ** k_bits
    # OUT_N: grid_n = cdiv(N, )

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # excumsum_bins_ptr: (m, n_passes, r)
    # flag_ptr: (m, r, OUT_N)

    # grid: (m, grid_r, grid_n)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2**k_bits
    pid_r = tl.program_id(1)
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # gcu400: the launch is capped at MAX_GRID_DIM, so walk the (pid_n, pid_m)
    # tiles with a grid-stride loop.  Tiles are still started in increasing
    # pid order, so the decoupled lookback below only ever waits on earlier
    # tiles.
    n_tiles = m * OUT_N
    for pid in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        pid_m = pid % m
        pid_n = pid // m

        # cumsum for a bin_index
        n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)  # (TILE_N, )
        mask = n_offsets < N
        arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask, other=0)
        arr_u = convert_to_uint_preverse_order(arr, descending)
        key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )
        if associate_arr_ptr is not None:
            associate_arr = tl.load(
                associate_arr_ptr + pid_m * N + n_offsets, mask=mask, other=0
            )
        # since triton can only use scalar as condition, loop by bin_index
        # status must be pre zero-initialized, or else we have to initialize it
        for bin_index in range(cta_r_start, cta_r_end):
            matches = (n_offsets < N) & (key == bin_index)  # (TILE_N, ) bool
            # cta level cumsum per bin
            # CAUTION: tl.sum in triton 3.2 does not promote type
            local_sum = tl.sum(matches.to(tl.uint32), axis=0)
            pack0 = aggregate_mask | local_sum
            status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
            tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

            # decoupled lookback
            exclusive_prefix = tl.zeros((), dtype=tl.uint32)
            i_lookback = pid_n - 1
            while i_lookback >= 0:
                flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)  # uin32
                while pack1 == 0:
                    pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
                exclusive_prefix += pack1 & v_mask
                if (pack1 & aggregate_mask) == aggregate_mask:
                    i_lookback -= 1
                else:
                    i_lookback = -1
            pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
            tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

            local_ex_cumsum = (
                tl.cumsum(matches.to(tl.uint32), axis=0) - matches
            )  # (TILE_N, )
            ex_cumsum_in_bin = (
                exclusive_prefix + local_ex_cumsum
            )  # global ex_cumsum_in_bin (TILE_N, )

            # ex_cumsum_bins (m, n_passes, r)
            ex_cumsum_bins = tl.load(
                excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
            )  # scalar
            pos = ex_cumsum_bins + ex_cumsum_in_bin  # (TILE_N, )

            # scatter
            tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
            if associate_arr_ptr is not None:
                tl.store(
                    associate_out_ptr + pid_m * N + pos,
                    associate_arr,
                    mask=matches,
                )


def _mode_radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    # gcu400: keep the launch at MAX_GRID_DIM programs; the histogram kernel
    # grid-strides over the remaining tiles.
    grid_for_global_hist = (min(m * grid_n, MAX_GRID_DIM), 1, 1)

    with torch_device_fn.device(arr.device):
        global_hist = torch.zeros(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )

        _mode_sort_histogram[grid_for_global_hist](
            arr,
            global_hist,
            n_passes,
            m,
            n,
            tiles_n_per_cta,
            TILE_N,
            TILE_R,
            k_bits,
            descending,
        )
        ex_cumsum_bins = torch.cumsum(global_hist, -1) - global_hist
        ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

        # sort
        arr_in = torch.clone(arr)
        indices_in = (
            torch.arange(0, n, dtype=torch.int64, device=arr_in.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        TILE_R = 8
        grid_r = triton.cdiv(num_bins, TILE_R)
        TILE_N = 2048
        grid_n = triton.cdiv(n, TILE_N)
        # Same MAX_GRID_DIM cap as the histogram kernel.
        grid_for_sweep = (min(m * grid_n, MAX_GRID_DIM), grid_r)

        status = torch.empty(
            (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
        )

        for i in range(0, n_passes):
            bit_offset = i * k_bits
            status.zero_()
            _mode_sort_sweep[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                ex_cumsum_bins,
                status,
                n_passes,
                i,
                bit_offset,
                m,
                n,
                grid_n,
                TILE_N,
                TILE_R,
                k_bits,
                descending,
            )
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


def _mode_sort(inp, dim=-1, descending=False):
    # Keep mode's radix sorting fixes independent of the public sort operator.
    if inp.shape[dim] == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)
    dim %= inp.ndim
    x = inp.movedim(dim, -1).contiguous()
    bits = 1 if inp.dtype == torch.bool else 4
    values, indices = _mode_radix_sort(x, bits, descending)
    return values.movedim(-1, dim), indices.movedim(-1, dim)


@triton.jit
def _mode_scan_body(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
):
    # gcu400 note: 8/16/32-bit tiles are lowered as 512-lane vectors and
    # 64-bit tiles as 256-lane vectors; their predicates follow the same
    # split.  Selecting a narrow value with a mask produced by an int64
    # compare makes llc pad the 256-bit predicate to 512 bits with
    #   concat_vectors(v64i1 x4, undef:v64i1 x4)
    # which is not selectable ("LLVM ERROR: Cannot select: v512i1 =
    # concat_vectors ... undef:v64i1").  Keep the run counters in the value's
    # own predicate class so the ``better`` mask never crosses classes.
    # Byte tiles are additionally scanned as int32 and stored without a mask;
    # the caller pads out_value/out_index by up to BLOCK_M.
    ELEM_TY: tl.constexpr = sorted_inp.dtype.element_ty
    IS_BYTE: tl.constexpr = ELEM_TY == tl.int8 or ELEM_TY == tl.uint8
    if ELEM_TY == tl.int64 or ELEM_TY == tl.uint64 or ELEM_TY == tl.float64:
        COUNT_TY: tl.constexpr = tl.int64
    else:
        COUNT_TY: tl.constexpr = tl.int32

    pid_m = tle.program_id(0)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M

    # Load first element
    n0 = tl.arange(0, 1)
    offset0 = m_offset[:, None] * N + n0[None, :]
    mask0 = mask_m[:, None] & (n0[None, :] < N)
    val0 = tl.load(sorted_inp + offset0, mask=mask0)
    idx0 = tl.load(sorted_indices + offset0, mask=mask0)

    # Squeeze the n-dimension (size 1)
    cur_value = val0.reshape([BLOCK_M])
    if IS_BYTE:
        cur_value = cur_value.to(tl.int32)
    cur_index = idx0.reshape([BLOCK_M])
    cur_count = tl.full([BLOCK_M], 1, dtype=COUNT_TY)
    best_value = cur_value
    best_index = cur_index
    best_count = tl.full([BLOCK_M], 1, dtype=COUNT_TY)

    # Scan remaining elements one by one
    for i in range(1, N):
        ni = tl.full([1], i, dtype=tl.int32)
        offset_i = m_offset[:, None] * N + ni[None, :]
        mask_i = mask_m[:, None] & (ni[None, :] < N)
        val_i = tl.load(sorted_inp + offset_i, mask=mask_i).reshape([BLOCK_M])
        if IS_BYTE:
            val_i = val_i.to(tl.int32)
        idx_i = tl.load(sorted_indices + offset_i, mask=mask_i).reshape([BLOCK_M])

        # gcu400: for int64 detect the run with two signed compares instead
        # of eq (equivalent: both are false only when the values are equal).
        # The int64 kernel is additionally capped at BLOCK_M <= 64, see
        # _MODE_I64_MAX_BLOCK_M.
        if ELEM_TY == tl.int64:
            same = ~((val_i - cur_value < 0) | (cur_value - val_i < 0))
        else:
            same = val_i == cur_value
        cur_count = tl.where(same, cur_count + 1, tl.full([BLOCK_M], 1, dtype=COUNT_TY))
        cur_value = tl.where(same, cur_value, val_i)
        cur_index = idx_i  # Always track the latest index in the run

        better = cur_count > best_count
        best_count = tl.where(better, cur_count, best_count)
        best_value = tl.where(better, cur_value, best_value)
        best_index = tl.where(better, cur_index, best_index)

    if IS_BYTE:
        # Keep only the low 8 bits; an i64->i8 truncate is illegal here too.
        narrowed = (best_value & 0xFF).to(tl.uint8)
        if ELEM_TY == tl.int8:
            narrowed = narrowed.to(tl.int8)
        tl.store(out_value + m_offset, narrowed)
    else:
        tl.store(out_value + m_offset, best_value, mask=mask_m)
    tl.store(out_index + m_offset, best_index, mask=mask_m)


def _int64_configs():
    return [
        conf
        for conf in runtime.get_tuned_config("naive_reduction")
        if conf.kwargs["BLOCK_M"] <= _MODE_I64_MAX_BLOCK_M
    ]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("naive_reduction"),
    key=["M", "N"],
)
@triton.jit
def mode_kernel(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _mode_scan_body(sorted_inp, sorted_indices, out_value, out_index, M, N, BLOCK_M)


@libentry()
@libtuner(
    configs=_int64_configs(),
    key=["M", "N"],
)
@triton.jit
def mode_kernel_i64(
    sorted_inp,
    sorted_indices,
    out_value,
    out_index,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    _mode_scan_body(sorted_inp, sorted_indices, out_value, out_index, M, N, BLOCK_M)


def mode(inp, dim=-1, keepdim=False):
    logger.debug("GEMS_ENFLAME_GCU400 MODE")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"

    shape = list(inp.shape)
    dim = dim % inp.ndim
    N = shape[dim]
    M = inp.numel() // N

    sorted_inp, sorted_indices = _mode_sort(inp, dim=dim)

    # Move dim to last for 2D processing
    sorted_inp = torch.movedim(sorted_inp, dim, -1).contiguous()
    sorted_indices = torch.movedim(sorted_indices, dim, -1).contiguous()

    sorted_flat = sorted_inp.reshape(M, N)
    indices_flat = sorted_indices.reshape(M, N)

    # The int8/uint8 scan stores its values without a mask (see mode_kernel),
    # so the value buffer has to cover the padded tail of the last block.
    pad = 0
    if inp.dtype in (torch.int8, torch.uint8):
        pad = _MODE_VALUE_MAX_BLOCK_M - 1
    out_value = torch.empty(M + pad, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(M + pad, dtype=torch.int64, device=inp.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    kernel = mode_kernel_i64 if inp.dtype == torch.int64 else mode_kernel
    with torch_device_fn.device(inp.device):
        kernel[grid](sorted_flat, indices_flat, out_value, out_index, M, N)

    if pad:
        out_value = out_value[:M]
        out_index = out_index[:M]

    out_shape = list(shape)
    out_shape[dim] = 1
    out_value = out_value.reshape(out_shape)
    out_index = out_index.reshape(out_shape)

    if not keepdim:
        out_value = out_value.squeeze(dim)
        out_index = out_index.squeeze(dim)

    Mode_out = namedtuple("mode", ["values", "indices"])
    return Mode_out(values=out_value, indices=out_index)

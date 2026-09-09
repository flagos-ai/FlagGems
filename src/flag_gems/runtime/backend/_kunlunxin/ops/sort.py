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
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.tensor_wrapper import StridedBuffer

from .cat import copy_func
from .cumsum import cumsum
from .topk import _get_finfo_val, _get_iinfo_val, argsort

logger = logging.getLogger(__name__)


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    if x.dtype.is_floating():
        if x.dtype == tl.bfloat16:
            x = x.to(tl.float32)
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


@triton.jit
def compute_global_hist_kernel(
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
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1  # a.k.a. 2 ** k_bits - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):  # parallel
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):  # parallel
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int32)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):  # sequantial
                n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                matches = tl.where(
                    mask, (bin_indices[:, None] == key), False
                )  # (TILE_R, TILE_N)
                acc += matches
            local_sum = tl.sum(acc, axis=1)
            tl.atomic_add(
                out_ptr + pid_m * num_passes * r + p * r + bin_indices,
                local_sum,
                sem="relaxed",
            )


@triton.jit
def sweep(
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

    # load data
    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # cumsum for a bin_index
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)  # (TILE_N, )
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )

    # since triton can only use scalar as condition, loop by bin_index
    # status must be pre zero-initialized, or else we have to initialize it
    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)  # (TILE_N, ) bool
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
            associate_arr = tl.load(
                associate_arr_ptr + pid_m * N + n_offsets, mask=mask
            )
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)


@triton.jit
def count_kernel(
    x_ptr,
    counts_ptr,  # Output: [M, R_PAD] int32, bin-major: bin * GRID_N + block
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
    GRID_N: tl.constexpr,
    R_PAD: tl.constexpr,
):
    # NOTE(kunlunxin): the histogram is written *bin-major* (all GRID_N block
    # counters of bin 0, then bin 1, ...) with a per-row pitch of R_PAD.  In
    # that layout the exclusive scan over the whole row is exactly
    #     global_offsets[b, i] = sum_{j<i} total[j] + sum_{b'<b} counts[b', i]
    # i.e. the value `scatter_kernel` needs, so a single contiguous 1D scan
    # (bin_prefix_kernel) replaces the previous host-side chain of
    # sum_dim + 2x cumsum + broadcast/clone + add.
    # GRID_N / R_PAD are constexpr on purpose: they remove the runtime cdiv and
    # the runtime div/mod on `pid`, and adding them as *runtime* i32 scalars is
    # a known 15-30x launch-cost cliff on this backend.
    pid = tl.program_id(0)

    row_idx = pid // GRID_N
    block_idx = pid % GRID_N

    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N

    # Masked loads on XPU can still issue the tail address.  Clamp inactive
    # lanes to the last real element and use the mask only for the histogram
    # predicate.
    n_offset_safe = tl.minimum(n_offset, N - 1)
    val = tl.load(x_ptr + row_start + n_offset_safe)
    val_u = convert_to_uint_preverse_order(val, descending)

    bfe_mask = num_bins - 1
    key = (val_u >> bit_offset) & bfe_mask

    counts_row = counts_ptr + row_idx * R_PAD + block_idx
    for i in range(num_bins):
        bin_mask = (key == i) & mask
        count = tl.sum(bin_mask.to(tl.int32))
        tl.store(counts_row + i * GRID_N, count)


@libentry()
@triton.jit
def bin_prefix_kernel(
    counts_ptr,  # [M, R_PAD] int32 (bin-major histogram)
    offsets_ptr,  # [M, R_PAD] int32 (exclusive prefix sums)
    R: tl.constexpr,  # num_bins * GRID_N valid entries per row
    R_PAD: tl.constexpr,  # padded row pitch, multiple of TILE
    TILE: tl.constexpr,
):
    """One program per row: exclusive scan of the bin-major histogram.

    Loads and stores are *unmasked* on purpose -- the caller allocates
    M * R_PAD elements and R_PAD is a multiple of TILE, so every lane of every
    tile stays inside this row's own padded slice.  That side-steps both the
    masked-store granule behaviour and `other=` contamination on this backend;
    the padding lanes are neutralised with an explicit `tl.where` instead.
    The carry follows the shape of cumsum_chunk_kernel in cumsum.py (vector
    carry + tl.sum), which is the proven in-loop 1D scan pattern here.
    """
    row = tl.program_id(0)
    base = row * R_PAD
    carry = tl.zeros([TILE], tl.int32)
    for start in range(0, R_PAD, TILE):
        offs = start + tl.arange(0, TILE)
        v = tl.load(counts_ptr + base + offs)
        v = tl.where(offs < R, v, 0)
        inclusive = tl.cumsum(v, axis=0)
        tl.store(offsets_ptr + base + offs, inclusive - v + carry)
        carry += tl.sum(v, axis=0)


@triton.jit
def scatter_kernel(
    x_ptr,
    x_out_ptr,
    idx_in_ptr,
    idx_out_ptr,
    global_offsets_ptr,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
    GRID_N: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // GRID_N
    block_idx = pid % GRID_N

    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N

    n_offset_safe = tl.minimum(n_offset, N - 1)
    val = tl.load(x_ptr + row_start + n_offset_safe)
    val_u = convert_to_uint_preverse_order(val, descending)

    idx = tl.load(idx_in_ptr + row_start + n_offset_safe)

    bfe_mask = num_bins - 1
    key = (val_u >> bit_offset) & bfe_mask

    # NOTE(kunlunxin): store masks are NOT honoured for data-dependent (scatter)
    # addresses on this backend -- every lane of the tile performs its write.
    # The previous form
    #     local_rank = tl.cumsum(bin_mask.to(tl.int32), axis=0) - 1
    #     tl.store(x_out_ptr + row_start + global_start + local_rank, val,
    #              mask=bin_mask)
    # therefore (a) wrote *out of bounds* at `global_start - 1` from every lane
    # that precedes the first match of a bin (local_rank == -1; quantified with
    # canary buffers in harness/probe/unique2_masked_scatter_probe.py -> one OOB
    # element per tile, reported by the driver as "axi wresp error" / status 700)
    # and (b) let every inactive lane after a match re-write the slot of the
    # previous match with its own value, silently corrupting the sorted output.
    # Fix: keep a per-lane destination, default it to a lane-unique scratch slot
    # in front of the output buffer (the caller over-allocates BLOCK_N elements
    # and passes a suffix view, so negative offsets in [-BLOCK_N, -1] are legal
    # and unique per lane), select the real destination with tl.where, and issue
    # a single *unmasked* store per lane.  Also avoid `bool.to(int32)`, which
    # trips triton_xpu.convert_layout at large BLOCK.
    lane = tl.arange(0, BLOCK_N)
    dest_idx = (lane - BLOCK_N).to(tl.int64)

    offsets_row = global_offsets_ptr + row_idx * R_PAD + block_idx
    for i in range(num_bins):
        bin_mask = (key == i) & mask
        local_rank = tl.cumsum(tl.where(bin_mask, 1, 0), axis=0) - 1

        global_start = tl.load(offsets_row + i * GRID_N)

        dest_idx = tl.where(
            bin_mask,
            (row_start + global_start + local_rank).to(tl.int64),
            dest_idx,
        )

    tl.store(x_out_ptr + dest_idx, val)
    tl.store(idx_out_ptr + dest_idx, idx)


@libentry()
@triton.jit
def init_indices_kernel(indices, total, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(indices + offsets, offsets % N, mask=offsets < total)


@libentry()
@triton.jit
def init_sort_buffers_kernel(
    source, values, indices, total, N, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    tl.store(values + offsets, tl.load(source + offsets, mask=mask), mask=mask)
    tl.store(indices + offsets, offsets % N, mask=mask)


def radix_sort_low_mem(arr, k_bits=4, descending=False):
    original_shape = arr.shape
    N = arr.shape[-1]
    arr = arr.reshape(-1, N)
    M = arr.shape[0]

    # NOTE(kunlunxin): BLOCK_N is the per-program tile of the count/scatter
    # passes.  512 leaves the discrete-scatter pass launch-bound: measured on
    # XPU 2 for torch.sort of 16,777,216 int32 elements, 512 -> 1876 ms,
    # 1024 -> 1247 ms, 2048 -> 764 ms, 4096 -> 590 ms.  8192 is *not* usable --
    # it is the same speed as 4096 but silently mis-sorts (values_ok=False), so
    # 4096 is the largest qualified tile.  Qualified with
    # harness/probe/unique2_sort_validate.py (290/290: 8 dtypes x 17 shapes x
    # asc/desc + constant inputs) and unique2_sort_sweep.py (exact and index
    # permutation clean at N = 1M .. 167.8M).
    #
    # NOTE(kunlunxin, sort_stable 2026-08-30): a *fixed* 4096 is only right for
    # long rows.  count/scatter cost tracks the padded row length
    # ceil(N / BLOCK_N) * BLOCK_N (every masked-off lane still pays the 16-bin
    # loop), so a 4096 tile makes a 64-wide row 64x more expensive than it needs
    # to be.  Measured on XPU 5 (fp32, median of 7, candidate chain):
    #   N=64    B=64 2.72 ms | 128 7.72 | 256 8.09 | 512 8.95 | 4096 11.94
    #   N=256   B=256 11.33  | 128 16.16 | 64 26.08 | 512 30.40 | 4096 41.71
    #   N=512   B=512 55.35  | 256 77.33 | 1024 140.67 | 4096 158.87
    #   N=1024  B=1024 67.87 | 512 93.31 | 2048 143.89 | 4096 161.03
    #   N=4096  B=4096 441.8 | 2048 611.8 | 1024 929.6 | 512 1364.2
    #   N=131072/262144: 4096 is the best of {512,1024,2048,4096}
    # i.e. the optimum is next_pow2(N) clamped to [64, 4096] on every shape
    # measured.  64 is the floor because a <= 32-lane tile is mis-lowered here,
    # 4096 the ceiling because 8192 silently mis-sorts (see above).
    _env_block_n = os.environ.get("GEMS_XPU_RADIX_BLOCK_N")
    if _env_block_n:
        BLOCK_N = int(_env_block_n)
    else:
        BLOCK_N = min(4096, max(64, triton.next_power_of_2(N)))
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (M * grid_n,)

    # NOTE(kunlunxin): scatter_kernel parks every inactive lane of a tile on a
    # lane-unique scratch slot at `dest - BLOCK_N` (store masks are ignored for
    # scatter addresses on this backend, see the comment there).  Allocate the
    # ping-pong buffers with a BLOCK_N-element head pad plus a 256-element tail
    # pad (masked *affine* stores on this backend touch a full 64-element
    # granule, and init_sort_buffers_kernel runs a 256-lane masked tile) and hand
    # the kernels a contiguous suffix view, so those writes stay inside our own
    # allocation instead of clobbering the neighbouring one.
    _HEAD_PAD = BLOCK_N
    _TAIL_PAD = 256
    _keepalive = []

    def _padded(dtype):
        buf = torch.empty(_HEAD_PAD + M * N + _TAIL_PAD, device=arr.device, dtype=dtype)
        _keepalive.append(buf)
        return buf[_HEAD_PAD : _HEAD_PAD + M * N].view(M, N)

    arr_in = _padded(arr.dtype)
    arr_out = _padded(arr.dtype)
    idx_in = _padded(torch.int64)
    idx_out = _padded(torch.int64)

    index_block = 256
    with torch_device_fn.device(arr.device):
        init_sort_buffers_kernel[(triton.cdiv(M * N, index_block),)](
            arr, arr_in, idx_in, M * N, N, BLOCK_SIZE=index_block
        )

    dtype = arr.dtype
    num_bits = 1
    if dtype == torch.bool:
        pass
    elif dtype == torch.bfloat16:
        num_bits = 4 * 8
    else:
        num_bits = arr.element_size() * 8
    num_passes = (num_bits + k_bits - 1) // k_bits
    num_bins = 2**k_bits

    # NOTE(kunlunxin): the per-pass histogram lives in a bin-major [M, R_PAD]
    # buffer so that its exclusive prefix sum (bin_prefix_kernel, one program
    # per row) *is* the scatter destination table.  This replaces the previous
    # per-pass host chain sum_dim + cumsum + cumsum + broadcast_to().clone()
    # + add, which allocated three int64 tensors of M*grid_n*num_bins elements
    # per pass (vendor `cumsum` promotes int32 -> int64) and, when grid_n was
    # small, degenerated into scan_then_fan with a 1-wide scan tile.
    # R_PAD is a multiple of TILE so bin_prefix_kernel needs no masking at all.
    r = num_bins * grid_n
    tile_r = max(64, min(4096, triton.next_power_of_2(r)))
    r_pad = triton.cdiv(r, tile_r) * tile_r

    with torch_device_fn.device(arr.device):
        counts = torch.empty(M * r_pad, device=arr.device, dtype=torch.int32)
        global_offsets = torch.empty(M * r_pad, device=arr.device, dtype=torch.int32)

        for p in range(num_passes):
            bit_offset = p * k_bits
            count_kernel[grid](
                arr_in,
                counts,
                M,
                N,
                bit_offset,
                num_bins,
                BLOCK_N,
                descending,
                GRID_N=grid_n,
                R_PAD=r_pad,
            )

            bin_prefix_kernel[(M,)](
                counts,
                global_offsets,
                R=r,
                R_PAD=r_pad,
                TILE=tile_r,
            )

            scatter_kernel[grid](
                arr_in,
                arr_out,
                idx_in,
                idx_out,
                global_offsets,
                M,
                N,
                bit_offset,
                num_bins,
                BLOCK_N,
                descending,
                GRID_N=grid_n,
                R_PAD=r_pad,
            )

            arr_in, arr_out = arr_out, arr_in
            idx_in, idx_out = idx_out, idx_in

    return arr_in.reshape(original_shape), idx_in.reshape(original_shape)


def _copy_bitview(t):
    # NOTE(kunlunxin): the tuned copy codegen hits an illegal memory access for
    # int32 on the strided-read path at some shapes (deterministic repro:
    # permute-read of a (65536, 4) / (32768, 8) int32 tensor, i.e.
    # M * N = 256 Ki; smaller shapes are clean).  int32 and float32 share
    # itemsize, so bit-cast both sides and reuse the proven fp32 copy path;
    # the copy is bit-exact.  Same workaround as the cat family.
    if t.dtype == torch.int32:
        return t.view(torch.float32)
    return t


def _permute_copy_to_last(inp, dim):
    """Materialize a dim-last view without routing a strided copy through copy_."""
    order = [i for i in range(inp.ndim) if i != dim] + [dim]
    shape = tuple(inp.shape[i] for i in order)
    strides = tuple(inp.stride()[i] for i in order)
    out = torch.empty(shape, dtype=inp.dtype, device=inp.device)
    src = _copy_bitview(inp)
    dst = _copy_bitview(out)
    in_view = StridedBuffer(src, shape, strides)
    out_view = StridedBuffer(dst, shape, dst.stride())
    copy_func.instantiate(inp.ndim)(in_view, out0=out_view)
    return out, order


def _permute_copy_from_last(inp, out_shape, order):
    """Copy a dim-last result into the original contiguous dimension order."""
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    shape = tuple(out_shape[i] for i in order)
    strides = tuple(out.stride()[i] for i in order)
    src = _copy_bitview(inp)
    dst = _copy_bitview(out)
    in_view = StridedBuffer(src, shape, src.stride())
    out_view = StridedBuffer(dst, shape, strides)
    copy_func.instantiate(len(order))(in_view, out0=out_view)
    return out


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.element_size() * 8)

    TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_for_global_hist = (m * grid_n, 1, 1)

    with torch_device_fn.device(arr.device):
        global_hist = torch.empty(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )
        zero_(global_hist)
        compute_global_hist_kernel[grid_for_global_hist](
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
        ex_cumsum_bins = cumsum(global_hist, dim=-1) - global_hist
        ex_cumsum_bins = ex_cumsum_bins.to(torch.uint32)

        # sort
        arr_in = torch.empty_like(arr)
        indices_in = torch.empty(arr.shape, dtype=torch.int64, device=arr.device)
        init_block = 256
        init_sort_buffers_kernel[(triton.cdiv(arr.numel(), init_block),)](
            arr, arr_in, indices_in, arr.numel(), n, BLOCK_SIZE=init_block
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        TILE_R = 8
        grid_r = triton.cdiv(num_bins, TILE_R)
        TILE_N = 2048
        grid_n = triton.cdiv(n, TILE_N)
        grid_for_sweep = (m * grid_n, grid_r)

        status = torch.empty(
            (m, num_bins, grid_n), device=arr.device, dtype=torch.uint32
        )

        for i in range(0, n_passes):
            bit_offset = i * k_bits
            status.zero_()
            sweep[grid_for_sweep](
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
            # print(f"< sorted last {bit_offset + k_bits:>2d} bits: {arr_out}")
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
        in_val = tl.where(in_val.dtype.is_fp64(), in_val, in_val.to(tl.float32))
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val).to(tl.int32)
    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN SORT")
    if inp.ndim == 0:
        return inp.clone(), torch.zeros_like(inp, dtype=torch.int64)
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0 or inp.numel() == 0:
        return inp, torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        indices = torch.empty_like(inp, dtype=torch.int64)
        with torch_device_fn.device(inp.device):
            init_indices_kernel[(triton.cdiv(inp.numel(), 256),)](
                indices, inp.numel(), 1, BLOCK_SIZE=256
            )
        return inp, indices
    # NOTE(kunlunxin): the bitonic argsort path (sort_kernel) mis-sorts /
    # faults the device on XPU (unrolled compare-and-swap chain, ~hundreds of
    # where-ops over BLOCK_SIZE lanes → miscompile + device kernel exception),
    # so every non-trivial size goes through the stable radix chain here
    # (identical to sort_stable); reference semantics of torch.sort with
    # stable=True are preserved and radix is stable by construction.
    return sort_stable(inp, stable=True, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN SORT_STABLE")
    # We only implement stable radix sort here
    _ = stable
    if inp.ndim == 0:
        return inp.clone(), torch.zeros_like(inp, dtype=torch.int64)
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0 or inp.numel() == 0:
        return inp, torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        indices = torch.empty_like(inp, dtype=torch.int64)
        with torch_device_fn.device(inp.device):
            init_indices_kernel[(triton.cdiv(inp.numel(), 256),)](
                indices, inp.numel(), 1, BLOCK_SIZE=256
            )
        return inp, indices

    if dim < 0:
        dim = dim + inp.ndim
    original_shape = inp.shape
    if dim != inp.ndim - 1:
        inp, order = _permute_copy_to_last(inp, dim)
    else:
        order = list(range(inp.ndim))
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort_low_mem(inp, num_bits_per_pass, descending)

    if dim != len(order) - 1:
        out = _permute_copy_from_last(out, original_shape, order)
        out_index = _permute_copy_from_last(out_index, original_shape, order)
    return out, out_index

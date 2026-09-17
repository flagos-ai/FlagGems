import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.zeros import zero_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.tle_copy import tle_copy
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
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
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
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
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
    pid = tl.program_id(0)
    pid_n = pid // m
    pid_m = pid % m

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int32)
            for n_start in range(cta_n_start, cta_n_end, TILE_N):
                n_offsets = n_start + tl.arange(0, TILE_N)
                mask = n_offsets < cta_n_end
                arr = tl.load(arr_ptr + pid_m * n + n_offsets, mask=mask)
                arr = convert_to_uint_preverse_order(arr, descending)
                key = (arr >> bit_offset) & bfe_mask
                matches = tl.where(mask, (bin_indices[:, None] == key), False)
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
    associate_arr_ptr,
    out_ptr,
    associate_out_ptr,
    excumsum_bins_ptr,
    status_ptr,
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

    pid = tl.program_id(0)
    pid_m = pid % m
    pid_n = pid // m
    pid_r = tl.program_id(1)

    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1

    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    arr = tl.load(arr_ptr + pid_m * N + n_offsets, mask=mask)
    arr_u = convert_to_uint_preverse_order(arr, descending)
    key = (arr_u >> bit_offset) & bfe_mask

    for bin_index in range(cta_r_start, cta_r_end):
        matches = tl.where(mask, key == bin_index, False)
        local_sum = tl.sum(matches.to(tl.uint32), axis=0)
        pack0 = aggregate_mask | local_sum
        status_offset = pid_m * (r * OUT_N) + bin_index * OUT_N + pid_n
        tl.store(status_ptr + status_offset, pack0, cache_modifier=".cg")

        exclusive_prefix = tl.zeros((), dtype=tl.uint32)
        i_lookback = pid_n - 1
        while i_lookback >= 0:
            flag_offset_i = pid_m * (r * OUT_N) + bin_index * OUT_N + i_lookback
            pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            while pack1 == 0:
                pack1 = tl.load(status_ptr + flag_offset_i, volatile=True)
            exclusive_prefix += pack1 & v_mask
            if (pack1 & aggregate_mask) == aggregate_mask:
                i_lookback -= 1
            else:
                i_lookback = -1
        pack2 = inclusive_prefix_mask | (exclusive_prefix + local_sum)
        tl.store(status_ptr + status_offset, pack2, cache_modifier=".cg")

        local_ex_cumsum = tl.cumsum(matches.to(tl.uint32), axis=0) - matches
        ex_cumsum_in_bin = exclusive_prefix + local_ex_cumsum

        ex_cumsum_bins = tl.load(
            excumsum_bins_ptr + pid_m * (n_passes * r) + pass_id * r + bin_index
        )
        pos = ex_cumsum_bins + ex_cumsum_in_bin

        tl.store(out_ptr + pid_m * N + pos, arr, mask=matches)
        if associate_arr_ptr is not None:
            associate_arr = tl.load(
                associate_arr_ptr + pid_m * N + n_offsets, mask=mask
            )
            tl.store(associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches)


@triton.jit
def count_kernel(
    x_ptr,
    counts_ptr,
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

    val = tl.load(x_ptr + row_start + n_offset, mask=mask, other=0)
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
    counts_ptr,
    offsets_ptr,
    R: tl.constexpr,
    R_PAD: tl.constexpr,
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

    val = tl.load(x_ptr + row_start + n_offset, mask=mask, other=0)
    val_u = convert_to_uint_preverse_order(val, descending)

    idx = tl.load(idx_in_ptr + row_start + n_offset, mask=mask, other=0)

    bfe_mask = num_bins - 1
    key = (val_u >> bit_offset) & bfe_mask

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

    _env_block_n = os.environ.get("GEMS_XPU_RADIX_BLOCK_N")
    if _env_block_n:
        BLOCK_N = int(_env_block_n)
    else:
        BLOCK_N = min(4096, max(64, triton.next_power_of_2(N)))
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (M * grid_n,)

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
                is_use_mask_zero=True,
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
                is_use_mask_zero=True,
            )

            arr_in, arr_out = arr_out, arr_in
            idx_in, idx_out = idx_out, idx_in

    return arr_in.reshape(original_shape), idx_in.reshape(original_shape)


@triton.jit
def build_packed_kernel(
    arr_ptr,
    packed_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    descending: tl.constexpr,
    GRID_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // GRID_N
    block_idx = pid % GRID_N
    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N
    val = tl.load(arr_ptr + row_start + n_offset, mask=mask, other=0)
    val_u = convert_to_uint_preverse_order(val, descending)
    idx = n_offset.to(tl.uint32)
    packed = (val_u.to(tl.uint32).to(tl.uint64) << 32) | idx.to(tl.uint64)
    tl.store(packed_ptr + row_start + n_offset, packed, mask=mask)


@triton.jit
def fused_pass_kernel(
    p_ptr,
    p_out_ptr,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n = tl.arange(0, BLOCK_N)
    mask = n < N
    packed = tl.load(p_ptr + pid * N + tl.minimum(n, N - 1))
    packed_u = packed.to(tl.uint64, bitcast=True)
    key = ((packed_u >> (32 + bit_offset)) & (num_bins - 1)).to(tl.int32)
    w = tl.where(mask, key >> 2, 0)
    f = tl.where(mask, key & 3, 0)
    one = tl.where(mask, tl.full((), 1, tl.uint64) << (16 * f), 0)
    m0 = tl.where(w == 0, one, 0)
    m1 = tl.where(w == 1, one, 0)
    m2 = tl.where(w == 2, one, 0)
    m3 = tl.where(w == 3, one, 0)
    h0 = tl.sum(m0, axis=0)
    h1 = tl.sum(m1, axis=0)
    h2 = tl.sum(m2, axis=0)
    h3 = tl.sum(m3, axis=0)
    s0 = tl.cumsum(m0, axis=0)
    s1 = tl.cumsum(m1, axis=0)
    s2 = tl.cumsum(m2, axis=0)
    s3 = tl.cumsum(m3, axis=0)
    s = tl.where(w == 0, s0, tl.where(w == 1, s1, tl.where(w == 2, s2, s3)))
    rank_incl = ((s >> (16 * f)) & 0xFFFF).to(tl.int32)
    local_rank = tl.where(mask, rank_incl - 1, 0)
    dest = (tl.arange(0, BLOCK_N) - BLOCK_N).to(tl.int64)
    acc = tl.zeros((), tl.int64)
    for b in range(num_bins):
        if b < 4:
            h = h0
            bf = b
        elif b < 8:
            h = h1
            bf = b - 4
        elif b < 12:
            h = h2
            bf = b - 8
        else:
            h = h3
            bf = b - 12
        cbb = ((h >> (16 * bf)) & 0xFFFF).to(tl.int64)
        dest = tl.where(
            key == b,
            (pid.to(tl.int64) * N + acc + local_rank.to(tl.int64)),
            dest,
        )
        acc += cbb
    dest = tl.where(mask, dest, (tl.arange(0, BLOCK_N) - BLOCK_N).to(tl.int64))
    tl.store(p_out_ptr + dest, packed)


@triton.jit
def count_packed_kernel(
    p_ptr,
    counts_ptr,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // GRID_N
    block_idx = pid % GRID_N
    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N
    packed = tl.load(p_ptr + row_start + tl.minimum(n_offset, N - 1))
    packed_u = packed.to(tl.uint64, bitcast=True)
    key = tl.where(
        mask,
        ((packed_u >> (32 + bit_offset)) & (num_bins - 1)).to(tl.int32),
        -1,
    )
    counts_row = counts_ptr + row_idx * R_PAD + block_idx
    for i in range(num_bins):
        bin_mask = key == i
        count = tl.sum(bin_mask.to(tl.int32))
        tl.store(counts_row + i * GRID_N, count)


@triton.jit
def scatter_packed_kernel(
    p_ptr,
    p_out_ptr,
    global_offsets_ptr,
    M,
    N,
    bit_offset,
    num_bins: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    R_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // GRID_N
    block_idx = pid % GRID_N
    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N
    packed = tl.load(p_ptr + row_start + tl.minimum(n_offset, N - 1))
    packed_u = packed.to(tl.uint64, bitcast=True)
    key = tl.where(
        mask,
        ((packed_u >> (32 + bit_offset)) & (num_bins - 1)).to(tl.int32),
        -1,
    )
    w = tl.where(mask, key >> 2, 0)
    f = tl.where(mask, key & 3, 0)
    one = tl.where(mask, tl.full((), 1, tl.uint64) << (16 * f), 0)
    m0 = tl.where(w == 0, one, 0)
    m1 = tl.where(w == 1, one, 0)
    m2 = tl.where(w == 2, one, 0)
    m3 = tl.where(w == 3, one, 0)
    s0 = tl.cumsum(m0, axis=0)
    s1 = tl.cumsum(m1, axis=0)
    s2 = tl.cumsum(m2, axis=0)
    s3 = tl.cumsum(m3, axis=0)
    s = tl.where(w == 0, s0, tl.where(w == 1, s1, tl.where(w == 2, s2, s3)))
    rank_incl = ((s >> (16 * f)) & 0xFFFF).to(tl.int32)
    local_rank = tl.where(mask, rank_incl - 1, 0)
    offsets_row = global_offsets_ptr + row_idx * R_PAD + block_idx
    global_start = tl.load(offsets_row + key * GRID_N)
    dest = tl.where(
        mask,
        (row_start + global_start + local_rank).to(tl.int64),
        (tl.arange(0, BLOCK_N) - BLOCK_N).to(tl.int64),
    )
    tl.store(p_out_ptr + dest, packed)


@triton.jit
def _u32_to_f32(u32, descending: tl.constexpr):
    sign = u32 >> 31
    if descending:
        bits = tl.where((sign == 1), u32, u32 ^ tl.full((), 0x7FFFFFFF, tl.uint32))
    else:
        bits = tl.where((sign == 1), u32 ^ tl.full((), 0x80000000, tl.uint32), ~u32)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _u16_to_f16(u16, descending: tl.constexpr):
    sign = u16 >> 15
    if descending:
        bits = tl.where((sign == 1), u16, u16 ^ tl.full((), 0x7FFF, tl.uint16))
    else:
        bits = tl.where((sign == 1), u16 ^ tl.full((), 0x8000, tl.uint16), ~u16)
    return bits.to(tl.float16, bitcast=True)


_SRC_CODE = {
    torch.float32: 0,
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.int32: 3,
    torch.int16: 4,
    torch.bool: 5,
}


@triton.jit
def uint_to_value(u: tl.tensor, src_code: tl.constexpr, descending: tl.constexpr):
    if src_code == 0:
        return _u32_to_f32(u, descending)
    elif src_code == 1:
        return _u32_to_f32(u, descending).to(tl.bfloat16)
    elif src_code == 2:
        return _u16_to_f16((u & 0xFFFF).to(tl.uint16), descending)
    elif src_code == 3:
        if descending:
            bits = u ^ tl.full((), 0x7FFFFFFF, tl.uint32)
        else:
            bits = u ^ tl.full((), 0x80000000, tl.uint32)
        return bits.to(tl.int32, bitcast=True)
    elif src_code == 4:
        u16 = (u & 0xFFFF).to(tl.uint16)
        if descending:
            bits = u16 ^ tl.full((), 0x7FFF, tl.uint16)
        else:
            bits = u16 ^ tl.full((), 0x8000, tl.uint16)
        return bits.to(tl.int16, bitcast=True)
    else:
        return (u & 1).to(tl.int1)


@triton.jit
def unpack_packed_kernel(
    p_ptr,
    out_ptr,
    value_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
    src_code: tl.constexpr,
    descending: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = pid // GRID_N
    block_idx = pid % GRID_N
    row_start = row_idx * N
    n_offset = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = n_offset < N
    packed = tl.load(p_ptr + row_start + n_offset, mask=mask, other=0)
    packed_u = packed.to(tl.uint64, bitcast=True)
    tl.store(
        out_ptr + row_start + n_offset, (packed_u & 0xFFFFFFFF).to(tl.int64), mask=mask
    )
    if value_ptr is not None:
        val_u = (packed_u >> 32).to(tl.uint32)
        tl.store(
            value_ptr + row_start + n_offset,
            uint_to_value(val_u, src_code, descending),
            mask=mask,
        )


def radix_argsort(inp, k_bits=4, descending=False):
    """Stable argsort (indices only) through packed (value, column) radix passes.

    NOTE(kunlunxin): the value+index radix chain (radix_sort_low_mem) issues
    TWO data-dependent stores per element per pass (2B value + 8B index), and
    a data-dependent (gather) store is the most expensive op on this backend:
    each one serialises an 8-byte DMA behind llvm_xpu.mfence (measured ~4.3ns
    vs ~0.5ns for an affine store on XPU; gather *loads* are only ~0.6ns and
    pipeline fine).  Packing (val_u, column) into one 64-bit word cuts the
    scatter to ONE 8-byte gather store per element, and the low-32-bit column
    makes (key, column) the stable order, so sort==argsort and no value
    reconstruction is needed here (the caller only wants indices).

    grid_n == 1 (rows of <= 4096 elements): the whole per-pass pipeline is
    fused into a single kernel (fused_pass_kernel) - no count/binscan.
    grid_n > 1: count_packed_kernel + bin_prefix_kernel + scatter_packed_kernel
    (the bin-major scan needs data from every block of the row).
    """
    original_shape = inp.shape
    N = inp.shape[-1]
    inp = inp.contiguous()
    arr = inp.reshape(-1, N)
    M = arr.shape[0]

    _env_block_n = os.environ.get("GEMS_XPU_RADIX_BLOCK_N")
    if _env_block_n:
        BLOCK_N = int(_env_block_n)
    else:
        BLOCK_N = min(4096, max(64, triton.next_power_of_2(N)))
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (M * grid_n,)

    dtype = inp.dtype
    num_bits = 1
    if dtype == torch.bool:
        pass
    elif dtype == torch.bfloat16:
        num_bits = 4 * 8
    else:
        num_bits = inp.element_size() * 8
    num_passes = (num_bits + k_bits - 1) // k_bits
    num_bins = 2**k_bits

    _HEAD_PAD = BLOCK_N
    _TAIL_PAD = 256
    _keepalive = []

    def _padded():
        buf = torch.empty(
            _HEAD_PAD + M * N + _TAIL_PAD, device=inp.device, dtype=torch.int64
        )
        _keepalive.append(buf)
        return buf[_HEAD_PAD : _HEAD_PAD + M * N].view(M, N)

    packed_in = _padded()
    packed_out = _padded()

    with torch_device_fn.device(inp.device):
        build_packed_kernel[grid](
            arr, packed_in, M, N, BLOCK_N, descending, GRID_N=grid_n
        )
        if grid_n == 1:
            for p in range(num_passes):
                fused_pass_kernel[(M,)](
                    packed_in,
                    packed_out,
                    M,
                    N,
                    p * k_bits,
                    num_bins,
                    BLOCK_N,
                )
                packed_in, packed_out = packed_out, packed_in
        else:
            r = num_bins * grid_n
            tile_r = max(64, min(4096, triton.next_power_of_2(r)))
            r_pad = triton.cdiv(r, tile_r) * tile_r
            counts = torch.empty(M * r_pad, device=inp.device, dtype=torch.int32)
            global_offsets = torch.empty(
                M * r_pad, device=inp.device, dtype=torch.int32
            )
            for p in range(num_passes):
                bit_offset = p * k_bits
                count_packed_kernel[grid](
                    packed_in,
                    counts,
                    M,
                    N,
                    bit_offset,
                    num_bins,
                    BLOCK_N,
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
                scatter_packed_kernel[grid](
                    packed_in,
                    packed_out,
                    global_offsets,
                    M,
                    N,
                    bit_offset,
                    num_bins,
                    BLOCK_N,
                    GRID_N=grid_n,
                    R_PAD=r_pad,
                )
                packed_in, packed_out = packed_out, packed_in
        indices = _padded()
        unpack_packed_kernel[grid](
            packed_in,
            indices,
            None,
            M,
            N,
            BLOCK_N,
            GRID_N=grid_n,
            src_code=_SRC_CODE[inp.dtype],
            descending=descending,
        )

    return indices.reshape(original_shape)


def radix_sort_packed(inp, k_bits=4, descending=False):
    """Stable sort (values + indices) through packed (value, column) passes.

    Same packed pipeline as radix_argsort, but the final unpack recovers BOTH
    the value (inverse of convert_to_uint_preverse_order) and the column.

    NOTE(kunlunxin, sort_stable): radix_sort_low_mem issues TWO data-dependent
    (gather) stores per element per pass (value + int64 index), and a gather
    store is the most expensive operation on this backend (~4.3ns each vs
    ~0.5ns for an affine store; each serialises an 8-byte DMA behind
    llvm_xpu.mfence).  Packing (val_u, column) into one 64-bit word cuts that
    to a single 8-byte gather store per element per pass -- the same
    ~2x reduction already captured by radix_argsort (see the note there) --
    and the low-32-bit column keeps (key, column) as the stable order, so the
    value+index pair needs no separate stable-index bookkeeping.  Only values
    whose transformed key fits in 32 bits (float16/32, bfloat16, int16/32,
    bool) can be packed; larger dtypes (int64/fp64) keep radix_sort_low_mem.
    """
    original_shape = inp.shape
    N = inp.shape[-1]
    inp = inp.contiguous()
    arr = inp.reshape(-1, N)
    M = arr.shape[0]

    _env_block_n = os.environ.get("GEMS_XPU_RADIX_BLOCK_N")
    if _env_block_n:
        BLOCK_N = int(_env_block_n)
    else:
        BLOCK_N = min(4096, max(64, triton.next_power_of_2(N)))
    grid_n = triton.cdiv(N, BLOCK_N)
    grid = (M * grid_n,)

    dtype = inp.dtype
    num_bits = 1
    if dtype == torch.bool:
        pass
    elif dtype == torch.bfloat16:
        num_bits = 4 * 8
    else:
        num_bits = inp.element_size() * 8
    num_passes = (num_bits + k_bits - 1) // k_bits
    num_bins = 2**k_bits

    _HEAD_PAD = BLOCK_N
    _TAIL_PAD = 256
    _keepalive = []

    def _padded(dtype_):
        buf = torch.empty(
            _HEAD_PAD + M * N + _TAIL_PAD, device=inp.device, dtype=dtype_
        )
        _keepalive.append(buf)
        return buf[_HEAD_PAD : _HEAD_PAD + M * N].view(M, N)

    packed_in = _padded(torch.int64)
    packed_out = _padded(torch.int64)

    with torch_device_fn.device(inp.device):
        build_packed_kernel[grid](
            arr, packed_in, M, N, BLOCK_N, descending, GRID_N=grid_n
        )
        if grid_n == 1:
            for p in range(num_passes):
                fused_pass_kernel[(M,)](
                    packed_in,
                    packed_out,
                    M,
                    N,
                    p * k_bits,
                    num_bins,
                    BLOCK_N,
                )
                packed_in, packed_out = packed_out, packed_in
        else:
            r = num_bins * grid_n
            tile_r = max(64, min(4096, triton.next_power_of_2(r)))
            r_pad = triton.cdiv(r, tile_r) * tile_r
            counts = torch.empty(M * r_pad, device=inp.device, dtype=torch.int32)
            global_offsets = torch.empty(
                M * r_pad, device=inp.device, dtype=torch.int32
            )
            for p in range(num_passes):
                bit_offset = p * k_bits
                count_packed_kernel[grid](
                    packed_in,
                    counts,
                    M,
                    N,
                    bit_offset,
                    num_bins,
                    BLOCK_N,
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
                scatter_packed_kernel[grid](
                    packed_in,
                    packed_out,
                    global_offsets,
                    M,
                    N,
                    bit_offset,
                    num_bins,
                    BLOCK_N,
                    GRID_N=grid_n,
                    R_PAD=r_pad,
                )
                packed_in, packed_out = packed_out, packed_in
        values = _padded(inp.dtype)
        indices = _padded(torch.int64)
        unpack_packed_kernel[grid](
            packed_in,
            indices,
            values,
            M,
            N,
            BLOCK_N,
            GRID_N=grid_n,
            src_code=_SRC_CODE[inp.dtype],
            descending=descending,
        )

    return values.reshape(original_shape), indices.reshape(original_shape)


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
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0:
        return inp, torch.empty_like(inp, dtype=torch.int64)
    if sort_elem_cnt == 1:
        indices = torch.empty_like(inp, dtype=torch.int64)
        with torch_device_fn.device(inp.device):
            init_indices_kernel[(triton.cdiv(inp.numel(), 256),)](
                indices, inp.numel(), 1, BLOCK_SIZE=256
            )
        return inp, indices
    return sort_stable(inp, stable=True, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_KUNLUNXIN SORT_STABLE")
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 0:
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
    if dim != inp.ndim - 1:
        view = torch.movedim(inp, dim, -1)
        inp = torch.empty(view.shape, device=inp.device, dtype=inp.dtype)
        if not tle_copy(view, inp):
            torch.ops.aten._copy_from(view, inp, False)
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    if dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.int16,
        torch.int32,
        torch.bool,
    ):
        out, out_index = radix_sort_packed(inp, num_bits_per_pass, descending)
    else:
        out, out_index = radix_sort_low_mem(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index

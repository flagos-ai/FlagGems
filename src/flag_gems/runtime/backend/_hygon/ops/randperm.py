import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.ops.topk import argsort
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

logger = logging.getLogger(__name__)
device_ = device

_MIN_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).min)
_MAX_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).max)
_MIN_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).min)
_MAX_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).max)
_MIN_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).min)
_MAX_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).max)
_MIN_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).min)
_MAX_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).max)
_MAX_UINT32_VAL = tl.constexpr((1 << 32) - 1)
_MIN_UINT32_VAL = tl.constexpr(0)
_MIN_INT24_VAL = tl.constexpr(-(2**23))
_MAX_INT24_VAL = tl.constexpr(2**23 - 1)


@triton.jit
def _get_iinfo_val(
    dtype,
    return_max,
):
    if dtype is tl.int64:
        if return_max:
            return _MAX_INT64_VAL
        else:
            return _MIN_INT64_VAL
    elif dtype is tl.int32:
        if return_max:
            return _MAX_INT32_VAL
        else:
            return _MIN_INT32_VAL
    elif dtype is tl.int16:
        if return_max:
            return _MAX_INT16_VAL
        else:
            return _MIN_INT16_VAL
    elif dtype is tl.int8:
        if return_max:
            return _MAX_INT8_VAL
        else:
            return _MIN_INT8_VAL
    elif dtype is tl.uint32:
        if return_max:
            return _MAX_UINT32_VAL
        else:
            return _MIN_UINT32_VAL
    else:
        raise ValueError("Unknown dtype")


@libentry()
@triton.jit
def bitonic_sortbykey_kernel(
    y_ptr,
    index_ptr,
    chunk_x,
    chunk_index,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    chunk_x += cur_batch * N
    chunk_index += cur_batch * N
    index_ptr += cur_batch * N
    y_ptr += cur_batch * N

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    mask_val = _get_iinfo_val(chunk_x.dtype.element_ty, return_max=not DESCENDING)

    chunk_x_val = tl.load(chunk_x + cols, mask=mask, other=mask_val)
    chunk_index_val = tl.load(chunk_index + cols, mask=mask)

    sorted_chunk_x, sorted_chunk_index = argsort(
        chunk_x_val, chunk_index_val, 0, descending=DESCENDING
    )
    tl.store(y_ptr + cols, sorted_chunk_x, mask=cols < N)
    tl.store(index_ptr + cols, sorted_chunk_index, mask=cols < N)


@triton.jit
def radix_type_convert(k):
    ik = k.to(tl.int64)
    if tl.constexpr(k.dtype == tl.int8):
        mask = (ik >> 7) & 0x1
        o = tl.where(mask, ik & 0x7F, ik | 0x80)
    elif tl.constexpr(k.dtype == tl.int16):
        mask = (ik >> 15) & 0x1
        o = tl.where(mask, ik & 0x7FFF, ik | 0x8000)
    elif tl.constexpr(k.dtype == tl.int32):
        mask = (ik >> 31) & 0x1
        o = tl.where(mask, ik & 0x7FFFFFFF, ik | 0x80000000)
    elif tl.constexpr(k.dtype == tl.int64):
        mask = (ik >> 63) & 0x1
        o = tl.where(mask, ik & 0x7FFFFFFFFFFFFFFF, ik | 0x8000000000000000)
    else:
        o = k
    return o


@libentry()
@triton.jit
def digit_hist_kernel(
    digit_hist,
    key,
    n_elements,
    bits_per_pass,
    bins,
    passes,
    bit_mask,
    bins_segment,
    BLOCK_SIZE: tl.constexpr,
):
    bin_segid = tl.program_id(1)
    pid0 = tl.program_id(0)
    grid0 = tl.num_programs(0)

    key_offset = pid0.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    key_mask = key_offset < n_elements
    key_data = tl.load(key + key_offset, mask=key_mask)
    ikey_data = radix_type_convert(key_data)
    bit_offset = 0
    for p in range(passes):
        key_digit = (ikey_data >> bit_offset) & bit_mask
        blk_bin_start = bin_segid * bins_segment
        for s in range(bins_segment):
            bin_id = s + blk_bin_start
            digit_mask = tl.where(key_digit == bin_id and key_mask, 1, 0)
            digit_sum = tl.sum(digit_mask)
            # +1 for exclusive
            bin_offset = p * (bins + 1) * grid0 + (bin_id + 1) * grid0 + pid0
            # reduce rather than global atomic for perf issue
            tl.store(digit_hist + bin_offset, digit_sum)
        tl.store(digit_hist + p * (bins + 1) * grid0 + pid0, 0, mask=bin_segid == 0)
        bit_offset += bits_per_pass


@libentry()
@triton.autotune(
    configs=runtime.get_tuned_config("randperm"),
    key=["n_elements"],
)
@triton.jit
def radix_sortbykey_scatter_kernel(
    key_out,
    value_out,
    key_in,
    value_in,
    digit_hist,
    d_lookback,
    n_elements,
    bit_offset,
    passes,
    p,
    num_portions,
    portion_size,
    portion_id,
    bit_mask,
    bins_segment,
    max_tiles_per_portion,
    bins: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    LOOKBACK_PARTIAL_MASK = 1 << 30
    LOOKBACK_GLOBAL_MASK = 1 << 31
    LOOKBACK_KIND_MASK = LOOKBACK_PARTIAL_MASK | LOOKBACK_GLOBAL_MASK
    LOOKBACK_VALUE_MASK = ~LOOKBACK_KIND_MASK

    pid0 = tl.program_id(0)
    portion_id_i64 = portion_id
    portion_id_i64 = portion_id_i64.to(tl.int64)
    key_offset = (
        portion_id_i64 * portion_size
        + pid0.to(tl.int64) * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    key_mask = key_offset < n_elements
    value_data = tl.load(value_in + key_offset, mask=key_mask)
    key_data = tl.load(key_in + key_offset, mask=key_mask)

    ikey_data = radix_type_convert(key_data)
    key_digit = (ikey_data >> bit_offset) & bit_mask

    blk_bin_start = tl.program_id(1) * bins_segment
    last_block = tl.program_id(0) == tl.num_programs(0) - 1
    for s in range(bins_segment):
        bin_id = s + blk_bin_start
        key_digit_mask = (key_digit == bin_id) & key_mask
        key_elem_mask = tl.where(key_digit_mask, 1, 0)
        key_block_rank = tl.cumsum(key_elem_mask)
        key_block_rank = tl.where(key_digit_mask, key_block_rank - 1, 0)
        bin_of_bucket = tl.sum(key_elem_mask)
        partial_counter = bin_of_bucket | LOOKBACK_PARTIAL_MASK
        tl.store(
            d_lookback
            + ((portion_id * passes + p) * max_tiles_per_portion + pid0) * bins
            + bin_id,
            partial_counter,
            cache_modifier=".cg",
        )
        bin_offset = p * (bins + 1) + bin_id
        prefix_offsets = tl.load(
            digit_hist + bin_offset + portion_id * passes * (bins + 1)
        )
        bk = pid0 - 1
        inc_sum = bin_of_bucket
        while bk >= 0:
            rd_lbk_offset = (
                (portion_id * passes + p) * max_tiles_per_portion + bk
            ) * bins + bin_id
            partial_prefix = 0
            while partial_prefix == 0:
                partial_prefix = tl.atomic_cas(
                    d_lookback + rd_lbk_offset, 0, 0, sem="acquire"
                )
            inc_sum += (partial_prefix & LOOKBACK_VALUE_MASK).to(tl.int32)
            if partial_prefix & LOOKBACK_GLOBAL_MASK:
                # break
                bk = -1
            else:
                bk -= 1
        global_counter = inc_sum | LOOKBACK_GLOBAL_MASK
        tl.store(
            d_lookback
            + ((portion_id * passes + p) * max_tiles_per_portion + pid0) * bins
            + bin_id,
            global_counter,
            cache_modifier=".cg",
        )
        inc_bucket_offset = prefix_offsets.to(tl.int64) + inc_sum.to(tl.int64)
        if last_block and portion_id < num_portions - 1:
            tl.store(
                digit_hist + bin_offset + (portion_id + 1) * passes * (bins + 1),
                inc_bucket_offset,
            )
        global_offsets = (
            inc_bucket_offset - bin_of_bucket.to(tl.int64) + key_block_rank.to(tl.int64)
        )
        tl.store(key_out + global_offsets, key_data, mask=key_digit_mask)
        tl.store(value_out + global_offsets, value_data, mask=key_digit_mask)


# for parallelization, randomly shuffle the entire block rather than adjacent equal elements as pytorch GPU backend
@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset"])
def duplicate_keys_shuffle_kernel(
    value_in, n_elements, philox_seed, philox_offset, BLOCK_SIZE: tl.constexpr
):
    pid0 = tl.program_id(0)
    offset_range = tl.arange(0, BLOCK_SIZE)
    value_offset = pid0.to(tl.int64) * BLOCK_SIZE + offset_range
    value_mask = value_offset < n_elements
    value_data = tl.load(value_in + value_offset, mask=value_mask)

    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c0 += i4
    _O = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0, c1, _O, _O)

    _block_size = BLOCK_SIZE
    r1 = r0 % _block_size.to(tl.uint32)
    mask_val = _get_iinfo_val(tl.uint32, True)
    r1 = tl.where(value_offset < n_elements, r1, mask_val)
    _, sorted_chunk_index = argsort(r1, offset_range, 0, descending=False)
    store_offset = pid0.to(tl.int64) * BLOCK_SIZE + sorted_chunk_index.to(tl.int64)
    tl.store(value_in + store_offset, value_data, mask=store_offset < n_elements)


def sort_by_key(key, value, valid_bits, generator=None):
    n_elements = key.numel()
    if n_elements > 2 * 1024:
        # radix method
        BLOCK_SIZE = 1024
        bits_per_pass = 4
        bits_per_segment = 3
        passes = triton.cdiv(valid_bits, bits_per_pass)
        bins = 2**bits_per_pass
        bins_per_sgement = 2**bits_per_segment
        bit_mask = bins - 1

        portion_size = 2**30  # 2 bits reserved for mask
        num_portions = triton.cdiv(n_elements, portion_size)
        max_portion_items = portion_size if num_portions > 1 else n_elements
        max_tiles_per_portion = triton.cdiv(max_portion_items, BLOCK_SIZE)

        hist_dtype = torch.int64 if num_portions > 1 else torch.int32
        grid_hist = (triton.cdiv(n_elements, BLOCK_SIZE), bins // bins_per_sgement)

        digit_hist_slice = torch.empty(
            (passes, bins + 1, grid_hist[0]), dtype=hist_dtype, device=key.device
        )

        digit_hist = torch.empty(
            (num_portions, passes, bins + 1), dtype=hist_dtype, device=key.device
        )
        d_lookback = torch.empty(
            num_portions * passes * bins * max_tiles_per_portion,
            dtype=torch.int32,
            device=key.device,
        )

        key_out_p = torch.empty_like(key)
        key_out_q = torch.empty_like(key)
        value_out_p = torch.empty_like(value)
        value_out_q = torch.empty_like(value)

        # step1
        d_lookback.zero_()
        with torch_device_fn.device(key.device):
            digit_hist_kernel[grid_hist](
                digit_hist_slice,
                key,
                n_elements,
                bits_per_pass,
                bins,
                passes,
                bit_mask,
                bins_per_sgement,
                BLOCK_SIZE,
            )

        # step2
        digit_hist_slice = torch.sum(digit_hist_slice, dim=2, keepdim=False)
        digit_hist_slice = digit_hist_slice.cumsum(dim=1)  # shape of [passes, bins + 1]
        digit_hist.copy_(digit_hist_slice)

        bit_offset = 0
        for p in range(passes):
            k_in = (key if p == 0 else key_out_p) if p % 2 == 0 else key_out_q
            v_in = (value if p == 0 else value_out_p) if p % 2 == 0 else value_out_q
            k_out = key_out_q if p % 2 == 0 else key_out_p
            v_out = value_out_q if p % 2 == 0 else value_out_p
            # step3
            for portion_id in range(num_portions):
                portion_items = min(
                    n_elements - portion_id * portion_size, portion_size
                )
                tiles_per_portion = triton.cdiv(portion_items, BLOCK_SIZE)
                grid_scatter = (tiles_per_portion, grid_hist[1])
                with torch_device_fn.device(key.device):
                    radix_sortbykey_scatter_kernel[grid_scatter](
                        k_out,
                        v_out,
                        k_in,
                        v_in,
                        digit_hist,
                        d_lookback,
                        n_elements,
                        bit_offset,
                        passes,
                        p,
                        num_portions,
                        portion_size,
                        portion_id,
                        bit_mask,
                        bins_per_sgement,
                        max_tiles_per_portion,
                        bins,
                        BLOCK_SIZE,
                    )
            bit_offset += bits_per_pass

        # last step, shuffle inner-block data
        BLOCK_SIZE_SHUFFLE = 512
        grid_shuffle = (triton.cdiv(n_elements, BLOCK_SIZE_SHUFFLE),)
        philox_seed, philox_offset = philox_backend_seed_offset(
            n_elements, generator=generator
        )
        with torch_device_fn.device(key.device):
            duplicate_keys_shuffle_kernel[grid_shuffle](
                v_out,
                n_elements,
                philox_seed,
                philox_offset,
                BLOCK_SIZE_SHUFFLE,
                num_warps=4,
            )
        return v_out
    else:
        # bitonic method
        BLOCK_SIZE = triton.next_power_of_2(n_elements)
        grid = (1,)
        k_out = torch.empty_like(key)
        v_out = torch.empty_like(value)
        with torch_device_fn.device(key.device):
            bitonic_sortbykey_kernel[grid](
                k_out, v_out, key, value, n_elements, BLOCK_SIZE, False
            )
        return v_out


def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS_MTHREADS RANDPERM")
    # import pdb
    # pdb.set_trace()
    assert dtype == torch.int16 or dtype == torch.int32 or dtype == torch.int64
    assert n <= _MAX_INT64_VAL, "n exceeds maximum int64"

    if device is None:
        device = torch.device(device_.name)
    in_range = torch.arange(n, dtype=dtype, device=device)

    u8max = 2**8
    u16max = 2**16
    u24max = 2**24
    u32max = 2**32

    if n <= u8max:
        valid_bits = 8
        key_dtype = torch.int8
        keymin = _MIN_INT8_VAL
        keymax = _MAX_INT8_VAL
    elif n <= u16max:
        valid_bits = 16
        key_dtype = torch.int16
        keymin = _MIN_INT16_VAL
        keymax = _MAX_INT16_VAL
    elif n <= u24max:
        valid_bits = 24
        key_dtype = torch.int32
        keymin = _MIN_INT24_VAL
        keymax = _MAX_INT24_VAL
    elif n <= u32max:
        valid_bits = 32
        key_dtype = torch.int32
        keymin = _MIN_INT32_VAL
        keymax = _MAX_INT32_VAL
    else:
        valid_bits = 64
        key_dtype = torch.int64
        keymin = _MIN_INT64_VAL
        keymax = _MAX_INT64_VAL

    rand_key = torch.randint(
        low=keymin, high=keymax, size=[n], dtype=key_dtype, device=device
    )
    perm_range = sort_by_key(rand_key, in_range, valid_bits, generator=generator)
    return perm_range

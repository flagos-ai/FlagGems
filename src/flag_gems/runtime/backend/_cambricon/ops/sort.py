import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

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
    M_PER_SPLIT: tl.constexpr,
):
    # grid layout:
    # program_id(0) -> split id s
    # program_id(1) -> pid_n
    # program_id(2) -> pid_m_idx (index inside split)
    s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m_idx = tl.program_id(2)
    pid_m = s * M_PER_SPLIT + pid_m_idx
    if pid_m >= m:
        return

    # arr_ptr: (m, n)
    # out_ptr: (m, n_passes, r), where r = 2 ** k_bits is the number of bins
    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for p in range(0, num_passes):
        bit_offset = p * num_bits_per_pass
        for r_start in range(0, r, TILE_R):
            bin_indices = r_start + tl.arange(0, TILE_R)
            acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int64)
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
    M_PER_SPLIT: tl.constexpr,
):
    # r: num_bins = 2 ** k_bits
    # OUT_N: grid_n = cdiv(N, )

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # excumsum_bins_ptr: (m, n_passes, r)
    # flag_ptr: (m, r, OUT_N)

    # grid: (S, grid_n, grid_r)
    # program_id(0) -> split id (s)
    # program_id(1) -> pid_n
    # program_id(2) -> pid_r

    s = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_r = tl.program_id(2)

    # bit masks
    aggregate_mask: tl.constexpr = 1 << 30
    inclusive_prefix_mask: tl.constexpr = 1 << 31
    v_mask: tl.constexpr = (1 << 30) - 1
    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    # initialize flag to zero-local sum is not ready
    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    for local_pid_m_idx in range(0, M_PER_SPLIT):
        pid_m = s * M_PER_SPLIT + local_pid_m_idx
        if pid_m < m:
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
                    tl.store(
                        associate_out_ptr + pid_m * N + pos, associate_arr, mask=matches
                    )


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    if arr.dtype == torch.int64:
        TILE_N = 512
    else:
        TILE_N = 1024
    tiles_n_per_cta = 8
    CTA_TILE_N = tiles_n_per_cta * TILE_N

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    TILE_R = 16

    grid_n = triton.cdiv(n, CTA_TILE_N)

    MAX_GRID = 65535
    S = (m + MAX_GRID - 1) // MAX_GRID
    M_PER_SPLIT = triton.cdiv(m, S)
    # grid_for_global_hist: 3D grid (S, grid_n, M_PER_SPLIT)
    grid_for_global_hist = (S, grid_n, M_PER_SPLIT)

    with torch_device_fn.device(arr.device):
        global_hist = torch.zeros(
            (m, n_passes, num_bins), device=arr.device, dtype=torch.int32
        )
        # launch compute_global_hist_kernel with M_PER_SPLIT passed
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
            M_PER_SPLIT,
        )

        # ex_cumsum_bins shape: (m, n_passes, num_bins)
        ex_cumsum_bins = torch.empty_like(global_hist, dtype=torch.uint32)
        # For each split, compute cumsum on the slice [s_start : s_end]
        for s in range(S):
            s_start = s * M_PER_SPLIT
            s_end = min(m, s_start + M_PER_SPLIT)
            if s_start >= s_end:
                continue
            # slice: shape (m_chunk, n_passes, num_bins)
            slice_hist = global_hist[s_start:s_end]  # this is a view
            # compute cumsum over last dim for this slice only (smaller kernel)
            slice_ex_cumsum = torch.cumsum(slice_hist, dim=-1) - slice_hist
            # write back to ex_cumsum_bins (and cast to uint32)
            ex_cumsum_bins[s_start:s_end] = slice_ex_cumsum.to(torch.uint32)

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
        TILE_N = 3072
        grid_n = triton.cdiv(n, TILE_N)

        # grid_for_sweep using same S (splits)
        grid_for_sweep = (S, grid_n, grid_r)

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
                M_PER_SPLIT,
            )
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in


def sort(inp, dim=-1, descending=False):
    # We only implement stable radix sort here
    logger.debug("GEMS_CAMBRICON SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_CAMBRICON SORT.STABLE")
    # We only implement stable radix sort here
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    num_bits_per_pass = 1 if dtype == torch.bool else 4
    out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index

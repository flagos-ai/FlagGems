import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Supported dtypes for the median operator
_SUPPORTED_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_INT_DTYPES = (torch.int16, torch.int32, torch.int64)
_SUPPORTED_DTYPES = _SUPPORTED_FLOAT_DTYPES + _SUPPORTED_INT_DTYPES


@libentry()
@triton.jit
def median_kernel_float(
    inp_ptr,
    out_val_ptr,
    out_idx_ptr,
    N: tl.constexpr,
    PADDED_N: tl.constexpr,
    MED_POS: tl.constexpr,
):
    """Compute median of a single row for floating-point types.

    Algorithm:
        1. Load row with power-of-2 padding (max float fill for sort stability)
        2. Sort ascending via tl.sort
        3. Extract median value at MED_POS
        4. Find original index by counting occurrences (handles duplicates)

    Args:
        inp_ptr: Pointer to input tensor (M, N)
        out_val_ptr: Pointer to output values (M,)
        out_idx_ptr: Pointer to output indices (M,)
        N: Actual row length (constexpr)
        PADDED_N: Power-of-2 padded length for tl.arange (constexpr)
        MED_POS: Median position index (constexpr)
    """
    pid = tle.program_id(0)
    inp_row_ptr = inp_ptr + pid * N
    out_val_ptr_pid = out_val_ptr + pid
    out_idx_ptr_pid = out_idx_ptr + pid

    cols = tl.arange(0, PADDED_N)
    mask = cols < N

    # Load values with padding - use max float (not inf, which breaks tl.sort on Iluvatar)
    pad_val = tl.full([PADDED_N], dtype=tl.float32, value=3.4028235e38)
    vals = tl.load(inp_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Detect NaN: NaN != NaN is True in IEEE 754
    has_nan = mask & (vals != vals)
    any_nan = tl.sum(has_nan.to(tl.int32)) > 0

    # Clamp inf/-inf to large finite values before sorting (tl.sort corrupts special floats on Iluvatar)
    max_f = tl.full([PADDED_N], dtype=tl.float32, value=3.4028235e38)
    neg_max_f = tl.full([PADDED_N], dtype=tl.float32, value=-3.4028235e38)
    has_pos_inf = mask & (vals > 3.4028235e38)
    has_neg_inf = mask & (vals < -3.4028235e38)
    vals = tl.where(has_pos_inf, max_f, vals)
    vals = tl.where(has_neg_inf, neg_max_f, vals)
    # Also clamp NaN to max_float to prevent sort corruption
    vals = tl.where(has_nan, max_f, vals)
    vals = tl.where(mask, vals, pad_val)

    # Sort ascending
    sorted_vals = tl.sort(vals, dim=0, descending=False)

    # Extract median value using positional mask
    med_val = tl.sum(tl.where(cols == MED_POS, sorted_vals, 0.0))

    # Restore inf: if the median equals max_float and input had +inf, restore to inf
    any_pos_inf = tl.sum(has_pos_inf.to(tl.int32)) > 0
    any_neg_inf = tl.sum(has_neg_inf.to(tl.int32)) > 0
    inf_val = tl.full([], dtype=tl.float32, value=float('inf'))
    neg_inf_val = tl.full([], dtype=tl.float32, value=float('-inf'))
    med_val = tl.where(any_pos_inf & (med_val >= 3.4028235e38), inf_val, med_val)
    med_val = tl.where(any_neg_inf & (med_val <= -3.4028235e38), neg_inf_val, med_val)

    # If any NaN in input, propagate NaN to output
    med_val = tl.where(any_nan, tl.full([], dtype=tl.float32, value=float('nan')), med_val)

    # Find original index of med_val in the input
    # For duplicates, we need the (MED_POS - count_less + 1)-th occurrence
    # Use clamped values for comparison since inf/nan were replaced before sort
    matches = mask & (vals == med_val)
    is_less = mask & (vals < med_val)
    count_less = tl.sum(is_less.to(tl.int32))
    # cumsum is 1-based, so +1 to match
    occurrence = MED_POS - count_less + 1
    match_cumsum = tl.cumsum(matches.to(tl.int32), axis=0)
    is_target = matches & (match_cumsum == occurrence)
    orig_idx = tl.sum(tl.where(is_target, cols, 0))

    tl.store(out_val_ptr_pid, med_val)
    tl.store(out_idx_ptr_pid, orig_idx.to(tl.int64))


@libentry()
@triton.jit
def median_kernel_int(
    inp_ptr,
    out_val_ptr,
    out_idx_ptr,
    N: tl.constexpr,
    PADDED_N: tl.constexpr,
    MED_POS: tl.constexpr,
    IS_INT64: tl.constexpr,
):
    """Compute median of a single row for integer types.

    Uses separate int32/int64 paths to avoid precision loss from float conversion.
    Algorithm is identical to median_kernel_float but operates in integer domain.
    """
    pid = tle.program_id(0)
    inp_row_ptr = inp_ptr + pid * N
    out_val_ptr_pid = out_val_ptr + pid
    out_idx_ptr_pid = out_idx_ptr + pid

    cols = tl.arange(0, PADDED_N)
    mask = cols < N

    if IS_INT64:
        pad_val = tl.full([PADDED_N], dtype=tl.int64, value=0x7FFFFFFFFFFFFFFF)
        vals = tl.load(inp_row_ptr + cols, mask=mask, other=0)
        vals = tl.where(mask, vals, pad_val)
        sorted_vals = tl.sort(vals, dim=0, descending=False)
        med_val = tl.sum(
            tl.where(cols == MED_POS, sorted_vals, tl.zeros([PADDED_N], dtype=tl.int64))
        )
        matches = mask & (vals == med_val)
        is_less = mask & (vals < med_val)
    else:
        pad_val = tl.full([PADDED_N], dtype=tl.int32, value=0x7FFFFFFF)
        vals = tl.load(inp_row_ptr + cols, mask=mask, other=0)
        vals = tl.where(mask, vals, pad_val)
        sorted_vals = tl.sort(vals, dim=0, descending=False)
        med_val = tl.sum(
            tl.where(cols == MED_POS, sorted_vals, tl.zeros([PADDED_N], dtype=tl.int32))
        )
        matches = mask & (vals == med_val)
        is_less = mask & (vals < med_val)

    count_less = tl.sum(is_less.to(tl.int32))
    occurrence = MED_POS - count_less + 1
    match_cumsum = tl.cumsum(matches.to(tl.int32), axis=0)
    is_target = matches & (match_cumsum == occurrence)
    orig_idx = tl.sum(tl.where(is_target, cols, 0))

    tl.store(out_val_ptr_pid, med_val)
    tl.store(out_idx_ptr_pid, orig_idx.to(tl.int64))


# ============================================================
# Histogram-based Radix Select: O(N) selection for large N
# ============================================================


def _to_sortable_bits(inp):
    """Convert tensor values to sortable int32 bit representation.

    For floats: sign-bit-flip on the 32-bit IEEE representation.
    For int32: XOR with 0x80000000 to map signed range to unsigned.
    Returns (int32_bits_tensor, num_bits=32).
    """
    if inp.dtype == torch.float32:
        bits = inp.view(torch.int32)
        # IEEE 754 sign-bit-flip: negative → invert all bits, positive → flip sign bit
        sign = (bits >> 31) & 1
        bits = torch.where(sign == 1, ~bits, bits ^ 0x80000000)
        return bits, 32
    elif inp.dtype in (torch.float16, torch.bfloat16):
        bits = inp.float().view(torch.int32)
        sign = (bits >> 31) & 1
        bits = torch.where(sign == 1, ~bits, bits ^ 0x80000000)
        return bits, 32
    elif inp.dtype == torch.int32:
        # Two's complement: simply flip sign bit to map signed → unsigned order
        return inp ^ 0x80000000, 32
    else:
        return inp.to(torch.int64) ^ 0x8000000000000000, 64


def _bits_to_value(bits, num_bits, orig_dtype):
    """Convert sortable int32 bits back to original dtype values.

    Reverse of sign-bit-flip: if top bit is 1, XOR with 0x80000000;
    if top bit is 0, XOR with 0xFFFFFFFF (invert all bits).
    """
    if num_bits == 64:
        return (bits ^ 0x8000000000000000).to(orig_dtype)

    bits32 = bits.to(torch.int32)
    if orig_dtype.is_floating_point:
        sign = (bits32 >> 31) & 1
        orig_bits = torch.where(sign == 1, bits32 ^ 0x80000000, ~bits32)
        return orig_bits.view(torch.float32).to(orig_dtype)
    else:
        # Reverse of inp ^ 0x80000000
        return (bits32 ^ 0x80000000).to(orig_dtype)


@libentry()
@triton.jit
def _radix_select_kernel(
    bits_ptr,
    out_ptr,
    k_val_ptr,
    N: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """Fused histogram-based radix select: find k-th smallest element per row.

    Uses tl.histogram with per-chunk accumulation. Each program handles one row.
    """
    pid = tle.program_id(0)
    row_start = bits_ptr + pid * N

    rank = tl.load(k_val_ptr + pid)
    prefix = tl.full([], dtype=tl.int32, value=0)

    for pass_idx in tl.static_range(4):
        shift = 24 - pass_idx * 8

        # Phase 1: Compute histogram across all chunks
        hist = tl.zeros([256], dtype=tl.int32)
        valid_zero_total = tl.full([], dtype=tl.int32, value=0)

        for chunk_start in range(0, N, CHUNK_SIZE):
            cols = tl.arange(0, CHUNK_SIZE)
            idx = chunk_start + cols
            mask = idx < N
            vals = tl.load(row_start + idx, mask=mask, other=0)

            # Check if element matches accumulated prefix
            # prefix stores bytes found so far in LSB-first order:
            # after pass P: byte[P-1]...(byte[1]<<8)|byte[0]
            # Extract bytes above current pass position and mask to avoid sign extension
            if pass_idx == 0:
                valid = mask
            else:
                check_shift = shift + 8
                check_mask = (1 << (pass_idx * 8)) - 1
                elem_prefix = (vals >> check_shift) & check_mask
                valid = mask & (elem_prefix == prefix)

            # Extract byte, zero invalid (they go to bin 0)
            byte_val = ((vals >> shift) & 0xFF).to(tl.int32)
            byte_val = tl.where(valid, byte_val, tl.zeros([], dtype=tl.int32))

            # Count valid elements with byte_val == 0 (for bin 0 correction)
            valid_zero_total += tl.sum((valid & (byte_val == 0)).to(tl.int32))

            # Accumulate histogram (includes padding in bin 0)
            hist += tl.histogram(byte_val, num_bins=256)

        # Correct bin 0: replace with actual count of valid elements with byte 0
        hist_corr = tl.where(
            tl.arange(0, 256) == 0,
            valid_zero_total,
            hist,
        )

        # Phase 2: Compute cumsum, find target bin
        cumsum = tl.cumsum(hist_corr, axis=0)

        # Find first bin where cumsum > rank (side="right")
        # Use argmax on comparison mask
        bin_idx = tl.argmax(cumsum > rank, axis=0)
        bin_idx = tl.minimum(bin_idx, 255)

        # Update rank: subtract elements in bins before target
        cols256b = tl.arange(0, 256)
        prev = tl.sum(tl.where(cols256b == bin_idx - 1, cumsum, 0))
        rank = rank - prev
        prefix = (prefix << 8) | bin_idx

    # Store raw sortable bits for host-side conversion
    tl.store(out_ptr + pid, prefix)


def _histogram_radix_select(inp_2d, k_val):
    """Find k-th smallest element per row using fused Triton histogram radix select.

    Uses a single Triton kernel with 4 passes of 8-bit histograms for 32-bit values.
    Falls back to torch.sort for int64 (64-bit values).
    """
    M, N = inp_2d.shape
    orig_dtype = inp_2d.dtype
    device = inp_2d.device

    # int64 needs 64-bit radix (8 passes) which isn't supported by the fused kernel
    if inp_2d.dtype == torch.int64:
        sorted_vals = torch.sort(inp_2d, dim=-1).values
        if isinstance(k_val, int):
            return sorted_vals[:, k_val]
        else:
            idx = k_val.unsqueeze(1).expand(-1, 1)
            return sorted_vals.gather(1, idx).squeeze(1)

    bits, num_bits = _to_sortable_bits(inp_2d)

    if isinstance(k_val, int):
        k_tensor = torch.full((M,), k_val, dtype=torch.int32, device=device)
    else:
        k_tensor = k_val.to(torch.int32)

    CHUNK_SIZE = 4096
    raw_bits = torch.empty(M, dtype=torch.int32, device=device)

    with torch_device_fn.device(device):
        _radix_select_kernel[(M,)](
            bits, raw_bits, k_tensor, N=N, CHUNK_SIZE=CHUNK_SIZE,
        )

    # Convert raw sortable bits back to original dtype
    return _bits_to_value(raw_bits, num_bits, orig_dtype)


def _median_fallback(inp, dim, keepdim):
    """Fallback for large N where tl.sort exceeds shared memory.

    Uses torch.kthvalue (O(N) selection) to find median values and indices.
    """
    if dim is None:
        flat = inp.contiguous().view(-1)
        n = flat.shape[0]
        if n == 0:
            return torch.tensor(float("nan"), dtype=inp.dtype, device=inp.device)
        if n == 1:
            return flat[0].to(inp.dtype)

        med_pos = n // 2 - 1 if n % 2 == 0 else (n - 1) // 2
        sorted_vals = torch.sort(flat).values
        return sorted_vals[med_pos].to(inp.dtype)

    dim = dim % inp.ndim
    n = inp.shape[dim]
    if n == 0:
        out_shape = list(inp.shape)
        if keepdim:
            out_shape[dim] = 1
        else:
            del out_shape[dim]
        return (
            torch.full(out_shape, float("nan"), dtype=inp.dtype, device=inp.device),
            torch.zeros(out_shape, dtype=torch.int64, device=inp.device),
        )

    # Move target dim to last (same pattern as sort-kernel path)
    inp = inp.contiguous()
    original_dim = dim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()

    M = inp.numel() // n
    pre_shape = list(inp.shape[:-1])
    inp_2d = inp.reshape(M, n)

    k = n // 2 if n % 2 == 0 else (n + 1) // 2
    out_val, out_idx = torch.kthvalue(inp_2d, k, dim=-1)

    out_val = out_val.reshape(pre_shape).to(inp.dtype)
    out_idx = out_idx.reshape(pre_shape)

    if keepdim:
        out_val = out_val.unsqueeze(-1)
        out_idx = out_idx.unsqueeze(-1)

    if original_dim != inp.ndim - 1:
        if keepdim:
            out_val = out_val.movedim(-1, original_dim).contiguous()
            out_idx = out_idx.movedim(-1, original_dim).contiguous()

    return out_val, out_idx


# Maximum N for which tl.sort fits in shared memory
# PADDED_N=8192 uses 32KB (fits in 48KB NVIDIA, 128KB Iluvatar)
_MAX_SORT_N = 2048


def median(inp, dim=None, keepdim=False):
    """Compute the median of a tensor along a given dimension.

    Follows PyTorch semantics:
        - For odd N: middle element
        - For even N: smaller of the two middle elements
        - dim=None: returns scalar tensor
        - dim specified: returns (values, indices) tuple

    Args:
        inp: Input tensor
        dim: Dimension to reduce (None for global median)
        keepdim: Whether to keep the reduced dimension

    Returns:
        Scalar tensor if dim=None, else (values, indices) tuple
    """
    logger.debug("GEMS MEDIAN")

    # Validate dtype
    assert inp.dtype in _SUPPORTED_DTYPES, (
        f"median: unsupported dtype {inp.dtype}. "
        f"Supported: float16, bfloat16, float32, int16, int32, int64"
    )

    # Cast int16 to int32 for processing (Triton doesn't natively sort int16)
    orig_dtype = inp.dtype
    if inp.dtype == torch.int16:
        inp = inp.to(torch.int32)

    is_float = inp.dtype.is_floating_point

    # Check if N exceeds shared memory limit for tl.sort
    if dim is not None:
        N = inp.shape[dim % inp.ndim]
    else:
        N = inp.numel()

    # For half-precision, tl.sort is slower than kthvalue for most N,
    # so use kthvalue fallback for N > 64
    sort_limit = 64 if inp.dtype in (torch.float16, torch.bfloat16) else _MAX_SORT_N
    if N > sort_limit:
        result = _median_fallback(inp, dim, keepdim)
        if dim is None:
            return result.to(orig_dtype)
        return result[0].to(orig_dtype), result[1]

    if dim is None:
        # Global median: flatten and compute single median
        inp_flat = inp.contiguous().view(-1)
        N = inp_flat.shape[0]
        if N == 0:
            return torch.tensor(float("nan"), dtype=orig_dtype, device=inp.device)
        if N == 1:
            return inp_flat[0].to(orig_dtype)

        PADDED_N = triton.next_power_of_2(N)
        IS_EVEN = N % 2 == 0
        MED_POS = N // 2 - 1 if IS_EVEN else (N - 1) // 2

        out_val = torch.empty(
            1, dtype=torch.float32 if is_float else inp.dtype, device=inp.device
        )
        out_idx = torch.empty(1, dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            if is_float:
                median_kernel_float[(1,)](
                    inp_flat,
                    out_val,
                    out_idx,
                    N=N,
                    PADDED_N=PADDED_N,
                    MED_POS=MED_POS,
                )
            else:
                median_kernel_int[(1,)](
                    inp_flat,
                    out_val,
                    out_idx,
                    N=N,
                    PADDED_N=PADDED_N,
                    MED_POS=MED_POS,
                    IS_INT64=(inp.dtype == torch.int64),
                )

        return out_val.squeeze().to(orig_dtype)
    else:
        # Dimension-wise median: returns (values, indices)
        assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
        dim = dim % inp.ndim
        N = inp.shape[dim]

        if N == 0:
            out_shape = list(inp.shape)
            if keepdim:
                out_shape[dim] = 1
            else:
                del out_shape[dim]
            return (
                torch.full(out_shape, float("nan"), dtype=orig_dtype, device=inp.device),
                torch.zeros(out_shape, dtype=torch.int64, device=inp.device),
            )

        if N == 1:
            out_val = inp.select(dim, 0).to(orig_dtype)
            out_idx = torch.zeros_like(out_val, dtype=torch.int64)
            if keepdim:
                out_val = out_val.unsqueeze(dim)
                out_idx = out_idx.unsqueeze(dim)
            return out_val, out_idx

        inp = inp.contiguous()
        original_dim = dim
        if dim != inp.ndim - 1:
            inp = torch.movedim(inp, dim, -1).contiguous()

        M = inp.numel() // N
        pre_shape = list(inp.shape[:-1])
        inp_2d = inp.reshape(M, N)

        PADDED_N = triton.next_power_of_2(N)
        IS_EVEN = N % 2 == 0
        MED_POS = N // 2 - 1 if IS_EVEN else (N - 1) // 2

        out_val = torch.empty(
            M, dtype=torch.float32 if is_float else inp.dtype, device=inp.device
        )
        out_idx = torch.empty(M, dtype=torch.int64, device=inp.device)

        with torch_device_fn.device(inp.device):
            if is_float:
                median_kernel_float[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N=N,
                    PADDED_N=PADDED_N,
                    MED_POS=MED_POS,
                )
            else:
                median_kernel_int[(M,)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N=N,
                    PADDED_N=PADDED_N,
                    MED_POS=MED_POS,
                    IS_INT64=(inp.dtype == torch.int64),
                )

        out_shape = pre_shape
        out_val = out_val.reshape(out_shape).to(orig_dtype)
        out_idx = out_idx.reshape(out_shape)

        if keepdim:
            out_val = out_val.unsqueeze(-1)
            out_idx = out_idx.unsqueeze(-1)

        if original_dim != inp.ndim - 1:
            if keepdim:
                out_val = out_val.movedim(-1, original_dim).contiguous()
                out_idx = out_idx.movedim(-1, original_dim).contiguous()

        return out_val, out_idx

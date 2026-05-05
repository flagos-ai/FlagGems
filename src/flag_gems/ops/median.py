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
    med_val = tl.where(any_pos_inf & (med_val >= 3.4028235e38), tl.full([], dtype=tl.float32, value=float('inf')), med_val)
    med_val = tl.where(any_neg_inf & (med_val <= -3.4028235e38), tl.full([], dtype=tl.float32, value=float('-inf')), med_val)

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


def _median_fallback(inp, dim, keepdim):
    """Fallback for large N where tl.sort exceeds shared memory.

    Implements median using torch.sort + indexing to avoid calling torch.median
    (which would be overridden by flag_gems and cause recursion).
    """
    if dim is None:
        flat = inp.contiguous().view(-1)
        n = flat.shape[0]
        if n == 0:
            return torch.tensor(float("nan"), dtype=inp.dtype, device=inp.device)
        sorted_vals, sorted_idx = torch.sort(flat)
        med_pos = n // 2 - 1 if n % 2 == 0 else (n - 1) // 2
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

    sorted_vals, sorted_idx = torch.sort(inp, dim=dim)
    is_even = n % 2 == 0
    med_pos = n // 2 - 1 if is_even else (n - 1) // 2

    out_val = sorted_vals.select(dim, med_pos)
    out_idx = sorted_idx.select(dim, med_pos)

    if keepdim:
        out_val = out_val.unsqueeze(dim)
        out_idx = out_idx.unsqueeze(dim)

    return out_val, out_idx


# Maximum N for which tl.sort fits in shared memory (128KB limit on Iluvatar BI-V150)
# PADDED_N=4096 uses ~256KB, PADDED_N=2048 uses ~128KB
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

    if N > _MAX_SORT_N:
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

        out_shape = list(inp.shape[:-1])
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

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

"""Kunlunxin XPU3 ``nanquantile`` override.

The generic implementation lowers the small-reduction path
(``reduction_size <= 2048``) to an in-register bitonic ``tl.sort`` and the
large path to ``flag_gems.ops.sort.sort_stable``; both fail to compile here
("no in-register permutation/sort primitive", see wiki pattern
``xpu3-no-inregister-sort-primitive``).  This override keeps the same
algorithmic structure but uses the proven backend radix sorter
(``_kunlunxin.ops.sort.sort_stable``) plus a NaN-aware gather:

1. canonicalize NaNs (``NaN -> float('nan')`` so they sort last),
2. radix sort each row ascending with the backend sorter,
3. binary-search the first NaN per row to get the valid count,
4. gather/interpolate the requested ranks.

No ``tl.sort``/``tl.flip``/``reshape()+reduce`` and no ATen/CPU compute
fallback is used.
"""

import logging

import torch
import triton
import triton.language as tl
from torch import Tensor

from flag_gems.ops.contiguous import contiguous
from flag_gems.ops.copy import copy_ as _gems_copy_
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

from .sort import sort_stable

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = ("linear", "lower", "higher", "nearest", "midpoint")
INTERPOLATION_CODES = {name: code for code, name in enumerate(INTERPOLATION_METHODS)}
MAX_REDUCTION_SIZE = 1 << 24

# Order-statistic selection fast path (single reduction row).  Measured on
# XPU3/P800: a per-row kernel that keeps the row resident and walks a *runtime*
# binary-search loop costs ~25 us for 32 steps at grid=1, but the very same
# loop at grid>=128 spends ~1 ms (the resident tile is spilled once the loop is
# dynamic and many programs contend).  So the selection path is enabled only
# when the reduction is a single row and the row fits one tile; everything else
# keeps the radix-sort path.
SELECT_MAX_REDUCTION_SIZE = 4096
SELECT_MAX_Q = 16
# Canonical NaN key in the signed-comparable key space below (0xFFC00000
# unsigned).  Every finite/inf value maps strictly below it.
_NAN_SKEY = tl.constexpr(0x7FC00000)
_MIN_SKEY = tl.constexpr(-(1 << 31))
_BIG_SKEY = tl.constexpr(1 << 62)
SEARCH_STEPS = 32


@triton.jit
def _skey_to_value(k):
    """Invert the monotone key back to the original float value."""
    key_u = (k & 0xFFFFFFFF) ^ 0x80000000
    neg = key_u < 0x80000000
    bits = tl.where(neg, key_u ^ 0xFFFFFFFF, key_u ^ 0x80000000)
    return (bits & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)


@libentry()
@triton.jit
def _nq_select_row_kernel(
    inp,
    q,
    out,
    q_scalar,
    reduction_size,
    q_size,
    n_steps,
    BLOCK_R: tl.constexpr,
    Q_SCALAR: tl.constexpr,
    interpolation: tl.constexpr,
):
    """Exact order-statistic selection for a single reduction row.

    One program per ``q`` element (or a single program when ``q`` is a Python
    scalar, passed by value so the host pays no device upload).  The row is
    loaded once into a clamped (tail re-reads the last element, gated by
    ``inb``) tile, NaNs are folded to the canonical key, and the floor/ceil
    ranks are resolved with a runtime binary search over the monotone integer
    key plus one successor reduction.  NaNs sort above every numeric key, so
    the valid count is a single ``skey < NAN`` reduction and the search bound
    excludes them for free.
    """
    if Q_SCALAR:
        qi = 0
        qv = q_scalar
    else:
        qi = tl.program_id(0)
        qv = tl.load(q + qi)
    offs = tl.arange(0, BLOCK_R)
    inb = offs < reduction_size
    off = tl.minimum(offs, reduction_size - 1)
    v = tl.load(inp + off)
    v = tl.where(v == v, v, float("nan"))
    bits = v.to(tl.int32, bitcast=True)
    key_u = bits ^ ((bits >> 31) | (-2147483648))
    skey = (key_u ^ (-2147483648)).to(tl.int64)

    cnt = tl.sum(tl.where(inb & (skey < _NAN_SKEY), 1, 0), axis=0)
    all_nan = cnt == 0
    cnt64 = tl.maximum(cnt, 1).to(tl.int64)
    last = cnt64 - 1
    qv = qv.to(tl.float32)
    ranks = last.to(tl.float32) * qv
    k_low = tl.floor(ranks).to(tl.int64)
    k_high = tl.ceil(ranks).to(tl.int64)
    k_low = tl.minimum(tl.maximum(k_low, 0), last)
    k_high = tl.minimum(tl.maximum(k_high, 0), last)

    # Smallest key K with count(skey <= K) > k_low  ==  the k_low-th value.
    lo = tl.full((), _MIN_SKEY, tl.int64)
    hi = tl.full((), _NAN_SKEY - 1, tl.int64)
    for _ in range(0, n_steps):
        mid = lo + (hi - lo) // 2
        c = tl.sum(tl.where(inb & (skey <= mid), 1, 0), axis=0).to(tl.int64)
        take = c > k_low
        lo = tl.where(take, lo, mid + 1)
        hi = tl.where(take, mid, hi)
    k_lo = lo

    # Rank ``k_high`` is either ``k_low`` or ``k_low + 1`` (ceil of a rank).
    # With ties the (k_low + 1)-th order statistic can still be the *same*
    # value, so it only advances to the next distinct key when k_low was the
    # last index of its equal group (k_high >= count(key <= k_lo)).
    cnt_le = tl.sum(tl.where(inb & (skey <= k_lo), 1, 0), axis=0).to(tl.int64)
    succ = tl.min(
        tl.where(inb & (skey > k_lo) & (skey < _NAN_SKEY), skey, _BIG_SKEY),
        axis=0,
    )
    same = k_high < cnt_le
    k_hi = tl.where(same, k_lo, succ)

    lower_values = _skey_to_value(k_lo)
    upper_values = _skey_to_value(k_hi)

    if interpolation == 0:
        result = lower_values + (upper_values - lower_values) * (
            ranks - k_low.to(tl.float32)
        )
    elif interpolation == 1:
        result = lower_values
    elif interpolation == 2:
        result = upper_values
    elif interpolation == 3:
        rounded = tl.extra.xpu.libdevice.rint(ranks).to(tl.int64)
        result = tl.where(rounded == k_high, upper_values, lower_values)
    else:
        result = lower_values + (upper_values - lower_values) * 0.5

    result = tl.where(all_nan, float("nan"), result)
    tl.store(out + qi, result)


@libentry()
@triton.jit
def _canonicalize_nan_kernel(inp, out, n_elements, BLOCK_SIZE: tl.constexpr):
    # Addresses are clamped so no masked load/store is needed: the trailing
    # lanes of the last block re-read/re-write the final element with its own
    # value (masked memory paths are unreliable in this backend's codegen).
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = tl.minimum(offsets, n_elements - 1)
    values = tl.load(inp + off)
    values = tl.where(values == values, values, float("nan"))
    tl.store(out + off, values)


@libentry()
@triton.jit
def _validate_q_kernel(q, q_size, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off = tl.minimum(offsets, q_size - 1)
    values = tl.load(q + off)
    inb = offsets < q_size
    valid = (~inb) | ((values >= 0.0) & (values <= 1.0))
    tl.device_assert(valid, "quantile() q values must be in the range [0, 1]")


@libentry()
@triton.jit
def _nanquantile_gather_kernel(
    sorted_inp,
    q,
    out,
    n_rows,
    reduction_size,
    q_size,
    n_steps,
    BLOCK_Q: tl.constexpr,
    interpolation: tl.constexpr,
):
    # One program per row.  q tail lanes are clamped to the last q element
    # (they redundantly write the same value back), so the kernel needs no
    # masked memory access at all.
    row = ext.program_id(1)
    q_offsets = ext.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_idx = tl.minimum(q_offsets, q_size - 1)
    compute_type = sorted_inp.dtype.element_ty
    row_start = row * reduction_size

    # Canonicalized NaNs sort after every numeric value, so the number of
    # valid entries in a row is the index of the first NaN.  Find it with a
    # binary search over the clamped, always in-bounds row.
    low = tl.zeros((1,), dtype=tl.int64)
    high = tl.full((1,), reduction_size, dtype=tl.int64)
    for _ in range(0, n_steps):
        middle = (low + high) // 2
        inb = middle < high
        probe = tl.load(sorted_inp + row_start + tl.minimum(middle, reduction_size - 1))
        probe_valid = inb & (probe == probe)
        low = tl.where(probe_valid, middle + 1, low)
        high = tl.where(probe_valid, high, middle)
    counts_value = low
    all_nan = counts_value == 0
    last_valid = tl.maximum(counts_value - 1, 0).to(compute_type)
    q_values = tl.load(q + q_idx).to(compute_type)
    ranks = last_valid * q_values
    lower = tl.floor(ranks).to(tl.int64)
    upper = tl.ceil(ranks).to(tl.int64)
    lower_values = tl.load(sorted_inp + row_start + lower)
    upper_values = tl.load(sorted_inp + row_start + upper)

    if interpolation == 0:
        result = lower_values + (upper_values - lower_values) * (ranks - lower)
    elif interpolation == 1:
        result = lower_values
    elif interpolation == 2:
        result = upper_values
    elif interpolation == 3:
        rounded = tl.extra.xpu.libdevice.rint(ranks).to(tl.int64)
        result = tl.where(rounded == upper, upper_values, lower_values)
    else:
        result = lower_values + (upper_values - lower_values) * 0.5

    result = tl.where(all_nan, float("nan"), result)
    tl.store(out + row * q_size + q_idx, result)


def _gather_blocks(q_size):
    return min(triton.next_power_of_2(q_size), 16)


def _validate_args(inp, q, dim, interpolation, out):
    if inp.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(
            "quantile() input tensor must be either float or double dtype"
        )
    if inp.numel() == 0:
        raise RuntimeError("quantile() input tensor must be non-empty")
    if interpolation not in INTERPOLATION_METHODS:
        raise RuntimeError(
            "quantile() interpolation must be one of linear, lower, higher, "
            f"midpoint or nearest, but got {interpolation}"
        )
    if dim is not None and not isinstance(dim, int):
        raise TypeError("quantile() dim must be an integer or None")
    if isinstance(q, torch.Tensor):
        if q.dim() > 1:
            raise RuntimeError("quantile() q must be a scalar or 1D tensor")
        if q.dtype != inp.dtype:
            raise RuntimeError(
                "quantile() q tensor must be same dtype as the input tensor"
            )
        if q.device != inp.device:
            raise RuntimeError(
                "quantile() q tensor must be on the same device as the input"
            )
    if out is not None:
        if out.dtype != inp.dtype:
            raise RuntimeError(
                "quantile() out tensor must be same dtype as the input tensor"
            )
        if out.device != inp.device:
            raise RuntimeError(
                "quantile() out tensor must be on the same device as the input"
            )


def _nanquantile_impl(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS_KUNLUNXIN NANQUANTILE")
    _validate_args(inp, q, dim, interpolation, out)

    original_ndim = inp.ndim
    dim_was_none = dim is None
    if dim_was_none:
        reduced = contiguous(inp).view(-1)
        dim = 0
    else:
        if inp.ndim == 0:
            if dim not in (-1, 0):
                raise IndexError("Dimension out of range")
            dim = 0
            reduced = inp.reshape(1)
        else:
            reduced = torch.movedim(inp, dim, -1)  # allow-torch-compute: view, no device compute  # fmt: skip
            dim %= inp.ndim
            reduced = contiguous(reduced)

    q_is_tensor = isinstance(q, torch.Tensor)
    q_is_scalar = not q_is_tensor or q.dim() == 0
    # A Python-scalar q is kept on the host so the selection path can pass it
    # by value (a per-call ``torch.tensor(..., device=)`` upload costs ~70 us
    # on this backend, comparable to the whole kernel).
    q_scalar_value = None
    if not q_is_tensor:
        q_scalar_value = float(q)
        if not 0.0 <= q_scalar_value <= 1.0:
            raise RuntimeError(
                f"quantile() q must be in the range [0, 1] but got {q_scalar_value}"
            )
        q_size = 1
        q_contiguous = None
    else:
        q_contiguous = contiguous(q).reshape(-1)
        q_size = q_contiguous.numel()
    if q_is_tensor and q_contiguous.numel():
        block_size = 256
        with torch_device_fn.device(inp.device):
            _validate_q_kernel[(triton.cdiv(q_contiguous.numel(), block_size),)](
                q_contiguous,
                q_contiguous.numel(),
                BLOCK_SIZE=block_size,
            )

    reduction_size = reduced.size(-1)
    if reduction_size > MAX_REDUCTION_SIZE:
        raise RuntimeError(
            "quantile() input tensor is too large; the reduction dimension "
            f"must not exceed {MAX_REDUCTION_SIZE} elements"
        )
    n_rows = reduced.numel() // reduction_size
    internal_shape = (*reduced.shape[:-1], q_size)
    if q_is_scalar:
        result_shape = list(reduced.shape[:-1])
        if keepdim:
            if dim_was_none:
                result_shape = [1] * original_ndim
            else:
                result_shape.insert(dim, 1)
    else:
        result_shape = [q_size, *reduced.shape[:-1]]
        if keepdim:
            if dim_was_none:
                result_shape = [q_size, *([1] * original_ndim)]
            else:
                result_shape.insert(dim + 1, 1)
    direct_out = False
    if out is not None and q_is_scalar:
        out.resize_(result_shape)
        out_view = out.reshape(reduced.shape[:-1])
        if out_view.is_contiguous():
            internal = out_view.unsqueeze(-1)
            direct_out = True
        else:
            internal = torch.empty(internal_shape, dtype=inp.dtype, device=inp.device)
    elif out is not None and n_rows == 1:
        out.resize_(result_shape)
        if out.is_contiguous():
            internal = out.reshape(internal_shape)
            direct_out = True
        else:
            internal = torch.empty(internal_shape, dtype=inp.dtype, device=inp.device)
    else:
        internal = torch.empty(internal_shape, dtype=inp.dtype, device=inp.device)

    if q_size and reduction_size:
        use_select = (
            reduced.ndim == 1
            and reduced.dtype == torch.float32
            and reduction_size <= SELECT_MAX_REDUCTION_SIZE
            and q_size <= SELECT_MAX_Q
        )
        if not use_select and q_contiguous is None:
            q_contiguous = contiguous(
                torch.tensor(q_scalar_value, dtype=inp.dtype, device=inp.device)
            ).reshape(-1)
        with torch_device_fn.device(inp.device):
            if use_select:
                block_r = triton.next_power_of_2(max(reduction_size, 2))
                q_scalar_arg = 0.0 if q_scalar_value is None else q_scalar_value
                _nq_select_row_kernel[(1 if q_is_scalar else q_size,)](
                    reduced,
                    reduced if q_scalar_value is not None else q_contiguous,
                    internal,
                    q_scalar_arg,
                    reduction_size,
                    q_size,
                    SEARCH_STEPS,
                    BLOCK_R=block_r,
                    Q_SCALAR=q_scalar_value is not None,
                    interpolation=INTERPOLATION_CODES[interpolation],
                )
            else:
                canonical = torch.empty_like(reduced)
                block = 1024
                _canonicalize_nan_kernel[(triton.cdiv(reduced.numel(), block),)](
                    reduced, canonical, reduced.numel(), BLOCK_SIZE=block
                )
                sorted_values, _ = sort_stable(
                    canonical, stable=False, dim=-1, descending=False
                )
                block_q = _gather_blocks(q_size)
                _nanquantile_gather_kernel[(triton.cdiv(q_size, block_q), n_rows)](
                    sorted_values,
                    q_contiguous,
                    internal,
                    n_rows,
                    reduction_size,
                    q_size,
                    max(1, int(reduction_size).bit_length()),
                    BLOCK_Q=block_q,
                    interpolation=INTERPOLATION_CODES[interpolation],
                )

    if q_is_scalar:
        result = internal.squeeze(-1)
    else:
        result = internal.movedim(-1, 0)
    if keepdim:
        result = result.reshape(result_shape)

    if out is not None:
        if direct_out:
            return out
        out.resize_(result.shape)
        _gems_copy_(out, result)
        return out
    return result


def nanquantile(inp, q, dim=None, keepdim=False, interpolation="linear") -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation)


def nanquantile_scalar(
    inp, q, dim=None, keepdim=False, interpolation="linear"
) -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation)


def nanquantile_out(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation, out)


def nanquantile_scalar_out(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    return _nanquantile_impl(inp, q, dim, keepdim, interpolation, out)

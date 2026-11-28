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
from torch import Tensor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, tl_extra_shim
from flag_gems.utils import triton_lang_extension as ext

from . import sort_mradix as _sort_mradix
from .contiguous import contiguous
from .copy import copy_
from .sort import sort_stable

logger = logging.getLogger(__name__)

INTERPOLATION_METHODS = ("linear", "lower", "higher", "nearest", "midpoint")
INTERPOLATION_CODES = {name: code for code, name in enumerate(INTERPOLATION_METHODS)}

# gcu400 hardware grid limits: grid.x <= 65535, grid.y <= 255.
_GRID_MAX = 65535
_GRID_Y_MAX = 255
# Gather row-block bound: measured best across 64..80000-row cases on GCU400
# (256-row blocks lose ~2.7x vs 512 for >64K rows; larger only gets worse).
# Grown past the cap only when n_rows would overflow grid.x.
_GATHER_BLOCK_N_CAP = 512
_GATHER_BLOCK_N_MAX = 2048
_GATHER_BLOCK_Q_CAP = 16


@libentry()
@triton.jit
def _quantile_gather_kernel(
    sorted_inp,
    q,
    out,
    n_rows,
    reduction_size,
    q_size,
    q_scalar,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    Q_SCALAR: tl.constexpr,
    interpolation: tl.constexpr,
):
    row_offsets = ext.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_row = row_offsets < n_rows
    q_offsets = ext.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    valid_q = q_offsets < q_size
    ctype = sorted_inp.dtype.element_ty

    if Q_SCALAR:
        q_vals = tl.zeros((BLOCK_Q,), dtype=tl.float32) + q_scalar
    else:
        q_vals = tl.load(q + q_offsets, mask=valid_q, other=0.0).to(tl.float32)
    # Out-of-range or NaN q would index past the row end (device-abort risk);
    # clamp instead of a host sync (torch.all + .item()) or a device assert,
    # both of which cost a kernel launch or an RAS-prone abort on this backend.
    q_vals = tl.where(q_vals == q_vals, tl.minimum(tl.maximum(q_vals, 0.0), 1.0), 0.0)
    ranks = q_vals * (reduction_size - 1)
    lower = tl.floor(ranks).to(tl.int32)
    upper = tl.ceil(ranks).to(tl.int32)
    # fp32 rounding of q * (M - 1) can push ceil one past the last column for
    # M > 2**24; clamp the upper index to keep the gather in bounds.
    upper = tl.minimum(upper, reduction_size - 1)

    row_base = row_offsets.to(tl.int64)[:, None] * reduction_size
    mask = valid_row[:, None] & valid_q[None, :]
    lower_vals = tl.load(sorted_inp + row_base + lower[None, :], mask=mask, other=0.0)
    upper_vals = tl.load(sorted_inp + row_base + upper[None, :], mask=mask, other=0.0)
    lower_f = lower_vals.to(tl.float32)
    upper_f = upper_vals.to(tl.float32)

    if interpolation == 0:
        # Deliberately unguarded: torch's own linear interpolation computes
        # lower + (upper - lower) * frac, which yields NaN for infinite
        # windows even at integral ranks; guard-free matches it exactly.
        result = lower_f + (upper_f - lower_f) * (
            ranks[None, :] - lower.to(tl.float32)[None, :]
        )
    elif interpolation == 1:
        result = lower_f
    elif interpolation == 2:
        result = upper_f
    elif interpolation == 3:
        rounded = tl_extra_shim.rint(ranks).to(tl.int32)
        result = tl.where(rounded[None, :] == upper[None, :], upper_f, lower_f)
    else:
        # Lerp form on purpose: torch's midpoint yields NaN for infinite
        # windows (a + (b - a) * 0.5), while (a + b) / 2 would give inf.
        result = lower_f + (upper_f - lower_f) * 0.5

    out_ptrs = out + row_offsets.to(tl.int64)[:, None] * q_size + q_offsets[None, :]
    tl.store(out_ptrs, result.to(ctype), mask=mask)


def _sort_rows(rows2d):
    """Ascending last-dim sort of a contiguous 2-D (n_rows, M) view.

    Goes through the hardware row-radix when the row fits its envelope and
    skips the int64 index widening torch.sort needs but quantile does not.
    Rows of width 1 are already sorted.
    """
    if rows2d.size(-1) == 1:
        return rows2d
    if _sort_mradix.sort_mradix_supported(rows2d):
        sorted_vals, _ = _sort_mradix.sort_mradix(
            rows2d, tuple(rows2d.shape), descending=False
        )
        return sorted_vals
    sorted_vals, _ = sort_stable(rows2d, stable=False, dim=-1, descending=False)
    return sorted_vals


def _gather_launch(
    sorted_vals, q, output, n_rows, M, Q, q_value, interpolation, device
):
    block_n = min(_GATHER_BLOCK_N_CAP, triton.next_power_of_2(max(n_rows, 1)))
    min_needed = triton.cdiv(max(n_rows, 1), _GRID_MAX)
    if block_n < min_needed:
        block_n = min(_GATHER_BLOCK_N_MAX, triton.next_power_of_2(min_needed))
    block_q = min(_GATHER_BLOCK_Q_CAP, triton.next_power_of_2(max(Q, 1)))
    q_blocks_needed = triton.cdiv(max(Q, 1), _GRID_Y_MAX)
    if block_q < q_blocks_needed:
        block_q = triton.next_power_of_2(q_blocks_needed)
    # grid.x = row blocks (<= 65535 by block_n sizing), grid.y = q blocks
    # (block_q grows to keep it <= 255 for any Q).
    grid = (triton.cdiv(max(n_rows, 1), block_n), triton.cdiv(max(Q, 1), block_q))
    with torch_device_fn.device(device):
        _quantile_gather_kernel[grid](
            sorted_vals,
            q,
            output,
            n_rows,
            M,
            Q,
            0.0 if q_value is None else q_value,
            BLOCK_N=block_n,
            BLOCK_Q=block_q,
            Q_SCALAR=q_value is not None,
            interpolation=interpolation,
            num_warps=4,
        )


def quantile(
    inp, q, dim=None, keepdim=False, interpolation="linear", out=None
) -> Tensor:
    logger.debug("GEMS_ENFLAME QUANTILE")
    assert torch.is_floating_point(inp)
    assert dim is None or isinstance(dim, int)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHODS

    if inp.dtype == torch.float64:
        raise RuntimeError("quantile() float64 input is not supported on this device")

    # Handle dim
    original_ndim = inp.ndim
    dim_was_none = dim is None
    if inp.ndim == 0:
        inp = inp.reshape(1)
        dim = 0
    if dim is None:
        inp = inp.ravel()
        dim = 0
    if dim < 0:
        dim = dim + inp.ndim

    # Handle q
    q_value = None
    q_all_ones = False
    q_all_zeros = False
    if isinstance(q, torch.Tensor):
        if q.dim() > 1:
            raise RuntimeError("quantile() q must be a scalar or 1D tensor")
        # Only move/cast q when it does not already live where the gather reads it: an
        # identity .to() still costs a dispatch on every call.
        if q.device != inp.device or q.dtype != inp.dtype:
            q = q.to(device=inp.device, dtype=inp.dtype)
        Q = 1 if q.numel() == 1 else len(q)
    else:
        q_value = float(q)
        q_all_ones = q_value == 1.0
        q_all_zeros = q_value == 0.0
        if not 0.0 <= q_value <= 1.0:
            raise RuntimeError(
                f"quantile() q must be in the range [0, 1] but got {q_value}"
            )
        Q = 1

    # Fast path: q == 0.0 -> min, q == 1.0 -> max (no sort needed)
    if q_all_ones or q_all_zeros:
        reduce_fn = torch.amax if q_all_ones else torch.amin
        if out is not None and Q == 1:
            reduce_fn(inp, dim=dim, keepdim=keepdim, out=out)
            return out
        output = reduce_fn(inp, dim=dim, keepdim=keepdim)
        if Q > 1:
            output = output.unsqueeze(0).expand(Q, *output.shape)
        if out is not None:
            copy_(out, output)
            return out
        return output

    # handle input tensor
    if dim != inp.ndim - 1:
        reduced = contiguous(torch.movedim(inp, dim, -1))
    else:
        reduced = contiguous(inp)

    M = reduced.size(-1)
    N = reduced.numel() // M

    output = torch.empty(reduced.shape[:-1] + (Q,), dtype=inp.dtype, device=inp.device)

    if N > 0 and M > 0 and Q > 0:
        sorted_vals = _sort_rows(reduced.view(N, M))
        _gather_launch(
            sorted_vals,
            q if q_value is None else reduced,
            output,
            N,
            M,
            Q,
            q_value,
            INTERPOLATION_CODES[interpolation],
            inp.device,
        )

    if Q == 1:
        output = output.squeeze(-1)
    else:
        output = output.movedim(-1, 0)
    if keepdim:
        if dim_was_none:
            # dim=None reduced every original dim, so keepdim keeps them all.
            if Q == 1:
                output = output.reshape([1] * original_ndim)
            else:
                output = output.reshape([Q] + [1] * original_ndim)
        else:
            output = output.unsqueeze(dim + (1 if Q != 1 else 0))

    if out is not None:
        copy_(out, output)
    return output

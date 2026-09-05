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

import torch
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

import triton
import triton.language as tl

logger = logging.getLogger(__name__)

INTERPOLATION_METHOD = ["linear", "lower", "higher", "nearest", "midpoint"]

# Resident selection covers a whole reduction slice in registers (value-only sort,
# no indices). Above this width the per-program register footprint of the sorted
# tile outweighs the benefit and the sort-based fallback wins.
RESIDENT_M_LIMIT = 2048
# Target program count for the resident kernel: enough tiles to fill the device
# without making each tile's sort wider than needed (measured optimum band).
RESIDENT_TARGET_PROGRAMS = 32768
RESIDENT_TILE_N_CAP = 16


def _pick_tile(block_m):
    """Rows-per-program for the resident kernel (measured optimum).

    Along inner lanes larger tiles trade program count for per-program sort
    width; ~32K programs is the sweet spot on S5000, capped at 16 lanes.
    Consecutive-row tiling (inner == 1) keeps 2 rows per program.
    """
    return 2


@triton.jit
def _quantile_ranks(
    q_ptr,
    M,
    Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    interpolation: tl.constexpr,
):
    # aten rank math (fp32 product, fp32 floor/ceil):
    #   p = q * (M-1); lower = floor(p)
    #   higher/midpoint: upper = ceil(p)  (equal to lower at integer ranks)
    #   linear:         upper = min(lower + 1, M-1)  (the +1 stays even at t=0)
    qoffs = tl.arange(0, BLOCK_Q)
    qmask = qoffs < Q
    qv = tl.load(q_ptr + qoffs, mask=qmask, other=0.0)
    p = qv * (M - 1)
    q_lower = tl.floor(p).to(tl.int32)
    if interpolation == "higher" or interpolation == "midpoint":
        q_upper = tl.ceil(p).to(tl.int32)
    else:
        q_upper = tl.minimum(q_lower + 1, M - 1)
    t = p - q_lower
    return qoffs, qmask, q_lower, q_upper, t


@triton.jit
def _quantile_interpolate(
    lower_vals,
    upper_vals,
    t2,
    ql2,
    interpolation: tl.constexpr,
):
    # aten lerp semantics (bit-exact, verified against reference):
    #   d = fp32(upper - lower)
    #   t <  0.5: a + t*d        (products/sums in fp64, single final round)
    #   t >= 0.5: b - (1-t)*d
    #   midpoint: b - 0.5*d      (the t=0.5 branch)
    #   nearest : rint(p) ties-to-even on the rank
    if interpolation == "linear":
        d = (upper_vals - lower_vals).to(tl.float64)
        t64 = t2.to(tl.float64)
        a64 = lower_vals.to(tl.float64)
        b64 = upper_vals.to(tl.float64)
        outv = tl.where(
            t2 < 0.5,
            (a64 + t64 * d).to(tl.float32),
            (b64 - (1.0 - t64) * d).to(tl.float32),
        )
    elif interpolation == "lower":
        outv = lower_vals
    elif interpolation == "higher":
        outv = upper_vals
    elif interpolation == "nearest":
        lower_even = (ql2 % 2) == 0
        pick_upper = (t2 > 0.5) | ((t2 == 0.5) & (~lower_even))
        outv = tl.where(pick_upper, upper_vals, lower_vals)
    else:  # midpoint
        d = (upper_vals - lower_vals).to(tl.float64)
        b64 = upper_vals.to(tl.float64)
        outv = (b64 - 0.5 * d).to(tl.float32)
    return outv


@libentry()
@triton.jit
def quantile_resident_kernel(
    inp,
    q_ptr,
    out,
    M,
    inner,
    Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_N: tl.constexpr,
    interpolation: tl.constexpr,
    INNER_TILE: tl.constexpr,
):
    # One program covers TILE_N reduction slices read directly from the native
    # [outer, M, inner] layout (strided loads, no materialization), value-only
    # sorts each slice in registers, and extracts the per-q order statistics.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_M)
    valid = cols < M

    if INNER_TILE:
        lanes = tl.arange(0, TILE_N)
        tiles_per_row = tl.cdiv(inner, TILE_N)
        outer_id = pid // tiles_per_row
        tile_id = pid % tiles_per_row
        lane_ids = tile_id * TILE_N + lanes
        lane_mask = lane_ids < inner
        ptrs = outer_id * (M * inner) + cols[None, :] * inner + lane_ids[:, None]
        out_off = (outer_id * inner + lane_ids) * Q
        out_mask = lane_mask
    else:
        lanes = tl.arange(0, TILE_N)
        rows = pid * TILE_N + lanes
        lane_mask = rows < inner  # `inner` carries the total slice count here
        ptrs = rows[:, None] * M + cols[None, :]
        out_off = rows * Q
        out_mask = lane_mask

    m2 = valid[None, :] & lane_mask[:, None]
    row = tl.load(inp + ptrs, mask=m2, other=float("inf"))

    # NaN in a slice -> NaN result for that slice; NaNs are sorted to the tail
    # as +inf and the output is overridden below.
    nan_mask = m2 & (row != row)
    has_nan = tl.max(nan_mask.to(tl.int32), axis=1) > 0
    sortable = tl.where(nan_mask, float("inf"), row)
    ordered = tl.sort(sortable, dim=1, descending=False)
    # Zero the padding tail: masked lanes hold +inf and inf*0 = nan in the
    # one-hot extraction below.
    ordered = tl.where(cols[None, :] < M, ordered, 0.0)

    qoffs, qmask, q_lower, q_upper, t = _quantile_ranks(
        q_ptr, M, Q, BLOCK_Q, interpolation
    )

    # one-hot extraction: lower/upper order statistics per q
    # select via where (not multiply) — data may contain +-inf and inf*0 = nan
    oh_l = cols[None, :] == q_lower[:, None]
    oh_u = cols[None, :] == q_upper[:, None]
    ord3 = tl.reshape(ordered, (TILE_N, 1, BLOCK_M))
    lower_vals = tl.sum(
        tl.where(tl.reshape(oh_l, (1, BLOCK_Q, BLOCK_M)), ord3, 0.0), axis=2
    )
    upper_vals = tl.sum(
        tl.where(tl.reshape(oh_u, (1, BLOCK_Q, BLOCK_M)), ord3, 0.0), axis=2
    )

    t2 = tl.broadcast_to(t[None, :], (TILE_N, BLOCK_Q))
    ql2 = tl.broadcast_to(q_lower[None, :], (TILE_N, BLOCK_Q))
    outv = _quantile_interpolate(lower_vals, upper_vals, t2, ql2, interpolation)

    outv = tl.where(tl.reshape(has_nan, (TILE_N, 1)), float("nan"), outv)
    st_mask = qmask[None, :] & out_mask[:, None]
    tl.store(out + out_off[:, None] + qoffs[None, :], outv, mask=st_mask)


@libentry()
@triton.jit
def quantile_gather_kernel(
    sorted_inp,
    q_ptr,
    out,
    M,
    inner,
    Q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_N: tl.constexpr,
    interpolation: tl.constexpr,
    INNER_TILE: tl.constexpr,
):
    # Post-sort gather/interpolation over rows already sorted along the last dim.
    pid = tl.program_id(0)
    qoffs, qmask, q_lower, q_upper, t = _quantile_ranks(
        q_ptr, M, Q, BLOCK_Q, interpolation
    )

    lanes = tl.arange(0, TILE_N)
    if INNER_TILE:
        tiles_per_row = tl.cdiv(inner, TILE_N)
        outer_id = pid // tiles_per_row
        tile_id = pid % tiles_per_row
        lane_ids = tile_id * TILE_N + lanes
        lane_mask = lane_ids < inner
        base = sorted_inp + outer_id * (M * inner) + lane_ids[:, None]
        out_off = (outer_id * inner + lane_ids) * Q
        row_stride = inner
    else:
        rows = pid * TILE_N + lanes
        lane_mask = rows < inner
        base = sorted_inp + rows[:, None] * M
        out_off = rows * Q
        row_stride = 1

    gmask = lane_mask[:, None] & qmask[None, :]
    lower_vals = tl.load(base + q_lower[None, :] * row_stride, mask=gmask, other=0.0)
    upper_vals = tl.load(base + q_upper[None, :] * row_stride, mask=gmask, other=0.0)

    t2 = tl.broadcast_to(t[None, :], (TILE_N, BLOCK_Q))
    ql2 = tl.broadcast_to(q_lower[None, :], (TILE_N, BLOCK_Q))
    outv = _quantile_interpolate(lower_vals, upper_vals, t2, ql2, interpolation)
    tl.store(out + out_off[:, None] + qoffs[None, :], outv, mask=gmask)


@libentry()
@triton.jit
def quantile_q_validate_kernel(
    q_ptr,
    status_ptr,
    Q,
    BLOCK_Q: tl.constexpr,
):
    # Single fused validity check: 0/1 status word, one launch, one host read.
    offs = tl.arange(0, BLOCK_Q)
    mask = offs < Q
    qv = tl.load(q_ptr + offs, mask=mask, other=0.0)
    bad = (qv < 0.0) | (qv > 1.0) | (qv != qv)
    any_bad = tl.max(bad.to(tl.int32), axis=0)
    tl.store(status_ptr, any_bad)


def _native_sort_rows(rows):
    """Sort rows along the last dim via the native out= overload.

    The python-registered flag_gems sort override covers torch.sort's functional
    form but not its out= form; the out= form reaches the native mudnn radix
    sort, which is an order of magnitude faster than the python radix sort on
    this stack. (Same trick as _mthreads/ops/unique.py.)
    """
    values = torch.empty_like(rows)
    indices = torch.empty(rows.shape, dtype=torch.int64, device=rows.device)
    torch.sort(rows, dim=-1, out=(values, indices))
    return values


def quantile(inp, q, dim=None, keepdim=False, interpolation="linear", out=None):
    logger.debug("GEMS_MTHREADS QUANTILE")
    assert torch.is_floating_point(inp)
    assert dim is None or isinstance(dim, int)
    assert isinstance(q, (float, torch.Tensor))
    assert interpolation in INTERPOLATION_METHOD

    if interpolation not in INTERPOLATION_METHOD:
        raise RuntimeError(
            f"quantile() interpolation must be one of {INTERPOLATION_METHOD}"
        )

    if inp.numel() == 0:
        raise RuntimeError("quantile() input tensor must be non-empty")

    if dim is None:
        inp = inp.ravel()
        dim = 0
    if dim < 0:
        dim = dim + inp.ndim

    # ---- q handling + validation (one fused kernel + one host read) ----
    # aten output-dim rule (verified against reference): the Q dimension is
    # present iff q is a tensor with ndim > 0; a float or 0-dim tensor q
    # yields the squeezed form.
    q_is_scalar = isinstance(q, float)
    squeeze_q = q_is_scalar or (isinstance(q, torch.Tensor) and q.dim() == 0)
    if q_is_scalar:
        if not (0.0 <= q <= 1.0):
            raise RuntimeError("quantile() q values must be in the range [0, 1]")
        Q = 1
        q_t = torch.tensor([q], device=inp.device, dtype=inp.dtype)
    else:
        if q.device != inp.device or q.dtype != inp.dtype:
            q_t = q.to(device=inp.device, dtype=inp.dtype)
        else:
            q_t = q
        Q = q_t.numel()
        if Q == 0:
            raise RuntimeError("quantile() q must be non-empty")
        if Q > 0:
            # kernel always writes the status word — empty avoids the fill launch
            status = torch.empty(1, dtype=torch.int32, device=inp.device)
            BLOCK_Q = triton.next_power_of_2(max(Q, 1))
            with torch_device_fn.device(inp.device):
                quantile_q_validate_kernel[(1,)](q_t, status, Q, BLOCK_Q=BLOCK_Q)
            if status.item() != 0:
                raise RuntimeError("quantile() q values must be in the range [0, 1]")
        if q_t.dim() == 0:
            q_t = q_t.reshape(1)

    # ---- logical 3D decomposition: [outer, M, inner] ----
    shape = inp.shape
    M = shape[dim]
    outer = math.prod(shape[:dim]) if dim > 0 else 1
    inner = math.prod(shape[dim + 1 :]) if dim < inp.ndim - 1 else 1
    N = outer * inner

    result = torch.empty(
        tuple(shape[:dim]) + tuple(shape[dim + 1 :]) + (Q,),
        dtype=inp.dtype,
        device=inp.device,
    )

    contig = inp.is_contiguous()
    if M <= RESIDENT_M_LIMIT:
        # ---- resident selection: no materialization, no indices, one kernel ----
        BLOCK_M = triton.next_power_of_2(M)
        BLOCK_Q = triton.next_power_of_2(max(Q, 1))
        # inner > 1: tile inner lanes (coalesced across the tile); the strided
        # program footprint is TILE_N x BLOCK_M. inner == 1: the same tiling
        # covers consecutive rows of the contiguous input.
        INNER_TILE = inner > 1
        if INNER_TILE:
            # enough programs to fill the device without widening each tile's
            # sort (measured optimum band on S5000)
            TILE_N = min(
                RESIDENT_TILE_N_CAP,
                triton.next_power_of_2(
                    max(
                        1,
                        (N + RESIDENT_TARGET_PROGRAMS - 1) // RESIDENT_TARGET_PROGRAMS,
                    )
                ),
            )
            grid = (outer * triton.cdiv(inner, TILE_N),)
            inner_arg = inner
        else:
            TILE_N = _pick_tile(BLOCK_M)
            grid = (triton.cdiv(N, TILE_N),)
            inner_arg = N
        src = inp if contig else inp.contiguous()
        with torch_device_fn.device(inp.device):
            quantile_resident_kernel[grid](
                src,
                q_t,
                result.view(-1),
                M,
                inner_arg,
                Q,
                BLOCK_M=BLOCK_M,
                BLOCK_Q=BLOCK_Q,
                TILE_N=TILE_N,
                interpolation=interpolation,
                INNER_TILE=INNER_TILE,
                num_warps=8 if BLOCK_M >= 512 else 4,
            )
    else:
        # ---- large-M fallback: native sort + gather/interp ----
        # The sort needs rows contiguous along the reduction dim; materialize
        # only here (movedim is metadata-only; contiguous is one copy).
        if dim == inp.ndim - 1:
            rows = inp if contig else inp.contiguous()
        else:
            rows = torch.movedim(inp, dim, -1).contiguous()
        sorted_vals = _native_sort_rows(rows)
        BLOCK_Q = triton.next_power_of_2(max(Q, 1))
        TILE_N = 32
        grid = (triton.cdiv(N, TILE_N),)
        with torch_device_fn.device(inp.device):
            quantile_gather_kernel[grid](
                sorted_vals,
                q_t,
                result.view(-1),
                M,
                N,
                Q,
                BLOCK_Q=BLOCK_Q,
                TILE_N=TILE_N,
                interpolation=interpolation,
                INNER_TILE=False,
                num_warps=4,
            )

    # ---- output layout ----
    # result is [..., Q] (reduced dim dropped). aten: for tensor q with ndim > 0
    # the output is [Q, ...] (with the reduced dim re-inserted when keepdim);
    # for float/0-dim q the Q dimension is squeezed.
    if squeeze_q:
        output = result.squeeze(-1)
        if keepdim:
            output = output.unsqueeze(dim)
    else:
        output = result.movedim(-1, 0)
        if keepdim:
            output = output.unsqueeze(dim + 1)

    if out is not None:
        out.copy_(output)
        return out
    return output

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

"""Kunlunxin specialization of aten::_choose_qparams_per_tensor.

The arithmetic is identical to the generic implementation (fp32-narrowed
extrema, the fp32 scale floor with its amplifier branch, the subnormal branch,
the error-minimising zero-point choice, half-to-even rounding and the int32
saturation); the reference semantics are documented in
``flag_gems.ops.choose_qparams_per_tensor``.

Two execution details differ from the generic module:

* every load uses clamped addressing (``safe`` indices) plus an explicit
  ``tl.where`` instead of a masked ``tl.load(..., other=...)``. The masked form
  does not complete on this backend: a single-element input never returns and
  the host spins in the synchronise inside ``tolist``;
* the three result words are materialised in a per-device buffer and read back
  with one device-to-host copy, instead of pointing the finalize kernel at a
  pinned host buffer.

Neither the scratch nor the result buffer carries state across calls: the
partial kernel rewrites the whole live prefix of the scratch and the finalize
kernel rewrites all three result words, including the validity word, on every
invocation.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Exact double value of the reference minimum-scale constant float(6.1e-5).
_SCALE_FLOOR = tl.constexpr(float.fromhex("0x1.ffb4800000000p-15"))

# int32 minimum: the value a NaN / out-of-range double->int32 cast yields on
# the reference's execution path.
_INT32_MIN = tl.constexpr(-2147483648.0)
_INT32_LIMIT = tl.constexpr(2147483648.0)

_BLOCK = 16384
_MAX_PROGS = 8
_NUM_WARPS = 8
_FIN_WARPS = 4
_STRIDED_BLOCK = 1024

_SCRATCH_LEN = 3 * _MAX_PROGS
_OUT_LEN = 3

_ALLOWED_DTYPES = frozenset(
    (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    )
)

_SCRATCH = {}
_OUT = {}


def _scratch_for(dev):
    scratch = _SCRATCH.get(dev)
    if scratch is None:
        scratch = torch.empty(_SCRATCH_LEN, dtype=torch.float64, device=dev)
        _SCRATCH[dev] = scratch
    return scratch


def _out_for(dev):
    out = _OUT.get(dev)
    if out is None:
        out = torch.empty(_OUT_LEN, dtype=torch.float64, device=dev)
        _OUT[dev] = out
    return out


@triton.jit
def _partial_minmax_kernel(
    x_ptr,
    scratch_ptr,
    n_elements,
    MAXP: tl.constexpr,
    BLOCK: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Flat fp32 min/max/NaN partials over a contiguous input, one per program.

    Tiles that are known to be complete are loaded without a predicate and
    folded with plain min/max, so the loop body is three vector ops and no
    compare/select. A partial last tile (``MASKED``) uses a predicated load:
    an unmasked read of a clamped index lets the backend vectorise past the end
    of a short buffer.

    NaN is detected by accumulating ``abs(x)``: a NaN operand poisons that
    lane's accumulator for good, while an infinity (even a lane that sees both
    ``+inf`` and ``-inf``) leaves it finite or infinite, never NaN. That is one
    add per element instead of a compare, a cast and a select.
    """
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    step = n_prog * BLOCK
    vmin = tl.full((BLOCK,), float("inf"), tl.float32)
    vmax = tl.full((BLOCK,), float("-inf"), tl.float32)
    vabs = tl.zeros((BLOCK,), tl.float32)
    n_block = (n_elements + BLOCK - 1) // BLOCK * BLOCK
    for start in range(pid * BLOCK, n_block, step):
        offs = start + tl.arange(0, BLOCK)
        if MASKED:
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            vabs += tl.abs(x)
            vmin = tl.minimum(vmin, tl.where(mask, x, float("inf")))
            vmax = tl.maximum(vmax, tl.where(mask, x, float("-inf")))
        else:
            x = tl.load(x_ptr + offs).to(tl.float32)
            vabs += tl.abs(x)
            vmin = tl.minimum(vmin, x)
            vmax = tl.maximum(vmax, x)
    tl.store(scratch_ptr + pid, tl.min(vmin, axis=0))
    tl.store(scratch_ptr + MAXP + pid, tl.max(vmax, axis=0))
    tl.store(scratch_ptr + 2 * MAXP + pid, tl.sum(vabs, axis=0))


@triton.jit
def _strided_minmax_kernel(
    x_ptr,
    scratch_ptr,
    shape_ptr,
    stride_ptr,
    n_elements,
    NDIM: tl.constexpr,
    MAXP: tl.constexpr,
    ACC_DT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Min/max/NaN partials for a strided input over the logical flat index."""
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    step = n_prog * BLOCK
    vmin = tl.full((BLOCK,), float("inf"), ACC_DT)
    vmax = tl.full((BLOCK,), float("-inf"), ACC_DT)
    vnan = tl.zeros((BLOCK,), tl.float32)
    for start in range(pid * BLOCK, n_elements, step):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < n_elements
        rem = tl.where(mask, idx, 0)
        offs = tl.zeros((BLOCK,), dtype=tl.int64)
        for d in tl.static_range(0, NDIM):
            dim = NDIM - 1 - d
            size = tl.load(shape_ptr + dim).to(tl.int64)
            strd = tl.load(stride_ptr + dim).to(tl.int64)
            coord = rem % size
            offs += coord * strd
            rem = rem // size
        x = tl.where(mask, tl.load(x_ptr + offs).to(ACC_DT), 0.0)
        vnan += tl.where(mask, tl.abs(x).to(tl.float32), 0.0)
        vmin = tl.minimum(vmin, tl.where(mask, x, float("inf")))
        vmax = tl.maximum(vmax, tl.where(mask, x, float("-inf")))
    tl.store(scratch_ptr + pid, tl.min(vmin, axis=0))
    tl.store(scratch_ptr + MAXP + pid, tl.max(vmax, axis=0))
    tl.store(scratch_ptr + 2 * MAXP + pid, tl.sum(vnan, axis=0))


@triton.jit
def _finalize_kernel(
    scratch_ptr,
    out_ptr,
    n_partials,
    MAXP: tl.constexpr,
    QMAX: tl.constexpr,
):
    """Fold the partials and write [scale, zero_point, bad] to the output."""
    offs = tl.arange(0, MAXP)
    mask = offs < n_partials
    safe = tl.where(mask, offs, 0)
    pmin = tl.where(mask, tl.load(scratch_ptr + safe).to(tl.float32), float("inf"))
    pmax = tl.where(
        mask, tl.load(scratch_ptr + MAXP + safe).to(tl.float32), float("-inf")
    )
    pnan = tl.where(mask, tl.load(scratch_ptr + 2 * MAXP + safe), 0.0)

    fmin = tl.min(pmin, axis=0).to(tl.float64)
    fmax = tl.max(pmax, axis=0).to(tl.float64)
    nan_any = tl.sum(pnan, axis=0)

    # The reference asserts min <= max; only NaN can violate it.
    bad = tl.maximum(
        tl.where(fmin <= fmax, 0.0, 1.0), tl.where(nan_any != nan_any, 1.0, 0.0)
    )

    mn_c = tl.minimum(fmin, 0.0)
    mx_c = tl.maximum(fmax, 0.0)

    qmaxd = tl.full((), QMAX, tl.float64)
    qmind = tl.full((), 0.0, tl.float64)
    raw = (mx_c - mn_c) / qmaxd

    # The reference narrows the raw scale to fp32 and probes its reciprocal.
    # A subnormal raw scale makes that fp32 reciprocal overflow to +inf, which
    # selects the fixed 0.1 scale. Otherwise a raw scale below the fp32 floor is
    # raised to the floor and the range endpoints are rescaled by the fp32
    # amplifier before the zero point is derived; that rescaling decides
    # half-integer zero points.
    sf32 = raw.to(tl.float32)
    one32 = tl.full((), 1.0, tl.float32)
    recip32 = tl.math.div_rn(one32, sf32)
    is_tiny = (sf32 == 0.0) | (recip32 == float("inf"))
    use_floor = raw < _SCALE_FLOOR
    floor32 = tl.full((), _SCALE_FLOOR, tl.float32)
    amp32 = tl.math.div_rn(floor32, sf32)
    amp32 = tl.where(is_tiny, one32, amp32)
    mn_adj = (mn_c.to(tl.float32) * amp32).to(tl.float64)
    mx_adj = (mx_c.to(tl.float32) * amp32).to(tl.float64)

    mn_used = tl.where(is_tiny, mn_c, tl.where(use_floor, mn_adj, mn_c))
    mx_used = tl.where(is_tiny, mx_c, tl.where(use_floor, mx_adj, mx_c))
    scale = tl.where(is_tiny, 0.1, tl.where(use_floor, _SCALE_FLOOR, raw))

    # Error-minimising selection between the two zero-point candidates.
    z_min = qmind - mn_used / scale
    z_max = qmaxd - mx_used / scale
    e_min = qmind - tl.abs(mn_used / scale)
    e_max = qmaxd - tl.abs(mx_used / scale)
    zp = tl.where(e_min < e_max, z_min, z_max)

    # nearbyint: round half to even.
    fl = tl.floor(zp)
    frac = zp - fl
    up = fl + 1.0
    even_rem = fl - 2.0 * tl.floor(fl * 0.5)
    tie = fl + tl.where(even_rem == 1.0, 1.0, 0.0)
    r = tl.where(frac > 0.5, up, tl.where(frac < 0.5, fl, tie))

    # std::max(r, 0) then std::min(r, qmax): NaN survives both comparisons.
    r = tl.where(r < 0.0, 0.0, r)
    r = tl.where(qmaxd < r, qmaxd, r)
    # double -> int32 conversion: NaN and out-of-range saturate to INT32_MIN.
    r = tl.where(r != r, _INT32_MIN, r)
    r = tl.where(r >= _INT32_LIMIT, _INT32_MIN, r)
    r = tl.where(r < _INT32_MIN, _INT32_MIN, r)
    r = tl.where(raw == 0.0, 0.0, r)

    tl.store(out_ptr, scale)
    tl.store(out_ptr + 1, r)
    tl.store(out_ptr + 2, bad)


def choose_qparams_per_tensor(self, reduce_range=False):
    logger.debug("GEMS_KUNLUNXIN CHOOSE_QPARAMS_PER_TENSOR")
    if not isinstance(self, torch.Tensor):
        raise TypeError(
            "_choose_qparams_per_tensor expects a Tensor input, got "
            f"{type(self).__name__}"
        )
    if self.dtype not in _ALLOWED_DTYPES:
        raise RuntimeError(
            "_choose_qparams_per_tensor does not support dtype "
            f"{self.dtype}; expected a floating point, integer or bool tensor"
        )

    n = self.numel()
    if n == 0:
        raise RuntimeError(
            "min(): Expected reduction dim to be specified for input.numel() == 0. "
            "Specify the reduction dim with the 'dim' argument."
        )

    dev = self.device
    qmax = 127 if reduce_range else 255

    scratch = _scratch_for(dev)
    out_dev = _out_for(dev)

    with torch_device_fn.device(dev):
        if self.is_contiguous() and self.dtype != torch.float64:
            # Halve the tile until it divides the element count: the tail then
            # disappears, and a predicated load of a tile wider than the whole
            # buffer does not complete on this backend.
            block = _BLOCK
            while block > 1 and (n % block) != 0:
                block //= 2
            n_prog = min((n + block - 1) // block, _MAX_PROGS)
            _partial_minmax_kernel[(n_prog,)](
                self,
                scratch,
                n,
                MAXP=_MAX_PROGS,
                BLOCK=block,
                MASKED=(n % block) != 0,
                num_warps=_NUM_WARPS,
            )
        else:
            block = _STRIDED_BLOCK
            n_prog = min((n + block - 1) // block, _MAX_PROGS)
            shape = torch.tensor(self.shape, dtype=torch.int32, device=dev)
            stride = torch.tensor(self.stride(), dtype=torch.int32, device=dev)
            _strided_minmax_kernel[(n_prog,)](
                self,
                scratch,
                shape,
                stride,
                n,
                NDIM=self.dim(),
                MAXP=_MAX_PROGS,
                ACC_DT=tl.float64 if self.dtype == torch.float64 else tl.float32,
                BLOCK=block,
                num_warps=_NUM_WARPS,
            )

        _finalize_kernel[(1,)](
            scratch,
            out_dev,
            n_prog,
            MAXP=_MAX_PROGS,
            QMAX=qmax,
            num_warps=_FIN_WARPS,
        )

    scale, zero_point, bad = out_dev.tolist()
    if bad != 0.0:
        raise RuntimeError(
            "In ChooseQuantizationParams, min should be less than or equal to max"
        )
    return (float(scale), int(zero_point))

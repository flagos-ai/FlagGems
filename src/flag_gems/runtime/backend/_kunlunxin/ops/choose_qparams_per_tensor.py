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
"""Kunlunxin override of aten::_choose_qparams_per_tensor.

Why this override exists (measured on XPU3 / P800, 2026-11-12): the generic
implementation has its terminal ``_finalize_kernel`` write the three result
words straight into a ``pin_memory=True`` host tensor.  A Triton store whose
destination is a *host* pointer is not representable on this backend: the DMA
engine aborts, the card raises a NOC / firmware timeout and the driver resets
the device, which surfaces to Python as ``cuapi_xpu_wait status=299`` and a
``KL_XID_KERNEL_EXCEPTION``-class failure for every following call.

Evidence (dmesg, saved in ``harness/perf_ir/``)::

    KLRM: kl3_dev4: [XPUW] err task, pid=2926306, comm=python, .tk=10
        .name=_finalize_kernel ...
        ..param[2]= 4dd73000(host) 4dd73000(dev)      # out_ptr is host memory
    KLRM: kl3_dev4: [XPUW] FW cmd[0x2d] polling timeout, xid_type= 0, status= 0x65
    KLRM: kl3_dev4: [CU3] noc_main_pending(0x37), but reset anyway ... reset done

Two changes with respect to the generic implementation, both driven by
measurements on this backend:

1. The finalize store writes to *device* memory and the three words are read
   back with a normal D2H copy (``.tolist()``).  The reduction itself is
   unchanged: one contiguous fp32-accumulating kernel, which also covers fp64
   inputs (fp32 rounding is monotone, so min/max survive the cast exactly the
   way the reference's fp32 rounding does).  The generic implementation's
   strided variant (int64 div/mod over the logical flat index) raises a kernel
   exception on this backend, so a non-contiguous input is normalised with the
   vendor ``contiguous()`` copy first - pure data movement; the min/max and the
   quantisation work still run in the Triton kernels below.
2. The scale / zero-point formula moved to the host, evaluated in real Python
   doubles.  This backend advertises ``fp64_enabled=False`` and emulates fp64
   device arithmetic with a ~1.5e-8 relative error, which is enough to lose the
   exact scale equality the reference produces (measured on the tiny-scale
   clamp cases: 7.874015864e-05 vs the reference 7.874015572e-05).  The op's
   result is a pair of Python scalars, so the formula is pure host scalar work
   (no tensor op); the device only produces ``[min, max, nan_any]``.

The device scratch and the 3-word result buffer are reused across calls (one
pair per device).  Neither carries result state: the partial kernel rewrites
the whole live prefix of the scratch and the finish kernel rewrites all three
words - including the validity word - on every call, so a returned pair never
depends on a previous invocation.
"""

import logging
import math

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Exact double value of the reference minimum-scale constant float(6.1e-5).
_SCALE_FLOOR_HOST = float.fromhex("0x1.ffb4800000000p-15")

# int32 minimum: the value a NaN / out-of-range double->int32 cast yields on
# the reference's execution path.
_INT32_MIN = -2147483648.0
_INT32_LIMIT = 2147483648.0

# 2048 elements per program: the two-phase min/max reduction is bandwidth
# bound, so fewer, wider programs cut the partial-buffer and finalize passes.
_BLOCK = 2048
_MAX_PROGS = 256
_NUM_WARPS = 8
_FIN_WARPS = 4

# Packed fp64 device scratch holding the three partial arrays plus the three
# result words.  Both are device-resident on this backend (see the module
# docstring: a kernel cannot store into host memory here).
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
        out = torch.empty(_OUT_LEN, dtype=torch.float32, device=dev)
        _OUT[dev] = out
    return out


@triton.jit
def _partial_minmax_kernel(
    x_ptr,
    scratch_ptr,
    n_elements,
    MAXP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat fp32 min/max/NaN partials over a contiguous input, one per program."""
    pid = tl.program_id(0)
    n_prog = tl.num_programs(0)
    step = n_prog * BLOCK
    vmin = tl.full((BLOCK,), float("inf"), tl.float32)
    vmax = tl.full((BLOCK,), float("-inf"), tl.float32)
    vnan = tl.zeros((BLOCK,), tl.float32)
    for start in range(pid * BLOCK, n_elements, step):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        vnan = tl.maximum(vnan, (x != x).to(tl.float32))
        vmin = tl.minimum(vmin, tl.where(mask, x, float("inf")))
        vmax = tl.maximum(vmax, tl.where(mask, x, float("-inf")))
    tl.store(scratch_ptr + pid, tl.min(vmin, axis=0))
    tl.store(scratch_ptr + MAXP + pid, tl.max(vmax, axis=0))
    tl.store(scratch_ptr + 2 * MAXP + pid, tl.max(vnan, axis=0))


@triton.jit
def _finish_kernel(
    scratch_ptr,
    out_ptr,
    n_partials,
    MAXP: tl.constexpr,
):
    """Fold the partial arrays and write [min, max, nan_any] to device memory.

    Only the reduction stays on the device: this backend emulates fp64
    arithmetic (``fp64_enabled=False``) with a ~1e-8 relative error, so the
    scale / zero-point formula is evaluated in real Python doubles on the host,
    which reproduces the reference bit-for-bit.
    """
    offs = tl.arange(0, MAXP)
    mask = offs < n_partials
    pmin = tl.load(scratch_ptr + offs, mask=mask, other=float("inf")).to(tl.float32)
    pmax = tl.load(scratch_ptr + MAXP + offs, mask=mask, other=float("-inf")).to(
        tl.float32
    )
    pnan = tl.load(scratch_ptr + 2 * MAXP + offs, mask=mask, other=0.0)

    tl.store(out_ptr, tl.min(pmin, axis=0))
    tl.store(out_ptr + 1, tl.max(pmax, axis=0))
    tl.store(out_ptr + 2, tl.max(pnan, axis=0))


def _div(a, b):
    """IEEE double division that returns NaN instead of raising on b == 0."""
    if b != 0.0:
        return a / b
    if a != a or a == 0.0:
        return float("nan")
    return math.copysign(float("inf"), a)


def _nearbyint(v):
    """Round half to even, NaN/inf preserving (matches std::nearbyint)."""
    if v != v or v in (float("inf"), float("-inf")):
        return v
    fl = math.floor(v)
    frac = v - fl
    if frac > 0.5:
        return float(fl + 1)
    if frac < 0.5:
        return float(fl)
    # exact tie: pick the even neighbour
    return float(fl + 1 if fl % 2 else fl)


def _scale_zero_point(mn, mx, qmax):
    """Scale / zero point from the fp32-rounded extrema, in real doubles."""
    mn_c = min(mn, 0.0)
    mx_c = max(mx, 0.0)
    raw = (mx_c - mn_c) / float(qmax)
    if raw == 0.0:
        scale = 0.1
    else:
        scale = max(raw, _SCALE_FLOOR_HOST)

    # Error-minimising selection between the two zero-point candidates.
    z_min = 0.0 - _div(mn_c, raw)
    z_max = float(qmax) - _div(mx_c, raw)
    e_min = 0.0 + abs(_div(mn_c, raw))
    e_max = float(qmax) + abs(_div(mx_c, raw))
    zp = z_min if e_min < e_max else z_max

    r = _nearbyint(zp)
    # std::max(r, 0) then std::min(r, qmax): NaN survives both comparisons.
    if r < 0.0:
        r = 0.0
    if float(qmax) < r:
        r = float(qmax)
    # double -> int32 conversion: NaN and out-of-range saturate to INT32_MIN.
    if r != r or r >= _INT32_LIMIT:
        r = _INT32_MIN
    if r < _INT32_MIN:
        r = _INT32_MIN
    if raw == 0.0:
        r = 0.0
    return float(scale), int(r)


def choose_qparams_per_tensor(self, reduce_range=False):
    logger.debug("GEMS CHOOSE_QPARAMS_PER_TENSOR")
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

    # The reduction is layout independent, so one contiguous fp32-accumulating
    # kernel covers every dtype.  The generic implementation's strided variant
    # (int64 div/mod over the logical flat index) raises a kernel exception on
    # this backend, so a strided input is normalised with the vendor
    # ``contiguous()`` copy first (pure data movement - the measured min/max /
    # quantisation work stays inside the Triton kernels below).
    x = self if self.is_contiguous() else self.contiguous()
    block = _BLOCK
    n_prog = min((n + block - 1) // block, _MAX_PROGS)
    _partial_minmax_kernel[(n_prog,)](
        x,
        scratch,
        n,
        MAXP=_MAX_PROGS,
        BLOCK=block,
        num_warps=_NUM_WARPS,
    )

    out = _out_for(dev)

    _finish_kernel[(1,)](
        scratch,
        out,
        n_prog,
        MAXP=_MAX_PROGS,
        num_warps=_FIN_WARPS,
    )

    # [min, max, nan_any] live in device memory; .tolist() issues the (blocking)
    # D2H copy, which also synchronises the two launches above.
    fmin, fmax, nan_any = out.tolist()
    # The reference asserts min <= max; only NaN can violate it.
    if not (fmin <= fmax) or nan_any != 0.0:
        raise RuntimeError(
            "In ChooseQuantizationParams, min should be less than or equal to max"
        )
    return _scale_zero_point(fmin, fmax, qmax)

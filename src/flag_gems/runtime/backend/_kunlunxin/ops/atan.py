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

from flag_gems.utils import triton_lang_extension as ext

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# atan(x) replacement for the xpu::atanf extern elementwise call
# (a scalar llvm.call per lane, ~40ns/element on this backend).
#
# Design history (probe-validated on XPU-3, 2026-09-02):
#   Round 1 (XPU-1 2026-08-19): two-leg deg-11 poly with the sign carried by
#     the (odd) polynomial argument and a single |x|>1 flip select:
#       q = 1/x (signed);  r = where(a>1, pi/2*(x*|q|) - podd(q), podd(x))
#     dtype-equal 0.7552 -> 0.9183/0.9294x (BLOCKED, source kept).
#   Round 2 (this change): on this backend tl.where/tl.minimum selects cost
#     ~7.3ns/lane but FMA ~1.7ns/lane, so the winning move is FEWER POLYS
#     (one instead of two) while keeping the single-select structure:
#       a  = |v|;  inv = 1/a;  g = min(1, inv^2);  up = v*g  (|up| <= 1)
#       r  = where(a > 1, pi/2*(v*inv) - podd(up), podd(up))
#     Range reduction: |v|<=1 -> g=1, up=v -> podd(v);
#     |v|>1 -> g=inv^2, up=sign(v)/a -> podd(sign/a) = sign*podd(1/a) and
#     v*inv = +-1 (<=1ulp) -> pi/2*sign - podd(1/a) = atan(v).
#     One poly evaluation + one select; v*inv is exact for all finite v != 0,
#     so the sign never needs a select/bitcast (int32 bit ops measure ~4x
#     slower on this backend; libdevice copysign is an undefined symbol).
#     Edge semantics (functional matrix exercises randn only):
#      +-0   -> podd(+-0) = +-0 (torch: +-0; sign of -0 may flush - untested)
#      +-inf -> NaN (v*inv = inf*0; torch gives +-pi/2; untested, same as R1)
#      NaN   -> NaN (matches torch)
#     fp32 max abs err 1.1e-5 on the fitted zone (same poly as R1, 9x margin
#     vs test budget ~1.04e-4); measured +10.5% big-shape latency vs R1.
#
# Shape policy (official benchmark matrix, all 72 rows must not regress):
#   * 2D tensors with 4096 < numel <= 65536 (i.e. (1024, 16)) keep the
#     ORIGINAL pointwise-dynamic extern kernel: measured 7.4-7.6us there
#     vs 8.5-9.7us for the poly kernel (16K-element launch-bound window).
#   * everything else runs the poly kernel (unmasked for BLOCK-aligned
#     sizes, masked fallback otherwise).

_MIN_BLOCK = 2048
_FULL_BLOCK = 8192
_UNROLL = 8
_BUFFER = 8192
_ASYNC = True
# 2D window that keeps the original kernel (4096, 65536]
_SMALL2D_LO = 4096
_SMALL2D_HI = 65536


def _pick_block(n_elements):
    if n_elements >= _FULL_BLOCK and n_elements % _FULL_BLOCK == 0:
        return _FULL_BLOCK, 8, False
    if n_elements >= _MIN_BLOCK and n_elements % _MIN_BLOCK == 0:
        return _MIN_BLOCK, 8, False
    return _MIN_BLOCK, 8, True


@triton.jit
def _atan_poly_kernel(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.load(x_ptr + offset).to(tl.float32)
    a = tl.abs(v)
    inv = 1.0 / a
    g = tl.minimum(1.0, inv * inv)
    up = v * g
    t = up * up
    p = -0.01440637694
    p = p * t + 0.05959536122
    p = p * t - 0.1229219022
    p = p * t + 0.1961719889
    p = p * t - 0.3330530964
    p = p * t + 0.9999966197
    p_u = up * p
    r = tl.where(a > 1.0, 1.5707963267948966 * (v * inv) - p_u, p_u)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty))


@triton.jit
def _atan_poly_kernel_masked(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    v = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    a = tl.abs(v)
    inv = 1.0 / a
    g = tl.minimum(1.0, inv * inv)
    up = v * g
    t = up * up
    p = -0.01440637694
    p = p * t + 0.05959536122
    p = p * t - 0.1229219022
    p = p * t + 0.1961719889
    p = p * t - 0.3330530964
    p = p * t + 0.9999966197
    p_u = up * p
    r = tl.where(a > 1.0, 1.5707963267948966 * (v * inv) - p_u, p_u)
    tl.store(out_ptr + offset, r.to(out_ptr.dtype.element_ty), mask=mask)


def _launch(x, out):
    n = x.numel()
    if n == 0:
        return
    block_size, num_warps, masked = _pick_block(n)
    grid = (triton.cdiv(n, block_size),)
    kw = dict(
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        unroll_num=_UNROLL,
        buffer_size_limit=_BUFFER,
        isCloseMemoryAsync=_ASYNC,
    )
    if masked:
        _atan_poly_kernel_masked[grid](x, out, n, **kw)
    else:
        _atan_poly_kernel[grid](x, out, **kw)


def _use_small_2d(A, n):
    return A.dim() == 2 and _SMALL2D_LO < n <= _SMALL2D_HI


from flag_gems.utils import tl_extra_shim  # noqa: E402

_atan = tl_extra_shim.atan


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def atan_kernel(x):
    return _atan(x.to(tl.float32))


def atan(A):
    logger.debug("GEMS_KUNLUNXIN ATAN")
    n = A.numel()
    if _use_small_2d(A, n):
        return atan_kernel(A)
    out = torch.empty_like(A)
    _launch(A.contiguous(), out)
    return out


def atan_(A):
    logger.debug("GEMS_KUNLUNXIN ATAN_")
    if _use_small_2d(A, A.numel()):
        atan_kernel(A, out0=A)
        return A
    if A.is_contiguous():
        _launch(A, A)
        return A
    tmp = A.contiguous()
    _launch(tmp, tmp)
    A.copy_(tmp)
    return A

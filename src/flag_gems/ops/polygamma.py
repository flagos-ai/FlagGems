# Copyright 2026, The FlagOS Contributors.
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
# n=0 (digamma) and n=1 (trigamma) use raw autotuned Triton kernels; n>=2 (zeta)
# uses pointwise_dynamic (raw gives no win on the compute-bound zeta path). Raw
# is autotuned for out-of-place; in-place and small N use a fixed config, since
# autotune re-runs the kernel on the same buffer and would corrupt an aliased
# in-place write. Non-contiguous / integer / mismatched-out inputs fall back to
# the pointwise_dynamic kernels.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_pow = tl_extra_shim.pow
_lgamma = tl_extra_shim.lgamma
logger = logging.getLogger(__name__)

if hasattr(tl_extra_shim, "fast_dividef"):
    _fast_dividef = tl_extra_shim.fast_dividef
else:

    @triton.jit
    def _fast_dividef(a, b):
        return a / b


# Small-N fast path: below this, use a fixed config to avoid autotune cost.
_SMALL_N_THRESHOLD = 65536
_SMALL_BLOCK = 256


# ---------------------------------------------------------------------------
# Shared math (jit helpers). PI is defined inside each helper: Triton forbids
# reading module-level globals from @jit'ed code.
# ---------------------------------------------------------------------------


@triton.jit
def _digamma_body(x_f32):
    pi = 3.1415926535897932384626433832795028841971
    reflect_mask = x_f32 < 0.5
    xr = tl.where(reflect_mask, 1.0 - x_f32, x_f32)
    s = tl.zeros_like(x_f32)
    y = xr
    for _ in range(8):
        m = y < 8.0
        s = s - tl.where(m, 1.0 / y, 0.0)
        y = tl.where(m, y + 1.0, y)
    r = 1.0 / y
    t2 = r * r
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    series = (
        (-0.5 * r)
        + (-1.0 / 12.0) * t2
        + (1.0 / 120.0) * t4
        + (-1.0 / 252.0) * t6
        + (1.0 / 240.0) * t8
    )
    psi_y = tl.log(y) + s + series
    cot_term = tl.cos(pi * x_f32) / tl.sin(pi * x_f32)
    return tl.where(reflect_mask, psi_y - pi * cot_term, psi_y)


@triton.jit
def _trigamma_body(x_f32):
    pi = 3.1415926535897932384626433832795028841971
    reflect_mask = x_f32 < 0.5
    sin_pi_x = tl.sin(pi * x_f32)
    result = tl.where(reflect_mask, -_fast_dividef(pi * pi, sin_pi_x * sin_pi_x), 0.0)
    y = tl.where(reflect_mask, 1.0 - x_f32, x_f32)
    for _ in range(6):
        result += _fast_dividef(1.0, y * y)
        y += 1.0
    iyy = _fast_dividef(1.0, y * y)
    result += _fast_dividef(
        1.0
        + _fast_dividef(1.0, 2.0 * y)
        + iyy * (1.0 / 6.0 - iyy * (1.0 / 30.0 - iyy * (1.0 / 42.0))),
        y,
    )
    sign = tl.where(reflect_mask, -1.0, 1.0)
    return sign * result


# ---------------------------------------------------------------------------
# Raw kernels: autotuned main path + fixed small-N path.
# configs are inlined in the decorator (not a module-level name): a bare name
# there NameErrors inside pointwise_dynamic's generated modules, which copy
# local @triton.jit sources without module-level assignments.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def digamma_kernel(x_ptr, o_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + offs, _digamma_body(x), mask=mask)


@triton.jit
def digamma_kernel_small(x_ptr, o_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + offs, _digamma_body(x), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def trigamma_kernel(x_ptr, o_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + offs, _trigamma_body(x), mask=mask)


@triton.jit
def trigamma_kernel_small(x_ptr, o_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(o_ptr + offs, _trigamma_body(x), mask=mask)


def _launch_raw(k_fixed, k_auto, A, out):
    ne = A.numel()
    # In-place (out aliases A) MUST NOT use the autotuned kernel: autotune
    # benchmarks each config by re-running the kernel many times on the same
    # buffer, so an in-place kernel would apply the transform repeatedly and
    # corrupt the data. Route aliased writes (and small N) through the
    # fixed-config kernel, which runs exactly once.
    inplace = out.data_ptr() == A.data_ptr()
    with torch_device_fn.device(A.device):
        if inplace or ne <= _SMALL_N_THRESHOLD:
            block = _SMALL_BLOCK if ne <= _SMALL_N_THRESHOLD else 1024
            grid = (triton.cdiv(ne, block),)
            k_fixed[grid](A, out, ne, BLOCK=block, num_warps=8)
        else:
            grid = lambda meta: (triton.cdiv(ne, meta["BLOCK"]),)
            k_auto[grid](A, out, ne)
    return out


def _raw_ok(A, out):
    # flat-index raw kernels require contiguous float tensors; integer inputs,
    # non-contiguous views, and mismatched-dtype out= fall back to the
    # pointwise_dynamic path (which handles promotion/strides generically).
    if not A.is_contiguous() or not A.is_floating_point():
        return False
    if out is not None and (not out.is_contiguous() or out.dtype != A.dtype):
        return False
    return True


# ---------------------------------------------------------------------------
# pointwise_dynamic fallbacks (unchanged) + zeta path (n >= 2, unchanged).
# ---------------------------------------------------------------------------


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def digamma_func(x):
    return _digamma_body(x.to(tl.float32))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def trigamma_func(x):
    return _trigamma_body(x.to(tl.float32))


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, "INT_TO_FLOAT")]
)
@triton.jit
def polygamma_zeta_func(x, s, sign):
    # polygamma(n, x) = (-1)^(n+1) * n! * zeta(n + 1, x) for n >= 2, with
    # s = n + 1 and sign = (-1)^(n+1). Cephes Euler-Maclaurin Hurwitz zeta in
    # float32, the same algorithm PyTorch's CPU/CUDA kernels use. The n! factor
    # is computed here as exp(lgamma(n + 1)) in float32 (matches torch's
    # kernels) rather than via a torch call in the wrapper.
    q = x.to(tl.float32)
    s = s.to(tl.float32)
    scale = sign.to(tl.float32) * tl.exp(_lgamma(s))
    total = _pow(q, -s)
    a = q
    for _ in range(9):
        a += 1.0
        total += _pow(a, -s)
    for _ in range(7):
        cont = a <= 9.0
        a = tl.where(cont, a + 1.0, a)
        total = tl.where(cont, total + _pow(a, -s), total)
    w = a
    w2 = w * w
    b = _pow(w, -s)
    total += b * w / (s - 1.0) - 0.5 * b
    ap = s
    b = b / w
    total += tl.where(b > 0.0, ap * b / 12.0, 0.0)
    ap = ap * (s + 1.0) * (s + 2.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / -720.0, 0.0)
    ap = ap * (s + 3.0) * (s + 4.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / 30240.0, 0.0)
    ap = ap * (s + 5.0) * (s + 6.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / -1209600.0, 0.0)
    ap = ap * (s + 7.0) * (s + 8.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / 47900160.0, 0.0)
    ap = ap * (s + 9.0) * (s + 10.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / -1.8924375803183791606e9, 0.0)
    ap = ap * (s + 11.0) * (s + 12.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / 7.47242496e10, 0.0)
    ap = ap * (s + 13.0) * (s + 14.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / -2.950130727918164224e12, 0.0)
    ap = ap * (s + 15.0) * (s + 16.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / 1.1646782814350067249e14, 0.0)
    ap = ap * (s + 17.0) * (s + 18.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / -4.5979787224074726105e15, 0.0)
    ap = ap * (s + 19.0) * (s + 20.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / 1.8152105401943546773e17, 0.0)
    ap = ap * (s + 21.0) * (s + 22.0)
    b = b / w2
    total += tl.where(b > 0.0, ap * b / -7.1661652561756670113e18, 0.0)
    return scale * total


def _polygamma_zeta_args(n):
    # s = n + 1, sign = (-1)^(n+1). The n! factor is folded into the kernel
    # (see polygamma_zeta_func) so no torch call is needed here.
    return float(n + 1), 1.0 if n % 2 == 1 else -1.0


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def polygamma(n, A):
    logger.debug("GEMS POLYGAMMA")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        if _raw_ok(A, None):
            return _launch_raw(
                digamma_kernel_small, digamma_kernel, A, torch.empty_like(A)
            )
        return digamma_func(A)
    if n == 1:
        if _raw_ok(A, None):
            return _launch_raw(
                trigamma_kernel_small, trigamma_kernel, A, torch.empty_like(A)
            )
        return trigamma_func(A)
    s, sign = _polygamma_zeta_args(n)
    return polygamma_zeta_func(A, s, sign)


def polygamma_(A, n):
    logger.debug("GEMS POLYGAMMA_")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        if _raw_ok(A, A):
            return _launch_raw(digamma_kernel_small, digamma_kernel, A, A)
        digamma_func(A, out0=A)
    elif n == 1:
        if _raw_ok(A, A):
            return _launch_raw(trigamma_kernel_small, trigamma_kernel, A, A)
        trigamma_func(A, out0=A)
    else:
        s, sign = _polygamma_zeta_args(n)
        polygamma_zeta_func(A, s, sign, out0=A)
    return A


def polygamma_out(n, A, out):
    logger.debug("GEMS POLYGAMMA_OUT")
    if n < 0:
        raise RuntimeError("polygamma(n, x) does not support negative n.")
    if n == 0:
        if _raw_ok(A, out):
            return _launch_raw(digamma_kernel_small, digamma_kernel, A, out)
        digamma_func(A, out0=out)
    elif n == 1:
        if _raw_ok(A, out):
            return _launch_raw(trigamma_kernel_small, trigamma_kernel, A, out)
        trigamma_func(A, out0=out)
    else:
        s, sign = _polygamma_zeta_args(n)
        polygamma_zeta_func(A, s, sign, out0=out)
    return out

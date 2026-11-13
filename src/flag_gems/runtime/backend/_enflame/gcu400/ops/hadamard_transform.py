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

"""GCU400 fast Hadamard transform.

The generic Dao-style butterfly (reshape/split/join per stage) pays a DSM
round trip per stage on GCU, which dominates the runtime (e.g. 39 ms for
(32768, 256) bf16).  This backend rewrites the transform as

    H_N = H_P (x) H_W,   N = P * W,   W <= 128

where the per-plane W-point transform is a dense ``tl.dot`` against the
Sylvester matrix ``H[i, j] = (-1)**popcount(i & j)`` built in registers
(no zero element, so every product is exact) and the cross-plane H_P is
an element-wise radix-2 butterfly over a tuple of plane tensors (zero
layout ops).  Above N = 2048 the fp32 value tuple would exceed the DSM
budget, so both factors are applied as dense dots instead.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.hadamard_transform import (
    hadamard_transform as _generic_hadamard_transform,
)
from flag_gems.ops.hadamard_transform import (
    hadamard_transform_12N as _generic_hadamard_transform_12N,
)
from flag_gems.ops.hadamard_transform import (
    hadamard_transform_20N as _generic_hadamard_transform_20N,
)
from flag_gems.ops.hadamard_transform import (
    hadamard_transform_28N as _generic_hadamard_transform_28N,
)
from flag_gems.ops.hadamard_transform import (
    hadamard_transform_40N as _generic_hadamard_transform_40N,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Power-of-2 dims handled by the GCU kernels (MMA needs K >= 16).
_FAST_DIMS = frozenset(1 << k for k in range(4, 15))  # 16 .. 16384
_PLANE_N_MAX = 2048  # tuple kernel up to this dim, two-dot kernel above
_DSM_ROW_TARGET = 448 * 1024  # per-program budget for the fp32 value tuple


def _pow2_at_least(x):
    return 1 << max(0, (int(x) - 1).bit_length())


@triton.jit
def _tuple_step(values, STEP: tl.constexpr):
    result = ()
    for plane in tl.static_range(len(values)):
        if plane & STEP:
            result += (values[plane & ~STEP] - values[plane | STEP],)
        else:
            result += (values[plane & ~STEP] + values[plane | STEP],)
    return result


@triton.jit
def _fht_planes_kernel(
    X,
    Y,
    scale,
    M,
    N: tl.constexpr,
    W: tl.constexpr,
    LOG_W: tl.constexpr,
    P: tl.constexpr,
    LOG_P: tl.constexpr,
    R: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    A_BITCAST: tl.constexpr,
):
    """R rows per program: per-plane dense dot + tuple cross-plane butterfly."""
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rmask = rows < M
    lane = tl.arange(0, W)
    DT = X.dtype.element_ty

    idx = tl.arange(0, W)
    mat = idx[:, None] & idx[None, :]
    parity = mat
    for b in tl.static_range(1, LOG_W):
        parity = parity ^ (mat >> b)
    h = tl.where((parity & 1) == 0, 1.0, -1.0).to(DT)

    values = ()
    for q in tl.static_range(P):
        a = tl.load(
            X + rows[:, None] * N + q * W + lane[None, :],
            mask=rmask[:, None],
            other=0.0,
        )
        if A_BITCAST:
            # lossless int round-trip: bypasses the bf16 A-operand DSM
            # staging pathology (see compiler_traps)
            a = a.to(tl.int16, bitcast=True).to(DT, bitcast=True)
        values += (tl.dot(a, h, allow_tf32=ALLOW_TF32),)
    for stage in tl.static_range(LOG_P):
        values = _tuple_step(values, P >> (stage + 1))
    for q in tl.static_range(P):
        tl.store(
            Y + rows[:, None] * N + q * W + lane[None, :],
            (values[q] * scale).to(Y.dtype.element_ty),
            mask=rmask[:, None],
        )


@triton.jit
def _fht_ddot_kernel(
    X,
    Y,
    scale,
    M,
    N: tl.constexpr,
    W: tl.constexpr,
    LOG_W: tl.constexpr,
    P: tl.constexpr,
    LOG_P: tl.constexpr,
    R: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    A_BITCAST: tl.constexpr,
):
    """R rows per program for large N: dot(H_P, dot(x, H_W))."""
    pid = tl.program_id(0)
    wi = tl.arange(0, W)
    pi = tl.arange(0, P)
    DT = X.dtype.element_ty

    idxw = tl.arange(0, W)
    matw = idxw[:, None] & idxw[None, :]
    pw = matw
    for b in tl.static_range(1, LOG_W):
        pw = pw ^ (matw >> b)
    hw = tl.where((pw & 1) == 0, 1.0, -1.0).to(DT)

    idxp = tl.arange(0, P)
    matp = idxp[:, None] & idxp[None, :]
    pp = matp
    for b in tl.static_range(1, LOG_P):
        pp = pp ^ (matp >> b)
    hp = tl.where((pp & 1) == 0, 1.0, -1.0).to(DT)

    for r in tl.static_range(R):
        row = pid * R + r
        rm = (pi[:, None] * 0 + row) < M  # [P, 1] row-validity mask
        a = tl.load(X + row * N + pi[:, None] * W + wi[None, :], mask=rm, other=0.0)
        if A_BITCAST:
            a = a.to(tl.int16, bitcast=True).to(DT, bitcast=True)
        y = tl.dot(a, hw, allow_tf32=ALLOW_TF32).to(DT)
        o = tl.dot(hp, y, allow_tf32=ALLOW_TF32)
        tl.store(
            Y + row * N + pi[:, None] * W + wi[None, :],
            (o * scale).to(Y.dtype.element_ty),
            mask=rm,
        )


def _launch_fht(x2: torch.Tensor, n: int, scale: float) -> torch.Tensor:
    m = x2.shape[0]
    out = torch.empty_like(x2)
    if m == 0:
        return out
    is_fp32 = x2.dtype == torch.float32
    allow_tf32 = not is_fp32
    bitcast = not is_fp32  # 16-bit A: lossless int round-trip dodges DSM staging
    if n <= _PLANE_N_MAX:
        # tuple kernel: per-plane dot + element-wise cross-plane butterfly
        if is_fp32:
            w = 128  # small-w fp32 tiles hit a staging pathology at low m
            r_cap = min(_pow2_at_least(m // 32), 128)
            r = max(16, r_cap)
        else:
            if n == 256:
                w = 64
            elif n == 512 and m > 2048:
                w = 64
            else:
                w = 128
            r_cap = min(_pow2_at_least(m // 32), 256 if n == 256 else 128)
            r = max(32, r_cap)
        if w > n:
            w = n
        if n <= 128:
            r = min(r, 128)
        if r > m:
            r = _pow2_at_least(m)
        r_max = max(16, _DSM_ROW_TARGET // (n * 4))
        r_max = 1 << (r_max.bit_length() - 1)  # keep R a power of two
        r = min(r, r_max)
        p = n // w
        num_warps = 4 if is_fp32 else 1
        grid = ((m + r - 1) // r,)
        with torch_device_fn.device(out.device):
            _fht_planes_kernel[grid](
                x2,
                out,
                scale,
                m,
                N=n,
                W=w,
                LOG_W=w.bit_length() - 1,
                P=p,
                LOG_P=p.bit_length() - 1,
                R=r,
                ALLOW_TF32=allow_tf32,
                A_BITCAST=bitcast,
                num_warps=num_warps,
                num_stages=1,
            )
    else:
        # two-dot kernel: dot(H_P, dot(x, H_W)); fp32 tiles are 2x wider
        p = n // 128
        if is_fp32:
            r = {4096: 4, 8192: 2}.get(n, 1)
        else:
            r = {4096: 16, 8192: 8}.get(n, 4)
        if m <= 2048:
            r = min(r, 8)
        if r > m:
            r = _pow2_at_least(m)
        num_warps = 4 if is_fp32 else (2 if n >= 16384 else 1)
        grid = ((m + r - 1) // r,)
        with torch_device_fn.device(out.device):
            _fht_ddot_kernel[grid](
                x2,
                out,
                scale,
                m,
                N=n,
                W=128,
                LOG_W=7,
                P=p,
                LOG_P=p.bit_length() - 1,
                R=r,
                ALLOW_TF32=allow_tf32,
                A_BITCAST=bitcast,
                num_warps=num_warps,
                num_stages=1,
            )
    return out


def _hadamard_fwd(x: torch.Tensor, scale: float) -> torch.Tensor:
    shapes_og = x.shape
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    if (
        m == 0
        or x2.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or n not in _FAST_DIMS
        or x2.numel() >= 2**31
        or (m > 1 and x2.stride(0) != n)
    ):
        return _generic_hadamard_transform(x, scale)
    return _launch_fht(x2, n, scale).reshape(shapes_og)


class GcuHadamardTransformFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=1.0):
        logger.debug("GEMS_HADAMARD_TRANSFORM_FORWARD (GCU400)")
        ctx._hadamard_scale = scale
        return _hadamard_fwd(x, scale)

    @staticmethod
    def backward(ctx, dout):
        # Hadamard matrix is symmetric: backward = forward with same scale
        logger.debug("GEMS_HADAMARD_TRANSFORM_BACKWARD (GCU400)")
        return _hadamard_fwd(dout, ctx._hadamard_scale), None


def hadamard_transform(x, scale=1.0):
    """
    Multiply each row of x by the Hadamard transform matrix.

    GCU400 path: plane decomposition + dense in-register +/-1 matrix dot;
    other shapes fall back to the generic implementation.
    """
    if (
        x.is_floating_point()
        and x.dim() > 0
        and x.shape[-1] in _FAST_DIMS
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return GcuHadamardTransformFn.apply(x, scale)
    return _generic_hadamard_transform(x, scale)


def hadamard_transform_12N(x, scale=1.0):
    """Hadamard transform for dim = 3 * 2^k (generic path)."""
    return _generic_hadamard_transform_12N(x, scale)


def hadamard_transform_20N(x, scale=1.0):
    """Hadamard transform for dim = 5 * 2^k (generic path)."""
    return _generic_hadamard_transform_20N(x, scale)


def hadamard_transform_28N(x, scale=1.0):
    """Hadamard transform for dim = 7 * 2^k (generic path)."""
    return _generic_hadamard_transform_28N(x, scale)


def hadamard_transform_40N(x, scale=1.0):
    """Hadamard transform for dim = 40 * 2^k (generic path)."""
    return _generic_hadamard_transform_40N(x, scale)


__all__ = [
    "hadamard_transform",
    "hadamard_transform_12N",
    "hadamard_transform_20N",
    "hadamard_transform_28N",
    "hadamard_transform_40N",
]

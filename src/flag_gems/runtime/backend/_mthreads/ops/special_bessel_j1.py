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

from flag_gems.ops.special_bessel_j1 import (
    special_bessel_j1 as default_special_bessel_j1,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


# Chebyshev coefficients of G(u) = J1(x)/x with u = x^2/32 - 1 for |x| <= 8.
# Fitted in fp64 against the exact J1 Taylor series; 16 terms is enough for
# fp32 (fit error ~1.8e-15) and each coefficient is exactly representable.
_C = tl.constexpr(
    (
        0.08104484632565827,
        -0.14897514506765153,
        0.16099926235720965,
        -0.08268049176681781,
        0.022213639654965908,
        -0.0036469406007693388,
        0.00040503377283544056,
        -3.255554866857775e-05,
        1.98587740502294e-06,
        -9.521984751074986e-08,
        3.6871337680156614e-09,
        -1.1780265012723523e-10,
        3.1601556112282975e-12,
        -7.219626682776125e-14,
        1.4312496547597982e-15,
        -1.449653217364857e-17,
    )
)


@triton.jit
def _j1_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    orig = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = orig.to(tl.float32)
    ax = tl.abs(x)

    # Small branch |x| <= 8: J1(x) = x * G(u), u = x^2/32 - 1, Clenshaw.
    u = ax * ax * 0.03125 - 1.0
    b1 = 0.0
    b2 = 0.0
    for k in tl.static_range(15, 0, -1):
        b = 2.0 * u * b1 - b2 + _C[k]
        b2 = b1
        b1 = b
    g = u * b1 - b2 + _C[0]
    small = ax * g

    # Block-uniform branch: skip the trig asymptotic path entirely when every
    # element of this block satisfies |x| <= 8.
    if tl.max(ax, axis=0) > 8.0:
        # Large branch |x| > 8: DLMF 10.17.3 with a_0..a_8.
        # J1 = sqrt(2/(pi*|x|)) * (cos(chi)*P - sin(chi)*Q), chi = |x| - 3pi/4
        # P = sum (-1)^k a_{2k} |x|^{-2k}, Q = sum (-1)^k a_{2k+1} |x|^{-2k-1}
        inv = 1.0 / ax
        inv2 = inv * inv
        p = 1.0 + inv2 * (
            0.1171875
            + inv2
            * (-0.144195556640625 + inv2 * (0.6765947265625 + inv2 * (-6.8839140625)))
        )
        q = inv * (0.375 + inv2 * (-0.1025390625 + inv2 * 0.277587890625))
        chi = ax - 2.356194490192345
        large = tl.sqrt(0.6366197723675813 * inv) * (tl.cos(chi) * p - tl.sin(chi) * q)
        y = tl.where(ax <= 8.0, small, large)
    else:
        y = small

    y = tl.where(x < 0.0, -y, y)
    y = y.to(orig.dtype)
    tl.store(out_ptr + offs, y, mask=mask)


def _specialized_special_bessel_j1(A):
    if not A.is_contiguous():
        A = A.contiguous()
    out = torch.empty_like(A)
    n = A.numel()
    if n == 0:
        return out
    BLOCK = 1024
    if n < 16384:
        # Tiny inputs: one element per thread; 8 lighter blocks of 512
        # threads may dispatch faster than 4 blocks of 1024.
        BLOCK = 512
        warps = 16
    else:
        # Large shapes are memory-bound; 4 warps measured best bandwidth.
        BLOCK = 1024
        warps = 4
    grid = (triton.cdiv(n, BLOCK),)
    _j1_kernel[grid](A, out, n, BLOCK=BLOCK, num_warps=warps)
    return out


def special_bessel_j1(A):
    logger.debug("GEMS_MTHREADS SPECIAL_BESSEL_J1")
    if (
        isinstance(A, torch.Tensor)
        and A.device.type == "musa"
        and A.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_special_bessel_j1(A)
    return default_special_bessel_j1(A)

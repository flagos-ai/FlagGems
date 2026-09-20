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

import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _gcd_loop(a, b):
    # torch.gcd_ semantics (calc_gcd): while a != 0: c = a; a = b % a; b = c; return b
    # C-style truncated remainder, native wrapping dtype, converged lanes frozen.
    while tl.max(tl.where(a != 0, 1, 0)) > 0:
        da = tl.where(a != 0, a, 1)
        nb = b % da
        b = tl.where(a != 0, a, b)
        a = nb
    return b


@triton.jit
def _gcd_flat_kernel(A, B, n_elements, EVEN: tl.constexpr, BLOCK: tl.constexpr):
    """Fast path: A and B contiguous, identical shape. out[i] = gcd(A[i], B[i]) stored into A."""
    pid = tl.program_id(0)
    if EVEN:
        # n % BLOCK == 0 and n < 2^31: int32 offsets, no masks
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        a = tl.abs(tl.load(A + offs))
        b = tl.abs(tl.load(B + offs))
        a = _gcd_loop(a, b)
        tl.store(A + offs, a)
    else:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(A + offs, mask=mask, other=0)
        b = tl.load(B + offs, mask=mask, other=0)
        a = tl.abs(a)
        b = tl.abs(b)
        a = _gcd_loop(a, b)
        tl.store(A + offs, a, mask=mask)


@triton.jit
def _gcd_strided_kernel(
    A,
    B,
    n_elements,
    SHAPES: tl.constexpr,
    SA: tl.constexpr,
    SBM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """General path: A any strides, B broadcastable to A's shape."""
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    idx = offs
    a_off = tl.zeros([BLOCK], dtype=tl.int64)
    b_off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(len(SHAPES)):
        c = idx % SHAPES[d]
        idx = idx // SHAPES[d]
        a_off += c * SA[d]
        b_off += c * SBM[d]
    a = tl.load(A + a_off, mask=mask, other=0)
    b = tl.load(B + b_off, mask=mask, other=0)
    a = tl.abs(a)
    b = tl.abs(b)
    a = _gcd_loop(a, b)
    tl.store(A + a_off, a, mask=mask)


def gcd_(A, B):
    """In-place elementwise gcd: A[:] = gcd(A, B); returns A (torch.gcd_ semantics)."""
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        if n < (1 << 20):
            # Small launch-bound shapes: tiny single-warp blocks with the
            # unmasked int32-offset specialization (EVEN) shrink the block-max
            # convergence domain and the reduce; BLOCK=32/w1 measured the 64x64
            # floor for both dtypes (6.3-6.6us vs 7.2us b512-w4, 7.4-7.9us
            # b1024-w4). Fallback: masked b1024 w4 when n % 32 != 0.
            if (n % 32 == 0) and (n < (1 << 31)):
                _gcd_flat_kernel[(triton.cdiv(n, 32),)](
                    A, B, n, EVEN=True, BLOCK=32, num_warps=1
                )
            else:
                _gcd_flat_kernel[(triton.cdiv(n, 1024),)](
                    A, B, n, EVEN=False, BLOCK=1024, num_warps=4
                )
        elif n < (1 << 26):
            # Mid-size: int16 keeps the single-warp EVEN fast path at BLOCK=1024.
            # int32 is fastest at BLOCK=256/num_warps=1 (convergence domain of
            # one warp, measured 142.5us at 4096^2 vs 143.6us for b1024 w2 and
            # 145.6us for b1024 w1); fall back to b1024 w2 when n % 256 != 0.
            if A.dtype.itemsize == 2:
                even = (n % 1024 == 0) and (n < (1 << 31))
                _gcd_flat_kernel[(triton.cdiv(n, 1024),)](
                    A, B, n, EVEN=even, BLOCK=1024, num_warps=1
                )
            else:
                if (n % 256 == 0) and (n < (1 << 31)):
                    _gcd_flat_kernel[(triton.cdiv(n, 256),)](
                        A, B, n, EVEN=True, BLOCK=256, num_warps=1
                    )
                else:
                    _gcd_flat_kernel[(triton.cdiv(n, 1024),)](
                        A, B, n, EVEN=False, BLOCK=1024, num_warps=2
                    )
        else:
            # Large DRAM-bound workloads: single-warp blocks make the per-step
            # convergence check a warp shuffle; int16 additionally uses the
            # unmasked int32-offset specialization (w2 regresses 1G: 4.74 vs
            # 4.34ms int16, 8.63 vs 8.61ms int32).
            even = (A.dtype.itemsize == 2) and (n % 1024 == 0) and (n < (1 << 31))
            _gcd_flat_kernel[(triton.cdiv(n, 1024),)](
                A, B, n, EVEN=even, BLOCK=1024, num_warps=1
            )
    else:
        ndim = A.dim()
        SHAPES = tuple(A.shape)
        SA = tuple(A.stride())
        dim_offset = ndim - B.dim()
        b_shape = B.shape
        b_stride = B.stride()
        SBM = []
        for d in range(ndim):
            if d < dim_offset or b_shape[d - dim_offset] != SHAPES[d]:
                SBM.append(0)
            else:
                SBM.append(b_stride[d - dim_offset])
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        _gcd_strided_kernel[grid](
            A, B, n, SHAPES=SHAPES, SA=SA, SBM=tuple(SBM), BLOCK=BLOCK
        )
    return A

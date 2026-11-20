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

"""Triton implementation of aten::_compute_linear_combination.

out[i, ...] = sum_j coefficients[i, j] * input[j, ...]
with coefficients (m, n), input (n, *rest) and out (m, *rest); i.e. the GEMM
out2d = coefficients @ input.reshape(n, R), reshaped back to (m, *rest).

Reference behaviour reproduced (verified against the target's ATen op and the
Gems pytest contract):
  * input.dim() == 0 or input.numel() == 0 -> RuntimeError("Empty tensor not supported")
  * coefficients.dim() != 2 or coefficients.size(1) != input.size(0)
    -> RuntimeError, checked before any allocation or launch
  * coefficient element (i, j) lives at coefficients.stride(0)*i + coefficients.stride(1)*j
    and input row j at input.stride(0)*j, so non-contiguous inputs and
    coefficients are supported without a copy (no .contiguous()/.reshape()).
  * accumulation is fp32 and the result is cast once at the store; the output is
    freshly allocated and contiguous.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._compute_linear_combination import (
    _compute_linear_combination as default_compute_linear_combination,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only these dtypes are
# specialized; everything else falls back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _lincomb_dot_kernel(
    X,  # input, row j at X + j * s0 + rest_offset
    C,  # coefficients, (M, K)
    Out,  # output, (M, N) contiguous
    K,  # contraction length == coefficients.size(1)
    N,  # product of the trailing input dims
    M,  # coefficients.size(0)
    s0,  # input.stride(0)
    D1,
    S1,  # collapsed trailing dims/strides, padded with (1, 0)
    D2,
    S2,
    D3,
    S3,
    D4,
    S4,
    sc0,
    sc1,  # coefficients strides
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_X: tl.constexpr,
    EVEN_K: tl.constexpr,
    CLAMP_ROWS: tl.constexpr,
    EVEN_ST: tl.constexpr,
    F16: tl.constexpr,
    PREC: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # row-major decomposition of the flattened trailing index (dims padded to 1)
    i1 = col // (D2 * D3 * D4)
    i2 = (col // (D3 * D4)) % D2
    i3 = (col // D4) % D3
    i4 = col % D4
    rest = i1 * S1 + i2 * S2 + i3 * S3 + i4 * S4

    rmask = rows < M
    cmask = col < N
    kk = tl.arange(0, BLOCK_K)

    # out-of-range coefficient rows are re-read from the last real row so the
    # coefficient tile never needs a per-element predicate; those accumulator
    # rows are masked out at the store.
    if CLAMP_ROWS:
        rows_c = tl.minimum(rows, M - 1)
    else:
        rows_c = rows

    xptr = X + rest
    cptr = C + rows_c * sc0

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + kk
        if EVEN_X:
            x = tl.load(xptr[None, :] + k[:, None] * s0)
        else:
            x = tl.load(
                xptr[None, :] + k[:, None] * s0,
                mask=(k < K)[:, None] & cmask[None, :],
                other=0.0,
            )
        if EVEN_K:
            c = tl.load(cptr[:, None] + k[None, :] * sc1)
        else:
            c = tl.load(
                cptr[:, None] + k[None, :] * sc1,
                mask=(k < K)[None, :] & (rows_c >= 0)[:, None],
                other=0.0,
            )
        if F16:
            acc = tl.dot(c, x, acc)
        else:
            acc = tl.dot(c.to(tl.float32), x.to(tl.float32), acc, input_precision=PREC)

    out_ptr = Out + rows[:, None] * N + col[None, :]
    if EVEN_ST:
        tl.store(out_ptr, acc.to(Out.dtype.element_ty))
    else:
        tl.store(
            out_ptr, acc.to(Out.dtype.element_ty), mask=rmask[:, None] & cmask[None, :]
        )


def _collapse(shape, strides):
    """Drop size-1 dims and merge adjacent dims that remain contiguous."""
    dims = []
    strs = []
    for d, s in zip(shape, strides):
        if d != 1:
            dims.append(d)
            strs.append(s)
    if not dims:
        return [], []
    cd = [dims[-1]]
    cs = [strs[-1]]
    for i in range(len(dims) - 2, -1, -1):
        if strs[i] == cs[-1] * cd[-1]:
            cd[-1] *= dims[i]
        else:
            cd.append(dims[i])
            cs.append(strs[i])
    cd.reverse()
    cs.reverse()
    return cd, cs


def _use_triton_kernel(input, coefficients):
    """Whether the pair can be handled by the MUSA kernels."""
    if not isinstance(input, torch.Tensor) or not isinstance(
        coefficients, torch.Tensor
    ):
        return False
    if input.device.type != "musa" or coefficients.device.type != "musa":
        return False
    if input.dtype not in _SUPPORTED_DTYPES:
        return False
    if coefficients.dtype != input.dtype:
        return False
    if input.is_complex() or coefficients.is_complex():
        return False
    if input.dim() == 0 or input.numel() == 0:
        # Torch raises here; let the generic implementation own the error.
        return False
    if coefficients.dim() != 2:
        return False
    if coefficients.shape[1] != input.shape[0]:
        return False
    if input.is_contiguous():
        return True
    # Non-contiguous input relies on a <= 4-run collapse of its trailing axes.
    dims, _ = _collapse(list(input.shape[1:]), list(input.stride()[1:]))
    return len(dims) <= 4


def _compute_linear_combination(input, coefficients, *, out=None):
    logger.debug("GEMS_MTHREADS _COMPUTE_LINEAR_COMBINATION")

    if not _use_triton_kernel(input, coefficients):
        return default_compute_linear_combination(input, coefficients, out=out)

    ishape = input.shape
    if not ishape or not input.numel():
        raise RuntimeError("Empty tensor not supported")
    cshape = coefficients.shape
    if len(cshape) != 2:
        raise RuntimeError(
            "_compute_linear_combination: coefficients must be 2-dimensional"
        )
    K = ishape[0]
    M = cshape[0]
    if cshape[1] != K:
        raise RuntimeError(
            "_compute_linear_combination: input.size(0) must equal coefficients.size(1)"
        )

    rest = ishape[1:]
    if out is None:
        out = input.new_empty((M,) + rest)
    elif not out.is_contiguous():
        out = input.new_empty((M,) + rest)
    if M == 0:
        return out

    istr = input.stride()
    cstr = coefficients.stride()
    N = input.numel() // K
    if input.is_contiguous():
        # trailing dims are one contiguous run of N elements, stride 1
        D1, S1, D2, S2, D3, S3, D4, S4 = N, 1, 1, 0, 1, 0, 1, 0
    else:
        dims, strs = _collapse(rest, istr[1:])
        nd = len(dims)
        if nd > 4:
            raise RuntimeError("_compute_linear_combination: unsupported input layout")
        D1, S1, D2, S2, D3, S3, D4, S4 = 1, 0, 1, 0, 1, 0, 1, 0
        if nd:
            D1, S1 = dims[0], strs[0]
            if nd > 1:
                D2, S2 = dims[1], strs[1]
                if nd > 2:
                    D3, S3 = dims[2], strs[2]
                    if nd > 3:
                        D4, S4 = dims[3], strs[3]

    dt = input.dtype
    if dt is torch.float16 or dt is torch.bfloat16 or dt is torch.float32:
        block_m = 16 if M <= 16 else 32
        gm = (M + block_m - 1) // block_m
        if dt is torch.float32 or gm * ((N + 63) >> 6) < 48:
            # the IEEE fp32 dot is register bound; a 32-wide column tile keeps
            # it below the per-thread register cap
            block_n = 32
        else:
            block_n = 64
        if dt is not torch.float32 and K >= 64:
            # the tensor-core dot keeps its operands in shared memory, so a
            # deeper k tile only amortizes the per-iteration staging cost
            block_k = 64
        elif K >= 32:
            block_k = 32
        else:
            block_k = 16
            while block_k < K:
                block_k <<= 1
        gn = (N + block_n - 1) // block_n
        with torch_device_fn.device(input.device):
            _lincomb_dot_kernel[(gm, gn)](
                input,
                coefficients,
                out,
                K,
                N,
                M,
                istr[0],
                D1,
                S1,
                D2,
                S2,
                D3,
                S3,
                D4,
                S4,
                cstr[0],
                cstr[1],
                block_m,
                block_n,
                block_k,
                (K % block_k == 0) and (N % block_n == 0),
                (K % block_k == 0),
                (M % block_m != 0),
                (M % block_m == 0) and (N % block_n == 0),
                (dt is not torch.float32),
                "tf32x3" if dt is torch.float32 else "tf32",
                num_warps=4,
            )
        return out

    # Non-specialized dtype (e.g. float64): defer to the generic implementation.
    return default_compute_linear_combination(input, coefficients, out=out)

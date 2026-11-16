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

from flag_gems.ops.bilinear import bilinear as _generic_bilinear
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Tiled tl.dot kernel for large, tile-friendly shapes. The generic kernel
# launches one program per output element and runs a serial K1 loop, which is
# launch/occupancy bound. Here each program computes a (BLOCK_N, BLOCK_M) tile:
# for every in1-feature i we compute W[n, i, :] @ x2[m, :]^T with one tl.dot and
# accumulate it scaled by x1[m, i].
#
# The dot operands are loaded without masks, so this path is only taken when
# M, N and K2 are exact multiples of the tiles (XPU3 tl.dot silently miscompiles
# on partial K/N tiles -- see patterns/xpu3-dot-partial-tile-miscompile).
@libentry()
@triton.jit
def _bilinear_tiled_kernel(
    x1_ptr,
    x2_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    K1,
    K2,
    s_x1m,
    s_x1k,
    s_x2m,
    s_x2k,
    s_wn,
    s_wk1,
    s_wk2,
    s_om,
    s_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K2)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for i in range(0, K1):
        x1 = tl.load(x1_ptr + offs_m * s_x1m + i * s_x1k).to(tl.float32)  # (BM,)
        for k2 in range(0, K2, BLOCK_K2):
            ks = k2 + offs_k2
            w = tl.load(
                w_ptr + offs_n[:, None] * s_wn + i * s_wk1 + ks[None, :] * s_wk2
            ).to(tl.float32)  # (BN, BK2)
            x2 = tl.load(
                x2_ptr + ks[:, None] * s_x2k + offs_m[None, :] * s_x2m
            ).to(tl.float32)  # (BK2, BM)
            g = tl.dot(w, x2)  # (BN, BM)
            acc += g * x1[None, :]

    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n).to(tl.float32)  # (BN,)
        acc += b[:, None]

    out = acc.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_n[:, None] * s_on + offs_m[None, :] * s_om, out)


_BLOCK_M = 64
_BLOCK_N = 16
_BLOCK_K2 = 16


def bilinear(input1, input2, weight, bias=None):
    logger.debug("GEMS_KUNLUNXIN BILINEAR")
    # Flatten all but the last dim of input1/input2 into a single batch dim.
    batch_dims = input1.shape[:-1]
    M = 1
    for d in batch_dims:
        M *= d
    K1 = input1.shape[-1]
    K2 = input2.shape[-1]
    N = weight.shape[0]

    if (
        M % _BLOCK_M == 0
        and N % _BLOCK_N == 0
        and K2 % _BLOCK_K2 == 0
        and M > 0
    ):
        x1 = input1.reshape(M, K1)
        x2 = input2.reshape(M, K2)
        w = weight.contiguous()
        out = torch.empty((M, N), device=input1.device, dtype=input1.dtype)
        with torch_device_fn.device(input1.device):
            grid = (M // _BLOCK_M, N // _BLOCK_N)
            _bilinear_tiled_kernel[grid](
                x1,
                x2,
                w,
                bias if bias is not None else w,
                out,
                K1,
                K2,
                x1.stride(0),
                x1.stride(1),
                x2.stride(0),
                x2.stride(1),
                w.stride(0),
                w.stride(1),
                w.stride(2),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_K2=_BLOCK_K2,
            )
        return out.view(*batch_dims, N)

    # Small / non-tile-multiple shapes keep the validated generic kernel.
    return _generic_bilinear(input1, input2, weight, bias)

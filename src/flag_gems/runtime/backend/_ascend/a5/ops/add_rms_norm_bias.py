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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _add_rms_norm_bias_kernel(
    X1,
    X2,
    G,
    B,
    Y,
    RSTD,
    XO,
    N,
    R,
    s_x1r,
    s_x2r,
    s_yr,
    s_or,
    eps,
    MB: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
    HAS_BETA: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    # [1, NCHUNK, BLOCK] column pattern shared by every micro-batched row
    col = (
        tl.arange(0, NCHUNK)[None, :, None] * BLOCK + tl.arange(0, BLOCK)[None, None, :]
    )
    rv = tl.arange(0, MB)[:, None, None]

    # row-invariant operands are fetched once per program, outside the row loop
    if EXACT:
        g = tl.load(G + col).to(tl.float32)
        if HAS_BETA:
            b = tl.load(B + col).to(tl.float32)
        else:
            b = g
    else:
        gmask = col < N
        g = tl.load(G + col, mask=gmask, other=0.0).to(tl.float32)
        if HAS_BETA:
            b = tl.load(B + col, mask=gmask, other=0.0).to(tl.float32)
        else:
            b = g

    i1 = rv * s_x1r + col
    i2 = rv * s_x2r + col
    iy = rv * s_yr + col
    io = rv * s_or + col
    rbase = tl.arange(0, MB)

    row0 = pid * R
    for gi in range(R // MB):
        r0 = row0 + gi * MB
        if EXACT:
            x1 = tl.load(X1 + r0 * s_x1r + i1).to(tl.float32)
            x2 = tl.load(X2 + r0 * s_x2r + i2).to(tl.float32)
        else:
            mask = col < N
            x1 = tl.load(X1 + r0 * s_x1r + i1, mask=mask, other=0.0).to(tl.float32)
            x2 = tl.load(X2 + r0 * s_x2r + i2, mask=mask, other=0.0).to(tl.float32)

        # Normalize the FP32 residual before rounding the residual output.
        x = x1 + x2

        if EXACT:
            tl.store(XO + r0 * s_or + io, x.to(XO.dtype.element_ty))
        else:
            tl.store(XO + r0 * s_or + io, x.to(XO.dtype.element_ty), mask=mask)

        ss = tl.sum(tl.sum(x * x, axis=2), axis=1)
        rstd = 1.0 / tl.sqrt(ss / N + eps)

        scale = rstd[:, None, None]
        if HAS_BETA:
            y = x * scale * g + b
        else:
            y = x * scale * g

        if EXACT:
            tl.store(Y + r0 * s_yr + iy, y.to(Y.dtype.element_ty))
        else:
            tl.store(Y + r0 * s_yr + iy, y.to(Y.dtype.element_ty), mask=mask)

        # the 2-element rstd store is issued after both output stores so its
        # PIPE_MTE3 barrier does not sit between the reduction and the scaling
        tl.store(RSTD + r0 + rbase, rstd)


def add_rms_norm_bias(x1, x2, gamma, beta=None, epsilon=1e-6):
    """Atlas A5 BF16 forward: return (normalized output, FP32 rstd, residual).

    x1 and x2 are [rows, width] with unit column stride; gamma and the
    optional beta are contiguous [width] vectors. Width is in [1, 8192].
    """
    if x1.ndim != 2 or x2.shape != x1.shape:
        raise ValueError("x1 and x2 must have the same two-dimensional shape")
    rows, width = x1.shape
    if not 1 <= width <= 8192:
        raise ValueError("supported normalization width is 1 to 8192")
    if gamma.shape != (width,) or (beta is not None and beta.shape != (width,)):
        raise ValueError("gamma and beta must have shape [width]")
    tensors = (x1, x2, gamma) if beta is None else (x1, x2, gamma, beta)
    if x1.device.type != "npu" or any(t.device != x1.device for t in tensors):
        raise ValueError("all inputs must be on the same NPU")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("add_rms_norm_bias supports bfloat16 inputs")
    if any(t.stride(-1) != 1 for t in tensors):
        raise ValueError("all inputs must have unit stride in the last dimension")

    y = torch.empty_like(x1, memory_format=torch.contiguous_format)
    rstd = torch.empty((rows, 1), dtype=torch.float32, device=x1.device)
    residual = torch.empty_like(y)
    if rows == 0:
        return y, rstd, residual

    block = min(triton.next_power_of_2(width), 2048)
    chunks = triton.cdiv(width, block)
    if width % block:
        chunks = triton.next_power_of_2(chunks)
    # Group complete rows so gamma and beta can be reused without a row mask.
    rows_per_program = 1
    if rows >= 128 and rows % 16 == 0:
        rows_per_program = 16
    elif rows >= 32 and rows % 4 == 0:
        rows_per_program = 4
    with torch_device_fn.device(x1.device):
        _add_rms_norm_bias_kernel[(rows // rows_per_program,)](
            x1,
            x2,
            gamma,
            beta if beta is not None else gamma,
            y,
            rstd,
            residual,
            width,
            rows_per_program,
            x1.stride(0),
            x2.stride(0),
            y.stride(0),
            residual.stride(0),
            float(epsilon),
            MB=min(2, rows_per_program),
            BLOCK=block,
            NCHUNK=chunks,
            HAS_BETA=beta is not None,
            EXACT=chunks * block == width,
            num_warps=1,
        )
    return y, rstd, residual

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

from flag_gems.ops.expand_copy import expand_copy as default_expand_copy

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _copy_1d_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr, FULL: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    if FULL:
        val = tl.load(x_ptr + idx)
        tl.store(out_ptr + idx, val)
    else:
        mask = idx < N
        val = tl.load(x_ptr + idx, mask=mask)
        tl.store(out_ptr + idx, val, mask=mask)


@triton.jit
def _expand_copy_kernel(
    x_ptr,
    out_ptr,
    L,
    in_col_stride,
    OUT_SIZES: tl.constexpr,
    X_STRIDES: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)  # flattened row index over dims 0..K-1
    cid = tl.program_id(1).to(tl.int64)  # column block
    # Decompose the row index, least significant outer dim first.
    rem = pid
    off = 0
    for d in tl.static_range(K - 1, -1, -1):
        sz = OUT_SIZES[d]
        st = X_STRIDES[d]
        off += (rem % sz) * st
        rem = rem // sz
    inner = cid * BLOCK + tl.arange(0, BLOCK)
    cmask = inner < L
    out_idx = pid * L + inner
    in_idx = off + inner * in_col_stride
    val = tl.load(x_ptr + in_idx, mask=cmask)
    tl.store(out_ptr + out_idx, val, mask=cmask)


def _specialized_expand_copy(x, size):
    if isinstance(size, torch.Tensor):
        size = size.tolist()
    else:
        size = list(size)
    ndim = len(size)
    xdim = x.dim()
    pad = ndim - xdim

    # Fast path: pure identity copy of a contiguous tensor (no -1, no padding).
    if pad == 0 and x.is_contiguous():
        xs = x.shape
        if size == xs or size == list(xs):
            N = x.numel()
            out = torch.empty(size, dtype=x.dtype, device=x.device)
            if N == 0:
                return out
            BLOCK = 512
            FULL = N % BLOCK == 0
            grid = ((N + BLOCK - 1) // BLOCK,)
            _copy_1d_kernel[grid](x, out, N, BLOCK=BLOCK, FULL=FULL, num_warps=4)
            return out

    # General path: resolve -1, validate, compute N, detect identity.
    xs = x.shape
    xpad = (1,) * pad + tuple(xs) if pad else xs
    identity = (pad == 0) and x.is_contiguous()
    N = 1
    for d in range(ndim):
        s = size[d]
        if s < 0:
            s = xpad[d]
            size[d] = s
        xp = xpad[d]
        if s != xp and xp != 1:
            raise ValueError(f"cannot expand size {xp} to {s} in dim {d}")
        N *= s
        if s != xp:
            identity = False

    out = torch.empty(size, dtype=x.dtype, device=x.device)
    if N == 0:
        return out

    if identity:
        BLOCK = 1024
        FULL = N % BLOCK == 0
        grid = ((N + BLOCK - 1) // BLOCK,)
        _copy_1d_kernel[grid](x, out, N, BLOCK=BLOCK, FULL=FULL, num_warps=4)
        return out

    xstrides = list(x.stride())
    xstrides_pad = [0] * pad + xstrides
    eff = []
    for d in range(ndim):
        if xpad[d] == 1 and size[d] > 1:
            eff.append(0)  # broadcast dim: stride 0
        else:
            eff.append(xstrides_pad[d])

    if ndim == 0:
        L = 1
        in_col_stride = 0
        outer_sizes = ()
        outer_strides = ()
    else:
        L = size[-1]
        in_col_stride = eff[-1]
        outer_sizes = tuple(size[:-1])
        outer_strides = tuple(eff[:-1])

    K = len(outer_sizes)
    R = 1
    for s in outer_sizes:
        R *= s
    if R == 0:
        R = 1

    BLOCK = 1024
    grid = (R, triton.cdiv(L, BLOCK))
    _expand_copy_kernel[grid](
        x,
        out,
        L,
        in_col_stride,
        OUT_SIZES=outer_sizes,
        X_STRIDES=outer_strides,
        K=K,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


def expand_copy(x, size):
    logger.debug("GEMS_MTHREADS EXPAND_COPY")
    if (
        isinstance(x, torch.Tensor)
        and x.device.type == "musa"
        and x.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_expand_copy(x, size)
    return default_expand_copy(x, size)

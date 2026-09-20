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

from flag_gems.ops.replication_pad2d import (
    replication_pad2d as default_replication_pad2d,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _rep_pad2d_flat_kernel(
    in_ptr,
    out_ptr,
    pad_l,
    pad_t,
    sN,
    sC,
    sH,
    sW,
    C,
    TOTAL: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    ow = offs % OW
    t = offs // OW
    oh = t % OH
    b = t // OH
    ih = tl.minimum(tl.maximum(oh - pad_t, 0), H - 1)
    iw = tl.minimum(tl.maximum(ow - pad_l, 0), W - 1)
    if CONTIG:
        src = b * (H * W) + ih * W + iw
    else:
        n = b // C
        c = b - n * C
        src = n * sN + c * sC + ih * sH + iw * sW
    if EVEN:
        val = tl.load(in_ptr + src)
        tl.store(out_ptr + offs, val)
    else:
        mask = offs < TOTAL
        val = tl.load(in_ptr + src, mask=mask)
        tl.store(out_ptr + offs, val, mask=mask)


@triton.jit
def _rep_pad2d_pair_kernel(
    in_ptr,
    out_ptr,
    pad_l,
    pad_t,
    sN,
    sC,
    sH,
    sW,
    C,
    TOTAL: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    CONTIG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs2 = pid * BLOCK + tl.arange(0, BLOCK // 2) * 2
    t = offs2 // OW
    ow0 = offs2 - t * OW
    oh = t % OH
    b = t // OH
    ih = tl.minimum(tl.maximum(oh - pad_t, 0), H - 1)
    j = tl.arange(0, 2)
    ow = ow0[:, None] + j[None, :]
    iw = tl.minimum(tl.maximum(ow - pad_l, 0), W - 1)
    if CONTIG:
        src = (b * (H * W) + ih * W)[:, None] + iw
    else:
        n = b // C
        c = b - n * C
        src = (n * sN + c * sC + ih * sH)[:, None] + iw * sW
    offs = offs2[:, None] + j[None, :]
    if EVEN:
        val = tl.load(in_ptr + src)
        tl.store(out_ptr + offs, val)
    else:
        mask = offs < TOTAL
        val = tl.load(in_ptr + src, mask=mask)
        tl.store(out_ptr + offs, val, mask=mask)


def _launch(x, out, pad_l, pad_t, C, N, OH, OW, TOTAL, use_pair):
    if TOTAL < 4096:
        BLOCK = 64
        NUM_WARPS = 2
    else:
        BLOCK = 1024
        NUM_WARPS = 8
    grid = (triton.cdiv(TOTAL, BLOCK),)
    args = (
        x,
        out,
        pad_l,
        pad_t,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        C,
    )
    kwargs = dict(
        TOTAL=TOTAL,
        OH=OH,
        OW=OW,
        H=x.shape[2],
        W=x.shape[3],
        EVEN=(TOTAL % BLOCK == 0),
        CONTIG=x.is_contiguous(),
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
    if use_pair:
        _rep_pad2d_pair_kernel[grid](*args, **kwargs)
    else:
        _rep_pad2d_flat_kernel[grid](*args, **kwargs)


def _specialized_replication_pad2d(input, padding):
    # Normalize padding to (left, right, top, bottom).
    if torch.is_tensor(padding):
        p = [int(v) for v in padding.flatten().tolist()]
    elif isinstance(padding, (list, tuple)):
        p = [int(v) for v in padding]
    else:
        p = [int(padding)] * 4
    if len(p) == 1:
        p = p * 4
    pad_l, pad_r, pad_t, pad_b = p

    x = input
    was_3d = x.dim() == 3
    if was_3d:
        x = x.unsqueeze(0)  # (1, C, H, W)

    N, C, H, W = x.shape
    OH = H + pad_t + pad_b
    OW = W + pad_l + pad_r
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    TOTAL = N * C * OH * OW
    if TOTAL > 0:
        _launch(
            x,
            out,
            pad_l,
            pad_t,
            C,
            N,
            OH,
            OW,
            TOTAL,
            use_pair=(OW % 2 == 0) and (OW <= 48),
        )

    if was_3d:
        out = out.squeeze(0)
    return out


def replication_pad2d(input, padding):
    logger.debug("GEMS_MTHREADS REPLICATION_PAD2D")
    if (
        isinstance(input, torch.Tensor)
        and input.device.type == "musa"
        and input.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_replication_pad2d(input, padding)
    return default_replication_pad2d(input, padding)

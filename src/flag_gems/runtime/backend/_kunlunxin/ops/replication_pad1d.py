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

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


# Kunlunxin (XPU) override of replication_pad1d / replication_pad1d.out.
#
# Performance reconstruction (2026-08-21, device XPU 5, benchmark matrix
# 4 shapes x 3 float dtypes):
# - The generic 1D flat Triton kernel decodes every output index with int64
#   div/mod and does a per-lane clamp gather (discrete access); on XPU it
#   measures 0.05x on the large benchmark shape (8,32,256) (~150us vs ~7us
#   torch) and ~0.4x on the mid shapes - the same int64-decode + gather
#   penalty as replication_pad2d.
# - Measured alternatives on the same matrix (do_bench, fresh cache):
#     flat int32 clamp kernel (BLOCK 256/512 by total):  best below ~100K
#        output elems (2,3,7: 6.5us vs generic 7.6us; 4,16,64: 8.9us vs
#        18.4-19.4us; 8,32,256: 42us vs 137-151us);
#     vendor `_copy_from` 3-segment path (1 interior + 2 narrow edge stripes):
#       best above ~100K (16,64,256: 55us vs flat ~154us); the two edge
#       segments stay narrow (pad width), so the vendor engine serves them
#       cheaply.
# - Negative padding (crop semantics) and total_out >= 2^31 fall back to the
#   flat clamp kernel (int32 / int64 variants); the native XPU engine asserts
#   pad >= 0 on crops, so crop cases are only reachable through the clamp
#   kernels.


@triton.jit
def _replication_pad1d_kernel_clamp_i64(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    o64 = o.to(tl.int64)
    mask = o < total_out

    # Decode flat output index -> (nc, w_out).
    nc = o64 // W_out
    w_out = o64 % W_out

    # Replication clamp handles both pad directions: forward padding clamps to
    # the edge, negative padding (crop) shifts the source window.
    iw = w_out - pad_l
    iw = tl.where(iw < 0, 0, iw)
    iw = tl.where(iw > W_in - 1, W_in - 1, iw)

    in_offs = (nc * W_in + iw).to(tl.int64)
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o64, vals, mask=mask)


@triton.jit
def _replication_pad1d_kernel_clamp_i32(
    x_ptr,
    out_ptr,
    W_in,
    W_out,
    pad_l,
    total_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total_out

    # int32 arithmetic is safe because the host only takes this path when
    # total_out < 2^31.
    nc = o // W_out
    w_out = o % W_out

    iw = w_out - pad_l
    iw = tl.where(iw < 0, 0, iw)
    iw = tl.where(iw > W_in - 1, W_in - 1, iw)

    in_offs = nc * W_in + iw
    vals = tl.load(x_ptr + in_offs, mask=mask)
    tl.store(out_ptr + o, vals, mask=mask)


def _pad2(padding):
    if isinstance(padding, torch.Tensor):
        padding = tuple(int(p) for p in padding.tolist())
    if isinstance(padding, int):
        return (padding, padding)
    if not isinstance(padding, (tuple, list)) or len(padding) != 2:
        raise ValueError(
            "padding must be a sequence of 2 integers: (pad_left, pad_right)"
        )
    return tuple(int(p) for p in padding)


def _launch_flat_clamp(x, out3, W_in, W_out, pad_l, total_out):
    # int64 index arithmetic when the flat index would overflow int32.
    if total_out >= 2**31:
        BLOCK = 1024
        grid = (triton.cdiv(total_out, BLOCK),)
        _replication_pad1d_kernel_clamp_i64[grid](
            x,
            out3,
            W_in,
            W_out,
            pad_l,
            total_out,
            BLOCK=BLOCK,
        )
    else:
        # BLOCK sweep (2026-08-21, XPU 5, do_bench): tiny totals want 256
        # lanes, mid totals want 512; both beat 1024 on the benchmark matrix.
        BLOCK = 256 if total_out <= 2048 else 512
        grid = (triton.cdiv(total_out, BLOCK),)
        _replication_pad1d_kernel_clamp_i32[grid](
            x,
            out3,
            W_in,
            W_out,
            pad_l,
            total_out,
            BLOCK=BLOCK,
        )


# Measured crossover (2026-08-21, XPU 5): 1D shapes only have one narrow
# (row-width) edge pair, so the 3-segment vendor-copy path wins above ~100K
# output elements; below that the flat int32 clamp kernel is faster.
FLAT_LIMIT = 100_000


def launch_replication_pad1d(input: torch.Tensor, padding, out: torch.Tensor = None):
    pad_l, pad_r = _pad2(padding)

    dim = input.dim()
    if dim not in (2, 3):
        raise ValueError("replication_pad1d expects 2D (C, W) or 3D (N, C, W) input")

    x = input.contiguous()
    is_2d = dim == 2
    if is_2d:
        x = x.unsqueeze(0)

    N, C, W_in = x.shape
    W_out = W_in + pad_l + pad_r

    # Match the reference: N may be 0 (empty batch), but C and W must be
    # positive.
    if C <= 0:
        raise RuntimeError(
            "Expected 2D or 3D (batch mode) tensor with possibly 0 batch size "
            "and other non-zero dimensions for input"
        )
    if W_in <= 0:
        raise ValueError("Input width must be greater than 0 for replication padding")
    if W_out <= 0:
        raise RuntimeError(
            f"replication_pad1d: output spatial dimension is non-positive: "
            f"output size {W_out}"
        )

    if out is None:
        out3 = torch.empty((N, C, W_out), device=x.device, dtype=x.dtype)
    else:
        expected = (C, W_out) if is_2d else (N, C, W_out)
        if tuple(out.shape) != expected:
            raise ValueError(
                f"Provided out tensor has shape {tuple(out.shape)}, expected {expected}"
            )
        if out.device != x.device:
            raise ValueError("Input and out must be on the same device")
        if out.dtype != x.dtype:
            raise ValueError("Input and out must have the same dtype")
        out3 = out.unsqueeze(0) if is_2d else out

    total_out = N * C * W_out
    if total_out == 0:
        return out3.squeeze(0) if is_2d else out3

    has_neg_pad = pad_l < 0 or pad_r < 0

    # Dispatch: flat clamp kernel under ~100K elements (measured crossover of
    # the i32 flat kernel vs the vendor-copy segment path; 1D only has one
    # narrow edge pair, so the segment path only pays off on large outputs)
    # and for any negative padding (crop semantics); 3 vendor `_copy_from`
    # segments (interior + left/right edge stripes) above that.
    if has_neg_pad or total_out <= FLAT_LIMIT:
        kout = out3 if out3.is_contiguous() else torch.empty_like(out3)
        with torch_device_fn.device(x.device):
            _launch_flat_clamp(x, kout, W_in, W_out, pad_l, total_out)
        if kout is not out3:
            with torch_device_fn.device(x.device):
                torch.ops.aten._copy_from(kout, out3)
        return out3.squeeze(0) if is_2d else out3

    # Fast path: 3 vendor strided-copy segments (interior + 2 edge stripes).
    # Segment order: interior first, then edges (edge sources read the
    # already-padded interior).
    with torch_device_fn.device(x.device):
        # 1. interior block
        torch.ops.aten._copy_from(x, out3[:, :, pad_l : pad_l + W_in])
        # 2-3. width edges (left / right first-interior-element replicated)
        if pad_l:
            torch.ops.aten._copy_from(
                out3[:, :, pad_l : pad_l + 1].expand(N, C, pad_l),
                out3[:, :, :pad_l],
            )
        if pad_r:
            torch.ops.aten._copy_from(
                out3[:, :, pad_l + W_in - 1 : pad_l + W_in].expand(N, C, pad_r),
                out3[:, :, pad_l + W_in :],
            )

    return out3.squeeze(0) if is_2d else out3


def replication_pad1d(input: torch.Tensor, padding):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D")
    return launch_replication_pad1d(input, padding, out=None)


def replication_pad1d_out(input: torch.Tensor, padding, out: torch.Tensor):
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD1D_OUT")
    return launch_replication_pad1d(input, padding, out=out)

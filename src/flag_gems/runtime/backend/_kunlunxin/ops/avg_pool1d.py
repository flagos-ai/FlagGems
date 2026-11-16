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
from flag_gems.utils import libentry

# Kunlunxin: use the vendor avg_pool2d, which replaced the scalar
# `if divisor_override != 0:` with tl.where (the generic one trips an
# 'scf.if ... control flow edge' MLIR pass failure on XPU).
from .avg_pool2d import avg_pool2d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XPU3 fast path for the dominant avg_pool1d configuration
# (kernel_size=3, stride=2, padding=1, ceil_mode=False).
#
# The generic path reshapes to 4-D and runs `avg_pool2d`, whose flat kernel
# gathers one tap per unrolled step: every lane issues a stride-`stride` load,
# i.e. a non-affine gather, and the 2-D delegation costs 4-16x vs ATen.
#
# For k=3/s=2/p=1 the window of output `o` is exactly
#     [2o-1, 2o+1]  ->  out[o] = (x[2o-1] + x[2o] + x[2o+1]) / div
# so with the input viewed as *pairs* (`int64` for fp32, `int32` for 16-bit
# dtypes) the two needed words are two dense, unit-stride streams:
#     self = pair[o]   (x[2o], x[2o+1])      -> x64 + q
#     left = pair[o-1] (x[2o-2], x[2o-1])    -> x64 + q - 1
# Both streams are affine and contiguous; the taps are recovered with 32/64-bit
# bit ops (measured: ~2x the time of a pure contiguous copy for the whole op,
# vs ~4.5x for the strided-gather formulation). `reshape`/`tl.split`/
# `tl.gather` are not usable here (see wiki: unsupported / uni_sram).
#
# `pair[o-1]` of the very first output is out of the allocation, so the head of
# the output is computed by a separate, compile-time "safe" specialization that
# clamps the index (it de-vectorizes, hence it only covers a few hundred
# outputs). Everything else runs the affine specialization.
# ---------------------------------------------------------------------------

# outputs covered by the clamped head launch. 256 is the measured sweet spot:
# a 64-lane program is latency-bound (14-16us!) while >=512 lanes re-introduces
# enough clamped-index work to cost more than the launch itself.
HEAD_OUT = 256
# below this output count a single clamped (de-vectorized) launch is faster than
# two specialized launches, because the whole op is launch-bound anyway
SINGLE_LAUNCH_MAX = 4096


def _pair_block(total: int) -> int:
    """Tile width for the affine pair kernel (measured medians on P800/XPU3)."""
    if total <= 4096:
        return 1024
    if total <= 16384:
        return 2048
    return 8192


@libentry()
@triton.jit
def _avg_pool1d_pair_kernel(
    xp,
    yp,
    total,
    off,
    OW: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    SAFE: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """`avg_pool1d(k=3, s=2, p=1, ceil_mode=False)` on the pair view.

    MODE 0: fp32 input viewed as int64 pairs, fp32 output.
    MODE 1: fp16 input viewed as int32 pairs, fp16 output.
    MODE 2: bf16 input viewed as int32 pairs, int16 output (raw bf16 bits).
    """
    lan = tl.arange(0, BLOCK)
    q = (tl.program_id(0) * BLOCK + lan + off).to(tl.int32)
    m = q < total
    if SAFE:
        # clamped index (de-vectorizes): only used for the small-output path and
        # for the head launch, where the extra offset of the first lane would
        # read pair[-1] (out of the allocation).
        left = tl.maximum(q - 1, 0)
    else:
        left = q - 1
    v0 = tl.load(xp + q, mask=m, other=0)
    v1 = tl.load(xp + left, mask=m, other=0)
    if MODE == 0:
        v0 = v0.to(tl.uint64, bitcast=True)
        v1 = v1.to(tl.uint64, bitcast=True)
        lo0 = (v0 & 0xFFFFFFFF).to(tl.uint32).to(tl.float32, bitcast=True)
        hi0 = (v0 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
        hi1 = (v1 >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    elif MODE == 1:
        v0 = v0.to(tl.uint32, bitcast=True)
        v1 = v1.to(tl.uint32, bitcast=True)
        lo0 = (v0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        hi0 = (v0 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        hi1 = (v1 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    else:
        v0 = v0.to(tl.uint32, bitcast=True)
        v1 = v1.to(tl.uint32, bitcast=True)
        lo0 = ((v0 & 0xFFFF) << 16).to(tl.float32, bitcast=True)
        hi0 = (v0 & 0xFFFF0000).to(tl.float32, bitcast=True)
        hi1 = (v1 & 0xFFFF0000).to(tl.float32, bitcast=True)
    ow = q % OW
    if COUNT_INCLUDE_PAD:
        acc = tl.where(ow > 0, lo0 + hi0 + hi1, lo0 + hi0)
        res = acc * (1.0 / 3.0)
    else:
        # leading edge window has 2 in-range taps
        acc = tl.where(ow > 0, lo0 + hi0 + hi1, lo0 + hi0)
        res = tl.where(ow > 0, acc * (1.0 / 3.0), acc * 0.5)
    if MODE == 2:
        rb = res.to(tl.uint32, bitcast=True)
        rounded = (rb + 0x7FFF + ((rb >> 16) & 1)) >> 16
        tl.store(yp + q, rounded.to(tl.uint16), mask=m)
    else:
        tl.store(yp + q, res.to(yp.dtype.element_ty), mask=m)


def _pair_fast_path(
    input: torch.Tensor, kernel_size, stride, padding, ceil_mode, count_include_pad
) -> bool:
    if ceil_mode:
        return False
    k = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
    if k != 3:
        return False
    if stride is None or (isinstance(stride, (list, tuple)) and len(stride) == 0):
        s = k
    elif isinstance(stride, int):
        s = stride
    else:
        s = stride[0]
    if s != 2:
        return False
    p = padding if isinstance(padding, int) else padding[0]
    if p != 1:
        return False
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    lin = input.shape[2]
    if lin % 2 != 0:
        return False
    # out_len must equal lin // 2 so that flat output index == flat pair index
    if (lin - k + 2 * p) // s + 1 != lin // 2:
        return False
    return True


def avg_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Average pooling operation over 1D input.

    Implemented by reshaping to 2D and calling avg_pool2d with kernel_size=(1, kernel_size).
    """
    logger.debug("GEMS AVG_POOL1D")

    # Input shape: (N, C, L) -> reshape to (N, C, 1, L)
    assert input.ndim == 3, f"avg_pool1d expects 3D input, got {input.ndim}D"

    if _pair_fast_path(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad
    ):
        return _avg_pool1d_pair(input, kernel_size, count_include_pad)

    input_4d = input.unsqueeze(2)  # (N, C, L) -> (N, C, 1, L)

    # Convert 1D parameters to 2D: kernel_size -> (1, kernel_size)
    if isinstance(kernel_size, int):
        kernel_size_1d = kernel_size
        kernel_size_2d = (1, kernel_size)
    else:
        kernel_size_1d = kernel_size[0]
        kernel_size_2d = (1, kernel_size[0])

    # stride=None or stride=[] means stride=kernel_size (PyTorch default behavior)
    if stride is None or (isinstance(stride, list) and len(stride) == 0):
        stride_2d = (1, kernel_size_1d)
    elif isinstance(stride, int):
        stride_2d = (1, stride)
    else:
        stride_2d = (1, stride[0])

    if isinstance(padding, int):
        padding_2d = (0, padding)
    else:
        padding_2d = (0, padding[0])

    # Call avg_pool2d
    output_4d = avg_pool2d(
        input_4d,
        kernel_size=kernel_size_2d,
        stride=stride_2d,
        padding=padding_2d,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )

    # Reshape back: (N, C, 1, L_out) -> (N, C, L_out)
    return output_4d.squeeze(2)


def _avg_pool1d_pair(input: torch.Tensor, kernel_size, count_include_pad: bool):
    logger.debug("GEMS_KUNLUNXIN AVG_POOL1D PAIR")
    inp = input.contiguous()
    n, c, lin = inp.shape
    out_len = lin // 2
    out = torch.empty((n, c, out_len), dtype=inp.dtype, device=inp.device)
    total = n * c * out_len
    if total == 0:
        return out

    if inp.dtype == torch.float32:
        mode = 0
        xv = inp.view(torch.int64)
        yv = out
    elif inp.dtype == torch.float16:
        mode = 1
        xv = inp.view(torch.int32)
        yv = out
    else:
        mode = 2
        xv = inp.view(torch.int32)
        yv = out.view(torch.int16)

    block = _pair_block(total)
    head = min(HEAD_OUT, total)
    with torch_device_fn.device(inp.device):
        if total <= SINGLE_LAUNCH_MAX:
            # whole op is launch-bound: one clamped launch, no second kernel
            _avg_pool1d_pair_kernel[(triton.cdiv(total, block),)](
                xv,
                yv,
                total,
                0,
                OW=out_len,
                COUNT_INCLUDE_PAD=count_include_pad,
                SAFE=True,
                MODE=mode,
                BLOCK=block,
                num_warps=4,
            )
        else:
            _avg_pool1d_pair_kernel[(triton.cdiv(head, head),)](
                xv,
                yv,
                total,
                0,
                OW=out_len,
                COUNT_INCLUDE_PAD=count_include_pad,
                SAFE=True,
                MODE=mode,
                BLOCK=head,
                num_warps=4,
            )
            rest = total - head
            _avg_pool1d_pair_kernel[(triton.cdiv(rest, block),)](
                xv,
                yv,
                total,
                head,
                OW=out_len,
                COUNT_INCLUDE_PAD=count_include_pad,
                SAFE=False,
                MODE=mode,
                BLOCK=block,
                num_warps=4,
            )
    return out

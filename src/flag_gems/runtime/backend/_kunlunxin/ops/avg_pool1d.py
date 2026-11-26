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

from flag_gems.utils import libentry

# Kunlunxin: use the vendor avg_pool2d, which replaced the scalar
# `if divisor_override != 0:` with tl.where (the generic one trips an
# 'scf.if ... control flow edge' MLIR pass failure on XPU).
from .avg_pool2d import avg_pool2d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pair-view fast path for k=3, s=2, p=1, ceil_mode=False and even row length.
#
# For that geometry `out[o] = (x[2o-1] + x[2o] + x[2o+1]) / div` and `out_l == L/2`,
# so with the input viewed as pairs (int64 for fp32, int32 for 16-bit) the flat
# output index q is also the flat pair index:
#
#     out[q] = (lo(pair[q]) + hi(pair[q]) + hi(pair[q-1])) / div
#
# Both streams are unit-stride and dense (no stride-2 gather, which is what the
# 4-D avg_pool2d delegation lowers to on this backend).  `pair[q-1]` is out of
# bounds for q == 0, so the first `_PAIR_HEAD_OUT` outputs are covered by a
# separate clamped launch and the rest runs on the affine body.  For tiny totals
# the whole op is launch-bound and a single clamped launch is faster.
#
# Only `o == 0` lanes (row starts) take the padding into account: their left tap
# is zero and `count_include_pad=False` divides by 2 instead of 3.  That is a
# value-level `tl.where`, which keeps the load addresses affine.
# ---------------------------------------------------------------------------

_PAIR_MIN_TOTAL = 4096  # below this a single clamped launch is faster
_PAIR_SMALL_BLOCK = 512
_PAIR_HEAD_OUT = 512  # SAFE head lanes (keeps the body base aligned)


def _pair_body_block(total):
    """Measured body tile (median of do_bench, 5 rounds, constexpr OUT_L)."""
    if total <= 65536:
        return 8192
    if total <= 131072:
        return 4096
    if total <= 262144:
        return 8192
    return 16384


def _pair_unroll(dtype, total):
    """Measured unroll factor: fp16 always 8, fp32 peaks at 4 below 512k."""
    if dtype == torch.float16:
        return 8
    if dtype == torch.float32:
        return 8 if total > 262144 else 4
    return 2


@triton.jit
def _pair_lo_f32(v, WIDE: tl.constexpr, IS_BF16: tl.constexpr):
    """Low half of a packed pair, exact in fp32."""
    if WIDE:
        return v.to(tl.int32).to(tl.float32, bitcast=True)
    elif IS_BF16:
        # Place the bf16 bit pattern in the high half of an fp32 (exact).
        return (v << 16).to(tl.float32, bitcast=True)
    else:
        return v.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _pair_hi_f32(v, WIDE: tl.constexpr, IS_BF16: tl.constexpr):
    """High half of a packed pair, exact in fp32."""
    if WIDE:
        return (v >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    elif IS_BF16:
        return (v & -65536).to(tl.float32, bitcast=True)
    else:
        return (v >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@libentry()
@triton.jit
def avg_pool1d_pair_safe_kernel(
    pair_ptr,
    out_ptr,
    total,
    OUT_L: tl.constexpr,
    WIDE: tl.constexpr,
    IS_BF16: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Clamped-left pair kernel: `left = pair[max(q-1, 0)]`.

    Used for the whole range when `total < _PAIR_MIN_TOTAL` and for the first
    `_PAIR_HEAD_OUT` outputs otherwise.  `OUT_L` is constexpr so the row-start
    test `q % OUT_L` folds into shift/multiply instead of a soft integer modulo.
    """
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    self_w = tl.load(pair_ptr + q, mask=mask)
    left_w = tl.load(pair_ptr + tl.maximum(q - 1, 0), mask=mask)
    lo = _pair_lo_f32(self_w, WIDE, IS_BF16)
    hi = _pair_hi_f32(self_w, WIDE, IS_BF16)
    lf = _pair_hi_f32(left_w, WIDE, IS_BF16)
    o = q % OUT_L
    has_left = o != 0
    acc = lo + hi + tl.where(has_left, lf, 0.0)
    if COUNT_INCLUDE_PAD:
        divisor = 3.0
    else:
        divisor = tl.where(has_left, 3.0, 2.0)
    res = acc / divisor
    tl.store(out_ptr + q, res.to(out_ptr.type.element_ty), mask=mask)


@libentry()
@triton.jit
def avg_pool1d_pair_body_kernel(
    pair_ptr,
    out_ptr,
    total,
    OUT_L: tl.constexpr,
    WIDE: tl.constexpr,
    IS_BF16: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    Q_OFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Affine pair kernel: `left = pair[q-1]` with `q >= Q_OFF > 0`."""
    q = Q_OFF + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = q < total
    self_w = tl.load(pair_ptr + q, mask=mask)
    left_w = tl.load(pair_ptr + q - 1, mask=mask)
    lo = _pair_lo_f32(self_w, WIDE, IS_BF16)
    hi = _pair_hi_f32(self_w, WIDE, IS_BF16)
    lf = _pair_hi_f32(left_w, WIDE, IS_BF16)
    o = q % OUT_L
    acc = lo + hi + tl.where(o != 0, lf, 0.0)
    if COUNT_INCLUDE_PAD:
        divisor = 3.0
    else:
        divisor = tl.where(o != 0, 3.0, 2.0)
    res = acc / divisor
    tl.store(out_ptr + q, res.to(out_ptr.type.element_ty), mask=mask)


def _pair_params(kernel_size, stride, padding, ceil_mode):
    """Return (k, s, p) as ints, or None when the geometry is not supported."""
    if ceil_mode:
        return None

    def _scalar(value):
        if isinstance(value, int):
            return value
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return value[0]
        return None

    k = _scalar(kernel_size)
    # stride=None/[] means stride == kernel_size (PyTorch default).
    s = (
        k
        if stride is None or (isinstance(stride, (list, tuple)) and not stride)
        else _scalar(stride)
    )
    p = _scalar(padding)
    if k is None or s is None or p is None:
        return None
    return k, s, p


def _avg_pool1d_pair(
    input: torch.Tensor, kernel_size, stride, padding, ceil_mode, count_include_pad
):
    """Pair-view fast path; returns None when the geometry is not covered."""
    params = _pair_params(kernel_size, stride, padding, ceil_mode)
    if params != (3, 2, 1):
        return None
    if input.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        return None
    if not input.is_contiguous():
        return None

    n, c, in_l = input.shape
    if in_l % 2 != 0 or in_l < 2:
        return None

    out_l = in_l // 2
    total = n * c * out_l
    if total > 2**31 - 1:
        return None

    out = torch.empty((n, c, out_l), device=input.device, dtype=input.dtype)
    if total == 0:
        return out

    wide = input.dtype == torch.float32
    pair = input.view(torch.int64 if wide else torch.int32).reshape(-1)
    kwargs = dict(
        WIDE=wide,
        IS_BF16=input.dtype == torch.bfloat16,
        COUNT_INCLUDE_PAD=bool(count_include_pad),
    )

    if total < _PAIR_MIN_TOTAL:
        block = _PAIR_SMALL_BLOCK
        avg_pool1d_pair_safe_kernel[(triton.cdiv(total, block),)](
            pair,
            out,
            total,
            OUT_L=out_l,
            BLOCK=block,
            num_warps=4,
            **kwargs,
        )
        return out

    head = min(_PAIR_HEAD_OUT, total)
    avg_pool1d_pair_safe_kernel[(1,)](
        pair,
        out,
        head,
        OUT_L=out_l,
        BLOCK=_PAIR_HEAD_OUT,
        num_warps=4,
        **kwargs,
    )
    block = _pair_body_block(total)
    rest = total - head
    avg_pool1d_pair_body_kernel[(triton.cdiv(rest, block),)](
        pair,
        out,
        total,
        OUT_L=out_l,
        Q_OFF=_PAIR_HEAD_OUT,
        BLOCK=block,
        num_warps=4,
        unroll_num=_pair_unroll(input.dtype, total),
        **kwargs,
    )
    return out


def avg_pool1d(
    input: torch.Tensor,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
):
    """Average pooling operation over 1D input.

    Runs a pair-view kernel for the k=3/s=2/p=1 geometry (the common case) and
    otherwise reshapes to 4-D and delegates to the vendor avg_pool2d.
    """
    logger.debug("GEMS_KUNLUNXIN AVG_POOL1D")

    # Input shape: (N, C, L)
    assert input.ndim == 3, f"avg_pool1d expects 3D input, got {input.ndim}D"

    fast = _avg_pool1d_pair(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad
    )
    if fast is not None:
        return fast

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

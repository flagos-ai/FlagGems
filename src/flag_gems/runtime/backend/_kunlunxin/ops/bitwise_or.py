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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def _pack_scalar_i16_to_i32(s):
    s = int(s) & 0xFFFF
    return s | (s << 16)


def _pack_scalar_bool_to_i32(b):
    v = int(b) & 0xFF
    return v | (v << 8) | (v << 16) | (v << 24)


def _try_view_i32(t):
    """Flatten and view as int32. Returns None if not possible."""
    if not t.is_contiguous():
        return None
    if t.numel() * t.element_size() % 4 != 0:
        return None
    try:
        return t.reshape(-1).view(torch.int32)
    except RuntimeError:
        return None


# Packing bool/int16 into int32 gives 4x/2x fewer elements, which is a large win
# on big, bandwidth-bound tensors.  On small, latency-bound tensors the extra
# reshape/view wrapping around the kernel dominates and a kernel on the natural
# element view is markedly faster (measured out-of-place: bool [64,64] 0.37 ->
# ~1.1x, int16 similar; the packed path only pulls ahead once the tensor is large
# enough to amortise that wrapping).  Only pack above this element count.
_PACK_MIN_NUMEL = 1 << 18


def bitwise_or_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR")
    if (
        A.dtype == B.dtype
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_strided(A.shape, A.stride(), dtype=A.dtype, device=A.device)
        return bitwise_or_func(A, B, out0=out)
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_")
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def bitwise_or_func_scalar(x, y):
    # `y` is a runtime scalar (do_not_specialize); `x | y` would promote x to
    # i32 (int16/int32) or emit a mixed-width `arith.ori (i8, i1 splat)` (bool),
    # both of which the XPU backend lowers far slower than the same-shape tensor
    # kernel. Casting the scalar to a matching width is bit-identical to torch's
    # scalar-truncation semantics (trunc(a | b) == trunc(a) | trunc(b)).
    if x.dtype == tl.int1:
        return (x | y.to(tl.int8)).to(tl.int1)
    return x | y.to(x.dtype)


def bitwise_or_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_SCALAR")
    if A.dtype == torch.bool:
        if A.numel() >= _PACK_MIN_NUMEL:
            a32 = _try_view_i32(A)
            if a32 is not None:
                return (
                    bitwise_or_func_scalar(a32, _pack_scalar_bool_to_i32(B))
                    .view(torch.bool)
                    .reshape(A.shape)
                )
        return bitwise_or_func_scalar(A.view(torch.int8), int(B)).view(torch.bool)
    if A.dtype == torch.int16:
        if A.numel() >= _PACK_MIN_NUMEL:
            a32 = _try_view_i32(A)
            if a32 is not None:
                return (
                    bitwise_or_func_scalar(a32, _pack_scalar_i16_to_i32(B))
                    .view(torch.int16)
                    .reshape(A.shape)
                )
    return bitwise_or_func_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_SCALAR_")
    if A.dtype == torch.bool:
        a32 = _try_view_i32(A)
        if a32 is not None:
            bitwise_or_func_scalar(a32, _pack_scalar_bool_to_i32(B), out0=a32)
            return A
        bitwise_or_func_scalar(A.view(torch.int8), int(B), out0=A.view(torch.int8))
        return A
    if A.dtype == torch.int16:
        a32 = _try_view_i32(A)
        if a32 is not None:
            bitwise_or_func_scalar(a32, _pack_scalar_i16_to_i32(B), out0=a32)
            return A
    return bitwise_or_func_scalar(A, B, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_OR_SCALAR_TENSOR")
    if B.dtype == torch.bool:
        b32 = _try_view_i32(B)
        if b32 is not None:
            return (
                bitwise_or_func_scalar(b32, _pack_scalar_bool_to_i32(A))
                .view(torch.bool)
                .reshape(B.shape)
            )
        return bitwise_or_func_scalar(B.view(torch.int8), int(A)).view(torch.bool)
    if B.dtype == torch.int16:
        b32 = _try_view_i32(B)
        if b32 is not None:
            return (
                bitwise_or_func_scalar(b32, _pack_scalar_i16_to_i32(A))
                .view(torch.int16)
                .reshape(B.shape)
            )
    return bitwise_or_func_scalar(B, A)

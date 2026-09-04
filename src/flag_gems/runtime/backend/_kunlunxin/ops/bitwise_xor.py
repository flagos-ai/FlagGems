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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

config_scalar = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_xor_func(x, y):
    return x ^ y


def bitwise_xor_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_TENSOR")
    return bitwise_xor_func(A, B)


def bitwise_xor_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_TENSOR_")
    return bitwise_xor_func(A, B, out0=A)


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_scalar,
)
@triton.jit
def bitwise_xor_func_scalar(x, y):
    return x ^ y


def _word_view(A, denom):
    """int32-word view of a contiguous bool/int16 tensor (4:1 / 2:1 packing).

    Returns None when the view is not applicable (non-contiguous, misaligned
    offset, or size not divisible), in which case callers fall back to the
    per-element codegen kernel. The last-dim check happens *before* the view
    attempt so the per-call overhead stays O(1) and no shape falls into a slow
    exception path; the returned view shares A's storage.
    """
    if A.numel() == 0 or A.numel() % denom != 0 or not A.is_contiguous():
        return None
    if A.size(-1) % denom != 0:
        return None
    try:
        return A.view(torch.int32)
    except (RuntimeError, TypeError):
        return None
    except Exception:
        return None


def _packed_scalar(A, B):
    """(word_val, denom) when the packed int32-word XOR path is lawful.

    bool -> 4 bytes per int32 word (scalar replicated to 0x01010101 / 0);
    int16 -> 2 words per int32 (scalar replicated in both 16-bit lanes).
    B must be a Python bool for bool tensors and an in-range int for int16;
    anything else falls back to the per-element kernel (torch semantics kept).
    """
    dtype = A.dtype
    if dtype == torch.bool:
        if not isinstance(B, bool):
            return None
        return (0x01010101 if B else 0, 4)
    if dtype == torch.int16:
        if not isinstance(B, int) or not (-(1 << 15) <= B < (1 << 15)):
            return None
        s = int(B) & 0xFFFF
        return (s | (s << 16), 2)
    return None


def bitwise_xor_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_SCALAR")
    pack = _packed_scalar(A, B)
    if pack is not None:
        word_val, denom = pack
        A_word = _word_view(A, denom)
        if A_word is not None:
            out = torch.empty_like(A)
            bitwise_xor_func_scalar(A_word, word_val, out0=_word_view(out, denom))
            return out
    return bitwise_xor_func_scalar(A, B)


def bitwise_xor_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_SCALAR_")
    pack = _packed_scalar(A, B)
    if pack is not None:
        word_val, denom = pack
        A_word = _word_view(A, denom)
        if A_word is not None:
            bitwise_xor_func_scalar(A_word, word_val, out0=A_word)
            return A
    return bitwise_xor_func_scalar(A, B, out0=A)


def bitwise_xor_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_XOR_SCALAR_TENSOR")
    return bitwise_xor_func_scalar(B, A)


xor = bitwise_xor_tensor
xor_ = bitwise_xor_tensor_
xor_scalar = bitwise_xor_scalar
xor_scalar_ = bitwise_xor_scalar_
xor_scalar_tensor = bitwise_xor_scalar_tensor

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
    # isCloseMemoryAsync must stay at its default (True = async copy closed).
    # Enabling async copy (=False) together with unroll_num=8 makes the LLVM
    # lowering materialize a ~478-pointer local-buffer struct that is re-printed
    # on every insertvalue, blowing the compiled IR up to ~9GB (see
    # benchmark/ir_dump/ir-bitwise_and_tensor-dev5.log). unroll_num/autoGrid are
    # kept for the #1277 speedup; only the async pipeline is dropped.
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")], config=config_)
@triton.jit
def bitwise_and_func(x, y):
    return x & y


def bitwise_and_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND")
    if (
        A.dtype == B.dtype
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        out = torch.empty_strided(A.shape, A.stride(), dtype=A.dtype, device=A.device)
        return bitwise_and_func(A, B, out0=out)
    return bitwise_and_func(A, B)


def bitwise_and_tensor_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_")
    return bitwise_and_func(A, B, out0=A)


# Scalar (tensor-vs-scalar) path.
@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")], config=config_
)
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def bitwise_and_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR")
    if (
        A.dtype in (torch.bool, torch.int16)
        and A.is_contiguous()
        and isinstance(B, (int, bool))
    ):
        nbytes = A.numel() * A.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            scalar = int(B)
            if A.dtype == torch.bool:
                if type(B) is not bool:
                    return bitwise_and_func_scalar(A, B)
                if nbytes < 8 * 1024:
                    return bitwise_and_func_scalar(A, B)
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                if not -0x8000 <= scalar <= 0x7FFF:
                    return bitwise_and_func_scalar(A, B)
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            try:
                in_view = A.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_and_func_scalar(A, B)
            n_words = nbytes // 4
            out = torch.empty_strided(
                (n_words,), (1,), dtype=torch.int32, device=A.device
            )
            bitwise_and_func_scalar(in_view, mask, out0=out)
            return out.view(A.dtype).reshape(A.shape)
    return bitwise_and_func_scalar(A, B)


def bitwise_and_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR_")
    if (
        A.dtype in (torch.bool, torch.int16)
        and A.is_contiguous()
        and isinstance(B, (int, bool))
    ):
        nbytes = A.numel() * A.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            scalar = int(B)
            if A.dtype == torch.bool:
                if type(B) is not bool:
                    return bitwise_and_func_scalar(A, B, out0=A)
                # torch bool conversion of a scalar = low bit (verified:
                # 2->False, 3->True, 5->True, -2->False on reference).
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                if not -0x8000 <= scalar <= 0x7FFF:
                    return bitwise_and_func_scalar(A, B, out0=A)
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            try:
                in_view = A.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_and_func_scalar(A, B, out0=A)
            bitwise_and_func_scalar(in_view, mask, out0=in_view)
            return A
    return bitwise_and_func_scalar(A, B, out0=A)


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR_TENSOR")
    if (
        B.dtype in (torch.bool, torch.int16)
        and B.is_contiguous()
        and isinstance(A, (int, bool))
    ):
        nbytes = B.numel() * B.element_size()
        if nbytes > 0 and nbytes % 4 == 0:
            scalar = int(A)
            if B.dtype == torch.bool:
                if type(A) is not bool:
                    return bitwise_and_func_scalar(B, A)
                # torch bool conversion of a scalar = low bit (verified:
                # 2->False, 3->True, 5->True, -2->False on reference).
                mask = 0x01010101 if (scalar & 1) else 0
            else:
                s = scalar & 0xFFFF
                mask = s | (s << 16)
            n_words = nbytes // 4
            out = torch.empty_strided(
                (n_words,), (1,), dtype=torch.int32, device=B.device
            )
            try:
                in_view = B.reshape(-1).view(torch.int32)
            except RuntimeError:  # e.g. unaligned storage offset
                return bitwise_and_func_scalar(B, A)
            bitwise_and_func_scalar(in_view, mask, out0=out)
            return out.view(B.dtype).reshape(B.shape)
    return bitwise_and_func_scalar(B, A)

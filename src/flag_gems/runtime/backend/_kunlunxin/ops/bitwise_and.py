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


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_,
)
@triton.jit
def bitwise_and_func_scalar(x, y):
    # Cast the scalar to x's dtype first. A Python scalar arrives as int32, and
    # `x & val0` promotes int16/bool x to i32 -> the i32 intermediate +
    # truncate-back scalarizes the kernel (int16 scalar bitwise_and_ 0.41x on
    # 1G elements vs 0.99x for the all-tensor variant, 2026-09-04). Keeping the
    # and in x's dtype keeps the vectorized load/store path.
    return x & y.to(x.dtype)


def _bool_scalar_result(t, scalar, inplace=False):
    # bool_tensor & bool_scalar simplifies exactly: x & True == x, x & False ==
    # False. Returns the result tensor or None if the kernel path must be used.
    # Rationale: the generic scalar kernel's i1 splat of the scalar scalarizes
    # the whole kernel (bool scalar bitwise_and_ 0.17x on 1G vs 1.0x for the
    # all-tensor variant, 2026-09-04), so the bool-scalar case takes this
    # 0-cost algebraic path instead. In-place callers get the input tensor
    # back; out-of-place callers get a fresh copy so the input is never aliased
    # or mutated (matching torch.bitwise_and out-of-place semantics).
    if (
        t.dtype == torch.bool
        and isinstance(scalar, (bool, int))
        and scalar in (0, 1, True, False)
    ):
        if scalar:
            return t if inplace else t.clone()
        if inplace:
            return t.fill_(False)
        return torch.zeros_like(t)
    return None


def bitwise_and_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR")
    r = _bool_scalar_result(A, B)
    if r is not None:
        return r
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
    r = _bool_scalar_result(A, B, inplace=True)
    if r is not None:
        return r
    return bitwise_and_func_scalar(A, B, out0=A)


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS_KUNLUNXIN BITWISE_AND_SCALAR_TENSOR")
    r = _bool_scalar_result(B, A)
    if r is not None:
        return r
    return bitwise_and_func_scalar(B, A)

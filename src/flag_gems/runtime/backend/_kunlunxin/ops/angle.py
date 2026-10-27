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
import math

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

atan2 = tl_extra_shim.atan2

config_complex_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)

config_float_int_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)

config_complex_packed_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    is_tensor=[True, True],
    promotion_methods=[(0, "DEFAULT")],
    config=config_complex_,
)
@triton.jit
def angle_func(real, imag):
    real_last, imag_last = (
        (real.to(tl.float32), imag.to(tl.float32))
        if real.dtype == tl.float16
        else (real, imag)
    )
    result = atan2(imag_last, real_last)
    return result


@pointwise_dynamic(
    is_tensor=[True],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    config=config_float_int_,
)
@triton.jit
def angle_float_and_int(real):
    if real.dtype == tl.int1:
        return tl.where(real >= 0, 0.0, math.pi)
    if real.dtype == tl.int8 or real.dtype == tl.int16:
        zero = 0.0
        pi = math.pi
        real_positive = real >= zero
        return tl.where(real_positive, zero, pi)
    real_last = real.to(tl.float32)
    pi = math.pi
    big = 8.6e37
    return pi * tl.minimum(tl.maximum(real_last * (0.0 - big), 0.0), 1.0)


@pointwise_dynamic(
    is_tensor=[True],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    config=config_complex_packed_,
)
@triton.jit
def angle_complex64_packed(packed):
    real = packed.to(tl.int32).to(tl.float32, bitcast=True)
    imag = (packed >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    return atan2(imag, real)


def angle(input_tensor: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN ANGLE")
    if input_tensor.dtype == torch.complex32 or input_tensor.dtype == torch.complex64:
        if input_tensor.dtype == torch.complex64 and input_tensor.is_contiguous():
            packed = input_tensor.view(torch.int64)
            return angle_complex64_packed(
                packed,
                out0=torch.empty(
                    packed.shape, dtype=torch.float32, device=packed.device
                ),
            )
        real = input_tensor.real
        imag = input_tensor.imag
        if (
            real.dtype == imag.dtype
            and real.dtype in (torch.float32, torch.float16)
            and real.shape == imag.shape
        ):
            return angle_func(
                real,
                imag,
                out0=torch.empty(real.shape, dtype=real.dtype, device=real.device),
            )
        return angle_func(real, imag)
    else:
        real = input_tensor
        if real.is_floating_point():
            out0 = torch.empty(real.shape, dtype=real.dtype, device=real.device)
        else:
            out0 = torch.empty(real.shape, dtype=torch.float32, device=real.device)
        return angle_float_and_int(real, out0=out0)

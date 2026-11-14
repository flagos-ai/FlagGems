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

import triton
import triton.language as tl

from flag_gems.ops.convolution_backward import (
    _convolution_backward_impl,
    _convolution_backward_out_impl,
    _input_gradient_vector,
    _spatial_offset,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _input_gradient_channel_tiled(
    GO,
    W,
    DX,
    N: tl.constexpr,
    CI: tl.constexpr,
    CO: tl.constexpr,
    GROUPS: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    I2: tl.constexpr,
    OUTPUT_SHAPE0: tl.constexpr,
    OUTPUT_SHAPE1: tl.constexpr,
    OUTPUT_SHAPE2: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    IS: tl.constexpr,
    OS: tl.constexpr,
    KS: tl.constexpr,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    STRIDE2: tl.constexpr,
    PAD0: tl.constexpr,
    PAD1: tl.constexpr,
    PAD2: tl.constexpr,
    DIL0: tl.constexpr,
    DIL1: tl.constexpr,
    DIL2: tl.constexpr,
    GS0: tl.constexpr,
    GS1: tl.constexpr,
    GS2: tl.constexpr,
    GS3: tl.constexpr,
    GS4: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    WS2: tl.constexpr,
    WS3: tl.constexpr,
    WS4: tl.constexpr,
    DS0: tl.constexpr,
    DS1: tl.constexpr,
    DS2: tl.constexpr,
    DS3: tl.constexpr,
    DS4: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    ACC: tl.constexpr,
    INDEX: tl.constexpr,
    BM: tl.constexpr,
    BR: tl.constexpr,
    PROGRAM0=None,
):
    spatial_tiles: tl.constexpr = (IS + BM - 1) // BM
    program = tl.program_id(0).to(INDEX)
    spatial = program % spatial_tiles * BM + tl.arange(0, BM)
    channel = program // spatial_tiles % CI
    batch = program // (spatial_tiles * CI)
    valid_m = spatial < IS
    group = channel // (CI // GROUPS)
    dest = (
        batch * DS0
        + channel * DS1
        + _spatial_offset(spatial, I0, I1, I2, DS2, DS3, DS4)
    )
    acc = tl.full((BM,), 0, ACC)
    for r in tl.static_range(CO // GROUPS * KS):
        oc = r // KS
        kernel = r % KS
        go_offset = batch * GS0 + (group * (CO // GROUPS) + oc) * GS1
        if TRANSPOSED:
            weight_offset = channel * WS0 + oc * WS1
        else:
            weight_offset = (group * (CO // GROUPS) + oc) * WS0 + channel % (
                CI // GROUPS
            ) * WS1
        valid = valid_m
        spos = spatial
        kpos = kernel
        for axis in tl.static_range(2, -1, -1):
            size = I0 if axis == 0 else I1 if axis == 1 else I2
            ksize = K0 if axis == 0 else K1 if axis == 1 else K2
            step = STRIDE0 if axis == 0 else STRIDE1 if axis == 1 else STRIDE2
            pad = PAD0 if axis == 0 else PAD1 if axis == 1 else PAD2
            dil = DIL0 if axis == 0 else DIL1 if axis == 1 else DIL2
            osize = (
                OUTPUT_SHAPE0
                if axis == 0
                else OUTPUT_SHAPE1 if axis == 1 else OUTPUT_SHAPE2
            )
            ip = spos % size
            kp = kpos % ksize
            if TRANSPOSED:
                op = ip * step - pad + kp * dil
            else:
                raw = ip + pad - kp * dil
                valid &= raw % step == 0
                op = raw // step
            valid &= (op >= 0) & (op < osize)
            go_offset += op * (GS2 if axis == 0 else GS3 if axis == 1 else GS4)
            weight_offset += kp * (WS2 if axis == 0 else WS3 if axis == 1 else WS4)
            spos = spos // size
            kpos = kpos // ksize
        go = tl.load(GO + go_offset, valid, 0).to(ACC)
        w = tl.load(W + weight_offset).to(ACC)
        # Invalid padding must not turn a masked zero into NaN via nonfinite weights.
        w = tl.where(valid, w, 0)
        acc += go * w
    tl.store(DX + dest, acc, valid_m)


def _input_gradient_config(
    kernel, grid, block, *, input_numel, input_spatial, reduction_size
):
    if kernel is _input_gradient_vector:
        kernel = _input_gradient_channel_tiled
        # A flat grid keeps channels uniform without a grid-y size limit.
        grid = (triton.cdiv(input_spatial, block) * (input_numel // input_spatial),)
    return kernel, grid, block


def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    logger.debug("GEMS_MTHREADS CONVOLUTION_BACKWARD")
    return _convolution_backward_impl(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        _input_gradient_config=_input_gradient_config,
    )


def convolution_backward_out(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
    *,
    out0,
    out1,
    out2,
):
    logger.debug("GEMS_MTHREADS CONVOLUTION_BACKWARD_OUT")
    return _convolution_backward_out_impl(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
        out0=out0,
        out1=out1,
        out2=out2,
        _input_gradient_config=_input_gradient_config,
    )

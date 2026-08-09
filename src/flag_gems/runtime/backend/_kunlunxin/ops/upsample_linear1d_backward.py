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
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def upsample_linear1d_backward_unique_writer_kernel(
    grad_output,
    grad_input,
    in_w: tl.constexpr,
    out_w: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    ALIGN_CORNERS: tl.constexpr,
):
    pid = ext.program_id(0)
    input_x = pid % in_w
    row = pid // in_w
    output_x = tl.arange(0, BLOCK_OUT)
    mask = output_x < out_w

    if ALIGN_CORNERS:
        if out_w > 1:
            source_x = output_x.to(tl.float32) * (in_w - 1) / (out_w - 1)
        else:
            source_x = tl.zeros((BLOCK_OUT,), dtype=tl.float32)
    else:
        source_x = (output_x.to(tl.float32) + 0.5) * in_w / out_w - 0.5

    source_x0 = tl.floor(source_x).to(tl.int32)
    source_x1 = source_x0 + 1
    weight1 = source_x - source_x0
    weight0 = 1.0 - weight1
    source_x0 = tl.minimum(tl.maximum(source_x0, 0), in_w - 1)
    source_x1 = tl.minimum(tl.maximum(source_x1, 0), in_w - 1)
    weight = tl.where(source_x0 == input_x, weight0, 0.0)
    weight += tl.where(source_x1 == input_x, weight1, 0.0)

    grad = tl.load(grad_output + row * out_w + output_x, mask=mask, other=0.0).to(
        tl.float32
    )
    value = tl.sum(grad * weight, axis=0)
    tl.store(grad_input + pid, value)


def upsample_linear1d_backward(
    grad_output,
    output_size,
    input_size,
    align_corners,
    scale_factors=None,
):
    logger.debug("GEMS_KUNLUNXIN UPSAMPLE_LINEAR1D_BACKWARD")
    if len(input_size) == 3:
        n, c, in_w = input_size
    elif len(input_size) == 2:
        n, c, in_w = input_size[0], 1, input_size[1]
    elif len(input_size) == 1:
        n, c, in_w = 1, 1, input_size[0]
    else:
        raise ValueError("input_size must have one to three dimensions")

    if output_size is not None:
        out_w = output_size[0]
    else:
        assert scale_factors is not None
        out_w = int(in_w * scale_factors[0])
    assert grad_output.shape[-1] == out_w

    grad_output = grad_output.contiguous().view(n, c, out_w)
    grad_input = torch.empty(
        (n, c, in_w), dtype=grad_output.dtype, device=grad_output.device
    )
    block_out = triton.next_power_of_2(out_w)
    with torch_device_fn.device(grad_output.device):
        upsample_linear1d_backward_unique_writer_kernel[(n * c * in_w, 1, 1)](
            grad_output,
            grad_input,
            in_w,
            out_w,
            block_out,
            ALIGN_CORNERS=align_corners,
            buffer_size_limit=2048,
            isCloseVectorization=True,
        )
    return grad_input.view(input_size)

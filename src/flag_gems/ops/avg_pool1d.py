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

from flag_gems.ops.avg_pool2d import avg_pool2d

logger = logging.getLogger(__name__)


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

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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.replication_pad3d_backward
# Covers unbatched, batched, singleton, zero-padding, asymmetric, and cropping cases.
@pytest.mark.parametrize("shape", [(3, 2, 4, 5), (2, 3, 4, 5, 6), (1, 2, 1, 1, 1)])
@pytest.mark.parametrize(
    "padding", [(0, 0, 0, 0, 0, 0), (1, 2, 0, 2, 3, 1), (-1, 2, 1, 0, 0, 1)]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad3d_backward(shape, padding, dtype):
    inp = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding
    output_shape = (
        *shape[:-3],
        shape[-3] + pad_front + pad_back,
        shape[-2] + pad_top + pad_bottom,
        shape[-1] + pad_left + pad_right,
    )
    grad_output = torch.randn(output_shape, device=flag_gems.device, dtype=dtype)
    expected = torch.ops.aten.replication_pad3d_backward(
        utils.to_reference(grad_output, True),
        utils.to_reference(inp, True),
        padding,
    )

    with flag_gems.use_gems():
        actual = torch.ops.aten.replication_pad3d_backward(grad_output, inp, padding)

    utils.gems_assert_close(actual, expected, dtype, reduce_dim=max(output_shape[-3:]))


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward_rejects_mismatched_leading_shape():
    inp = torch.randn((2, 3, 4, 5, 6), device=flag_gems.device)
    grad_output = torch.randn((1, 3, 6, 7, 8), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(ValueError, match="leading dimensions"):
        torch.ops.aten.replication_pad3d_backward(grad_output, inp, (1, 1, 1, 1, 1, 1))

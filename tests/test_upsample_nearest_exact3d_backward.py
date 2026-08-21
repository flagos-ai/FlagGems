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

CASES = [
    ((1, 1, 2, 3, 4), (4, 6, 8), (None, None, None)),
    ((2, 3, 4, 5, 6), (6, 8, 3), (None, None, None)),
    ((1, 2, 5, 4, 3), (3, 7, 8), (0.6, 1.75, 8.0 / 3.0)),
    ((1, 1, 2, 3, 4), (4, 6, 8), (1.5, 2.5, 0.75)),
    ((2, 4, 6, 7, 8), (6, 7, 8), (None, 1.0, -1.0)),
]


@pytest.mark.upsample_nearest_exact3d_backward
@pytest.mark.parametrize("input_size,output_size,scales", CASES)
@pytest.mark.parametrize("dtype", [*utils.FLOAT_DTYPES, torch.float64])
def test_upsample_nearest_exact3d_backward(input_size, output_size, scales, dtype):
    grad_output = torch.randn(
        (*input_size[:2], *output_size), dtype=dtype, device=flag_gems.device
    )
    if scales == (1.5, 2.5, 0.75):
        # Native CPU ignores inconsistent explicit scales in this overload,
        # whereas CUDA uses them.  FlagGems implements the CUDA dispatch key.
        ref = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            grad_output, output_size, input_size, *scales
        )
        if utils.TO_CPU:
            ref = ref.cpu()
    else:
        ref = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            utils.to_reference(grad_output), output_size, input_size, *scales
        )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            grad_output, output_size, input_size, *scales
        )

    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.upsample_nearest_exact3d_backward
def test_upsample_nearest_exact3d_backward_uint8():
    input_size = (1, 2, 3, 4, 5)
    output_size = (7, 6, 9)
    grad_output = torch.randint(
        0,
        256,
        (*input_size[:2], *output_size),
        dtype=torch.uint8,
        device=flag_gems.device,
    )
    ref = torch.ops.aten._upsample_nearest_exact3d_backward.default(
        grad_output, output_size, input_size
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            grad_output, output_size, input_size
        )
    assert torch.equal(result, ref)


@pytest.mark.upsample_nearest_exact3d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact3d_backward_noncontiguous(dtype):
    input_size = (2, 3, 3, 4, 5)
    output_size = (4, 5, 6)
    base = torch.randn((2, 3, 8, 10, 12), dtype=dtype, device=flag_gems.device)
    grad_output = base[:, :, ::2, ::2, ::2]
    assert not grad_output.is_contiguous()

    ref = torch.ops.aten._upsample_nearest_exact3d_backward.default(
        utils.to_reference(grad_output), output_size, input_size
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            grad_output, output_size, input_size
        )
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.upsample_nearest_exact3d_backward_grad_input
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_upsample_nearest_exact3d_backward_grad_input(noncontiguous):
    input_size = (2, 3, 3, 4, 5)
    output_size = (5, 7, 9)
    grad_output = torch.randn(
        (*input_size[:2], *output_size),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref = torch.ops.aten._upsample_nearest_exact3d_backward.default(
        utils.to_reference(grad_output), output_size, input_size
    )

    if noncontiguous:
        grad_input = torch.empty(
            (2, 3, 3, 5, 4), dtype=torch.float32, device=flag_gems.device
        ).transpose(-1, -2)
        assert not grad_input.is_contiguous()
    else:
        # The native out overload resizes an empty destination in place.
        grad_input = torch.empty(0, dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact3d_backward.grad_input(
            grad_output,
            output_size,
            input_size,
            None,
            None,
            None,
            grad_input=grad_input,
        )

    assert result is grad_input
    assert tuple(result.shape) == input_size
    utils.gems_assert_close(result, ref, torch.float32)

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
from .conftest import QUICK_MODE, TO_CPU

SHAPES = (
    [(1, 2, 7, 13)]
    if QUICK_MODE
    else [
        (1, 1, 1, 9),
        (1, 2, 7, 13),
        (2, 3, 17, 8),
        (4, 8, 64, 127),
    ]
)
DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES + [torch.float64]


def _make_grad_output(shape, dtype, noncontiguous=False):
    batch, channels, _, output_w = shape
    if noncontiguous:
        storage = torch.randn(
            (batch, channels, output_w * 2),
            dtype=dtype,
            device=flag_gems.device,
        )
        return storage[..., ::2]
    return torch.randn(
        (batch, channels, output_w),
        dtype=dtype,
        device=flag_gems.device,
    )


@pytest.mark.upsample_nearest_exact1d_backward
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest_exact1d_backward(shape, dtype, noncontiguous):
    batch, channels, input_w, output_w = shape
    grad_output = _make_grad_output(shape, dtype, noncontiguous)
    ref_grad_output = utils.to_reference(grad_output)
    output_size = (output_w,)
    input_size = (batch, channels, input_w)

    reference = torch.ops.aten._upsample_nearest_exact1d_backward.default(
        ref_grad_output, output_size, input_size
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact1d_backward.default(
            grad_output, output_size, input_size
        )

    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.upsample_nearest_exact1d_backward
@pytest.mark.parametrize(
    "input_w,output_w,scale", [(3, 5, 1.6), (4, 7, 2.0), (9, 4, 0.5)]
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest_exact1d_backward_with_scale(input_w, output_w, scale, dtype):
    grad_output = torch.randn((2, 3, output_w), dtype=dtype, device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output)
    output_size = (output_w,)
    input_size = (2, 3, input_w)

    reference = torch.ops.aten._upsample_nearest_exact1d_backward.default(
        ref_grad_output, output_size, input_size, scale
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact1d_backward.default(
            grad_output, output_size, input_size, scale
        )

    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.upsample_nearest_exact1d_backward_grad_input
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_nearest_exact1d_backward_grad_input(dtype, noncontiguous):
    grad_output = torch.randn((2, 3, 13), dtype=dtype, device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output)
    input_size = (2, 3, 7)

    if noncontiguous:
        grad_input = torch.empty((2, 3, 14), dtype=dtype, device=flag_gems.device)[
            ..., ::2
        ]
        ref_grad_input = None
    else:
        grad_input = torch.empty(0, dtype=dtype, device=flag_gems.device)
        ref_grad_input = torch.empty(0, dtype=dtype, device=ref_grad_output.device)

    if ref_grad_input is None:
        # PyTorch 2.9's native out kernel assumes contiguous storage. Compare
        # the logical values with the default overload while still exercising
        # the GEMS grad_input overload and its stride-aware writes.
        reference = torch.ops.aten._upsample_nearest_exact1d_backward.default(
            ref_grad_output, (13,), input_size
        )
    else:
        reference = torch.ops.aten._upsample_nearest_exact1d_backward.grad_input(
            ref_grad_output, (13,), input_size, grad_input=ref_grad_input
        )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact1d_backward.grad_input(
            grad_output, (13,), input_size, grad_input=grad_input
        )

    assert result is grad_input
    if ref_grad_input is not None:
        assert reference is ref_grad_input
    utils.gems_assert_close(result, reference, dtype)


@pytest.mark.upsample_nearest_exact1d_backward
@pytest.mark.parametrize("shape", [(0, 3, 7, 13), (2, 0, 7, 13)])
def test_upsample_nearest_exact1d_backward_empty(shape):
    batch, channels, input_w, output_w = shape
    grad_output = torch.empty(shape[:2] + (output_w,), device=flag_gems.device)
    ref_grad_output = utils.to_reference(grad_output)
    input_size = (batch, channels, input_w)
    reference = torch.ops.aten._upsample_nearest_exact1d_backward.default(
        ref_grad_output, (output_w,), input_size
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact1d_backward.default(
            grad_output, (output_w,), input_size
        )
    utils.gems_assert_equal(result, reference)


@pytest.mark.upsample_nearest_exact1d_backward
@pytest.mark.skipif(TO_CPU, reason="native CPU backward does not support uint8")
def test_upsample_nearest_exact1d_backward_uint8():
    grad_output = torch.randint(
        0, 256, (2, 3, 17), dtype=torch.uint8, device=flag_gems.device
    )
    reference = torch.ops.aten._upsample_nearest_exact1d_backward.default(
        grad_output, (17,), (2, 3, 5)
    )
    with flag_gems.use_gems():
        result = torch.ops.aten._upsample_nearest_exact1d_backward.default(
            grad_output, (17,), (2, 3, 5)
        )
    utils.gems_assert_equal(result, reference)


@pytest.mark.upsample_nearest_exact1d_backward
@pytest.mark.parametrize(
    "output_size,input_size",
    [((0,), (1, 1, 3)), ((6,), (1, 1, 0)), ((6, 7), (1, 1, 3))],
)
def test_upsample_nearest_exact1d_backward_invalid_size(output_size, input_size):
    grad_output = torch.randn((1, 1, max(output_size[0], 1)), device=flag_gems.device)
    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.ops.aten._upsample_nearest_exact1d_backward.default(
            grad_output, output_size, input_size
        )

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

SUPPORTED_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]
if flag_gems.runtime.device.support_fp64:
    SUPPORTED_DTYPES.append(torch.float64)


def _randn(shape, dtype):
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _to_cross_reference(tensor, dtype):
    reference = utils.to_reference(tensor)
    if dtype in (torch.float16, torch.bfloat16):
        # CPU and accelerator cross implementations round low-precision
        # intermediates differently. FP32 opmath is the accurate common
        # reference for both execution locations.
        reference = reference.to(torch.float32)
    return reference


def _assert_cross_close(result, reference, dtype):
    # A cross-product component is a difference of two products, so catastrophic
    # cancellation can amplify the low-precision quantization step. The Triton
    # kernel accumulates in FP32 and is bit-identical to eager, so the residual
    # is purely the input quantization measured against the FP32 reference.
    if dtype == torch.float16:
        utils.gems_assert_close(result, reference, dtype, atol=8e-3)
    elif dtype == torch.bfloat16:
        # BF16 has a wider quantization step than FP16 for unit-scale values.
        utils.gems_assert_close(result, reference, dtype, atol=4e-2)
    else:
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.cross
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "input_shape,other_shape,dim",
    [
        ((3, 4), (3, 4), 0),
        ((2, 3, 4), (1, 3, 4), 1),
        ((4096, 3, 4), (1, 3, 4), 1),
        ((2, 4, 3), (1, 4, 3), 2),
        ((2, 3, 4), (1, 3, 4), -2),
        ((1, 3), (5, 3), -1),
        ((2, 4, 3), (2, 4, 3), -1),
        ((2, 4, 3, 5), (1, 4, 3, 5), 2),
    ],
)
def test_cross(input_shape, other_shape, dim, dtype):
    input = _randn(input_shape, dtype)
    other = _randn(other_shape, dtype)
    ref_input = _to_cross_reference(input, dtype)
    ref_other = _to_cross_reference(other, dtype)

    ref_out = torch.cross(ref_input, ref_other, dim=dim)
    with flag_gems.use_gems(include=["cross"]):
        result = torch.cross(input, other, dim=dim)

    _assert_cross_close(result, ref_out, dtype)


@pytest.mark.cross
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(
    "input_shape,other_shape",
    [
        ((4, 3), (4, 3)),  # first size-3 axis is dim 1
        ((3, 5), (3, 5)),  # first size-3 axis is dim 0
        ((2, 3, 4), (2, 3, 4)),  # first size-3 axis is dim 1
        ((5, 3, 3), (5, 3, 3)),  # ambiguous: first size-3 axis is dim 1
    ],
)
def test_cross_default_dim(input_shape, other_shape, dtype):
    # dim=None must select the first dimension of size 3, matching eager
    # torch.cross semantics.
    input = _randn(input_shape, dtype)
    other = _randn(other_shape, dtype)
    ref_input = _to_cross_reference(input, dtype)
    ref_other = _to_cross_reference(other, dtype)

    ref_out = torch.cross(ref_input, ref_other)
    with flag_gems.use_gems(include=["cross"]):
        result = torch.cross(input, other)

    _assert_cross_close(result, ref_out, dtype)


@pytest.mark.cross_out
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_cross_noncontiguous_input_and_out(dtype):
    input = _randn((2, 4, 3), dtype).transpose(1, 2)
    other = _randn((1, 4, 3), dtype).transpose(1, 2)
    out = torch.empty((2, 4, 3), dtype=dtype, device=flag_gems.device).transpose(1, 2)
    ref_input = _to_cross_reference(input, dtype)
    ref_other = _to_cross_reference(other, dtype)
    ref_out = torch.empty(
        (2, 4, 3), dtype=ref_input.dtype, device=ref_input.device
    ).transpose(1, 2)
    torch.ops.aten.cross.out(ref_input, ref_other, dim=1, out=ref_out)
    with flag_gems.use_gems(include=["cross_out"]):
        result = torch.ops.aten.cross.out(input, other, dim=1, out=out)

    assert result is out
    _assert_cross_close(out, ref_out, dtype)


@pytest.mark.cross
def test_cross_rejects_different_input_ranks():
    input = _randn((3,), torch.float32)
    other = _randn((1, 3), torch.float32)

    with (
        flag_gems.use_gems(include=["cross"]),
        pytest.raises(RuntimeError, match="same number of dimensions"),
    ):
        torch.cross(input, other, dim=0)

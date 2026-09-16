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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_BASE_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8, torch.bool, torch.complex64]
)
if utils.fp64_is_supported:
    _BASE_DTYPES.append(torch.complex128)
TRANSPOSE_COPY_DTYPES = list(dict.fromkeys(_BASE_DTYPES))


def _float8_dtypes():
    dtype_names = [
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ]
    dtype_params = []
    for name in dtype_names:
        if not hasattr(torch, name):
            continue
        dtype = getattr(torch, name)
        try:
            input_probe = torch.empty(1, dtype=torch.uint8, device=flag_gems.device)
            input_probe.view(dtype)
            output_probe = torch.empty(1, dtype=dtype, device=flag_gems.device)
            output_probe.view(torch.uint8)
        except (NotImplementedError, RuntimeError, TypeError) as error:
            dtype_params.append(
                pytest.param(
                    dtype,
                    marks=pytest.mark.skip(
                        reason=(
                            f"{flag_gems.vendor_name} does not support {dtype} "
                            f"byte storage views: {error}"
                        )
                    ),
                )
            )
        else:
            dtype_params.append(dtype)
    return dtype_params


FLOAT8_DTYPES = _float8_dtypes()

TRANSPOSE_COPY_CASES = [
    ((7,), 0, 0),
    ((2, 3), 0, 1),
    ((2, 3, 5), 0, -1),
    ((2, 3, 4, 5), -3, -1),
    ((1, 7, 1), 1, -1),
    ((0, 3, 5), 0, 2),
]

TRANSPOSE_COPY_FLOAT8_CASES = [
    ((), 0, 0, False),
    ((7,), 0, 0, False),
    ((4, 6), 0, 1, False),
    ((2, 3, 4), 0, -1, False),
    ((2, 3, 4, 5), 1, -1, False),
    ((2, 3, 4), 1, 1, False),
    ((2, 3, 5), 0, -1, True),
]


def _make_input(shape, dtype, device):
    numel = math.prod(shape)
    if dtype == torch.bool:
        values = torch.arange(numel, dtype=torch.int64) % 2 == 0
    elif dtype.is_complex:
        real = torch.arange(numel, dtype=torch.float32) - numel // 2
        values = torch.complex(real, real + 1).to(dtype)
    elif dtype.is_floating_point:
        values = torch.arange(numel, dtype=torch.float32).to(dtype)
    else:
        values = torch.arange(numel, dtype=torch.int64).to(dtype)
    return values.reshape(shape).to(device)


def _assert_copy_layout(result, reference, input):
    utils.gems_assert_equal(result, reference)
    assert result.shape == reference.shape
    assert result.stride() == reference.stride()
    assert result.is_contiguous()
    assert not torch._C._is_alias_of(input, result)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape,dim0,dim1", TRANSPOSE_COPY_CASES)
@pytest.mark.parametrize("dtype", TRANSPOSE_COPY_DTYPES)
def test_accuracy_transpose_copy(shape, dim0, dim1, dtype):
    input = _make_input(shape, dtype, flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, dim0, dim1)

    result = flag_gems.transpose_copy(input, dim0, dim1)

    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("dim0,dim1", [(0, 0), (-1, 0), (0, -1), (-1, -1)])
def test_accuracy_transpose_copy_scalar(dim0, dim1):
    input = torch.tensor(3.0, device=flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, dim0, dim1)

    result = flag_gems.transpose_copy(input, dim0, dim1)

    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("dtype", TRANSPOSE_COPY_DTYPES)
def test_accuracy_transpose_copy_non_contiguous(dtype):
    base = _make_input((4, 3, 5), dtype, flag_gems.device)
    input = base[::2]
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 0, -1)

    result = flag_gems.transpose_copy(input, 0, -1)

    assert not input.is_contiguous()
    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
def test_accuracy_transpose_copy_same_dim_does_not_alias():
    input = _make_input((2, 3, 4), torch.float32, flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 1, 1)

    result = flag_gems.transpose_copy(input, 1, 1)

    _assert_copy_layout(result, reference, input)
    assert result.data_ptr() != input.data_ptr()


@pytest.mark.transpose_copy
@pytest.mark.parametrize(
    "shape,dim0,dim1",
    [
        ((), 1, 0),
        ((), -2, 0),
        ((2, 3), 0, 2),
        ((2, 3), -3, 0),
    ],
)
def test_transpose_copy_invalid_dims(shape, dim0, dim1):
    input = _make_input(shape, torch.float32, flag_gems.device)

    with pytest.raises(IndexError, match="Dimension out of range"):
        flag_gems.transpose_copy(input, dim0, dim1)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape,dim0,dim1,non_contiguous", TRANSPOSE_COPY_FLOAT8_CASES)
@pytest.mark.parametrize("dtype", FLOAT8_DTYPES)
def test_accuracy_transpose_copy_float8(shape, dim0, dim1, non_contiguous, dtype):
    base_shape = (shape[0] * 2, *shape[1:]) if non_contiguous else shape
    input_bytes = torch.arange(
        math.prod(base_shape), dtype=torch.uint8, device=flag_gems.device
    ).reshape(base_shape)
    if non_contiguous:
        input_bytes = input_bytes[::2]
    input = (
        input_bytes.reshape(1).view(dtype).reshape(())
        if input_bytes.ndim == 0
        else input_bytes.view(dtype)
    )
    reference = (
        utils.to_reference(input_bytes)
        .transpose(dim0, dim1)
        .clone(memory_format=torch.contiguous_format)
    )

    result = flag_gems.transpose_copy(input, dim0, dim1)

    _assert_copy_layout(result.view(torch.uint8), reference.view(torch.uint8), input)

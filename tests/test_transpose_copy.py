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

_MIN_FP8_E4M3FN_CUDA_CAPABILITY = (8, 9)


def _float8_dtypes():
    dtype_names = ["float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"]
    if flag_gems.vendor_name == "nvidia" and torch.cuda.is_available():
        if torch.cuda.get_device_capability() < _MIN_FP8_E4M3FN_CUDA_CAPABILITY:
            dtype_names.remove("float8_e4m3fn")
    return [getattr(torch, name) for name in dtype_names if hasattr(torch, name)]


FLOAT8_DTYPES = _float8_dtypes()

TRANSPOSE_COPY_CASES = [
    ((7,), 0, 0),
    ((2, 3), 0, 1),
    ((2, 3, 5), 0, -1),
    ((2, 3, 4, 5), -3, -1),
    ((1, 7, 1), 1, -1),
    ((0, 3, 5), 0, 2),
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


def _skip_if_native_clone_is_unsupported(input, dim0, dim1):
    try:
        input.transpose(dim0, dim1).clone(memory_format=torch.contiguous_format)
    except (NotImplementedError, RuntimeError) as error:
        message = str(error).lower()
        if "not implemented" not in message and "not support" not in message:
            raise
        pytest.skip(f"backend clone/copy does not support {input.dtype}: {error}")


@pytest.mark.transpose_copy
@pytest.mark.parametrize("shape,dim0,dim1", TRANSPOSE_COPY_CASES)
@pytest.mark.parametrize("dtype", TRANSPOSE_COPY_DTYPES)
def test_accuracy_transpose_copy(shape, dim0, dim1, dtype):
    input = _make_input(shape, dtype, flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, dim0, dim1)

    with flag_gems.use_gems():
        result = torch.ops.aten.transpose_copy.int(input, dim0, dim1)

    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("dim0,dim1", [(0, 0), (-1, 0), (0, -1), (-1, -1)])
def test_accuracy_transpose_copy_scalar(dim0, dim1):
    input = torch.tensor(3.0, device=flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, dim0, dim1)

    with flag_gems.use_gems():
        result = torch.ops.aten.transpose_copy.int(input, dim0, dim1)

    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
@pytest.mark.parametrize("dtype", TRANSPOSE_COPY_DTYPES)
def test_accuracy_transpose_copy_non_contiguous(dtype):
    base = _make_input((4, 3, 5), dtype, flag_gems.device)
    input = base[::2]
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 0, -1)

    with flag_gems.use_gems():
        result = torch.ops.aten.transpose_copy.int(input, 0, -1)

    assert not input.is_contiguous()
    _assert_copy_layout(result, reference, input)


@pytest.mark.transpose_copy
def test_accuracy_transpose_copy_same_dim_does_not_alias():
    input = _make_input((2, 3, 4), torch.float32, flag_gems.device)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 1, 1)

    with flag_gems.use_gems():
        result = torch.ops.aten.transpose_copy.int(input, 1, 1)

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

    with (
        flag_gems.use_gems(),
        pytest.raises(IndexError, match="Dimension out of range"),
    ):
        torch.ops.aten.transpose_copy.int(input, dim0, dim1)


@pytest.mark.transpose_copy
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not torch.cuda.is_available() or not FLOAT8_DTYPES,
    reason="float8 coverage requires a CUDA-compatible backend and PyTorch float8",
)
@pytest.mark.parametrize("dtype", FLOAT8_DTYPES)
def test_accuracy_transpose_copy_float8(dtype):
    input = torch.arange(24, dtype=torch.uint8, device=flag_gems.device).reshape(4, 6)
    input = input.view(dtype)
    _skip_if_native_clone_is_unsupported(input, 0, 1)
    ref_input = utils.to_reference(input)
    reference = torch.ops.aten.transpose_copy.int(ref_input, 0, 1)

    with flag_gems.use_gems():
        result = torch.ops.aten.transpose_copy.int(input, 0, 1)

    _assert_copy_layout(result.view(torch.uint8), reference.view(torch.uint8), input)

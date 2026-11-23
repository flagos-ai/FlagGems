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
from flag_gems.ops.to import to_copy

from . import accuracy_utils as utils

DTYPES = list(
    dict.fromkeys(
        utils.ALL_FLOAT_DTYPES
        + utils.ALL_INT_DTYPES
        + [torch.int8, torch.uint8, torch.bool]
    )
)
LAYOUTS = [
    ((7, 8), (8, 1), 0),
    ((8, 7), (1, 8), 0),
    ((7, 8), (19, 2), 3),
    ((7, 8), (8, 1), 5),
    ((3, 7, 8), (0, 8, 1), 0),
    ((2, 2, 2, 7), (28, 1, 14, 2), 0),
    ((2, 2, 1, 2, 7), (28, 1, 28, 14, 2), 0),
]


def _values(dtype):
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
        smallest = info.tiny * info.eps
        values = [
            -0.5,
            -0.125,
            -0.0,
            0.0,
            0.125,
            0.5,
            1.0,
            -1.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            info.tiny,
            -info.tiny,
            smallest,
            -smallest,
        ]
    elif dtype == torch.bool:
        values = [False, True, False, True]
    else:
        info = torch.iinfo(dtype)
        values = [0, 1, info.min, info.max]
        if info.min < 0:
            values += [-1, -2, 2]
        if dtype == torch.int64:
            values += [2**40, -(2**40)]
    return torch.tensor(values, dtype=dtype)


def _view(dtype, shape, stride, offset):
    count = offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride)) + 13
    values = _values(dtype)
    storage = values.repeat(math.ceil(count / values.numel()))[:count].clone()
    device_storage = storage.to(flag_gems.device)
    cpu = torch.as_strided(storage, shape, stride, offset)
    device = torch.as_strided(device_storage, shape, stride, offset)
    return cpu, device, storage, device_storage


def _check(cpu, device, storage, device_storage, memory_format):
    expected = cpu.to(dtype=torch.bool, copy=True, memory_format=memory_format)
    actual = to_copy(device, dtype=torch.bool, memory_format=memory_format)
    assert actual is not device
    assert actual.dtype == torch.bool and actual.device == device.device
    assert actual.shape == expected.shape and actual.stride() == expected.stride()
    if device.numel():
        assert not torch._C._is_alias_of(actual, device)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.equal(
        device_storage.cpu().view(torch.uint8), storage.view(torch.uint8)
    )


@pytest.mark.to_copy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,stride,offset", LAYOUTS)
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
def test_to_copy_bool_semantics(dtype, shape, stride, offset, memory_format):
    cpu, device, storage, device_storage = _view(dtype, shape, stride, offset)
    _check(cpu, device, storage, device_storage, memory_format)


@pytest.mark.to_copy
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_to_copy_bool_scalar_and_empty(dtype, shape):
    values = _values(dtype)
    cpu = (
        values[0].clone().reshape(()) if not shape else torch.empty(shape, dtype=dtype)
    )
    device = cpu.to(flag_gems.device)
    actual = to_copy(device, dtype=torch.bool)
    expected = cpu.to(dtype=torch.bool, copy=True)
    assert actual is not device and actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.to_copy
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize(
    "memory_format,shape",
    [(torch.channels_last, (2, 2, 2, 7)), (torch.channels_last_3d, (2, 2, 1, 2, 7))],
)
def test_to_copy_bool_requested_memory_format(dtype, memory_format, shape):
    strides = torch.empty(shape).stride()
    cpu, device, storage, device_storage = _view(dtype, shape, strides, 3)
    _check(cpu, device, storage, device_storage, memory_format)


@pytest.mark.to_copy
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_to_copy_other_destination_controls(dtype):
    cpu = torch.tensor([-2, -1, 0, 1, 2, 3], dtype=torch.float32)
    device = cpu.to(flag_gems.device)
    actual = to_copy(device, dtype=dtype)
    assert actual is not device and actual.dtype == dtype
    torch.testing.assert_close(actual.cpu(), cpu.to(dtype), rtol=0, atol=0)


@pytest.mark.to_copy
@pytest.mark.parametrize("direction", ["cpu_to_device", "device_to_cpu", "cpu_to_cpu"])
def test_to_copy_bool_device_transfer_controls(direction):
    cpu = torch.tensor(
        [-0.5, -0.125, -0.0, 0.0, 0.125, 0.5, float("nan"), float("inf")]
    )
    if direction == "cpu_to_device":
        actual = to_copy(cpu, dtype=torch.bool, device=flag_gems.device)
    elif direction == "device_to_cpu":
        actual = to_copy(cpu.to(flag_gems.device), dtype=torch.bool, device="cpu")
    else:
        actual = to_copy(cpu, dtype=torch.bool, device="cpu")
    torch.testing.assert_close(actual.cpu(), cpu.to(torch.bool), rtol=0, atol=0)

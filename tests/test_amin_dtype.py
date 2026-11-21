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
from flag_gems.ops.amin import amin as generic_amin

DTYPES = [
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]


def make_input(dtype):
    if dtype == torch.bool:
        return torch.arange(105).reshape(3, 5, 7) % 3 == 0
    if dtype.is_floating_point:
        x = torch.arange(105, dtype=torch.float64).reshape(3, 5, 7) / 8
        if dtype == torch.float64:
            x = 2**40 + x
        return x.to(dtype)
    info = torch.iinfo(dtype)
    x = (torch.arange(105).reshape(3, 5, 7) % 31).to(dtype)
    x[0, 0, 0] = info.max
    x[2, 4, 6] = info.min
    return x


@pytest.mark.amin
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", [None, (), 0, -1, (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_amin_dtype_coverage(dtype, dim, keepdim, transpose):
    cpu = make_input(dtype)
    inp = cpu.to(flag_gems.device)
    if transpose:
        cpu = cpu.transpose(0, 2)
        inp = inp.transpose(0, 2)
    expected = torch.amin(cpu, dim=dim, keepdim=keepdim)
    actual = generic_amin(inp, dim=dim, keepdim=keepdim).cpu()
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize("dim", [None, 0, -1, (0, 1)])
def test_amin_large_int64(dim):
    cpu = torch.tensor(
        [
            [2**60 + 3, 2**60 + 1, 2**60 + 2],
            [2**60 + 9, 2**60 + 7, 2**60 + 8],
        ],
        dtype=torch.int64,
    )
    actual = generic_amin(cpu.to(flag_gems.device), dim=dim).cpu()
    torch.testing.assert_close(actual, torch.amin(cpu, dim=dim), rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_amin_nan_inf(dtype):
    cpu = torch.tensor(
        [[float("nan"), 1, float("inf")], [float("inf"), float("inf"), float("inf")]],
        dtype=dtype,
    )
    for dim in (None, 0, 1):
        actual = generic_amin(cpu.to(flag_gems.device), dim=dim).cpu()
        torch.testing.assert_close(
            actual, torch.amin(cpu, dim=dim), rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.amin
@pytest.mark.parametrize("dim", [None, 0, -1, ()])
def test_amin_scalar(dim):
    cpu = torch.tensor(2**60 + 1, dtype=torch.int64)
    actual = generic_amin(cpu.to(flag_gems.device), dim=dim).cpu()
    torch.testing.assert_close(actual, torch.amin(cpu, dim=dim), rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((0, 3), None),
        ((0, 3), 0),
        ((0, 3), 1),
        ((2, 0), 1),
        ((2, 3), 2),
        ((2, 3), (0, 0)),
    ],
)
def test_amin_empty_and_invalid(shape, dim):
    cpu = torch.empty(shape, dtype=torch.int64)
    inp = cpu.to(flag_gems.device)
    try:
        expected = torch.amin(cpu, dim=dim)
    except (IndexError, RuntimeError) as exc:
        with pytest.raises(type(exc)):
            generic_amin(inp, dim=dim)
    else:
        actual = generic_amin(inp, dim=dim).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_amin_integer_extrema(dtype):
    for value in (torch.iinfo(dtype).min, torch.iinfo(dtype).max):
        cpu = torch.full((3, 5), value, dtype=dtype)
        for dim in (None, 0, -1):
            actual = generic_amin(cpu.to(flag_gems.device), dim=dim).cpu()
            torch.testing.assert_close(actual, torch.amin(cpu, dim=dim), rtol=0, atol=0)


@pytest.mark.amin
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float64])
def test_amin_long_reduction(dtype):
    if dtype == torch.bool:
        cpu = torch.ones((3, 2051), dtype=dtype)
    else:
        cpu = torch.arange(6153, dtype=torch.int64).reshape(3, 2051)
        cpu = (cpu + (2**60 if dtype == torch.int64 else 0)).to(dtype)
    for dim in (None, 1):
        actual = generic_amin(cpu.to(flag_gems.device), dim=dim).cpu()
        torch.testing.assert_close(actual, torch.amin(cpu, dim=dim), rtol=0, atol=0)

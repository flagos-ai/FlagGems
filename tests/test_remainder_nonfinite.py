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

import importlib

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = [
    pytest.mark.remainder,
    pytest.mark.remainder_tensor,
    pytest.mark.remainder_tensor_,
    pytest.mark.remainder_scalar,
    pytest.mark.remainder_scalar_,
    pytest.mark.remainder_scalar_tensor,
]
IMPLEMENTATIONS = [
    "generic",
    pytest.param(
        "hygon",
        marks=pytest.mark.skipif(
            flag_gems.vendor_name != "hygon", reason="Hygon override"
        ),
    ),
]
LAYOUTS = ["contiguous", "transpose", "offset_stride"]
SCALARS = [-float("inf"), -2.0, -0.0, 0.0, 2.0, float("inf"), float("nan")]
CASES = [
    ("tt", shape, mode, None)
    for shape in ["matrix", "rhs", "both"]
    for mode in ["functional", "inplace", "out"]
    if not (shape == "both" and mode == "inplace")
]
CASES += [
    ("ts", "matrix", mode, value)
    for value in SCALARS
    for mode in ["functional", "inplace", "out"]
]
CASES += [
    ("st", "matrix", mode, value) for value in SCALARS for mode in ["functional", "out"]
]


def _module(implementation):
    name = (
        "flag_gems.ops.remainder"
        if implementation == "generic"
        else "flag_gems.runtime.backend._hygon.ops.div"
    )
    return importlib.import_module(name)


def _view(storage, layout, width):
    if layout == "transpose":
        return storage.t()
    if layout == "offset_stride":
        return storage[:, 1 : 1 + 2 * width : 2]
    return storage


def _storage(data, layout):
    width = data.shape[1]
    if layout == "transpose":
        storage = data.t().contiguous()
    elif layout == "offset_stride":
        storage = torch.full((data.shape[0], 2 * width + 3), 57, dtype=data.dtype)
        _view(storage, layout, width).copy_(data)
    else:
        storage = data.clone()
    device_storage = storage.to(flag_gems.device)
    return (
        _view(storage, layout, width),
        storage,
        _view(device_storage, layout, width),
        device_storage,
    )


def _assert_exact(actual, expected):
    actual = actual.cpu()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    if expected.is_floating_point():
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
        assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
        zeros = (actual == 0) & (expected == 0)
        assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _run_case(module, case, left, right, dtype, layout):
    form, shape, mode, scalar = case
    records = []
    if form != "st":
        cpu_a, backing_a, a, device_a = _storage(left, layout)
        records.append(
            (backing_a, device_a, a, (a.shape, a.stride(), a.storage_offset()))
        )
    else:
        cpu_a = a = scalar
    if form != "ts":
        cpu_b, backing_b, b, device_b = _storage(right, layout)
        records.append(
            (backing_b, device_b, b, (b.shape, b.stride(), b.storage_offset()))
        )
    else:
        cpu_b = b = scalar
    if mode == "inplace":
        expected = cpu_a.remainder_(cpu_b)
        actual = module.remainder_(a, b)
        assert actual is a
    elif mode == "out":
        expected = torch.remainder(cpu_a, cpu_b)
        cpu_out, backing_out, out, device_out = _storage(
            torch.zeros_like(expected), layout
        )
        cpu_out.copy_(expected)
        actual = getattr(module, "rem_" + form)(a, b, out0=out)
        assert actual is out
        _assert_exact(device_out, backing_out)
    else:
        expected = torch.remainder(cpu_a, cpu_b)
        actual = module.remainder(a, b)
    _assert_exact(actual, expected)
    for cpu_storage, device_storage, tensor, metadata in records:
        _assert_exact(device_storage, cpu_storage)
        assert (tensor.shape, tensor.stride(), tensor.storage_offset()) == metadata


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", CASES)
def test_remainder_nonfinite_contract(implementation, dtype, layout, case):
    values = torch.tensor(
        [
            -float("inf"),
            -7.5,
            -2.0,
            -0.0,
            0.0,
            2.0,
            7.5,
            float("inf"),
            float("nan"),
            1e-45,
            -1e-45,
            1e30,
            -1e30,
        ],
        dtype=torch.float32,
    ).to(dtype)
    divisors = torch.tensor(
        [-float("inf"), -2.0, -0.5, -0.0, 0.0, 0.5, 2.0, float("inf"), float("nan")],
        dtype=dtype,
    )
    left = values[:, None].expand(-1, divisors.numel()).clone()
    right = divisors[None, :].expand(values.numel(), -1).clone()
    if case[1] == "rhs":
        right = divisors[None, :]
    elif case[1] == "both":
        left, right = values[:, None], divisors[None, :]
    _run_case(_module(implementation), case, left, right, dtype, layout)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mode", ["functional", "inplace", "out"])
def test_remainder_integer_controls(implementation, dtype, layout, mode):
    wide = 2**60 + 3 if dtype == torch.int64 else 1027
    values = torch.tensor([-wide, -17, -1, 0, 1, 17, wide], dtype=dtype)
    divisors = torch.tensor([-7, -1, 1, 7], dtype=dtype)
    a = values[:, None].expand(-1, 4).clone()
    b = divisors[None, :].expand(7, -1).clone()
    _run_case(
        _module(implementation), ("tt", "matrix", mode, None), a, b, dtype, layout
    )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_remainder_empty_contract(implementation, dtype, shape):
    cpu = torch.empty(shape, dtype=dtype)
    actual = _module(implementation).remainder(cpu.to(flag_gems.device), float("inf"))
    _assert_exact(actual, torch.remainder(cpu, float("inf")))

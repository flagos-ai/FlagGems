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
    pytest.mark.clamp,
    pytest.mark.clamp_,
    pytest.mark.clamp_min,
    pytest.mark.clamp_min_,
    pytest.mark.clamp_max,
    pytest.mark.clamp_max_,
    pytest.mark.clamp_tensor,
    pytest.mark.clamp_tensor_,
]
DTYPES = utils.FLOAT_DTYPES + [torch.float64]
LAYOUTS = ["contiguous", "transpose", "offset_stride"]
SCALAR_OPS = [
    "clamp",
    "clamp_",
    "clamp_min",
    "clamp_min_",
    "clamp_max",
    "clamp_max_",
    "clamp_min_only",
    "clamp_min_only_",
    "clamp_max_only",
    "clamp_max_only_",
    "clamp_out",
    "clamp_min_out",
    "clamp_max_out",
]


def _view(storage, layout):
    if layout == "transpose":
        return storage.t()
    if layout == "offset_stride":
        return storage[:, 1:515:2]
    return storage


def _inputs(dtype, layout):
    values = torch.tensor(
        [
            -float("inf"),
            -2.0,
            -0.0,
            0.0,
            2.0,
            float("inf"),
            float("nan"),
            1e-45,
            -1e-45,
            1e30,
            -1e30,
        ],
        dtype=torch.float32,
    ).to(dtype)
    data = values.repeat(71)[:771].reshape(3, 257)
    if layout == "transpose":
        storage = data.t().contiguous()
    elif layout == "offset_stride":
        storage = torch.full((3, 517), 77.0, dtype=dtype)
        _view(storage, layout).copy_(data)
    else:
        storage = data.clone()
    device_storage = storage.to(flag_gems.device)
    return (
        _view(storage, layout),
        storage,
        _view(device_storage, layout),
        device_storage,
    )


def _assert_exact(actual, expected):
    actual = actual.cpu()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = (actual == 0) & (expected == 0)
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("bounds", ["finite", "nan", "infinite"])
@pytest.mark.parametrize("op", SCALAR_OPS)
def test_clamp_scalar_nonfinite_contract(dtype, layout, bounds, op):
    ref, backing, inp, device_backing = _inputs(dtype, layout)
    metadata = (inp.shape, inp.stride(), inp.storage_offset())
    lo, hi = {
        "finite": (-0.5, 0.5),
        "nan": (float("nan"), float("nan")),
        "infinite": (-float("inf"), float("inf")),
    }[bounds]
    inplace = op.endswith("_")
    out_variant = op.endswith("_out")
    name = op[:-4] if out_variant else op.rstrip("_")
    only = (
        "min"
        if name in ("clamp_min", "clamp_min_only")
        else ("max" if name in ("clamp_max", "clamp_max_only") else None)
    )
    args = (lo,) if only == "min" else ((hi,) if only == "max" else (lo, hi))
    cpu_name = (
        "clamp_min" if only == "min" else ("clamp_max" if only == "max" else "clamp")
    )
    cpu_fn = getattr(torch, cpu_name + ("_" if inplace else ""))
    expected = cpu_fn(ref, *args)
    if out_variant:
        ref_out, out_backing, target, target_backing = _inputs(dtype, layout)
        ref_out.copy_(expected)
        module = importlib.import_module("flag_gems.ops.clamp")
        fn = getattr(module, "clamp_func" + ("_" + only if only else ""))
        actual = fn(inp, *args, out0=target)
        assert actual is target
        _assert_exact(target_backing, out_backing)
    elif name.endswith("_only"):
        fn = flag_gems.clamp_ if inplace else flag_gems.clamp
        actual = fn(inp, **{"mini" if only == "min" else "maxi": args[0]})
    else:
        actual = getattr(flag_gems, op)(inp, *args)
    _assert_exact(actual, expected)
    _assert_exact(device_backing, backing)
    assert (inp.shape, inp.stride(), inp.storage_offset()) == metadata
    if inplace:
        assert actual is inp
        assert actual.data_ptr() == inp.data_ptr()


TENSOR_CASES = [
    ("both", "finite"),
    ("both", "lower_nan"),
    ("both", "upper_nan"),
    ("both", "mixed_nan"),
    ("min", "finite"),
    ("min", "mixed_nan"),
    ("max", "finite"),
    ("max", "mixed_nan"),
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("case", TENSOR_CASES)
@pytest.mark.parametrize("mode", ["functional", "inplace", "out"])
def test_clamp_tensor_bound_nonfinite_contract(dtype, layout, case, mode):
    form, pattern = case
    ref, backing, inp, device_backing = _inputs(dtype, layout)
    lo_storage = torch.full((1, 514), -0.5, dtype=dtype)
    hi_storage = torch.full((6, 1), 0.5, dtype=dtype)
    lo, hi = lo_storage[:, ::2], hi_storage[::2, :]
    if pattern in ("lower_nan", "mixed_nan"):
        lo[:, ::7] = float("nan")
    if pattern in ("upper_nan", "mixed_nan"):
        hi[::2] = float("nan")
    device_lo_storage = lo_storage.to(flag_gems.device)
    device_hi_storage = hi_storage.to(flag_gems.device)
    device_lo = device_lo_storage[:, ::2] if form != "max" else None
    device_hi = device_hi_storage[::2, :] if form != "min" else None
    cpu_lo = lo if form != "max" else None
    cpu_hi = hi if form != "min" else None
    expected = (
        torch.clamp_(ref, cpu_lo, cpu_hi)
        if mode == "inplace"
        else torch.clamp(ref, cpu_lo, cpu_hi)
    )
    if mode == "out":
        ref_out, out_backing, target, target_backing = _inputs(dtype, layout)
        ref_out.copy_(expected)
        module = importlib.import_module("flag_gems.ops.clamp")
        if form == "both":
            actual = module.clamp_func_tensor(inp, device_lo, device_hi, out0=target)
        elif form == "min":
            actual = module.clamp_func_min_tensor(inp, device_lo, out0=target)
        else:
            actual = module.clamp_func_max_tensor(inp, device_hi, out0=target)
        assert actual is target
        _assert_exact(target_backing, out_backing)
    else:
        fn = flag_gems.clamp_tensor_ if mode == "inplace" else flag_gems.clamp_tensor
        actual = fn(inp, device_lo, device_hi)
    _assert_exact(actual, expected)
    _assert_exact(device_backing, backing)
    _assert_exact(device_lo_storage, lo_storage)
    _assert_exact(device_hi_storage, hi_storage)
    if mode == "inplace":
        assert actual is inp

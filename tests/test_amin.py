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
from .conftest import QUICK_MODE

# QUICK_MODE keeps the matrix small so the smoke run stays under the CI timeout.
# The full matrix covers every rank from 0-D to 5-D, every dim form (positive,
# negative, multi-dim list, None) and keepdim in both states.
DIM_CASES = [
    # (shape, [dim, ...])
    ((), [None]),
    ((256,), [0, -1, None]),
    ((1024, 1024), [0, 1, -1, [0, 1], None]),
    ((20, 320, 15), [0, 1, -1, [0, 1], [0, 2], None]),
    ((16, 128, 64, 60), [0, 1, 2, -1, [0, 1], [2, 3], None]),
    ((16, 7, 57, 32, 29), [0, 1, 4, -1, -5, [0, 1], [2, 3], None]),
]
if QUICK_MODE:
    DIM_CASES = [((2, 19, 7), [1, None])]

KEEPDIM_VALUES = [True, False]

# Value ranges from the shared test spec: symmetric, positive-only,
# negative-only, and the two half-open dtype extremes.
VALUE_RANGES = [
    (-1, 1),
    (0, 1),
    (-1, 0),
    ("dtype_max",),  # [0, dtype_max]
    ("dtype_min",),  # [dtype_min, 0]
]

# Dtypes that run the full shape/dim matrix. float8 is handled separately
# below because only its full-reduction (dim=None) path compiles.
FULL_MATRIX_DTYPES = (
    [torch.float32]
    if QUICK_MODE
    else [
        torch.float32,
        torch.bfloat16 if utils.bf16_is_supported else torch.float32,
        torch.float16,
        torch.int8,
        torch.uint8,
        torch.int32,
        torch.int64 if utils.int64_is_supported else torch.int32,
        torch.bool,
    ]
)

# fp8 reductions compile only on the dim=None (full-reduction) path; the
# dtype list is gated on the actual device capability in accuracy_utils
# (fp8e4nv pointers only lower to Triton kernels on sm89+).
FP8_DTYPES = utils.FP8_DTYPES if not QUICK_MODE else []


def _make_input(shape, dtype, value_range):
    """Build an input on the active device inside the requested value range."""
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    if dtype.is_floating_point:
        info = torch.finfo(dtype)
        lo, hi = info.min, info.max
    else:
        info = torch.iinfo(dtype)
        lo, hi = int(info.min), int(info.max)

    if len(value_range) == 2:
        low, high = value_range
        low, high = max(low, lo), min(high, hi)
    elif value_range[0] == "dtype_max":
        low, high = max(0, lo), hi
    else:  # "dtype_min": [dtype_min, 0)
        low, high = lo, max(0, lo) + 1

    if low >= high:
        high = low + 1

    if dtype.is_floating_point:
        span = torch.empty(shape, dtype=torch.float64, device="cpu").uniform_()
        # Bias away from the exact endpoints so float16 with a huge range does
        # not overflow to inf.
        vals = (low + span * (high - low)).to(dtype)
        return vals.to(flag_gems.device)
    return torch.randint(low, high, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )


@pytest.mark.amin
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", FULL_MATRIX_DTYPES)
@pytest.mark.parametrize("shape, dims", DIM_CASES)
def test_amin(shape, dims, dtype, value_range):
    inp = _make_input(shape, dtype, value_range)
    ref_inp = utils.to_reference(inp)

    for dim in dims:
        for keepdim in KEEPDIM_VALUES:
            ref_out = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
            # amin normalises an int dim to a list the same way the dispatcher
            # lifts an int dim to IntList, so both call forms are exercised.
            res_out = flag_gems.amin(inp, dim=dim, keepdim=keepdim)

            utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amin_
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", FULL_MATRIX_DTYPES)
@pytest.mark.parametrize("shape, dims", DIM_CASES)
def test_amin_(shape, dims, dtype, value_range):
    for dim in dims:
        for keepdim in KEEPDIM_VALUES:
            inp = _make_input(shape, dtype, value_range)
            ref_inp = utils.to_reference(inp.clone())

            ref_out = torch.amin(ref_inp, dim=dim, keepdim=True)
            ref_out = ref_out.expand_as(ref_inp)
            res_out = flag_gems.amin_(inp, dim=dim, keepdim=keepdim)

            assert res_out is inp
            utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amin
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("shape, _dims", DIM_CASES)
def test_amin_fp8_full_reduction(shape, _dims, dtype, value_range):
    """fp8 inputs exercise only the dim=None (full reduction) kernel."""
    inp = _make_input(shape, dtype, value_range)
    ref_inp = utils.to_reference(inp, upcast=True)

    for keepdim in KEEPDIM_VALUES:
        ref_out = torch.amin(ref_inp, dim=None, keepdim=keepdim).to(dtype)
        res_out = flag_gems.amin(inp, dim=None, keepdim=keepdim)

        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amin_
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("shape, _dims", DIM_CASES)
def test_amin__fp8_full_reduction(shape, _dims, dtype, value_range):
    inp = _make_input(shape, dtype, value_range)
    ref_inp = utils.to_reference(inp.clone(), upcast=True)

    for keepdim in KEEPDIM_VALUES:
        ref_out = torch.amin(ref_inp, dim=None, keepdim=True).to(dtype)
        ref_out = ref_out.expand_as(ref_inp)
        res_out = flag_gems.amin_(inp, dim=None, keepdim=keepdim)

        assert res_out is inp
        utils.gems_assert_equal(res_out, ref_out)


# Compact cases cover scalar, aligned, tail and multi-axis reduction layouts.
AMIN_BACKWARD_CASES = [
    ((), None, False),
    ((31, 47), (1,), False),
    ((17, 33, 5), (0, 2), False),
    ((8, 9, 10), (1,), True),
]


@pytest.mark.amin_backward
@pytest.mark.parametrize("shape, dim, keepdim", AMIN_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_amin_backward(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_shape = torch.amin(inp, dim=dim, keepdim=keepdim).shape
    grad = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amin(ref_inp, dim=dim, keepdim=keepdim),
        ref_inp,
        utils.to_reference(grad, False),
    )[0]
    output = flag_gems.amin(inp, dim=dim, keepdim=keepdim)
    result = flag_gems.amin_backward(grad, inp, output, dim=dim, keepdim=keepdim)
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.amin_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_amin_backward_ties(dtype):
    inp = torch.tensor(
        [[-2.0, -2.0, 1.0], [-3.0, 1.0, -3.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad = torch.tensor([2.0, -3.0], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amin(ref_inp, dim=1), ref_inp, utils.to_reference(grad, False)
    )[0]
    output = flag_gems.amin(inp, dim=(1,))
    result = flag_gems.amin_backward(grad, inp, output, dim=(1,))
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.amin_backward
def test_amin_backward_registered_dispatch():
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn(
        (31, 47), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((31,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amin(ref_inp, dim=1), ref_inp, utils.to_reference(grad, False)
    )[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["amin"])
        out = torch.amin(inp, dim=1)
        assert "AminFunction" in type(out.grad_fn).__name__
        result = torch.autograd.grad(out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.amin_backward
def test_amin_backward_noncontiguous_and_nan():
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn((13, 17, 19), dtype=dtype, device=flag_gems.device).transpose(
        0, 2
    )
    inp[0, 0, 0] = float("nan")
    grad = torch.randn((19, 17), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amin(ref_inp, dim=2), ref_inp, utils.to_reference(grad, False)
    )[0]
    output = flag_gems.amin(inp, dim=(2,))
    result = flag_gems.amin_backward(grad, inp, output, dim=(2,))
    utils.gems_assert_close(result, expected, dtype, equal_nan=True)


@pytest.mark.amin_backward
def test_amin_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((64, 512), dtype=torch.float16, device=flag_gems.device)
    grad = torch.randn((64,), dtype=torch.float16, device=flag_gems.device)
    output = flag_gems.amin(inp, dim=(1,))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.amin_backward(grad, inp, output, dim=(1,))
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "amin" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

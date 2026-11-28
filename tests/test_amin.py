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

# QUICK_MODE limits to a single float32 dtype to avoid CI timeout
FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES
DIM_LIST = [1] if QUICK_MODE else [0, 1]
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS_SHAPE = (
    [(True, DIMS_LIST[0], utils.REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(
        zip([True, False] * 2, DIMS_LIST, utils.REDUCTION_SHAPES + [(7, 4, 11, 1)])
    )
)


@pytest.mark.amin
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize(
    "dtype", FLOAT_DTYPES + utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES
)
def test_amin(shape, dim, keepdim, dtype):
    inp = utils.make_reduction_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
    # amin normalises an int dim to a list the same way the dispatcher lifts an
    # int dim to IntList, so both call forms are exercised here.
    res_out = flag_gems.amin(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amin_
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize(
    "dtype", FLOAT_DTYPES + utils.SMALL_INT_DTYPES + utils.ALL_INT_DTYPES
)
def test_amin_(shape, dim, keepdim, dtype):
    inp = utils.make_reduction_input(shape, dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amin(ref_inp, dim=dim, keepdim=True)
    ref_out = ref_out.expand_as(inp)
    res_out = flag_gems.amin_(inp, dim=dim, keepdim=keepdim)

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


AMIN_DTYPES = [
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


def make_amin_dtype_input(dtype):
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
@pytest.mark.parametrize("dtype", AMIN_DTYPES)
@pytest.mark.parametrize("dim", [None, (), 0, -1, (0, 2)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_amin_dtype_coverage(dtype, dim, keepdim, transpose):
    cpu = make_amin_dtype_input(dtype)
    inp = cpu.to(flag_gems.device)
    if transpose:
        cpu = cpu.transpose(0, 2)
        inp = inp.transpose(0, 2)
    expected = torch.amin(cpu, dim=dim, keepdim=keepdim)
    actual = flag_gems.amin(inp, dim=dim, keepdim=keepdim).cpu()
    assert actual.dtype == expected.dtype
    utils.gems_assert_equal(actual, expected)


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
    actual = flag_gems.amin(cpu.to(flag_gems.device), dim=dim).cpu()
    utils.gems_assert_equal(actual, torch.amin(cpu, dim=dim))


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
        actual = flag_gems.amin(cpu.to(flag_gems.device), dim=dim).cpu()
        utils.gems_assert_close(actual, torch.amin(cpu, dim=dim), dtype, equal_nan=True)


@pytest.mark.amin
@pytest.mark.parametrize("dim", [None, 0, -1, ()])
def test_amin_scalar(dim):
    cpu = torch.tensor(2**60 + 1, dtype=torch.int64)
    actual = flag_gems.amin(cpu.to(flag_gems.device), dim=dim).cpu()
    utils.gems_assert_equal(actual, torch.amin(cpu, dim=dim))


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
            flag_gems.amin(inp, dim=dim)
    else:
        actual = flag_gems.amin(inp, dim=dim).cpu()
        utils.gems_assert_equal(actual, expected)


@pytest.mark.amin
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_amin_integer_extrema(dtype):
    for value in (torch.iinfo(dtype).min, torch.iinfo(dtype).max):
        cpu = torch.full((3, 5), value, dtype=dtype)
        for dim in (None, 0, -1):
            actual = flag_gems.amin(cpu.to(flag_gems.device), dim=dim).cpu()
            utils.gems_assert_equal(actual, torch.amin(cpu, dim=dim))


@pytest.mark.amin
@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float64])
def test_amin_long_reduction(dtype):
    if dtype == torch.bool:
        cpu = torch.ones((3, 2051), dtype=dtype)
    else:
        cpu = torch.arange(6153, dtype=torch.int64).reshape(3, 2051)
        cpu = (cpu + (2**60 if dtype == torch.int64 else 0)).to(dtype)
    for dim in (None, 1):
        actual = flag_gems.amin(cpu.to(flag_gems.device), dim=dim).cpu()
        utils.gems_assert_equal(actual, torch.amin(cpu, dim=dim))

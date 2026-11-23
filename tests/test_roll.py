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

ROLL_SHIFTS_DIMS = [
    (1, 0),
    (-1, 0),
    (2, -1),
    (3, 1),
]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("shifts_dims", ROLL_SHIFTS_DIMS)
def test_roll_single_dim(shape, dtype, shifts_dims):
    shifts, dims = shifts_dims
    ndim = len(shape)
    # Adjust dims if it's out of range for this shape
    if dims >= ndim or dims < -ndim:
        # Skip test if dims is out of range for the specified shape
        return

    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.roll(ref_inp, shifts, dims)
    res_out = flag_gems.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


ROLL_MULTI_DIMS = [
    ((1, 2), (0, 1)),
    ((-1, 1), (0, -1)),
    ((2, -2), (-2, -1)),
]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shifts_dims", ROLL_MULTI_DIMS)
def test_roll_multi_dims(shape, dtype, shifts_dims):
    shifts, dims = shifts_dims
    ndim = len(shape)
    # Check all dims are valid for this shape
    for d in dims:
        if d >= ndim or d < -ndim:
            # Skip the case when dims is out of range for the shape
            return

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    ref_out = torch.roll(ref_inp, shifts, dims)
    res_out = flag_gems.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


ROLL_FLATTEN_SHIFTS = [1, -1, 5, -3]


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("shifts", ROLL_FLATTEN_SHIFTS)
def test_roll_flatten(shape, dtype, shifts):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, False)

    # Roll without specifying dims (flatten case)
    ref_out = torch.roll(ref_inp, shifts)
    res_out = flag_gems.roll(inp, shifts)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_roll_with_non_dense_input(shape, dtype):
    if len(shape) < 2:
        # Need at least 2D for non-dense test
        return

    shape_dilated = tuple(item * 2 for item in shape)
    inp = torch.randn(shape_dilated, dtype=dtype, device=flag_gems.device)[::2, ::2]
    ref_inp = utils.to_reference(inp, False)

    shifts = 2
    dims = 0

    ref_out = torch.roll(ref_inp, shifts, dims)
    res_out = flag_gems.roll(inp, shifts, dims)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.roll_backward
@pytest.mark.parametrize(
    "shape,shifts,dims",
    [
        ((7, 11), 3, 0),
        ((4, 5, 6), -2, -1),
        ((3, 5, 7), (1, -2), (0, 2)),
        ((3, 5, 7), (2, -1), (1, 1)),
        ((17,), 5, None),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_roll_backward(shape, shifts, dims, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_out = (
        torch.roll(ref_inp, shifts, dims)
        if dims is not None
        else torch.roll(ref_inp, shifts)
    )
    res_out = flag_gems.roll(res_inp, shifts, dims)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad))
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)
    direct_grad = flag_gems.roll_backward(grad, shifts, dims)

    utils.gems_assert_equal(res_grad, ref_grad)
    utils.gems_assert_equal(direct_grad, ref_grad)


@pytest.mark.roll_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_roll_backward_registered_dispatch(dtype):
    inp = torch.randn((7, 11), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.randn((7, 11), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.roll(ref_inp, (2, -3), (0, 1)),
        ref_inp,
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["roll"])
        out = torch.roll(inp, (2, -3), (0, 1))
        assert "RollAutogradFunction" in type(out.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(out, inp, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.roll_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_roll_backward_noncontiguous(dtype):
    inp = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.roll(ref_inp, 2, 0), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(flag_gems.roll(res_inp, 2, 0), res_inp, grad)
    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.roll_backward
def test_roll_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (32, 64), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((32, 64), dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.roll(inp, (3, -5), (0, 1))

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    kernels = [item for item in launches if "roll" in item[0].lower()]
    assert len(kernels) == 1 and kernels[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

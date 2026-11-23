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


def replace_zeros(inp):
    return torch.where(inp == 0, 1, inp)


@pytest.mark.remainder
@pytest.mark.remainder_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    inp1 = replace_zeros(inp1)
    inp2 = replace_zeros(inp2)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1 % ref_inp2
    res_out = flag_gems.remainder(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)

    for d in inp2.flatten()[:2]:
        d = d.item()
        ref_out = ref_inp1 % d
        res_out = flag_gems.remainder(inp1, d)
        utils.gems_assert_equal(res_out, ref_out)

        ref_out = d % ref_inp1
        res_out = flag_gems.remainder(d, inp1)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_(shape, dtype):
    inp1 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    inp2 = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    inp1 = replace_zeros(inp1.clone())
    inp2 = replace_zeros(inp2)
    ref_inp1 = utils.to_reference(inp1.clone(), False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = ref_inp1.remainder_(ref_inp2)

    res_out = flag_gems.remainder_(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)

    ref_inp1 = utils.to_reference(inp1.clone(), False)
    for d in inp2.flatten()[:2]:
        d = d.item()
        ref_out = ref_inp1.remainder_(d)

        res_out = flag_gems.remainder_(inp1, d)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("scalar", [7, -7])
def test_remainder_scalar(shape, dtype, scalar):
    inp = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)

    ref_inp = utils.to_reference(inp, False)
    ref_out = torch.remainder(ref_inp, scalar)

    res_out = flag_gems.remainder(inp, scalar)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_scalar_(shape, dtype):
    inp = torch.randint(
        torch.iinfo(dtype).min, torch.iinfo(dtype).max, shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    scalar = (
        torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            (1,),
            dtype=dtype,
            device="cpu",
        )
        .to(flag_gems.device)
        .item()
    )

    if scalar == 0:
        scalar = 1

    ref_inp = utils.to_reference(inp.clone(), False)
    ref_out = ref_inp.remainder_(scalar)

    res_out = flag_gems.remainder_(inp, scalar)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_scalar_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_remainder_scalar_tensor(shape, dtype):
    inp = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device="cpu",
    ).to(flag_gems.device)

    inp = replace_zeros(inp)

    ref_inp = utils.to_reference(inp, False)

    scalar = 7
    ref_out = torch.remainder(torch.tensor(scalar, dtype=dtype), ref_inp)
    res_out = flag_gems.remainder(scalar, inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.remainder_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_remainder_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.5
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()
    res_other = other.detach().requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.remainder(ref_inp, ref_other),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.remainder(res_inp, res_other),
        (res_inp, res_other),
        grad,
    )
    direct_grads = flag_gems.remainder_backward(grad, inp, other)

    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)
    for result, reference in zip(direct_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.remainder_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_remainder_backward_scalar(dtype):
    inp = torch.randn((31, 47), dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.remainder(ref_inp, -1.75), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(
        flag_gems.remainder(res_inp, -1.75), res_inp, grad
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.remainder_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_remainder_backward_registered_dispatch(dtype):
    inp = torch.randn(
        (64, 64), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    other = (
        torch.rand((64, 64), dtype=dtype, device=flag_gems.device) + 0.5
    ).requires_grad_()
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    ref_grads = torch.autograd.grad(
        torch.remainder(ref_inp, ref_other),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["remainder"])
        out = torch.remainder(inp, other)
        assert "RemainderFunction" in type(out.grad_fn).__name__
        res_grads = torch.autograd.grad(out, (inp, other), grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.remainder_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape,other_shape",
    [((8, 7, 5), (5,)), ((8, 7, 5), (1, 7, 1)), ((1, 7, 5), (8, 1, 5))],
)
def test_remainder_backward_broadcast(shape, other_shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.rand(other_shape, dtype=dtype, device=flag_gems.device) + 0.5
    grad = torch.randn_like(inp + other)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()
    res_other = other.detach().requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.remainder(ref_inp, ref_other),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.remainder(res_inp, res_other),
        (res_inp, res_other),
        grad,
    )
    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.remainder_backward
@pytest.mark.parametrize("required_input", [0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_remainder_backward_selective(required_input, dtype):
    inp = torch.randn((17, 29), dtype=dtype, device=flag_gems.device)
    other = torch.rand((17, 29), dtype=dtype, device=flag_gems.device) + 0.5
    values = [inp, other]
    values[required_input].requires_grad_()
    refs = [utils.to_reference(value.detach()) for value in values]
    refs[required_input].requires_grad_()
    grad = torch.randn_like(inp)

    (ref_grad,) = torch.autograd.grad(
        torch.remainder(*refs), refs[required_input], utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(
        flag_gems.remainder(*values), values[required_input], grad
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.remainder_backward
def test_remainder_backward_mixed_dtype():
    inp = torch.randint(-8, 8, (17, 29), dtype=torch.int32, device=flag_gems.device)
    other = (
        torch.rand((17, 29), dtype=torch.float32, device=flag_gems.device) + 0.5
    ).requires_grad_()
    grad = torch.randn_like(other)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other.detach()).requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.remainder(ref_inp, ref_other), ref_other, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(flag_gems.remainder(inp, other), other, grad)
    utils.gems_assert_close(res_grad, ref_grad, torch.float32)


@pytest.mark.remainder_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_remainder_backward_noncontiguous(dtype):
    inp = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()
    other = (torch.rand((47, 31), dtype=dtype, device=flag_gems.device) + 0.5).t()
    grad = torch.randn((47, 31), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()
    res_other = other.detach().requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.remainder(ref_inp, ref_other),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.remainder(res_inp, res_other),
        (res_inp, res_other),
        grad,
    )
    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.remainder_backward
def test_remainder_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (64, 129), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    other = (
        torch.rand((64, 129), dtype=torch.float32, device=flag_gems.device) + 0.5
    ).requires_grad_()
    grad = torch.randn_like(inp)
    out = flag_gems.remainder(inp, other)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (inp, other), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "remainder_backward" in item[0].lower()]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

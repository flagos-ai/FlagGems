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

import random

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.add
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add(shape, alpha, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.add(ref_inp1, ref_inp2, alpha=alpha)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add)):
        res_out = flag_gems.add(inp1, inp2, alpha=alpha)

    assert f"{utils.gems_log_prefix(flag_gems.add)} ADD" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Issues #3267: Ascend NPU does not support complex32 dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3897: TX81 does not support complex32 dtype",
)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("complex_dtype", utils.COMPLEX_DTYPES)
@pytest.mark.parametrize(
    "other_type", ["complex", "float_tensor", "int_tensor", "int_scalar"]
)
def test_add_complex(shape, complex_dtype, other_type, caplog):
    inp1 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)

    if other_type == "complex":
        inp2 = torch.randn(shape, dtype=complex_dtype, device=flag_gems.device)
    elif other_type == "float_tensor":
        if complex_dtype == torch.complex64:
            float_dtype = torch.float32
        elif complex_dtype == torch.complex32:
            float_dtype = torch.float16
        else:
            raise ValueError(f"Unsupported complex_dtype: {complex_dtype}")
        inp2 = torch.randn(shape, dtype=float_dtype, device=flag_gems.device)
    elif other_type == "int_tensor":
        inp2 = torch.randint(10, 20, shape, device=flag_gems.device)
    elif other_type == "int_scalar":
        inp2 = 3
    else:
        raise ValueError(f"Unknown other_type: {other_type}")

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = (
        utils.to_reference(inp2, True) if isinstance(inp2, torch.Tensor) else inp2
    )

    ref_out = torch.add(ref_inp1, ref_inp2)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add)):
        res_out = flag_gems.add(inp1, inp2)

    assert f"{utils.gems_log_prefix(flag_gems.add)} ADD" in caplog.text
    utils.gems_assert_close(res_out, ref_out, complex_dtype)


@pytest.mark.add_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_(shape, alpha, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.add_(ref_inp2, alpha=alpha)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add_)):
        res_out = flag_gems.add_(inp1, inp2, alpha=alpha)

    assert f"{utils.gems_log_prefix(flag_gems.add_)} ADD_" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_scalar(shape, scalar, alpha, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1, True)

    ref_out = torch.add(ref_inp1, inp2, alpha=alpha)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add)):
        res_out = flag_gems.add(inp1, inp2, alpha=alpha)

    assert f"{utils.gems_log_prefix(flag_gems.add)} ADD" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_tensor_scalar_(shape, scalar, alpha, dtype, caplog):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar
    ref_inp1 = utils.to_reference(inp1.clone(), True)

    ref_out = ref_inp1.add_(inp2, alpha=alpha)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add_)):
        res_out = flag_gems.add_(inp1, inp2, alpha=alpha)

    assert f"{utils.gems_log_prefix(flag_gems.add_)} ADD_" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_scalar_tensor(shape, scalar, alpha, dtype, caplog):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.add(inp1, ref_inp2, alpha=alpha)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add)):
        res_out = flag_gems.add(inp1, inp2, alpha=alpha)

    assert f"{utils.gems_log_prefix(flag_gems.add)} ADD" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_add_scalar_scalar(dtype, caplog):
    if dtype == torch.float32:
        inp1 = float(np.float32(random.random()))
        inp2 = float(np.float32(random.random()))
        alpha = float(np.float32(random.random()))
    else:
        inp1 = random.randint(0, 100)
        inp2 = random.randint(0, 100)
        alpha = random.randint(0, 100)

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(flag_gems.add)):
        res_out = flag_gems.add(inp1, inp2, alpha=alpha)

    assert f"{utils.gems_log_prefix(flag_gems.add)} ADD" in caplog.text
    if dtype == torch.int64:
        utils.gems_assert_equal(res_out, ref_out)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.add_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_backward(shape, alpha, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    ref_grad_output = utils.to_reference(grad)
    res_inp = inp.detach().requires_grad_()
    res_other = other.detach().requires_grad_()

    ref_out = torch.add(ref_inp, ref_other, alpha=alpha)
    res_out = flag_gems.add(res_inp, res_other, alpha=alpha)
    ref_grads = torch.autograd.grad(ref_out, (ref_inp, ref_other), ref_grad_output)
    res_grads = torch.autograd.grad(res_out, (res_inp, res_other), grad)

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype)
    direct_grads = flag_gems.add_backward(grad, alpha)
    assert direct_grads[0].data_ptr() == grad.data_ptr()
    utils.gems_assert_close(direct_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(direct_grads[1], ref_grads[1], dtype)


@pytest.mark.add_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape,other_shape",
    [((8, 7, 5), (5,)), ((8, 7, 5), (1, 7, 1)), ((1, 7, 5), (8, 1, 5))],
)
def test_add_backward_broadcast(shape, other_shape, dtype):
    # These pairs exercise both leading-dimension and singleton broadcasting.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()
    res_other = other.detach().requires_grad_()
    grad = torch.randn_like(res_inp + res_other)

    ref_out = torch.add(ref_inp, ref_other, alpha=-0.75)
    res_out = flag_gems.add(res_inp, res_other, alpha=-0.75)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_inp, ref_other), utils.to_reference(grad)
    )
    res_grads = torch.autograd.grad(res_out, (res_inp, res_other), grad)

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype)


@pytest.mark.add_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_backward_registered_dispatch(dtype):
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device).requires_grad_()
    other = torch.randn_like(inp, requires_grad=True)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    ref_grads = torch.autograd.grad(
        torch.add(ref_inp, ref_other, alpha=0.5),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["add"])
        res_out = torch.add(inp, other, alpha=0.5)
        assert "AddFunction" in type(res_out.grad_fn).__name__
        res_grads = torch.autograd.grad(res_out, (inp, other), grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype)


@pytest.mark.add_backward
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_add_backward_complex(dtype):
    inp = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    other = torch.randn_like(inp)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()
    res_other = other.detach().requires_grad_()
    # The existing complex add forward accepts real alpha values; backward still
    # exercises interleaved complex gradients and the conjugate-alpha formula.
    alpha = 0.5

    ref_grads = torch.autograd.grad(
        torch.add(ref_inp, ref_other, alpha=alpha),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.add(res_inp, res_other, alpha=alpha),
        (res_inp, res_other),
        grad,
    )

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype)


@pytest.mark.add_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("self_requires_grad", [True, False])
def test_add_backward_single_input(dtype, self_requires_grad):
    inp = torch.randn(
        (64, 64),
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=self_requires_grad,
    )
    other = torch.randn_like(inp, requires_grad=not self_requires_grad)
    grad = torch.randn_like(inp)
    target = inp if self_requires_grad else other

    res_grad = torch.autograd.grad(flag_gems.add(inp, other, alpha=0.5), target, grad)[
        0
    ]

    expected = grad if self_requires_grad else grad * 0.5
    utils.gems_assert_equal(res_grad, expected)


@pytest.mark.add_backward
@pytest.mark.parametrize(
    "self_dtype,other_dtype",
    [(torch.float16, torch.float32), (torch.float32, torch.float16)],
)
def test_add_backward_mixed_dtype(self_dtype, other_dtype):
    # Mixed dtypes exercise the self-gradient cast inside the same fused kernel.
    inp = torch.randn(
        (64, 64), dtype=self_dtype, device=flag_gems.device, requires_grad=True
    )
    other = torch.randn(
        (64, 64), dtype=other_dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp + other)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_other = utils.to_reference(other.detach()).requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.add(ref_inp, ref_other, alpha=-0.75),
        (ref_inp, ref_other),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.add(inp, other, alpha=-0.75), (inp, other), grad
    )

    utils.gems_assert_close(res_grads[0], ref_grads[0], self_dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], other_dtype)


@pytest.mark.add_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_backward_empty(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device, requires_grad=True)
    other = torch.empty_like(inp, requires_grad=True)
    out = flag_gems.add(inp, other, alpha=0.5)
    grads = torch.autograd.grad(out, (inp, other), torch.empty_like(out))

    assert grads[0].shape == inp.shape
    assert grads[1].shape == other.shape
    assert grads[0].dtype == dtype
    assert grads[1].dtype == dtype


@pytest.mark.add_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_backward_fused_kernel_count(dtype):
    """Identity self-grad plus the scaled other-grad use one Triton launch."""
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((4096,), dtype=dtype, device=flag_gems.device).requires_grad_()
    other = torch.randn_like(inp, requires_grad=True)
    out = flag_gems.add(inp, other, alpha=0.5)
    grad = torch.randn_like(out)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (inp, other), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "add_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

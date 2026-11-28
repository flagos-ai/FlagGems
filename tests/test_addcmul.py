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
from . import conftest as cfg

if cfg.QUICK_MODE:
    ADDCMUL_BROADCAST_SHAPES = [
        ((128, 256), (128, 256), (128, 256), (1,)),
    ]
else:
    ADDCMUL_BROADCAST_SHAPES = [
        ((1, 256, 1, 1), (1, 256, 56, 56), (1, 256, 56, 56), (1, 256, 1, 1)),
        ((1, 3), (2, 1), (2, 3), (1, 1)),
        ((4, 1, 16), (1, 8, 1), (4, 8, 16), (4, 1, 1)),
        ((128, 256), (128, 256), (128, 256), (1,)),
    ]


@pytest.mark.addcmul
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_addcmul(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out = torch.addcmul(ref_inp, ref_t1, ref_t2, value=v)
    res_out = flag_gems.addcmul(res_inp, t1, t2, value=v)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.addcmul_out
@pytest.mark.parametrize(
    "inp_shape, t1_shape, t2_shape, out_shape",
    ADDCMUL_BROADCAST_SHAPES,
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_addcmul_out_broadcast(inp_shape, t1_shape, t2_shape, out_shape, dtype):
    res_inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    t1 = torch.randn(t1_shape, dtype=dtype, device=flag_gems.device)
    t2 = torch.randn(t2_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(res_inp, True)
    ref_t1 = utils.to_reference(t1, True)
    ref_t2 = utils.to_reference(t2, True)

    v = float(np.float32(random.random()))

    ref_out_tensor = torch.randn(out_shape, dtype=dtype, device=ref_inp.device)
    ref_result = torch.addcmul(ref_inp, ref_t1, ref_t2, value=v, out=ref_out_tensor)

    res_out_tensor = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    res_result = flag_gems.addcmul_out(res_inp, t1, t2, value=v, out=res_out_tensor)

    broadcast_shape = torch.broadcast_shapes(inp_shape, t1_shape, t2_shape)
    assert list(res_out_tensor.shape) == list(
        broadcast_shape
    ), f"out tensor was not resized: expected {broadcast_shape}, got {res_out_tensor.shape}"
    utils.gems_assert_close(res_result, ref_result, dtype)


# Compact scalar, aligned and non-aligned cases exercise the fused backward path.
ADDCMUL_BACKWARD_SHAPES = [(), (4096,), (31, 47)]


@pytest.mark.addcmul_backward
@pytest.mark.parametrize("shape", ADDCMUL_BACKWARD_SHAPES)
@pytest.mark.parametrize("value", [0.5, -0.75])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcmul_backward(shape, value, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    tensor1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    tensor2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_tensor1 = utils.to_reference(tensor1.detach()).requires_grad_()
    ref_tensor2 = utils.to_reference(tensor2.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()
    res_tensor1 = tensor1.detach().requires_grad_()
    res_tensor2 = tensor2.detach().requires_grad_()

    ref_out = torch.addcmul(ref_inp, ref_tensor1, ref_tensor2, value=value)
    res_out = flag_gems.addcmul(res_inp, res_tensor1, res_tensor2, value=value)
    ref_grads = torch.autograd.grad(
        ref_out,
        (ref_inp, ref_tensor1, ref_tensor2),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(res_out, (res_inp, res_tensor1, res_tensor2), grad)
    direct_grads = flag_gems.addcmul_backward(grad, tensor1, tensor2, value=value)

    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)
    for result, reference in zip(direct_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.addcmul_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcmul_backward_registered_dispatch(dtype):
    shape = (64, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    tensor1 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    tensor2 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_tensor1 = utils.to_reference(tensor1.detach()).requires_grad_()
    ref_tensor2 = utils.to_reference(tensor2.detach()).requires_grad_()
    ref_grads = torch.autograd.grad(
        torch.addcmul(ref_inp, ref_tensor1, ref_tensor2, value=-0.75),
        (ref_inp, ref_tensor1, ref_tensor2),
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["addcmul"])
        res_out = torch.addcmul(inp, tensor1, tensor2, value=-0.75)
        assert "AddcmulFunction" in type(res_out.grad_fn).__name__
        res_grads = torch.autograd.grad(res_out, (inp, tensor1, tensor2), grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.addcmul_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcmul_backward_broadcast(dtype):
    inp = torch.randn(
        (8, 7, 5), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    tensor1 = torch.randn(
        (5,), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    tensor2 = torch.randn(
        (1, 7, 1), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    ref_tensor1 = utils.to_reference(tensor1.detach()).requires_grad_()
    ref_tensor2 = utils.to_reference(tensor2.detach()).requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.addcmul(ref_inp, ref_tensor1, ref_tensor2, value=0.5),
        (ref_inp, ref_tensor1, ref_tensor2),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addcmul(inp, tensor1, tensor2, value=0.5),
        (inp, tensor1, tensor2),
        grad,
    )

    reduction_dims = (1, 8 * 7, 8 * 5)
    for result, reference, reduce_dim in zip(res_grads, ref_grads, reduction_dims):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addcmul_backward
@pytest.mark.parametrize("required_input", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcmul_backward_single_input(required_input, dtype):
    shape = (17, 29)
    inputs = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device),
        torch.randn(shape, dtype=dtype, device=flag_gems.device),
        torch.randn(shape, dtype=dtype, device=flag_gems.device),
    ]
    inputs[required_input].requires_grad_()
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inputs = [utils.to_reference(value.detach()) for value in inputs]
    ref_inputs[required_input].requires_grad_()

    ref_grad = torch.autograd.grad(
        torch.addcmul(*ref_inputs, value=-0.75),
        ref_inputs[required_input],
        utils.to_reference(grad),
    )[0]
    res_grad = torch.autograd.grad(
        flag_gems.addcmul(*inputs, value=-0.75),
        inputs[required_input],
        grad,
    )[0]

    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.addcmul_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addcmul_backward_noncontiguous(dtype):
    shape = (17, 33)
    inputs = [
        torch.randn(shape[::-1], dtype=dtype, device=flag_gems.device)
        .transpose(0, 1)
        .requires_grad_()
        for _ in range(3)
    ]
    grad = torch.randn(shape[::-1], dtype=dtype, device=flag_gems.device).transpose(
        0, 1
    )
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_() for value in inputs
    ]

    ref_grads = torch.autograd.grad(
        torch.addcmul(*ref_inputs, value=0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(flag_gems.addcmul(*inputs, value=0.5), inputs, grad)

    for result, reference in zip(res_grads, ref_grads):
        utils.gems_assert_close(result, reference, dtype)


@pytest.mark.addcmul_backward
def test_addcmul_backward_fused_kernel_count():
    """The three addcmul gradients are emitted by one Triton kernel."""
    from torch.profiler import ProfilerActivity, profile

    shape = (4096,)
    dtype = torch.float16
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    tensor1 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    tensor2 = torch.randn(
        shape, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    out = flag_gems.addcmul(inp, tensor1, tensor2, value=-0.75)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (inp, tensor1, tensor2), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "addcmul" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

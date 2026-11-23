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

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_close,
    to_reference,
)


@pytest.mark.rsub
@pytest.mark.rsub_tensor
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rsub_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.rsub(ref_inp1, ref_inp2)
    res_out = flag_gems.rsub_tensor(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rsub
@pytest.mark.rsub_scalar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rsub_scalar(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    inp2 = 0.5

    ref_out = torch.rsub(ref_inp1, inp2)
    res_out = flag_gems.rsub_scalar(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.rsub_backward
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_rsub_backward_tensor(shape, dtype, alpha):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.detach()).requires_grad_()
    ref_inp2 = to_reference(inp2.detach()).requires_grad_()
    res_inp1 = inp1.detach().requires_grad_()
    res_inp2 = inp2.detach().requires_grad_()

    ref_out = torch.rsub(ref_inp1, ref_inp2, alpha=alpha)
    res_out = flag_gems.rsub_tensor(res_inp1, res_inp2, alpha=alpha)
    ref_grads = torch.autograd.grad(ref_out, (ref_inp1, ref_inp2), to_reference(grad))
    res_grads = torch.autograd.grad(res_out, (res_inp1, res_inp2), grad)
    direct_grads = flag_gems.rsub_backward(grad, alpha)

    for res_grad, direct_grad, ref_grad in zip(res_grads, direct_grads, ref_grads):
        gems_assert_close(res_grad, ref_grad, dtype)
        gems_assert_close(direct_grad, ref_grad, dtype)


@pytest.mark.rsub_backward
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rsub_backward_scalar(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_out = torch.rsub(ref_inp, 0.5, alpha=2.0)
    res_out = flag_gems.rsub_scalar(res_inp, 0.5, alpha=2.0)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, to_reference(grad))
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)

    gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.rsub_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rsub_backward_registered_dispatch(dtype):
    inp1 = torch.randn(
        (7, 11), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    inp2 = torch.randn(
        (7, 11), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((7, 11), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.detach()).requires_grad_()
    ref_inp2 = to_reference(inp2.detach()).requires_grad_()
    ref_grads = torch.autograd.grad(
        torch.rsub(ref_inp1, ref_inp2, alpha=0.5),
        (ref_inp1, ref_inp2),
        to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["rsub_tensor"])
        out = torch.rsub(inp1, inp2, alpha=0.5)
        assert "RsubTensorAutogradFunction" in type(out.grad_fn).__name__
        res_grads = torch.autograd.grad(out, (inp1, inp2), grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    for res_grad, ref_grad in zip(res_grads, ref_grads):
        gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.rsub_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rsub_backward_noncontiguous(dtype):
    inp1 = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    inp2 = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    ref_inp1 = to_reference(inp1.detach()).requires_grad_()
    ref_inp2 = to_reference(inp2.detach()).requires_grad_()
    res_inp1 = inp1.detach().requires_grad_()
    res_inp2 = inp2.detach().requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.rsub(ref_inp1, ref_inp2, alpha=0.5),
        (ref_inp1, ref_inp2),
        to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.rsub_tensor(res_inp1, res_inp2, alpha=0.5),
        (res_inp1, res_inp2),
        grad,
    )
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.rsub_backward
def test_rsub_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp1 = torch.randn(
        (64, 96), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    inp2 = torch.randn(
        (64, 96), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((64, 96), dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.rsub_tensor(inp1, inp2, alpha=0.5)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (inp1, inp2), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    kernels = [item for item in launches if "rsub_backward" in item[0].lower()]
    assert len(kernels) == 1 and kernels[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

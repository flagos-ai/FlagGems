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
from . import conftest as cfg

if cfg.QUICK_MODE:
    MN_SHAPES = [
        (1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.addr
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addr(M, N, dtype):
    input_tensor = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    alpha = torch.randn((), dtype=dtype, device=flag_gems.device)
    beta = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(input_tensor, True)
    ref_vec1 = utils.to_reference(vec1, True)
    ref_vec2 = utils.to_reference(vec2, True)

    ref_out = torch.addr(ref_inp, ref_vec1, ref_vec2, alpha=alpha, beta=beta)
    res_out = flag_gems.addr(input_tensor, vec1, vec2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Compact aligned, rectangular and tail cases exercise all fused grid regions.
ADDR_BACKWARD_SHAPES = [(17, 31), (64, 64), (65, 129)]


@pytest.mark.addr_backward
@pytest.mark.parametrize("M, N", ADDR_BACKWARD_SHAPES)
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (0.75, -0.5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addr_backward(M, N, alpha, beta, dtype):
    input_tensor = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_()
        for value in (input_tensor, vec1, vec2)
    ]

    ref_grads = torch.autograd.grad(
        torch.addr(*ref_inputs, alpha=alpha, beta=beta),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = flag_gems.addr_backward(grad, vec1, vec2, alpha=alpha, beta=beta)

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, N, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addr_backward
def test_addr_backward_registered_dispatch():
    # Dtype coverage is provided by the primitive test above; this checks dispatch.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    inputs = [
        torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((M,), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((N,), dtype=dtype, device=flag_gems.device, requires_grad=True),
    ]
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_() for value in inputs
    ]
    ref_grads = torch.autograd.grad(
        torch.addr(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["addr"])
        res_out = torch.addr(*inputs, alpha=0.75, beta=-0.5)
        assert "AddrFunction" in type(res_out.grad_fn).__name__
        res_grads = torch.autograd.grad(res_out, inputs, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, N, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addr_backward
@pytest.mark.parametrize("input_shape", [(), (1, 31), (17, 1)])
def test_addr_backward_broadcast_input(input_shape):
    # Dtype coverage is provided by the primitive test above.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    inputs = [
        torch.randn(input_shape, dtype=dtype, device=flag_gems.device),
        torch.randn((M,), dtype=dtype, device=flag_gems.device),
        torch.randn((N,), dtype=dtype, device=flag_gems.device),
    ]
    inputs = [value.requires_grad_() for value in inputs]
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_() for value in inputs
    ]

    ref_grads = torch.autograd.grad(
        torch.addr(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addr(*inputs, alpha=0.75, beta=-0.5), inputs, grad
    )

    input_reduce = M * N // max(1, inputs[0].numel())
    for result, reference, reduce_dim in zip(
        res_grads, ref_grads, (input_reduce, N, M)
    ):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addr_backward
@pytest.mark.parametrize("required_input", [0, 1, 2])
def test_addr_backward_single_input(required_input):
    # Dtype coverage is provided by the primitive test above.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    inputs = [
        torch.randn((M, N), dtype=dtype, device=flag_gems.device),
        torch.randn((M,), dtype=dtype, device=flag_gems.device),
        torch.randn((N,), dtype=dtype, device=flag_gems.device),
    ]
    inputs[required_input].requires_grad_()
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [utils.to_reference(value.detach()) for value in inputs]
    ref_inputs[required_input].requires_grad_()

    ref_grad = torch.autograd.grad(
        torch.addr(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs[required_input],
        utils.to_reference(grad),
    )[0]
    res_grad = torch.autograd.grad(
        flag_gems.addr(*inputs, alpha=0.75, beta=-0.5),
        inputs[required_input],
        grad,
    )[0]

    utils.gems_assert_close(
        res_grad, ref_grad, dtype, reduce_dim=(1, N, M)[required_input]
    )


@pytest.mark.addr_backward
def test_addr_backward_noncontiguous():
    # Dtype coverage is provided by the primitive test above.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    input_tensor = torch.randn((N, M), dtype=dtype, device=flag_gems.device).t()
    vec1 = torch.randn((M * 2,), dtype=dtype, device=flag_gems.device)[::2]
    vec2 = torch.randn((N * 2,), dtype=dtype, device=flag_gems.device)[::2]
    grad = torch.randn((N, M), dtype=dtype, device=flag_gems.device).t()
    inputs = [value.requires_grad_() for value in (input_tensor, vec1, vec2)]
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_() for value in inputs
    ]

    ref_grads = torch.autograd.grad(
        torch.addr(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addr(*inputs, alpha=0.75, beta=-0.5), inputs, grad
    )

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, N, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addr_backward
@pytest.mark.parametrize("M, N", [(0, 17), (17, 0)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addr_backward_empty_dimensions(M, N, dtype):
    input_tensor = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_()
        for value in (input_tensor, vec1, vec2)
    ]

    ref_grads = torch.autograd.grad(
        torch.addr(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = flag_gems.addr_backward(grad, vec1, vec2, alpha=0.75, beta=-0.5)

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, N, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addr_backward
def test_addr_backward_fused_kernel_count():
    """Input and both vector gradients are emitted by one Triton launch."""
    from torch.profiler import ProfilerActivity, profile

    M = N = 64
    dtype = torch.float16
    vec1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.addr_backward(grad, vec1, vec2, alpha=0.75, beta=-0.5)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "addr" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

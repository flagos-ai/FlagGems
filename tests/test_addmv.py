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
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]


@pytest.mark.addmv
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmv(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_vec = utils.to_reference(vec, True)
    ref_bias1 = utils.to_reference(bias1, True)

    alpha = beta = scalar

    ref_out1 = torch.addmv(ref_bias1, ref_mat, ref_vec, alpha=alpha, beta=beta)
    res_out1 = flag_gems.addmv(bias1, mat, vec, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out1, ref_out1, dtype, reduce_dim=N)

    # broadcast bias scalar
    bias2 = torch.randn((), dtype=dtype, device=flag_gems.device)
    ref_bias2 = utils.to_reference(bias2, True)

    ref_out2 = torch.addmv(ref_bias2, ref_mat, ref_vec, alpha=alpha, beta=beta)
    res_out2 = flag_gems.addmv(bias2, mat, vec, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out2, ref_out2, dtype, reduce_dim=N)


@pytest.mark.addmv_out
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmv_out(M, N, scalar, dtype):
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M,), dtype=dtype, device=flag_gems.device)
    ref_mat = utils.to_reference(mat, True)
    ref_vec = utils.to_reference(vec, True)
    ref_bias = utils.to_reference(bias, True)
    ref_out = utils.to_reference(out, True)

    alpha = beta = scalar

    torch.addmv(ref_bias, ref_mat, ref_vec, alpha=alpha, beta=beta, out=ref_out)
    flag_gems.addmv_out(bias, mat, vec, alpha=alpha, beta=beta, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=N)


# Compact aligned, rectangular and tail cases exercise both fused grid regions.
ADDMV_BACKWARD_SHAPES = [(17, 31), (64, 64), (65, 129)]


@pytest.mark.addmv_backward
@pytest.mark.parametrize("M, N", ADDMV_BACKWARD_SHAPES)
@pytest.mark.parametrize("alpha, beta", [(1.0, 1.0), (0.75, -0.5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmv_backward(M, N, alpha, beta, dtype):
    bias = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_()
        for value in (bias, mat, vec)
    ]
    ref_grads = torch.autograd.grad(
        torch.addmv(*ref_inputs, alpha=alpha, beta=beta),
        ref_inputs,
        utils.to_reference(grad),
    )
    direct_grads = flag_gems.addmv_backward(grad, mat, vec, alpha=alpha, beta=beta)

    for result, reference, reduce_dim in zip(direct_grads, ref_grads, (1, 1, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addmv_backward
def test_addmv_backward_registered_dispatch():
    # Dtype coverage is provided by the primitive test above; this checks dispatch.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    inputs = [
        torch.randn((M,), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True),
        torch.randn((N,), dtype=dtype, device=flag_gems.device, requires_grad=True),
    ]
    grad = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_() for value in inputs
    ]
    ref_grads = torch.autograd.grad(
        torch.addmv(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["addmv"])
        res_out = torch.addmv(*inputs, alpha=0.75, beta=-0.5)
        assert "AddmvFunction" in type(res_out.grad_fn).__name__
        res_grads = torch.autograd.grad(res_out, inputs, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, 1, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addmv_backward
def test_addmv_backward_broadcast_bias():
    # Dtype coverage is provided by the primitive test above.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    bias = torch.randn((), dtype=dtype, device=flag_gems.device)
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_()
        for value in (bias, mat, vec)
    ]
    res_inputs = [value.detach().requires_grad_() for value in (bias, mat, vec)]

    ref_grads = torch.autograd.grad(
        torch.addmv(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addmv(*res_inputs, alpha=0.75, beta=-0.5), res_inputs, grad
    )

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (M, 1, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addmv_backward
@pytest.mark.parametrize("required_input", [0, 1, 2])
def test_addmv_backward_single_input(required_input):
    # Dtype coverage is provided by the primitive test above.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    inputs = [
        torch.randn((M,), dtype=dtype, device=flag_gems.device),
        torch.randn((M, N), dtype=dtype, device=flag_gems.device),
        torch.randn((N,), dtype=dtype, device=flag_gems.device),
    ]
    inputs[required_input].requires_grad_()
    grad = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_inputs = [utils.to_reference(value.detach()) for value in inputs]
    ref_inputs[required_input].requires_grad_()

    ref_grad = torch.autograd.grad(
        torch.addmv(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs[required_input],
        utils.to_reference(grad),
    )[0]
    res_grad = torch.autograd.grad(
        flag_gems.addmv(*inputs, alpha=0.75, beta=-0.5),
        inputs[required_input],
        grad,
    )[0]

    utils.gems_assert_close(
        res_grad, ref_grad, dtype, reduce_dim=(1, 1, M)[required_input]
    )


@pytest.mark.addmv_backward
def test_addmv_backward_noncontiguous():
    # Dtype coverage is provided by the primitive test above.
    dtype = utils.FLOAT_DTYPES[0]
    M, N = 17, 31
    bias = torch.randn((M * 2,), dtype=dtype, device=flag_gems.device)[::2]
    mat = torch.randn((N, M), dtype=dtype, device=flag_gems.device).t()
    vec = torch.randn((N * 2,), dtype=dtype, device=flag_gems.device)[::2]
    grad = torch.randn((M * 2,), dtype=dtype, device=flag_gems.device)[::2]
    inputs = [value.requires_grad_() for value in (bias, mat, vec)]
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_() for value in inputs
    ]

    ref_grads = torch.autograd.grad(
        torch.addmv(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addmv(*inputs, alpha=0.75, beta=-0.5), inputs, grad
    )

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, 1, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addmv_backward
@pytest.mark.parametrize("M, N", [(0, 17), (17, 0)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addmv_backward_empty_dimensions(M, N, dtype):
    bias = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_inputs = [
        utils.to_reference(value.detach()).requires_grad_()
        for value in (bias, mat, vec)
    ]

    ref_grads = torch.autograd.grad(
        torch.addmv(*ref_inputs, alpha=0.75, beta=-0.5),
        ref_inputs,
        utils.to_reference(grad),
    )
    res_grads = flag_gems.addmv_backward(grad, mat, vec, alpha=0.75, beta=-0.5)

    for result, reference, reduce_dim in zip(res_grads, ref_grads, (1, 1, M)):
        utils.gems_assert_close(result, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.addmv_backward
def test_addmv_backward_fused_kernel_count():
    """Bias, matrix and vector gradients are emitted by one Triton launch."""
    from torch.profiler import ProfilerActivity, profile

    M = N = 64
    dtype = torch.float16
    mat = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    grad = torch.randn((M,), dtype=dtype, device=flag_gems.device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.addmv_backward(grad, mat, vec, alpha=0.75, beta=-0.5)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "addmv" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

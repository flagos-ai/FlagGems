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

import os

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    # Reduced shapes for CI quick mode
    MNK_SHAPES = [
        (1, 1, 32),
        (16, 16, 64),
    ]
else:
    # Standard test shapes covering small, medium, large and non-aligned dimensions
    MNK_SHAPES = [
        (1, 1, 32),
        (16, 16, 64),
        (15, 160, 1024),
        (128, 256, 512),
        (256, 256, 256),
        (495, 5333, 71),
        (1024, 1024, 128),
    ]


@pytest.mark.addbmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("scalar", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_addbmm(M, N, K, scalar, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Skiping fp32 addbmm test on tsingmicro platform")

    if flag_gems.vendor_name == "mthreads" and dtype in [torch.float16, torch.bfloat16]:
        os.environ["MUSA_ENABLE_SQMMA"] = "1"
    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bias = utils.to_reference(bias, True)

    alpha = beta = scalar

    ref_out = torch.addbmm(ref_bias, ref_mat1, ref_mat2, alpha=alpha, beta=beta)
    res_out = flag_gems.addbmm(bias, mat1, mat2, beta=beta, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    if flag_gems.vendor_name == "mthreads" and dtype in [torch.float16, torch.bfloat16]:
        del os.environ["MUSA_ENABLE_SQMMA"]


# Backward computes three matrix gradients, so use compact aligned and non-aligned
# cases instead of the much larger forward-only stress shapes above.
ADDBMM_BACKWARD_SHAPES = [
    (2, 3, 5, 4),
    (3, 16, 24, 32),
    (4, 31, 17, 29),
]


def _addbmm_backward_atol(dtype):
    # Native CUDA addbmm backward itself differs from the CPU FP64 reference by
    # up to roughly 8e-4 per reduction element for FP32 GEMMs on H20.
    return 1e-3 if dtype == torch.float32 else 1e-4


@pytest.mark.addbmm_backward
@pytest.mark.parametrize("B, M, N, K", ADDBMM_BACKWARD_SHAPES)
@pytest.mark.parametrize("alpha,beta", [(0.5, -0.75), (-1.25, 0.25)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addbmm_backward(B, M, N, K, alpha, beta, dtype):
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    batch1 = torch.randn((B, M, K), dtype=dtype, device=flag_gems.device)
    batch2 = torch.randn((B, K, N), dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(bias)
    ref_bias = utils.to_reference(bias.detach()).requires_grad_()
    ref_batch1 = utils.to_reference(batch1.detach()).requires_grad_()
    ref_batch2 = utils.to_reference(batch2.detach()).requires_grad_()
    res_bias = bias.detach().requires_grad_()
    res_batch1 = batch1.detach().requires_grad_()
    res_batch2 = batch2.detach().requires_grad_()

    ref_out = torch.addbmm(ref_bias, ref_batch1, ref_batch2, beta=beta, alpha=alpha)
    res_out = flag_gems.addbmm(res_bias, res_batch1, res_batch2, beta=beta, alpha=alpha)
    ref_grads = torch.autograd.grad(
        ref_out,
        (ref_bias, ref_batch1, ref_batch2),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(res_out, (res_bias, res_batch1, res_batch2), grad)
    direct_grads = flag_gems.addbmm_backward(
        grad, batch1, batch2, beta=beta, alpha=alpha
    )

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    atol = _addbmm_backward_atol(dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype, reduce_dim=N, atol=atol)
    utils.gems_assert_close(res_grads[2], ref_grads[2], dtype, reduce_dim=M, atol=atol)
    utils.gems_assert_close(direct_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(
        direct_grads[1], ref_grads[1], dtype, reduce_dim=N, atol=atol
    )
    utils.gems_assert_close(
        direct_grads[2], ref_grads[2], dtype, reduce_dim=M, atol=atol
    )


@pytest.mark.addbmm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addbmm_backward_registered_dispatch(dtype):
    B, M, N, K = 2, 8, 12, 16
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True)
    batch1 = torch.randn(
        (B, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    batch2 = torch.randn(
        (B, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(bias)
    ref_bias = utils.to_reference(bias.detach()).requires_grad_()
    ref_batch1 = utils.to_reference(batch1.detach()).requires_grad_()
    ref_batch2 = utils.to_reference(batch2.detach()).requires_grad_()
    ref_grads = torch.autograd.grad(
        torch.addbmm(ref_bias, ref_batch1, ref_batch2, beta=0.75, alpha=-0.5),
        (ref_bias, ref_batch1, ref_batch2),
        utils.to_reference(grad),
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["addbmm"])
        res_out = torch.addbmm(bias, batch1, batch2, beta=0.75, alpha=-0.5)
        assert "AddbmmFunction" in type(res_out.grad_fn).__name__
        res_grads = torch.autograd.grad(res_out, (bias, batch1, batch2), grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    atol = _addbmm_backward_atol(dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype, reduce_dim=N, atol=atol)
    utils.gems_assert_close(res_grads[2], ref_grads[2], dtype, reduce_dim=M, atol=atol)


@pytest.mark.addbmm_backward
@pytest.mark.parametrize("required_input", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addbmm_backward_single_input(required_input, dtype):
    B, M, N, K = 2, 7, 11, 13
    inputs = [
        torch.randn((M, N), dtype=dtype, device=flag_gems.device),
        torch.randn((B, M, K), dtype=dtype, device=flag_gems.device),
        torch.randn((B, K, N), dtype=dtype, device=flag_gems.device),
    ]
    inputs[required_input].requires_grad_()
    grad = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    ref_inputs = [utils.to_reference(value.detach()) for value in inputs]
    ref_inputs[required_input].requires_grad_()

    ref_out = torch.addbmm(*ref_inputs, beta=0.75, alpha=-0.5)
    res_out = flag_gems.addbmm(*inputs, beta=0.75, alpha=-0.5)
    ref_grad = torch.autograd.grad(
        ref_out, ref_inputs[required_input], utils.to_reference(grad)
    )[0]
    res_grad = torch.autograd.grad(res_out, inputs[required_input], grad)[0]

    reduce_dim = N if required_input == 1 else M if required_input == 2 else 1
    utils.gems_assert_close(
        res_grad,
        ref_grad,
        dtype,
        reduce_dim=reduce_dim,
        atol=_addbmm_backward_atol(dtype),
    )


@pytest.mark.addbmm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_addbmm_backward_noncontiguous(dtype):
    B, M, N, K = 2, 9, 7, 11
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True)
    batch1 = torch.randn((B, K, M), dtype=dtype, device=flag_gems.device).transpose(
        1, 2
    )
    batch2 = torch.randn((B, N, K), dtype=dtype, device=flag_gems.device).transpose(
        1, 2
    )
    batch1.requires_grad_()
    batch2.requires_grad_()
    grad = torch.randn_like(bias)
    ref_bias = utils.to_reference(bias.detach()).requires_grad_()
    ref_batch1 = utils.to_reference(batch1.detach()).requires_grad_()
    ref_batch2 = utils.to_reference(batch2.detach()).requires_grad_()

    ref_grads = torch.autograd.grad(
        torch.addbmm(ref_bias, ref_batch1, ref_batch2, beta=0.75, alpha=-0.5),
        (ref_bias, ref_batch1, ref_batch2),
        utils.to_reference(grad),
    )
    res_grads = torch.autograd.grad(
        flag_gems.addbmm(bias, batch1, batch2, beta=0.75, alpha=-0.5),
        (bias, batch1, batch2),
        grad,
    )

    utils.gems_assert_close(res_grads[0], ref_grads[0], dtype)
    atol = _addbmm_backward_atol(dtype)
    utils.gems_assert_close(res_grads[1], ref_grads[1], dtype, reduce_dim=N, atol=atol)
    utils.gems_assert_close(res_grads[2], ref_grads[2], dtype, reduce_dim=M, atol=atol)


@pytest.mark.addbmm_backward
def test_addbmm_backward_fused_kernel_count():
    """All three gradients are emitted by one target Triton kernel."""
    from torch.profiler import ProfilerActivity, profile

    B, M, N, K = 2, 64, 48, 32
    dtype = torch.float16
    bias = torch.randn((M, N), dtype=dtype, device=flag_gems.device, requires_grad=True)
    batch1 = torch.randn(
        (B, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    batch2 = torch.randn(
        (B, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(bias)
    out = flag_gems.addbmm(bias, batch1, batch2, beta=0.75, alpha=-0.5)

    # Warm up autotuning before profiling the steady-state launch count.
    torch.autograd.grad(out, (bias, batch1, batch2), grad, retain_graph=True)
    flag_gems.runtime.torch_device_fn.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (bias, batch1, batch2), grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "addbmm_backward" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

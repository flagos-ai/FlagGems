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

# Quick mode uses minimal shape; full mode covers various ranks and sizes
RENORM_SHAPES = (
    [(2, 8)]
    if QUICK_MODE
    else [
        (10, 20),
        (20, 10),
        (5, 32, 20),
        (4, 8, 16),
        (2, 4, 8, 16),
        # Along dim=0 this reduces over N = 2*64 = 128 elements per row;
        # keeping a case whose N differs from the other shapes guards the
        # segmented loop over N against shape-specific indexing errors.
        (5, 2, 64),
    ]
)


@pytest.mark.renorm
@pytest.mark.parametrize("shape", RENORM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_renorm(shape, dtype, p, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    maxnorm = 1.0

    ref_inp = utils.to_reference(inp)

    ref_out = torch.renorm(ref_inp, p, dim, maxnorm)
    res_out = flag_gems.renorm(inp, p, dim, maxnorm)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.renorm_
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
@pytest.mark.parametrize("shape", RENORM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_renorm_(shape, dtype, p, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    maxnorm = 1.0

    ref_out = ref_inp.renorm_(p, dim, maxnorm)
    original_pointer = inp.data_ptr()
    res_out = flag_gems.renorm_(inp, p, dim, maxnorm)
    assert res_out is inp
    assert res_out.data_ptr() == original_pointer

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.renorm_backward
@pytest.mark.parametrize("shape", [RENORM_SHAPES[0], RENORM_SHAPES[-1]])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_renorm_backward(shape, dtype, p, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.renorm(ref_inp, p, dim, 1.0), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(
        flag_gems.renorm(res_inp, p, dim, 1.0), res_inp, grad
    )
    direct_grad = flag_gems.renorm_backward(grad, inp, p, dim, 1.0)

    # Backward performs both a norm and an inner-product reduction per slice.
    reduce_dim = 2 * inp.numel() // inp.shape[dim]
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim)
    utils.gems_assert_close(direct_grad, ref_grad, dtype, reduce_dim=reduce_dim)


@pytest.mark.renorm_backward
def test_renorm_backward_bfloat16_dot_rounding():
    # Exact BF16 inputs whose product needs rounding before the reduction.
    inp = torch.full((2, 1), 0.75, dtype=torch.bfloat16, device=flag_gems.device)
    grad = torch.full_like(inp, 1.0078125)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    (expected,) = torch.autograd.grad(
        torch.renorm(ref_inp, 1.0, 1, 1.0), ref_inp, utils.to_reference(grad)
    )
    actual = flag_gems.renorm_backward(grad, inp, 1.0, 1, 1.0)

    utils.gems_assert_close(actual, expected, inp.dtype)


@pytest.mark.renorm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_renorm_backward_unclipped(dtype):
    inp = torch.randn((7, 11), dtype=dtype, device=flag_gems.device) * 0.01
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.renorm(ref_inp, 2.0, 0, 10.0), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(
        flag_gems.renorm(res_inp, 2.0, 0, 10.0), res_inp, grad
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=2 * inp.shape[0])


@pytest.mark.renorm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_renorm_backward_registered_dispatch(dtype):
    inp = torch.randn((8, 16), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.renorm(ref_inp, 2.0, 1, 1.0), ref_inp, utils.to_reference(grad)
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["renorm"])
        out = torch.renorm(inp, 2.0, 1, 1.0)
        assert "RenormFunction" in type(out.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(out, inp, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=2 * inp.shape[0])


@pytest.mark.renorm_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_renorm_backward_noncontiguous(dtype):
    inp = torch.randn((19, 13), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((19, 13), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.renorm(ref_inp, 3.0, 1, 1.0), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(
        flag_gems.renorm(res_inp, 3.0, 1, 1.0), res_inp, grad
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=2 * inp.shape[0])


@pytest.mark.renorm_backward
def test_renorm_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (32, 64), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    out = flag_gems.renorm(inp, 2.0, 1, 1.0)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "renorm_backward" in item[0].lower()]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

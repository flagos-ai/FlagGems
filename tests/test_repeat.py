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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    REPEAT_SIZES = [(2, 3, 4, 5)]
else:
    REPEAT_SIZES = [(2, 3, 4, 5), (5, 0, 4)]


@pytest.mark.repeat
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("sizes", REPEAT_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_repeat(shape, sizes, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    sizes = utils.unsqueeze_tuple(sizes, inp.ndim)

    ref_out = ref_inp.repeat(*sizes)
    res_out = flag_gems.repeat(inp, sizes)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.repeat_backward
@pytest.mark.parametrize(
    "shape,repeats",
    [((3, 5), (2, 3)), ((2, 3), (2, 1, 3)), ((2, 3, 4), (1, 2, 1))],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_backward(shape, repeats, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    padded_shape = (1,) * (len(repeats) - len(shape)) + shape
    grad = torch.randn(
        tuple(a * b for a, b in zip(repeats, padded_shape)),
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_out = ref_inp.repeat(*repeats)
    res_out = flag_gems.repeat(res_inp, repeats)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad))
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)
    direct_grad = flag_gems.repeat_backward(grad, repeats, shape)

    reduce_dim = math.prod(repeats)
    utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=reduce_dim, atol=1e-3)
    utils.gems_assert_close(
        direct_grad, ref_grad, dtype, reduce_dim=reduce_dim, atol=1e-3
    )


@pytest.mark.repeat_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_backward_zero_repeat(dtype):
    inp = torch.randn((3, 5), dtype=dtype, device=flag_gems.device)
    grad = torch.empty((6, 0), dtype=dtype, device=flag_gems.device)
    result = flag_gems.repeat_backward(grad, (2, 0), inp.shape)
    expected = torch.zeros_like(utils.to_reference(inp))
    utils.gems_assert_equal(result, expected)


@pytest.mark.repeat_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_backward_registered_dispatch(dtype):
    inp = torch.randn((7, 11), dtype=dtype, device=flag_gems.device, requires_grad=True)
    repeats = (2, 3)
    grad = torch.randn((14, 33), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        ref_inp.repeat(*repeats), ref_inp, utils.to_reference(grad)
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["repeat"])
        out = inp.repeat(*repeats)
        assert "RepeatAutogradFunction" in type(out.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(out, inp, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(
        res_grad, ref_grad, dtype, reduce_dim=math.prod(repeats), atol=1e-3
    )


@pytest.mark.repeat_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_repeat_backward_noncontiguous(dtype):
    inp = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    repeats = (2, 3)
    grad = torch.randn((15, 14), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        ref_inp.repeat(*repeats), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(flag_gems.repeat(res_inp, repeats), res_inp, grad)
    utils.gems_assert_close(
        res_grad, ref_grad, dtype, reduce_dim=math.prod(repeats), atol=1e-3
    )


@pytest.mark.repeat_backward
def test_repeat_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (32, 64), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    repeats = (2, 3)
    grad = torch.randn((64, 192), dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.repeat(inp, repeats)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [item for item in launches if "repeat_backward" in item[0].lower()]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

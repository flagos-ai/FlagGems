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

# rot90 test shapes - the rotation plane is dims=[0, 1]; higher-rank shapes
# check that the trailing dims are carried through unchanged
if cfg.QUICK_MODE:
    ROT90_SHAPES = [(2, 3), (100, 128), (2, 3, 2), (16, 24, 3, 5)]
    ROT90_K_VALUES = [0, 1, -1]
else:
    ROT90_SHAPES = [(2, 3), (5, 7), (100, 128), (2, 3, 2), (5, 7, 3), (16, 24, 3, 5)]
    ROT90_K_VALUES = [0, 1, 2, 3, -1, -2]


@pytest.mark.rot90
@pytest.mark.parametrize("shape", ROT90_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("k", ROT90_K_VALUES)
def test_rot90(shape, dtype, k):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.rot90(ref_inp, k, [0, 1])
    res_out = flag_gems.rot90(inp, k, [0, 1])

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.rot90_backward
@pytest.mark.parametrize("shape", ROT90_SHAPES_2D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("k", ROT90_K_VALUES)
def test_rot90_backward(shape, dtype, k):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_shape = shape if k % 2 == 0 else (shape[1], shape[0])
    grad = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    ref_out = torch.rot90(ref_inp, k, [0, 1])
    res_out = flag_gems.rot90(res_inp, k, [0, 1])
    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, utils.to_reference(grad))
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad)
    direct_grad = flag_gems.rot90_backward(grad, k, [0, 1])

    utils.gems_assert_equal(res_grad, ref_grad)
    utils.gems_assert_equal(direct_grad, ref_grad)


@pytest.mark.rot90_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_backward_registered_dispatch(dtype):
    inp = torch.randn((7, 11), dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.randn((11, 7), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    (ref_grad,) = torch.autograd.grad(
        torch.rot90(ref_inp, 1, [0, 1]), ref_inp, utils.to_reference(grad)
    )

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["rot90"])
        out = torch.rot90(inp, 1, [0, 1])
        assert "Rot90AutogradFunction" in type(out.grad_fn).__name__
        (res_grad,) = torch.autograd.grad(out, inp, grad)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.rot90_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_backward_noncontiguous(dtype):
    inp = torch.randn((5, 7), dtype=dtype, device=flag_gems.device).t()
    grad = torch.randn((7, 5), dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp.detach()).requires_grad_()
    res_inp = inp.detach().requires_grad_()

    (ref_grad,) = torch.autograd.grad(
        torch.rot90(ref_inp, 1, [0, 1]), ref_inp, utils.to_reference(grad)
    )
    (res_grad,) = torch.autograd.grad(
        flag_gems.rot90(res_inp, 1, [0, 1]), res_inp, grad
    )
    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.rot90_backward
def test_rot90_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn(
        (64, 96), dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((96, 64), dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.rot90(inp, 1, [0, 1])

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, inp, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    kernels = [item for item in launches if "rot90" in item[0].lower()]
    assert len(kernels) == 1 and kernels[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

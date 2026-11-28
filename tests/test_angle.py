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


@pytest.mark.angle
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.COMPLEX_DTYPES + utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_angle(shape, dtype):
    if flag_gems.vendor_name == "enflame" and dtype == torch.complex32:
        pytest.skip("enflame does not support ComplexHalf")

    if cfg.TO_CPU and dtype == torch.complex32:
        # Complex32 on CPU is not supported
        return

    if not cfg.TO_CPU and dtype in [torch.float16, torch.bfloat16]:
        # Half is treated as an unsupported data type on GPU
        return

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    elif dtype in utils.COMPLEX_DTYPES + utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.angle(ref_inp)

    res_out = flag_gems.angle(inp)

    dtype_out = res_out.dtype
    utils.gems_assert_close(res_out, ref_out, dtype_out)


ANGLE_BACKWARD_SHAPES = [(), (1,), (37,), (31, 47), (4, 5, 7)]


@pytest.mark.angle_backward
@pytest.mark.parametrize("shape", ANGLE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_angle_backward_complex(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    grad = torch.randn(shape, dtype=inp.real.dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.angle(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]
    result = flag_gems.angle_backward(grad, inp)
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.angle_backward
@pytest.mark.parametrize("shape", ANGLE_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_angle_backward_real(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad = torch.randn_like(inp)
    result = flag_gems.angle_backward(grad, inp)
    expected = torch.zeros_like(utils.to_reference(inp))
    utils.gems_assert_equal(result, expected)


@pytest.mark.angle_backward
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_angle_backward_zero_and_axes(dtype):
    inp = torch.tensor(
        [0.0j, 1.0 + 0.0j, 0.0 + 1.0j, -2.0 + 3.0j],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=inp.real.dtype).to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.angle(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]
    result = flag_gems.angle_backward(grad, inp)
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.angle_backward
def test_angle_backward_registered_dispatch():
    inp = torch.randn(
        (31, 47), dtype=torch.complex64, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((31, 47), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.angle(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["angle"])
        output = torch.angle(inp)
        assert "AngleFunction" in type(output.grad_fn).__name__
        result = torch.autograd.grad(output, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(result, expected, torch.complex64)


@pytest.mark.angle_backward
def test_angle_backward_noncontiguous():
    inp = torch.randn(
        (13, 17, 19), dtype=torch.complex64, device=flag_gems.device
    ).transpose(0, 2)
    grad = torch.randn(
        (13, 17, 19), dtype=torch.float32, device=flag_gems.device
    ).transpose(0, 2)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.angle(ref_inp), ref_inp, utils.to_reference(grad, False)
    )[0]
    result = flag_gems.angle_backward(grad, inp)
    utils.gems_assert_close(result, expected, torch.complex64)


@pytest.mark.angle_backward
def test_angle_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((64, 512), dtype=torch.complex64, device=flag_gems.device)
    grad = torch.randn((64, 512), dtype=torch.float32, device=flag_gems.device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.angle_backward(grad, inp)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "angle" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

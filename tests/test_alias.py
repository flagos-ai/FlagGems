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

ALIAS_BACKWARD_SHAPES = [(), (0,), (4096,), (31, 47), (17, 29, 5)]


@pytest.mark.alias_backward
@pytest.mark.parametrize("shape", ALIAS_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alias_backward(shape, dtype):
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    result = flag_gems.alias_backward(grad)
    utils.gems_assert_equal(result, utils.to_reference(grad, False))


@pytest.mark.alias_backward
@pytest.mark.parametrize("layout", ["transpose", "sliced"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alias_backward_noncontiguous(layout, dtype):
    grad = torch.randn((34, 58), dtype=dtype, device=flag_gems.device)
    grad = grad.T if layout == "transpose" else grad[::2, ::2]
    result = flag_gems.alias_backward(grad)
    utils.gems_assert_equal(result, utils.to_reference(grad, False))


@pytest.mark.alias_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_alias_backward_registered_dispatch(dtype):
    inp = torch.randn(
        (31, 47), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn_like(inp)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.ops.aten.alias.default(ref_inp),
        ref_inp,
        utils.to_reference(grad, False),
    )[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["alias"])
        out = torch.ops.aten.alias.default(inp)
        assert "AliasFunction" in type(out.grad_fn).__name__
        assert torch._C._is_alias_of(out, inp)
        result = torch.autograd.grad(out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_equal(result, expected)


@pytest.mark.alias_backward
def test_alias_backward_preserves_forward_view_semantics():
    inp = torch.randn((7, 11), device=flag_gems.device)
    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["alias"])
        out = torch.ops.aten.alias.default(inp)
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    assert out is not inp
    assert torch._C._is_alias_of(out, inp)
    assert out.shape == inp.shape
    assert out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()


@pytest.mark.alias_backward
def test_alias_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    grad = torch.randn((4096,), dtype=torch.float16, device=flag_gems.device)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.alias_backward(grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "alias" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

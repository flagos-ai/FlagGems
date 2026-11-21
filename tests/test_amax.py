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
    DIMS_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
    KEEPDIM_DIMS_SHAPE = [(True, DIMS_LIST[0], utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KEEPDIM_DIMS_SHAPE = list(
        zip([True, False] * 2, DIMS_LIST, utils.REDUCTION_SHAPES + [(7, 4, 11, 1)])
    )


@pytest.mark.amax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_amax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    # Direct GEMS call: the dispatcher lifts an int dim to IntList, so normalise
    # it here to match amax's own contract.
    dims = dim if isinstance(dim, (list, tuple)) else [dim]
    res_out = flag_gems.amax(inp, dim=dims, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


# Compact cases cover scalar, aligned, tail and multi-axis reduction layouts.
AMAX_BACKWARD_CASES = [
    ((), None, False),
    ((31, 47), (1,), False),
    ((17, 33, 5), (0, 2), False),
    ((8, 9, 10), (1,), True),
]


@pytest.mark.amax_backward
@pytest.mark.parametrize("shape, dim, keepdim", AMAX_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_amax_backward(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_shape = torch.amax(inp, dim=dim, keepdim=keepdim).shape
    grad = torch.randn(grad_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amax(ref_inp, dim=dim, keepdim=keepdim),
        ref_inp,
        utils.to_reference(grad, False),
    )[0]
    output = flag_gems.amax(inp, dim=dim, keepdim=keepdim)
    result = flag_gems.amax_backward(grad, inp, output, dim=dim, keepdim=keepdim)
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.amax_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_amax_backward_ties(dtype):
    inp = torch.tensor(
        [[2.0, 2.0, 1.0], [3.0, 1.0, 3.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    grad = torch.tensor([2.0, -3.0], dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amax(ref_inp, dim=1), ref_inp, utils.to_reference(grad, False)
    )[0]
    output = flag_gems.amax(inp, dim=(1,))
    result = flag_gems.amax_backward(grad, inp, output, dim=(1,))
    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.amax_backward
def test_amax_backward_registered_dispatch():
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn(
        (31, 47), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    grad = torch.randn((31,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amax(ref_inp, dim=1), ref_inp, utils.to_reference(grad, False)
    )[0]

    lib = torch.library.Library("aten", "IMPL")
    previous_registrar = getattr(flag_gems, "current_work_registrar", None)
    try:
        flag_gems.only_enable(lib=lib, include=["amax"])
        out = torch.amax(inp, dim=1)
        assert "AmaxFunction" in type(out.grad_fn).__name__
        result = torch.autograd.grad(out, inp, grad)[0]
    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar

    utils.gems_assert_close(result, expected, dtype)


@pytest.mark.amax_backward
def test_amax_backward_noncontiguous_and_nan():
    dtype = utils.FLOAT_DTYPES[0]
    inp = torch.randn((13, 17, 19), dtype=dtype, device=flag_gems.device).transpose(
        0, 2
    )
    inp[0, 0, 0] = float("nan")
    grad = torch.randn((19, 17), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()
    expected = torch.autograd.grad(
        torch.amax(ref_inp, dim=2), ref_inp, utils.to_reference(grad, False)
    )[0]
    output = flag_gems.amax(inp, dim=(2,))
    result = flag_gems.amax_backward(grad, inp, output, dim=(2,))
    utils.gems_assert_close(result, expected, dtype, equal_nan=True)


@pytest.mark.amax_backward
def test_amax_backward_single_kernel():
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((64, 512), dtype=torch.float16, device=flag_gems.device)
    grad = torch.randn((64,), dtype=torch.float16, device=flag_gems.device)
    output = flag_gems.amax(inp, dim=(1,))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        flag_gems.amax_backward(grad, inp, output, dim=(1,))
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    backward = [(name, count) for name, count in launches if "amax" in name]
    assert len(backward) == 1 and backward[0][1] == 1, launches
    assert sum(count for _, count in launches) == 1, launches

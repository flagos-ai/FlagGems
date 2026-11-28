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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [1]
    SHAPES = [(1, 256)]
    BACKWARD_SHAPES = [(1, 256)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    SHAPES = [(1, 256), (4096, 256), (200, 2560, 3), (1, 0, 128, 512)]
    BACKWARD_SHAPES = [(1, 256), (4096, 256), (200, 2560, 3)]

random.seed(time.time() // 100)


# Regression test for issue #6109: `softmax_heur_tile_n_inner` used to return
# next_power_of_2(N) for every N up to 32768, so rows of 16385..32768 widened to
# a 32768-element tile.  The ONE_TILE_PER_CTA branch of `softmax_kernel_inner`
# keeps that row plus its fp32 exp() result live, which does not fit Ascend's
# 192 KiB unified buffer, and the compiler fails with
# "ub overflow ... while 1572864 bits available".  32769 is the control: it
# already used TILE_N == 4096 and kept working.
_WIDE_INNER_ROWS = [16385] if cfg.QUICK_MODE else [16385, 20000, 32768, 32769]


@pytest.mark.softmax
@pytest.mark.parametrize("n", _WIDE_INNER_ROWS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_wide_inner_row(n, dtype, neg_inf):
    inp = torch.randn((2, n), dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=-1)
    res_out = flag_gems.softmax(inp, -1)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Issue 2852: This fails at (1, 2) (200, 40999, 3)
@pytest.mark.softmax
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.softmax(ref_inp, dim=dim)
    res_out = flag_gems.softmax(inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.softmax_out
@pytest.mark.parametrize(
    "shape",
    (
        [(1, 256)]
        if cfg.QUICK_MODE
        else [(1, 256), (4096, 256), (200, 2560, 3), (1, 0, 128, 512)]
    ),
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_out(shape, dtype, dim, neg_inf):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        inp = torch.where(inp < 0.0, float("-inf"), inp)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._softmax.out(ref_inp, dim, False, out=ref_out)

    res_out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.softmax_out(inp, dim, False, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.softmax_backward_out
@pytest.mark.parametrize(
    "shape", [(1, 256)] if cfg.QUICK_MODE else [(1, 256), (4096, 256), (200, 2560, 3)]
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_backward_out(shape, dtype, dim, neg_inf):
    if shape[dim] == 1 and flag_gems.vendor_name == "kunlunxin":
        pytest.skip(
            "Issue #2851 _softmax_backward_data short-circuits to zero when reduction dim "
            "is 1, while the Triton kernel computes normally with synthetic inputs, "
            "causing a mismatch that does not reflect a real correctness issue."
        )
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.empty(shape, dtype=ref_grad.dtype, device=ref_grad.device)
    torch.ops.aten._softmax_backward_data.out(
        ref_grad, ref_out, dim, ref_grad.dtype, grad_input=ref_in_grad
    )

    res_in_grad = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    flag_gems.softmax_backward_out(
        res_grad, res_out, dim, dtype, grad_input=res_in_grad
    )
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


@pytest.mark.softmax_backward
@pytest.mark.softmax_backward_data
@pytest.mark.parametrize("shape", BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("neg_inf", [True, False])
def test_softmax_backward(shape, dtype, dim, neg_inf):
    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if neg_inf:
        res_grad = torch.where(res_grad < 0.0, float("-inf"), res_grad)
    res_out = torch.randn_like(res_grad)

    ref_grad = utils.to_reference(res_grad, True)
    ref_out = utils.to_reference(res_out, True)

    ref_in_grad = torch.ops.aten._softmax_backward_data(
        ref_grad, ref_out, dim, ref_grad.dtype
    )
    res_in_grad = flag_gems.softmax_backward(res_grad, res_out, dim, dtype)

    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=shape[dim], equal_nan=True
    )


@pytest.mark.softmax
@pytest.mark.softmax_out
@pytest.mark.parametrize("dim", [1, -1])
@pytest.mark.parametrize("use_out", [False, True])
def test_softmax_noncontiguous_layout(dim, use_out):
    cpu = torch.randn(2, 4, 5, 5).permute(0, 2, 3, 1)
    inp = cpu.permute(0, 3, 1, 2).to(flag_gems.device).permute(0, 2, 3, 1)
    assert not inp.is_contiguous()
    expected = torch.softmax(cpu, dim=dim)

    if use_out:
        out = torch.empty((2, 5, 5, 8), device=inp.device, dtype=inp.dtype)[..., ::2]
        stride = out.stride()
        actual = flag_gems.softmax_out(inp, dim, out=out)
        assert actual is out
        assert actual.stride() == stride
    else:
        actual = flag_gems.softmax(inp, dim)
        assert actual.is_contiguous()

    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.softmax_backward_data
@pytest.mark.softmax_backward_out
@pytest.mark.parametrize("dim", [1, -1])
@pytest.mark.parametrize("use_out", [False, True])
def test_softmax_backward_noncontiguous_layout(dim, use_out):
    cpu = torch.randn(2, 5, 5, 4)
    expected_out = torch.softmax(cpu, dim=dim)
    expected_grad = torch.randn_like(expected_out)
    expected = torch.ops.aten._softmax_backward_data(
        expected_grad, expected_out, dim, cpu.dtype
    )

    def to_strided_device(tensor):
        return (
            tensor.permute(0, 3, 1, 2)
            .contiguous()
            .to(flag_gems.device)
            .permute(0, 2, 3, 1)
        )

    out = to_strided_device(expected_out)
    grad = to_strided_device(expected_grad)
    assert not out.is_contiguous() and not grad.is_contiguous()

    if use_out:
        grad_input = torch.empty(
            (2, 5, 5, 8), device=out.device, dtype=out.dtype
        )[..., ::2]
        stride = grad_input.stride()
        actual = flag_gems.softmax_backward_out(
            grad, out, dim, cpu.dtype, grad_input=grad_input
        )
        assert actual is grad_input
        assert actual.stride() == stride
    else:
        actual = flag_gems.softmax_backward(grad, out, dim, cpu.dtype)
        assert actual.is_contiguous()

    torch.testing.assert_close(actual.cpu(), expected)

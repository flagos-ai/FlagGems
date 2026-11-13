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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


_SHAPES = [*utils.REDUCTION_SHAPES, (0,), (2, 0, 3)]
_REDUCTIONS = ["none", "mean", "sum"]
_REDUCTION_ENUM = {"none": 0, "mean": 1, "sum": 2}


def _inputs(shape, dtype, weighted):
    torch.manual_seed(0)
    inp = torch.sigmoid(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    target = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    weight = (
        torch.rand(shape, dtype=dtype, device=flag_gems.device) if weighted else None
    )
    return inp, target, weight


def _assert_loss(result, reference, dtype, reduction, numel):
    utils.gems_assert_close(
        result,
        reference,
        dtype,
        equal_nan=True,
        reduce_dim=max(numel, 1) if reduction == "sum" else 1,
    )


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_cross_entropy(shape, dtype, reduction, weighted):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and shape in ((0,), (2, 0, 3))
    ):
        pytest.skip(
            "Kunlunxin native binary-cross-entropy reference rejects empty input"
        )
    if (
        flag_gems.vendor_name == "mthreads"
        and not utils.TO_CPU
        and reduction == "mean"
        and 0 in shape
    ):
        pytest.skip("MUSA native BCE mean returns zero for empty input instead of NaN")
    inp, target, weight = _inputs(shape, dtype, weighted)
    inp_before = inp.clone()
    target_before = target.clone()
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_weight = utils.to_reference(weight, True) if weight is not None else None
    reference = torch.nn.functional.binary_cross_entropy(
        ref_inp, ref_target, weight=ref_weight, reduction=reduction
    )

    result = flag_gems.binary_cross_entropy(
        inp, target, weight=weight, reduction=reduction
    )

    _assert_loss(result, reference, dtype, reduction, inp.numel())
    assert inp.device == inp_before.device
    assert inp.dtype == inp_before.dtype and inp.shape == inp_before.shape
    assert target.device == target_before.device
    assert target.dtype == target_before.dtype and target.shape == target_before.shape
    if utils.TO_CPU:
        inp_before = inp_before.cpu()
        target_before = target_before.cpu()
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(target, target_before)


@pytest.mark.binary_cross_entropy_out
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_cross_entropy_out(shape, dtype, reduction, weighted):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and not utils.TO_CPU
        and shape in ((0,), (2, 0, 3))
    ):
        pytest.skip(
            "Kunlunxin native binary-cross-entropy reference rejects empty input"
        )
    inp, target, weight = _inputs(shape, dtype, weighted)
    reference = torch.nn.functional.binary_cross_entropy(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        weight=utils.to_reference(weight, True) if weight is not None else None,
        reduction=reduction,
    )
    out_shape = shape if reduction == "none" else ()
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    pointer = out.data_ptr()

    result = flag_gems.binary_cross_entropy_out(
        inp, target, weight, _REDUCTION_ENUM[reduction], out=out
    )

    assert result is out
    assert out.data_ptr() == pointer
    _assert_loss(out, reference, dtype, reduction, inp.numel())


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_binary_cross_entropy_boundary(dtype, reduction, weighted):
    inp = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.5], dtype=dtype, device=flag_gems.device)
    target = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 1.0], dtype=dtype, device=flag_gems.device
    )
    weight = torch.rand_like(inp) if weighted else None
    reference = torch.nn.functional.binary_cross_entropy(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        weight=utils.to_reference(weight, True) if weight is not None else None,
        reduction=reduction,
    )

    result = flag_gems.binary_cross_entropy(
        inp, target, weight=weight, reduction=reduction
    )

    _assert_loss(result, reference, dtype, reduction, inp.numel())


@pytest.mark.binary_cross_entropy
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_binary_cross_entropy_autograd(reduction):
    torch.manual_seed(0)
    inp = torch.sigmoid(
        torch.randn((3, 5), device=flag_gems.device, requires_grad=True)
    )
    target = torch.rand_like(inp)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    ref_target = utils.to_reference(target, True)
    reference = torch.nn.functional.binary_cross_entropy(
        ref_inp, ref_target, reduction=reduction
    )
    result = flag_gems.binary_cross_entropy(inp, target, reduction=reduction)
    ref_grad_output = torch.ones_like(reference)
    grad_output = torch.ones_like(result)
    (reference_grad,) = torch.autograd.grad(reference, ref_inp, ref_grad_output)
    (result_grad,) = torch.autograd.grad(result, inp, grad_output)
    utils.gems_assert_close(result_grad, reference_grad, torch.float32)

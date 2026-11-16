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


_SHAPES = list(dict.fromkeys([*utils.REDUCTION_SHAPES, (2, 3), (4, 5, 7), (0,), (2, 0, 3)]))
_REDUCTIONS = [0, 1, 2]


def _make_inputs(shape, dtype, weighted):
    inp = torch.sigmoid(torch.randn(shape, dtype=dtype, device=flag_gems.device))
    target = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    weight = torch.rand(shape, dtype=dtype, device=flag_gems.device) if weighted else None
    return inp, target, weight


def _reference(inp, target, weight, reduction):
    names = {0: "none", 1: "mean", 2: "sum"}
    if flag_gems.vendor_name == "mthreads" and inp.numel() == 0 and reduction == 1:
        return torch.nn.functional.binary_cross_entropy(
            inp.cpu(), target.cpu(), None if weight is None else weight.cpu(), reduction="mean"
        )
    return torch.nn.functional.binary_cross_entropy(
        utils.to_reference(inp, True),
        utils.to_reference(target, True),
        weight=utils.to_reference(weight, True) if weight is not None else None,
        reduction=names[reduction],
    )


def _call(inp, target, weight, reduction, out):
    return flag_gems.binary_cross_entropy_out(inp, target, weight, reduction, out=out)


def _assert_close(result, expected, dtype, reduction, numel):
    utils.gems_assert_close(
        result,
        expected,
        dtype,
        equal_nan=True,
        reduce_dim=max(numel, 1) if reduction == 2 else 1,
    )


@pytest.mark.binary_cross_entropy_out
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", _REDUCTIONS)
@pytest.mark.parametrize("weighted", [False, True])
def test_binary_cross_entropy_out(shape, dtype, reduction, weighted):
    inp, target, weight = _make_inputs(shape, dtype, weighted)
    inp_before = inp.clone()
    target_before = target.clone()
    weight_before = weight.clone() if weight is not None else None
    expected = _reference(inp, target, weight, reduction)
    out_shape = shape if reduction == 0 else ()
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    pointer = out.data_ptr()
    stride = out.stride()
    storage_offset = out.storage_offset()

    result = _call(inp, target, weight, reduction, out)

    assert result is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    assert out.storage_offset() == storage_offset
    if flag_gems.vendor_name == "mthreads" and inp.numel() == 0 and reduction == 1:
        assert out.device == inp.device
        _assert_close(out.cpu(), expected, dtype, reduction, inp.numel())
    else:
        _assert_close(out, expected, dtype, reduction, inp.numel())
    utils.gems_assert_equal(inp, inp_before)
    utils.gems_assert_equal(target, target_before)
    if weight is not None:
        utils.gems_assert_equal(weight, weight_before)


@pytest.mark.binary_cross_entropy_out
def test_binary_cross_entropy_out_preserves_noncontiguous_storage():
    inp, target, weight = _make_inputs((3, 4), torch.float32, True)
    expected = _reference(inp, target, weight, 0)
    backing = torch.empty((3, 8), device=flag_gems.device)
    out = backing[:, 2:6]
    backing_before = backing.clone()
    pointer = out.data_ptr()
    stride = out.stride()
    storage_offset = out.storage_offset()

    result = _call(inp, target, weight, 0, out)

    assert result is out
    assert out.data_ptr() == pointer
    assert out.stride() == stride
    assert out.storage_offset() == storage_offset
    utils.gems_assert_close(out, expected, torch.float32)
    utils.gems_assert_equal(backing[:, :2], backing_before[:, :2])
    utils.gems_assert_equal(backing[:, 6:], backing_before[:, 6:])


@pytest.mark.binary_cross_entropy_out
@pytest.mark.parametrize("reduction", _REDUCTIONS)
def test_binary_cross_entropy_out_resizes_empty_buffer(reduction):
    inp, target, weight = _make_inputs((3, 5), torch.float32, False)
    expected = _reference(inp, target, weight, reduction)
    out = torch.empty((0,), device=flag_gems.device)

    result = _call(inp, target, weight, reduction, out)

    assert result is out
    assert out.shape == expected.shape
    _assert_close(out, expected, torch.float32, reduction, inp.numel())


@pytest.mark.binary_cross_entropy_out
def test_binary_cross_entropy_out_cast_contract():
    inp, target, weight = _make_inputs((3, 5), torch.float32, False)
    expected = _reference(inp, target, weight, 0)
    wide_out = torch.empty(inp.shape, dtype=torch.float64, device=flag_gems.device)
    result = _call(inp, target, weight, 0, wide_out)
    assert result is wide_out
    assert wide_out.device == inp.device
    assert wide_out.dtype == torch.float64
    utils.gems_assert_close(wide_out.cpu().float(), expected.cpu(), torch.float32)

    int_out = torch.empty(inp.shape, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        _call(inp, target, weight, 0, int_out)


@pytest.mark.binary_cross_entropy_out
def test_binary_cross_entropy_out_resizes_wrong_shape_buffer():
    inp, target, _ = _make_inputs((3, 5), torch.float32, False)
    expected = _reference(inp, target, None, 0)
    out = torch.empty((2, 2), device=flag_gems.device)

    result = _call(inp, target, None, 0, out)

    assert result is out
    assert out.shape == expected.shape
    utils.gems_assert_close(out, expected, torch.float32)


@pytest.mark.binary_cross_entropy_out
def test_binary_cross_entropy_out_rejects_invalid_weight_and_device():
    inp, target, _ = _make_inputs((3, 5), torch.float32, False)
    bad_weight = torch.rand((2,), device=flag_gems.device)
    out = torch.empty_like(inp)
    with pytest.raises((RuntimeError, AssertionError)):
        _call(inp, target, bad_weight, 0, out)

    cpu_out = torch.empty(inp.shape)
    with pytest.raises((RuntimeError, AssertionError)):
        _call(inp, target, None, 0, cpu_out)

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
from . import test_utils as tu

# Gather values at the sparse mask positions; both operands must have the same shape.
# The reference rejects FP8 self operands during dtype promotion.
_SPARSE_MASK_DTYPES = (
    [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
    ]
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
)

_SPARSE_MASK_STRUCT_SHAPES = [
    (16,),
    (2, 3),
    (8, 8),
    (16, 32),
    (4, 8, 16),
    (3, 7, 5, 4),
]

_SPARSE_MASK_BACKWARD_SHAPES = [(16, 32), (4, 8, 16), (3, 7, 5, 4)]

_SPARSE_MASK_NANINF_SHAPES = [(16,), (8, 8), (3, 7, 5, 4)]

# (base shape, shape after slicing the last dimension by 2).
_SPARSE_MASK_NON_CONTIGUOUS_CASES = [
    ((4, 8, 16), (4, 8, 8)),
    ((6, 10), (6, 5)),
]

_SPARSE_MASK_VALUE_RANGE_CASES = [
    (shape, value_range, dtype)
    for dtype in _SPARSE_MASK_DTYPES
    for value_range in tu.selected_ranges()
    for shape in tu.selected_shapes()
]


def _make_mask(shape, density=0.5):
    # Keep values above the threshold, then convert the bool mask to coalesced COO.
    return (torch.rand(shape, device=flag_gems.device) > density).to_sparse()


def _assert_masked(res_out, ref_out):
    assert res_out.layout == ref_out.layout
    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.is_coalesced() == ref_out.is_coalesced()
    tu.assert_result_equal(res_out.indices(), ref_out.indices())
    tu.assert_result_equal(res_out.values(), ref_out.values())


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape,value_range,dtype", _SPARSE_MASK_VALUE_RANGE_CASES)
def test_sparse_mask_value_ranges(shape, value_range, dtype):
    # Use fewer mask entries for large tensors.
    numel = math.prod(shape)
    keep_threshold = 0.5 if numel <= 4096 else 0.9
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    mask = _make_mask(shape, density=keep_threshold)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = flag_gems.sparse_mask(inp, mask)

    _assert_masked(res_out, ref_out)
    # The gather must not mutate either operand: the reference was computed on
    # a pristine clone, so any candidate mutation of ``inp``/``mask`` shows up
    # here.
    tu.assert_result_equal(inp, ref_inp)
    tu.assert_result_equal(mask, ref_mask)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_sparse_self(shape, dtype):
    dense = tu.make_input(dtype, shape, ["-1", "1"])
    inp = dense.to_sparse()
    mask = _make_mask(shape)
    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = flag_gems.sparse_mask(inp, mask)

    _assert_masked(res_out, ref_out)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("base_shape,shape", _SPARSE_MASK_NON_CONTIGUOUS_CASES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_non_contiguous(base_shape, shape, dtype):
    base = tu.make_input(dtype, base_shape, ["-1", "1"])
    ref_base = tu.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]
    assert not inp.is_contiguous()
    mask = _make_mask(shape)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = flag_gems.sparse_mask(inp, mask)

    _assert_masked(res_out, ref_out)


@pytest.mark.sparse_mask_out
@pytest.mark.parametrize("shape", _SPARSE_MASK_STRUCT_SHAPES)
@pytest.mark.parametrize("dtype", _SPARSE_MASK_DTYPES)
def test_sparse_mask_out(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    mask = _make_mask(shape)
    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    out = torch.empty_like(mask, dtype=dtype)
    ref_out = torch.empty_like(ref_mask, dtype=dtype)

    torch.ops.aten.sparse_mask.out(ref_inp, ref_mask, out=ref_out)
    res_ret = flag_gems.sparse_mask(inp, mask, out=out)

    assert res_ret is out
    _assert_masked(res_ret, ref_out)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_NANINF_SHAPES)
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_SPARSE_MASK_DTYPES))
)
def test_sparse_mask_nan_inf(shape, dtype, scenario):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    specials = tu.make_special_input(dtype, scenario)
    n = min(inp.numel(), specials.numel())
    inp.flatten()[:n] = specials[:n]

    # Mask every position holding a special value (plus random extras) so the
    # nan/inf entries are guaranteed to flow into the result.
    mask_dense = torch.rand(shape, device=flag_gems.device) > 0.7
    mask_dense.flatten()[:n] = True
    mask = mask_dense.to_sparse()

    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    res_out = flag_gems.sparse_mask(inp, mask)

    _assert_masked(res_out, ref_out)


@pytest.mark.sparse_mask
@pytest.mark.parametrize("shape", _SPARSE_MASK_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_sparse_mask_backward(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_()
    mask = _make_mask(shape)
    dense_grad = tu.make_input(dtype, shape, ["-1", "1"])
    # grad_output for the sparse result: sparse COO with the mask's indices and
    # values zero outside the mask.
    grad_out = (dense_grad * mask.to_dense()).to_sparse()

    ref_inp = tu.to_reference(inp)
    ref_mask = tu.to_reference(mask)
    ref_grad_out = tu.to_reference(grad_out)

    ref_out = torch.ops.aten.sparse_mask(ref_inp, ref_mask)
    ref_in_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)[0]

    res_out = flag_gems.sparse_mask(inp, mask)
    _assert_masked(res_out, ref_out)

    # The candidate must retain the reference's autograd behavior.
    assert res_out.requires_grad
    res_in_grad = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)[0]
    tu.assert_result_close(res_in_grad, ref_in_grad)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_shape_mismatch():
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    mask = _make_mask((4, 6))
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(
            tu.to_reference(self_t),
            tu.to_reference(mask),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_mask(self_t, mask)


@pytest.mark.sparse_mask_negative
def test_sparse_mask_rejects_dense_mask():
    self_t = tu.make_input(torch.float32, (4, 5), ["-1", "1"])
    dense_mask = torch.rand(4, 5, device=flag_gems.device) > 0.5
    with pytest.raises(RuntimeError):
        torch.ops.aten.sparse_mask(
            tu.to_reference(self_t),
            tu.to_reference(dense_mask),
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.sparse_mask(self_t, dense_mask)

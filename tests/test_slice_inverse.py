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

"""Correctness tests for ``aten::slice_inverse``.

``slice_inverse(self, src, dim=0, start=None, end=None, step=1)`` returns
``self``'s storage reinterpreted with ``src``'s sizes, strides and storage
offset; the slice arguments are only read by the backward pass.  The checks
below assert that view contract (values, shape, strides, storage offset,
storage aliasing and write-through), and ``src`` always lives in a separate
allocation so that the storage assertion rejects a candidate returning ``src``.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

pytestmark = pytest.mark.slice_inverse

DTYPES = list(tu.REQUIRED_DTYPES)
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)
DTYPES += [torch.bool, torch.complex64]

# Payload for the independent src allocation; its storage must never supply
# the result, even when its values happen to match self.
SRC_RANGE = ["0", "max"]


def _half(size):
    return max(1, size // 2)


def _half_view(tensor):
    """``src`` of the default cases: a half slice of ``tensor`` (0-D: itself)."""
    if tensor.dim() == 0:
        return tensor
    return tensor[: _half(tensor.shape[0])]


def _region(tensor, region):
    dim, start, end, step = region
    index = [slice(None)] * tensor.dim()
    index[dim % tensor.dim()] = slice(start, end, step)
    return tensor[tuple(index)]


def _region_inputs(tensor, region):
    """``(src, slice_args)`` for one region row; ``None`` means the defaults."""
    if region is None:
        return _half_view(tensor), ()
    return _region(tensor, region), region


def _meta(tensor):
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())


def _snapshot(*tensors):
    return [_meta(tensor) for tensor in tensors]


def _assert_view(result, reference, src, base, operands):
    """Values match the reference, metadata comes from src, inputs stay intact."""
    tu.assert_result_equal(result, reference)
    assert result.shape == src.shape
    assert result.stride() == src.stride()
    assert result.storage_offset() == src.storage_offset()
    assert result.stride() == reference.stride()
    assert result.storage_offset() == reference.storage_offset()
    assert result.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()
    if src.numel():
        assert result.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()
    assert _snapshot(base, src) == operands


@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", DTYPES)
def test_slice_inverse(shape, value_range, dtype):
    base = tu.make_input(dtype, shape, value_range)
    src_plain = tu.make_input(dtype, shape, SRC_RANGE)
    src = _half_view(src_plain)
    args = () if not shape else (0, 0, _half(shape[0]), 1)
    operands = _snapshot(base, src)

    ref_base = tu.to_reference(base)
    ref_src = _half_view(tu.to_reference(src_plain))
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *args)
    res = flag_gems.slice_inverse(base, src, *args)

    _assert_view(res, ref, src, base, operands)


# Region rows are (dim, start, end, step); ``None`` omits every slice argument,
# i.e. the schema defaults.  The rows cover positive/negative/zero and boundary
# values of each integer parameter: dim 0/1/-1, start -3/0/1, end -1/0/2/3,
# step 1/2 and a degenerate empty region.
REGION_ROWS = [
    None,
    (0, None, None, 1),
    (0, 0, 2, 1),
    (0, 1, 3, 1),
    (0, -3, -1, 1),
    (0, 0, None, 2),
    (0, 0, 0, 1),
    (1, 0, 2, 2),
    (-1, 0, 2, 1),
]
REGION_DTYPES = [torch.bfloat16, torch.int32]


def _region_cases(shapes):
    cases = []
    for shape in shapes:
        rank = len(shape)
        for row in REGION_ROWS:
            # A row naming another dimension does not apply to this rank.  A
            # negative start such as -3 stays valid for dim0 < 3 because Python
            # clamps it into the slice bounds, giving an empty region.
            if row is not None and (rank == 0 or not -rank <= row[0] < rank):
                continue
            cases.extend((shape, row, dtype) for dtype in REGION_DTYPES)
    return cases


REGION_CASES = tu.selected_cases(
    _region_cases(tu.REQUIRED_SHAPES),
    quick=_region_cases(tu.QUICK_SHAPES),
)


@pytest.mark.parametrize("shape,region,dtype", REGION_CASES)
def test_slice_inverse_region(shape, region, dtype):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    src_plain = tu.make_input(dtype, shape, SRC_RANGE)
    src, args = _region_inputs(src_plain, region)
    operands = _snapshot(base, src)

    ref_base = tu.to_reference(base)
    ref_src, _ = _region_inputs(tu.to_reference(src_plain), region)
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *args)
    res = flag_gems.slice_inverse(base, src, *args)

    _assert_view(res, ref, src, base, operands)


# The forward pass never reads dim/start/end/step, so the metadata must still be
# src's for arguments that do not describe src: an out-of-range dim, a zero step
# and a reversed start/end.
IGNORED_ARG_ROWS = [(5, 0, 1, 1), (0, 0, 1, 0), (-1, 3, 1, 1)]
IGNORED_ARG_SHAPES = [(4, 6), (20, 320, 15)]
IGNORED_ARG_CASES = tu.selected_cases(
    [(shape, row) for shape in IGNORED_ARG_SHAPES for row in IGNORED_ARG_ROWS],
    quick=[(shape, row) for shape in IGNORED_ARG_SHAPES for row in IGNORED_ARG_ROWS],
)


@pytest.mark.parametrize("shape,params", IGNORED_ARG_CASES)
def test_slice_inverse_ignores_slice_args(shape, params):
    base = tu.make_input(torch.float32, shape, ["-1", "1"])
    src_plain = tu.make_input(torch.float32, shape, SRC_RANGE)
    src = _half_view(src_plain)
    operands = _snapshot(base, src)

    ref_base = tu.to_reference(base)
    ref_src = _half_view(tu.to_reference(src_plain))
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *params)
    res = flag_gems.slice_inverse(base, src, *params)

    _assert_view(res, ref, src, base, operands)


# Non-contiguous, offset and transposed inputs: the result must reuse self's
# storage and reproduce src's sizes, strides and storage offset exactly.
LAYOUTS = {
    "contiguous": lambda tensor: tensor,
    "dim1_step2": lambda tensor: tensor[:, ::2],
    "offset": lambda tensor: tensor[1:],
    "offset_step2": lambda tensor: tensor[1:, ::2],
    "transposed": lambda tensor: tensor.transpose(0, 1),
}


@pytest.mark.parametrize("layout", LAYOUTS.values(), ids=list(LAYOUTS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slice_inverse_layout(layout, dtype):
    plain = tu.make_input(dtype, (6, 8), ["-1", "1"])
    src_plain = tu.make_input(dtype, (6, 8), SRC_RANGE)
    base = layout(plain)
    src = layout(src_plain)[:2]
    args = (0, 0, 2, 1)
    operands = _snapshot(base, src)

    ref_base = layout(tu.to_reference(plain))
    ref_src = layout(tu.to_reference(src_plain))[:2]
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *args)
    res = flag_gems.slice_inverse(base, src, *args)

    _assert_view(res, ref, src, base, operands)


# Scalar, size-1 and empty extents also run in quick mode.
DEGENERATE_ROWS = [
    (torch.float32, ()),
    (torch.float32, (1,)),
    (torch.int32, (1, 1)),
    (torch.float32, (0, 4)),
    (torch.float32, (4, 0)),
    (torch.uint8, (1, 5)),
]
DEGENERATE_CASES = DEGENERATE_ROWS


@pytest.mark.parametrize("dtype,shape", DEGENERATE_CASES)
def test_slice_inverse_degenerate(dtype, shape):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    src_plain = tu.make_input(dtype, shape, SRC_RANGE)
    src = _half_view(src_plain)
    args = () if not shape else (0, 0, _half(shape[0]), 1)
    operands = _snapshot(base, src)

    ref_base = tu.to_reference(base)
    ref_src = _half_view(tu.to_reference(src_plain))
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *args)
    res = flag_gems.slice_inverse(base, src, *args)

    _assert_view(res, ref, src, base, operands)


SPECIAL_DTYPES = [dtype for dtype in DTYPES if dtype.is_floating_point]
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[])


@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_slice_inverse_special_values(dtype, scenario):
    base = tu.make_special_input(dtype, scenario)
    src_plain = tu.make_input(dtype, tuple(base.shape), ["0", "1"])
    src = _half_view(src_plain)
    args = () if base.dim() == 0 else (0, 0, src.shape[0], 1)
    operands = _snapshot(base, src)

    ref_base = tu.to_reference(base)
    ref_src = _half_view(tu.to_reference(src_plain))
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *args)
    res = flag_gems.slice_inverse(base, src, *args)

    _assert_view(res, ref, src, base, operands)


def test_slice_inverse_reads_self_and_writes_through():
    base = torch.full((4, 6), 1.0, device=flag_gems.device)
    src = torch.full((2, 6), 7.0, device=flag_gems.device)
    args = (0, 0, 2, 1)
    operands = _snapshot(base, src)

    ref_base = tu.to_reference(base)
    ref_src = tu.to_reference(src)
    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, *args)
    res = flag_gems.slice_inverse(base, src, *args)

    _assert_view(res, ref, src, base, operands)
    res.fill_(3.0)
    ref.fill_(3.0)
    tu.assert_result_equal(base, ref_base)
    tu.assert_result_equal(src, ref_src)
    assert _snapshot(base, src) == operands


BACKWARD_CASES = tu.selected_cases(
    [
        (dtype, shape)
        for shape in [(4, 6), (5,), (2, 3, 4)]
        for dtype in DTYPES
        if dtype.is_floating_point
    ],
    quick=[],
)


@pytest.mark.parametrize("dtype,shape", BACKWARD_CASES)
def test_slice_inverse_backward(dtype, shape):
    # Native complex autograd is unsupported. Partial regions return gradients
    # with an incompatible shape, so positive backward cases use full regions.
    # The output and full-region gradient are pure views and compare exactly.
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    src = tu.make_input(dtype, shape, SRC_RANGE)
    upstream = tu.make_input(dtype, shape, ["-1", "1"])
    args = (0, 0, shape[0], 1)

    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.slice_inverse(ref_inp, tu.to_reference(src), *args)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )

    out = flag_gems.slice_inverse(inp, src, *args)
    (grad,) = torch.autograd.grad(out, inp, grad_outputs=upstream)

    tu.assert_result_equal(out, ref_out)
    tu.assert_result_equal(grad, ref_grad)


# src's metadata must fit inside self's storage: the native operator raises
# RuntimeError ("setStorage: sizes ..., storage offset ... requiring a storage
# size of ...") when the described region reaches past the end of self.
OUT_OF_STORAGE_CASES = [((4, 6), (10, 6), 5), ((2, 3), (6, 3), 2), ((4,), (8,), 4)]


@pytest.mark.parametrize("self_shape,src_shape,offset", OUT_OF_STORAGE_CASES)
def test_slice_inverse_src_out_of_storage(self_shape, src_shape, offset):
    inp = tu.make_input(torch.float32, self_shape, ["-1", "1"])
    too_large = tu.make_input(torch.float32, src_shape, ["-1", "1"])[offset:]
    with pytest.raises(RuntimeError):
        flag_gems.slice_inverse(inp, too_large)


@pytest.mark.parametrize("bad_src", [[1.0, 2.0], 3.0])
def test_slice_inverse_non_tensor_src(bad_src):
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.slice_inverse(inp, bad_src)


@pytest.mark.parametrize("src_dtype", DTYPES)
def test_slice_inverse_src_dtype_does_not_change_result(src_dtype):
    base = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    src = tu.make_input(src_dtype, (2, 6), ["-1", "1"])
    operands = _snapshot(base, src)
    ref_base, ref_src = tu.to_reference(base), tu.to_reference(src)

    ref = torch.ops.aten.slice_inverse(ref_base, ref_src, 0, 0, 2, 1)
    res = flag_gems.slice_inverse(base, src, 0, 0, 2, 1)

    _assert_view(res, ref, src, base, operands)

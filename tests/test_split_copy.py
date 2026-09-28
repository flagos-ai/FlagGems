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
from . import test_utils as tu

# aten::split_copy.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
# aten::split_copy.Tensor_out(Tensor self, SymInt split_size, int dim=0, *,
#                             Tensor(a!)[] out) -> ()
# split_copy is a packet without a `.default` overload: the reference is reached
# through the packet call form and, for the out variant, through `.Tensor_out`.
# The candidate is always the single public `flag_gems.split_copy` name.

# The native kernel requires rank >= 1 (0-dim inputs raise), so the spec's scalar
# shape is covered by the dedicated rejection test instead of the shape grid.
_SPLIT_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 1]

# Required dtypes minus the ones the backend does not report, plus the extra
# types this operator supports. bf16/fp8/int64/fp64 kernels are gated on the
# existing static capability flags; int8/uint8 have no such flag.
_SPLIT_DTYPES = (
    [
        dtype
        for dtype in tu.REQUIRED_DTYPES
        if not (
            (dtype == torch.bfloat16 and not utils.bf16_is_supported)
            or (
                dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                and not utils.fp8_is_supported
            )
            or (dtype == torch.int64 and not utils.int64_is_supported)
        )
    ]
    + [torch.int16, torch.complex64]
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
)

# The shared special-value generator only builds real nan/inf payloads - it
# returns no rows for complex dtypes - so those stay in the value-range grid.
_SPECIAL_DTYPES = [
    dtype
    for dtype in _SPLIT_DTYPES
    if dtype
    in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
]

# (shape, split_size, dim) rows: remainder, exact division, singletons, a
# split_size equal to and larger than the axis length, the trailing axis and the
# lower dim boundary for rank 3 and rank 4. Explicit parameter workload: the
# plain grid above already covers the quick-mode parameter choice.
_PARAM_CASES = tu.selected_cases(
    [
        ((20, 320, 15), 4, 0),
        ((20, 320, 15), 3, -1),
        ((20, 320, 15), 5, 0),
        ((20, 320, 15), 1, 0),
        ((20, 320, 15), 7, -3),
        ((256,), 7, 0),
        ((1024, 1024), 1024, 0),
        ((1024, 1024), 2048, 0),
        ((1024, 1024), 512, 1),
        ((16, 128, 64, 60), 5, 1),
        ((16, 128, 64, 60), 3, -4),
    ],
    quick=[],
)

# Empty results, including a zero-sized dimension that is not the split axis, and
# a zero split_size on an empty split axis (the only native-valid split_size 0).
_EMPTY_CASES = tu.selected_cases(
    [((0,), 3, 0), ((0, 4), 3, 1), ((3, 0, 5), 2, 0), ((0, 0), 1, 0), ((0, 4), 0, 0)],
    quick=[],
)

# Out-buffer workloads. The plain quick-mode smoke row stays so the out overload
# is exercised in both modes.
_OUT_CASES = tu.selected_cases(
    [((4, 6, 8), 2, 0), ((4, 6, 8), 3, 0), ((4, 6, 8), 2, 2), ((4, 6, 8), 4, -1)],
    quick=[((2, 19, 7), 2, 0)],
)

# Out buffers that are offset views of a larger sentinel-filled parent.
_OUT_OFFSET_CASES = tu.selected_cases(
    [((6, 6, 8), 2, 0), ((6, 6, 8), 5, 1)],
    quick=[],
)

# Out buffers whose trailing axis is the ::2 view of a padded parent.
_OUT_STRIDED_CASES = tu.selected_cases([((4, 6, 8), 2, -1)], quick=[])

# Split sizes that produce a remainder on both the leading and the trailing axis.
_NON_CONTIGUOUS_CASES = tu.selected_cases([(3, 0), (4, 1), (3, 2)], quick=[])

# Zero-stride input: the base row is retained to prove nothing wrote through it.
_EXPANDED_CASES = tu.selected_cases([((6, 4), 4, 0)], quick=[])

_BACKWARD_CASES = tu.selected_cases(
    [((6, 8), 3, 0), ((6, 8), 3, 1), ((5, 9, 4), 4, -1), ((7, 3), 2, -2)],
    quick=[],
)

# Independence rows: (parent_shape, view, split_size, dim), where view None means
# the parent itself. The view row feeds an offset + strided input and the last
# row has zero-size parts.
_FRESH_CASES = tu.selected_cases(
    [
        ((7, 4), None, 3, 0),
        ((8, 6, 16), (slice(1, None), slice(None), slice(None, None, 2)), 3, 1),
        ((0, 4), None, 3, 1),
    ],
    quick=[],
)


def _main_split(shape):
    """Remainder-producing split of the leading axis for the grid tests."""
    return max(1, shape[0] // 3), 0


def _assert_parts(res_parts, ref_parts, device):
    # The contract is a Python sequence of distinct tensors; a single stacked
    # tensor is not an acceptable split result. The part device is checked
    # against the input because the shared comparison moves the oracle to the
    # reference device and therefore cannot observe a wrong-device result.
    assert not torch.is_tensor(res_parts), type(res_parts)
    assert isinstance(res_parts, (list, tuple)), type(res_parts)
    assert len(res_parts) == len(ref_parts)
    for res, ref in zip(res_parts, ref_parts):
        assert res.device == device
        tu.assert_result_equal(res, ref)


def _padded_views(parts, device):
    """Out buffers that are trailing views of a larger sentinel-filled parent."""
    parents = [
        torch.full(
            (part.shape[0] + 1,) + tuple(part.shape[1:]),
            7.0,
            dtype=part.dtype,
            device=device,
        )
        for part in parts
    ]
    return parents, [parent[1:] for parent in parents]


def _strided_views(parts, device):
    """Out buffers whose trailing axis is the ::2 view of a padded parent."""
    parents = [
        torch.full(
            tuple(part.shape[:-1]) + (part.shape[-1] * 2,),
            7.0,
            dtype=part.dtype,
            device=device,
        )
        for part in parts
    ]
    return parents, [parent[..., ::2] for parent in parents]


@pytest.mark.split_copy
@pytest.mark.parametrize("shape", _SPLIT_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _SPLIT_DTYPES)
def test_split_copy_value_ranges(shape, value_range, dtype):
    split_size, dim = _main_split(shape)
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    # The snapshot was taken before the call, so this also proves the input is
    # left untouched.
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape", _SPLIT_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_split_copy_bool(shape, value_range):
    split_size, dim = _main_split(shape)
    inp = tu.make_input(torch.bool, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _PARAM_CASES)
def test_split_copy_split_size_and_dim(shape, split_size, dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize(
    "shape",
    tu.selected_cases(
        [(1024, 1024), (20, 320, 15), (16, 128, 64, 60), (16, 7, 57, 32, 29)],
        quick=[],
    ),
)
def test_split_copy_default_dim(shape):
    # `dim` has a schema default of 0; omitting it checks that the candidate
    # implements the default rather than only an explicitly passed argument.
    split_size = max(1, shape[0] // 4)
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size)
    res_parts = flag_gems.split_copy(inp, split_size)

    _assert_parts(res_parts, ref_parts, inp.device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _EMPTY_CASES)
def test_split_copy_empty_tensors(shape, split_size, dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("split_size,dim", _NON_CONTIGUOUS_CASES)
def test_split_copy_non_contiguous(split_size, dim):
    # Offset storage plus a strided last axis: the copies must come from the
    # viewed values, not from a contiguous re-reading of the parent buffer.
    parent = tu.make_input(torch.float32, (8, 6, 16), ["-1", "1"])
    inp = parent[1:, :, ::2]
    ref_parent = tu.to_reference(parent)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    # The whole parent is compared, padding cells included, so a write anywhere
    # in the input storage is detected.
    tu.assert_result_equal(parent, ref_parent)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _EXPANDED_CASES)
def test_split_copy_expanded_input(shape, split_size, dim):
    # Zero-stride input: every row aliases the only row of the base storage.
    source = tu.make_input(torch.float32, (1, shape[1]), ["-1", "1"])
    inp = source.expand(*shape)
    ref_source = tu.to_reference(source)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    tu.assert_result_equal(source, ref_source)


@pytest.mark.split_copy
@pytest.mark.parametrize("parent_shape,view,split_size,dim", _FRESH_CASES)
def test_split_copy_parts_are_fresh_copies(parent_shape, view, split_size, dim):
    parent = tu.make_input(torch.float32, parent_shape, ["-1", "1"])
    inp = parent if view is None else parent[view]
    ref_parent = tu.to_reference(parent)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    # Values are compared before anything is written, so a candidate that only
    # produces correct parts on a later call cannot pass by being called again.
    _assert_parts(res_parts, ref_parts, inp.device)

    for index, part in enumerate(res_parts):
        # Writing through one part must not reach the input, the parent storage
        # or any other part: every part is an independent copy. The expectation
        # is built on the reference device, never on the candidate's device.
        part.fill_(-1.0 - index)
        tu.assert_result_equal(
            part,
            torch.full(
                ref_parts[index].shape,
                -1.0 - index,
                dtype=ref_parts[index].dtype,
                device=ref_parts[index].device,
            ),
        )
        tu.assert_result_equal(inp, ref_inp)
        if view is not None:
            tu.assert_result_equal(parent, ref_parent)
        for other_index, other in enumerate(res_parts):
            if other_index == index:
                continue
            tu.assert_result_equal(other, ref_parts[other_index])
        # Restore the part so later iterations still compare pristine values.
        part.copy_(ref_parts[index])


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _OUT_CASES)
def test_split_copy_out(shape, split_size, dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    # The buffers start at a sentinel that only a real in-place write removes, so
    # an unwritten buffer fails the comparison below.
    ref_out = [torch.full_like(part, 7.0) for part in ref_parts]
    out = [
        torch.full(part.shape, 7.0, dtype=inp.dtype, device=inp.device)
        for part in ref_parts
    ]
    given = list(out)

    torch.ops.aten.split_copy.Tensor_out(ref_inp, split_size, dim, out=ref_out)
    res_ret = flag_gems.split_copy(inp, split_size, dim, out=out)

    # The native out overload writes into the caller's buffers, returns None and
    # leaves the caller's list entries in place.
    assert res_ret is None
    assert len(out) == len(given)
    assert all(entry is original for entry, original in zip(out, given))
    _assert_parts(out, ref_out, inp.device)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _OUT_OFFSET_CASES)
def test_split_copy_out_offset_buffers(shape, split_size, dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)

    ref_parents, ref_out = _padded_views(ref_parts, ref_inp.device)
    parents, out = _padded_views(ref_parts, inp.device)
    given = list(out)

    torch.ops.aten.split_copy.Tensor_out(ref_inp, split_size, dim, out=ref_out)
    res_ret = flag_gems.split_copy(inp, split_size, dim, out=out)

    assert res_ret is None
    assert len(out) == len(given)
    assert all(entry is original for entry, original in zip(out, given))
    for part, ref_part, parent, ref_parent in zip(out, ref_out, parents, ref_parents):
        assert part.device == inp.device
        # The buffer must still be the same view of the storage the caller
        # provided: the whole parent, sentinel cells included, is compared below.
        assert part.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
        tu.assert_result_equal(part, ref_part)
        tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _OUT_STRIDED_CASES)
def test_split_copy_out_strided_buffers(shape, split_size, dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)

    ref_parents, ref_out = _strided_views(ref_parts, ref_inp.device)
    parents, out = _strided_views(ref_parts, inp.device)
    given = list(out)

    torch.ops.aten.split_copy.Tensor_out(ref_inp, split_size, dim, out=ref_out)
    res_ret = flag_gems.split_copy(inp, split_size, dim, out=out)

    assert res_ret is None
    assert len(out) == len(given)
    assert all(entry is original for entry, original in zip(out, given))
    for part, ref_part, parent, ref_parent in zip(out, ref_out, parents, ref_parents):
        assert part.device == inp.device
        assert part.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
        tu.assert_result_equal(part, ref_part)
        # Every cell is compared, so the skipped padding cells still hold the
        # sentinel while the viewed cells hold the copied values.
        tu.assert_result_equal(parent, ref_parent)
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.split_copy
@pytest.mark.parametrize("shape,split_size,dim", _BACKWARD_CASES)
def test_split_copy_backward(shape, split_size, dim):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)
    _assert_parts(res_parts, ref_parts, inp.device)

    # Distinct per-part upstream gradients: a constant upstream, or a permuted
    # scatter, would be invisible in the summed input gradient.
    upstream = [
        (
            torch.arange(part.numel(), dtype=torch.float32, device=inp.device).reshape(
                part.shape
            )
            + 1.0
        )
        * float(index + 1)
        for index, part in enumerate(res_parts)
    ]
    res_grad = torch.autograd.grad(list(res_parts), inp, grad_outputs=upstream)[0]
    ref_grad = torch.autograd.grad(
        list(ref_parts), ref_inp, grad_outputs=[tu.to_reference(up) for up in upstream]
    )[0]

    assert res_grad.device == inp.device
    tu.assert_result_equal(res_grad, ref_grad)
    tu.assert_result_equal(inp.detach(), ref_inp.detach())


@pytest.mark.split_copy
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[]),
)
def test_split_copy_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    split_size, dim = _main_split(inp.shape)
    ref_inp = tu.to_reference(inp)

    ref_parts = torch.ops.aten.split_copy(ref_inp, split_size, dim)
    res_parts = flag_gems.split_copy(inp, split_size, dim)

    _assert_parts(res_parts, ref_parts, inp.device)
    tu.assert_result_equal(inp, ref_inp)


# Native error contract on the active backend: split_size < 0 -> RuntimeError
# ("split expects split_size be non-negative"), split_size = 0 on a non-empty
# axis -> RuntimeError ("split_size can only be 0 if dimension size is 0"), and
# a dim outside [-rank, rank-1] -> IndexError ("Dimension out of range").
@pytest.mark.split_copy
@pytest.mark.parametrize(
    "split_size,dim,expected",
    [
        (-1, 0, RuntimeError),
        (0, 0, RuntimeError),
        (0, 2, RuntimeError),
        (1, 3, IndexError),
        (1, -4, IndexError),
    ],
)
def test_split_copy_rejects_invalid_params(split_size, dim, expected):
    inp = tu.make_input(torch.float32, (4, 6, 8), ["-1", "1"])

    with pytest.raises(expected):
        flag_gems.split_copy(inp, split_size, dim)


@pytest.mark.split_copy
def test_split_copy_rejects_0dim():
    # Native rejects rank 0 with "split expects at least a 1-dimensional tensor".
    inp = torch.tensor(3.0, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.split_copy(inp, 1, 0)


@pytest.mark.split_copy
@pytest.mark.parametrize("args", [(2.5, 0), (2, 2.5)])
def test_split_copy_rejects_non_integer_params(args):
    # A non-integer split_size/dim matches no native schema (RuntimeError); a
    # Python implementation reports the same mistake as a TypeError.
    inp = tu.make_input(torch.float32, (4, 6, 8), ["-1", "1"])

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.split_copy(inp, *args)


@pytest.mark.split_copy
def test_split_copy_rejects_non_tensor_input():
    # A non-tensor `self` matches no native schema ("failed to match any schema").
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.split_copy([1.0, 2.0, 3.0], 2, 0)


@pytest.mark.split_copy
@pytest.mark.parametrize("out_shape,count", [((2, 6, 8), 1), ((3, 6, 8), 2)])
def test_split_copy_rejects_invalid_out(out_shape, count):
    # Native rejects a buffer count that does not match the number of parts and
    # a buffer geometry that cannot hold its part. A dtype-mismatched buffer is
    # not an error natively, so it is deliberately not asserted here.
    inp = tu.make_input(torch.float32, (4, 6, 8), ["-1", "1"])
    out = [
        torch.empty(*out_shape, dtype=inp.dtype, device=inp.device)
        for _ in range(count)
    ]

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.split_copy(inp, 2, 0, out=out)

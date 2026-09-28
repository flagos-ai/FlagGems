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

# aten::_reshape_from_tensor(Tensor self, Tensor shape) reshapes `self` to the extents
# listed in the 1-D int64 metadata tensor `shape`.  Native aliases the operand's
# storage when the requested target is stride-expressible from the operand's strides
# and otherwise materializes a fresh contiguous tensor, so the cases below compare the
# values as well as the resulting alias and mutation relation.

_MUTATION_SENTINEL = -3.5

# tu.make_input resolves each range bound through test_utils' symbol table, so the
# range has to be given in the same symbolic form tu.selected_ranges() returns.
_VALUE_RANGE = ["-1", "1"]


def _shape_tensor(*dims):
    # The extents are metadata: native requires a 1-D int64 tensor and reads the
    # entries as host values, so it is created on the CPU.
    return torch.tensor(list(dims), dtype=torch.long)


def _dtype_is_supported(dtype):
    # Static capability flags published by the shared helpers; no native probe runs
    # while this file is collected.
    if dtype is torch.bfloat16:
        return utils.bf16_is_supported
    if dtype is torch.int64:
        return utils.int64_is_supported
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return utils.fp8_is_supported
    if dtype is torch.float64:
        return utils.fp64_is_supported
    return True


_VALUE_DTYPES = [dtype for dtype in tu.REQUIRED_DTYPES if _dtype_is_supported(dtype)]
_VALUE_DTYPES += [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _VALUE_DTYPES.append(torch.float64)

_LAYOUT_DTYPES = [
    dtype
    for dtype in (torch.float32, torch.bfloat16, torch.int64)
    if _dtype_is_supported(dtype)
]

_BACKWARD_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _dtype_is_supported(dtype)
]
if utils.fp64_is_supported:
    _BACKWARD_DTYPES.append(torch.float64)

_SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _dtype_is_supported(dtype)
]
if utils.fp64_is_supported:
    _SPECIAL_DTYPES.append(torch.float64)


def _observed(tensor):
    # The stored values behind a lazy conjugate or negative view.
    return tensor.detach().resolve_conj().resolve_neg()


def _shares_storage(result, operand):
    if result.numel() == 0 or operand.numel() == 0:
        return None
    return result.untyped_storage().data_ptr() == operand.untyped_storage().data_ptr()


def _assert_alias_semantics(result, operand, reference, reference_operand):
    # The view/copy decision depends on stride compatibility, so the candidate has to
    # reproduce the native extents, strides and storage offset; storage identity is
    # only observable when there is an element to alias.
    assert tuple(result.shape) == tuple(reference.shape)
    assert result.stride() == reference.stride()
    assert result.storage_offset() == reference.storage_offset()
    reference_alias = _shares_storage(reference, reference_operand)
    if reference_alias is not None:
        assert _shares_storage(result, operand) == reference_alias


def _write_first_element(tensor):
    # Indexed assignment is valid for the strided results that sliced operands
    # produce, where view(-1) would raise.
    tensor[0] = _MUTATION_SENTINEL


def _assert_mutation_matches(result, operand, reference, reference_operand):
    # A candidate that clones its input, or aliases where native copies, must not slip
    # through: write through both results and require the same observable operand
    # change.
    tu.assert_result_equal(_observed(result), _observed(reference))

    observed_operand = _observed(operand).clone()
    observed_reference_operand = _observed(reference_operand).clone()

    _write_first_element(result)
    result_changed = not torch.equal(_observed(operand), observed_operand)
    _write_first_element(reference)
    reference_changed = not torch.equal(
        _observed(reference_operand), observed_reference_operand
    )

    assert result_changed == reference_changed
    tu.assert_result_equal(_observed(operand), _observed(reference_operand))


def _offset_flatten(operand):
    return operand[1:]


def _offset_columns(operand):
    return operand[1:, 2:8]


def _even_columns(operand):
    return operand[:, ::2]


def _every_other_row(operand):
    return operand[::2]


def _transpose(operand):
    return operand.t()


def _conjugate_view(operand):
    return operand.conj()


def _negated_view(operand):
    return torch._neg_view(operand)


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _VALUE_DTYPES)
def test__reshape_from_tensor_value_range(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    shape_tensor = _shape_tensor(-1)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)

    # The exact shared result comparison already pins the extent and element count.
    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_semantics(res_out, inp, ref_out, ref_inp)
    tu.assert_result_equal(inp, ref_inp)


_TARGET_CASES = [
    ("infer_all", (4, 6), (-1,)),
    ("merge_dims", (2, 3, 4), (6, 4)),
    ("split_dims", (24,), (2, 3, 4)),
    ("infer_leading_dim", (2, 12), (-1, 12)),
    ("infer_middle_dim", (2, 12), (2, -1, 3)),
    ("add_leading_dim", (24,), (1, 24)),
    ("scalar_target", (1,), ()),
    ("zero_dim_input", (), (1, 1)),
    ("empty_input", (0,), (3, 0)),
]

_TARGET_ROWS = tu.selected_cases(_TARGET_CASES, quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("case_name,shape,dims", _TARGET_ROWS)
def test__reshape_from_tensor_target_shapes(case_name, shape, dims):
    inp = tu.make_input(torch.float32, shape, _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)

    shape_tensor = _shape_tensor(*dims)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_semantics(res_out, inp, ref_out, ref_inp)
    tu.assert_result_equal(inp, ref_inp)


_LAYOUT_CASES = [
    ("offset_flatten", (4, 12), _offset_flatten, (36,)),
    ("offset_strided_copy", (4, 12), _offset_columns, (18,)),
    ("strided_view", (4, 12), _even_columns, (2, 12)),
    ("transposed_copy", (4, 6), _transpose, (24,)),
]

_LAYOUT_ROWS = tu.selected_cases(_LAYOUT_CASES, quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
@pytest.mark.parametrize("case_name,base_shape,transform,dims", _LAYOUT_ROWS)
def test__reshape_from_tensor_layouts(case_name, base_shape, transform, dims, dtype):
    base = tu.make_input(dtype, base_shape, _VALUE_RANGE)
    inp = base if transform is None else transform(base)
    ref_inp = tu.to_reference(inp)

    shape_tensor = _shape_tensor(*dims)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_semantics(res_out, inp, ref_out, ref_inp)
    tu.assert_result_equal(inp, ref_inp)


# Two distinct view paths: `(4, 12)[:, ::2]` keeps every other column and its
# stride-(12, 2) result is still expressible as a (2, 12) target, while
# `(4, 6)[:, ::2]` yields strides (6, 2) that flatten to stride 2.
_ALIAS_CASES = [
    ("contiguous_flatten", (4, 6), None, (24,)),
    ("merged_dims", (2, 3, 4), None, (6, 4)),
    ("transposed_copy", (4, 6), _transpose, (24,)),
    ("offset_flatten", (4, 12), _offset_flatten, (36,)),
    ("offset_strided_copy", (4, 12), _offset_columns, (18,)),
    ("column_strided_view", (4, 12), _even_columns, (2, 12)),
    ("column_strided_flatten", (4, 6), _even_columns, (12,)),
    ("singleton_strided_flatten", (2, 4), _every_other_row, (4,)),
]

_ALIAS_ROWS = tu.selected_cases(_ALIAS_CASES, quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("case_name,base_shape,transform,dims", _ALIAS_ROWS)
def test__reshape_from_tensor_alias_effects(case_name, base_shape, transform, dims):
    # float32 keeps the mutation sentinel representable so the write-through check is
    # meaningful.
    base = tu.make_input(torch.float32, base_shape, _VALUE_RANGE)
    inp = base if transform is None else transform(base)
    ref_inp = tu.to_reference(inp)

    shape_tensor = _shape_tensor(*dims)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)

    _assert_mutation_matches(res_out, inp, ref_out, ref_inp)


_LAZY_CASES = [
    ("lazy_conj_view", (4, 6), torch.complex64, _conjugate_view),
    ("lazy_neg_view", (4, 6), torch.float32, _negated_view),
]

_LAZY_ROWS = tu.selected_cases(_LAZY_CASES, quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("case_name,base_shape,dtype,transform", _LAZY_ROWS)
def test__reshape_from_tensor_lazy_bit_operands(
    case_name, base_shape, dtype, transform
):
    base = tu.make_input(dtype, base_shape, _VALUE_RANGE)
    inp = transform(base)
    ref_inp = tu.to_reference(inp)

    shape_tensor = _shape_tensor(-1)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)

    # Values are compared with the lazy bits resolved, but the alias/mutation relation
    # is compared without resolving them: materializing into independent storage is not
    # equivalent to the native result.
    tu.assert_result_equal(_observed(res_out), _observed(ref_out))
    _assert_alias_semantics(res_out, inp, ref_out, ref_inp)
    _assert_mutation_matches(res_out, inp, ref_out, ref_inp)


_KEYWORD_CASES = [
    ("keyword_flatten", (2, 3), (-1,)),
    ("keyword_merge", (2, 3), (3, 2)),
]

_KEYWORD_ROWS = tu.selected_cases(_KEYWORD_CASES, quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("case_name,shape,dims", _KEYWORD_ROWS)
def test__reshape_from_tensor_keyword_shape(case_name, shape, dims):
    # Native also accepts the extents as the keyword argument `shape`.
    inp = tu.make_input(torch.float32, shape, _VALUE_RANGE)
    ref_inp = tu.to_reference(inp)

    shape_tensor = _shape_tensor(*dims)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape=shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape=shape_tensor)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_semantics(res_out, inp, ref_out, ref_inp)
    tu.assert_result_equal(inp, ref_inp)


def _upstream_gradient(reference):
    # Distinct per-element values, so a permuted or transposed gradient cannot pass.
    # The upstream gradient is an ordinary tensor operand (unlike the host shape
    # metadata), so it is allocated on the execution device and only cast to the
    # tested dtype there.
    values = torch.linspace(
        0.5,
        1.5,
        reference.numel(),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    return values.reshape(reference.shape).to(dtype=reference.dtype)


_BACKWARD_CASES = [
    ("contiguous_view", (4, 6), None, (24,)),
    ("merged_dims", (2, 3, 4), None, (6, 4)),
    ("offset_strided_copy", (4, 12), _offset_columns, (18,)),
    ("column_strided_view", (4, 6), _even_columns, (12,)),
]

_BACKWARD_ROWS = tu.selected_cases(_BACKWARD_CASES, quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("dtype", _BACKWARD_DTYPES)
@pytest.mark.parametrize("case_name,base_shape,transform,dims", _BACKWARD_ROWS)
def test__reshape_from_tensor_backward(case_name, base_shape, transform, dims, dtype):
    base = tu.make_input(dtype, base_shape, _VALUE_RANGE).requires_grad_()
    inp = base if transform is None else transform(base)
    ref_base = tu.to_reference(base).detach().requires_grad_()
    ref_inp = ref_base if transform is None else transform(ref_base)

    shape_tensor = _shape_tensor(*dims)
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)
    tu.assert_result_equal(res_out.detach(), ref_out.detach())

    upstream = _upstream_gradient(ref_out)
    ref_upstream = tu.to_reference(upstream)
    ref_grad = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_upstream)[0]
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    tu.assert_result_equal(res_grad.detach(), ref_grad.detach())


_SPECIAL_ROWS = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test__reshape_from_tensor_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    # A leading singleton target keeps every element addressable for any input shape.
    shape_tensor = _shape_tensor(1, inp.numel())
    ref_out = torch.ops.aten._reshape_from_tensor(ref_inp, shape_tensor)
    res_out = flag_gems._reshape_from_tensor(inp, shape_tensor)

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_semantics(res_out, inp, ref_out, ref_inp)
    tu.assert_result_equal(inp, ref_inp)


_NEGATIVE_DTYPE_CASES = [torch.int32, torch.int16, torch.float32, torch.bool]


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("bad_dtype", _NEGATIVE_DTYPE_CASES)
def test__reshape_from_tensor_rejects_shape_dtype(bad_dtype):
    # Native reads the metadata as int64: any other dtype is rejected before the
    # reshape runs.
    inp = tu.make_input(torch.float32, (4, 6), _VALUE_RANGE)
    bad_shape = torch.tensor([24], dtype=bad_dtype)

    with pytest.raises(RuntimeError):
        flag_gems._reshape_from_tensor(inp, bad_shape)


_NEGATIVE_DIMS_CASES = [
    ("rank_zero", 6),
    ("rank_two", [[6]]),
    ("numel_mismatch", [2, 5]),
    ("two_inferred", [-1, -1]),
    ("dim_below_minus_one", [-2, 12]),
    ("empty_on_nonempty", []),
]


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize("case_name,bad_dims", _NEGATIVE_DIMS_CASES)
def test__reshape_from_tensor_rejects_invalid_shape(case_name, bad_dims):
    # The rows are handed straight to torch.tensor, so the bare integer builds rank-0
    # metadata and the nested list builds rank-2 metadata.
    inp = tu.make_input(torch.float32, (4, 6), _VALUE_RANGE)
    bad_shape = torch.tensor(bad_dims, dtype=torch.long)

    with pytest.raises(RuntimeError):
        flag_gems._reshape_from_tensor(inp, bad_shape)


_NEGATIVE_METADATA_CASES = [
    ("omitted", (), {}),
    ("none", (None,), {}),
    ("list", ([24],), {}),
    ("int", (24,), {}),
    ("tuple", ((24,),), {}),
]


@pytest.mark.reshape_from_tensor
@pytest.mark.parametrize(
    "args,kwargs",
    [row[1:] for row in _NEGATIVE_METADATA_CASES],
    ids=[row[0] for row in _NEGATIVE_METADATA_CASES],
)
def test__reshape_from_tensor_rejects_non_tensor_metadata(args, kwargs):
    # Native needs the metadata argument: omitting it, passing None, or passing a
    # Python list/int/tuple each raise their own RuntimeError.
    inp = tu.make_input(torch.float32, (4, 6), _VALUE_RANGE)

    with pytest.raises(RuntimeError):
        flag_gems._reshape_from_tensor(inp, *args, **kwargs)

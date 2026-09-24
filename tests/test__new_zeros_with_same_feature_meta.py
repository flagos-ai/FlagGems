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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# Register underscore-prefixed pytest markers explicitly.
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta_out",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta_out", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# Allocate zeros with shape self.shape[:N] + other.shape and other's dtype/device.
# Cases are (self_shape, other_shape, N), where N is self_num_batch_dims.
_NEW_ZEROS_WITH_SAME_FEATURE_META_CASES = [
    pytest.param((2, 3, 4, 5), (7, 8, 9), 0, id="N0"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 1, id="N1"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 3, id="N3"),
    pytest.param((2, 3, 4, 5), (7, 8, 9), 4, id="N_self_rank"),
    pytest.param((2, 3, 4), (7, 8, 9), 3, id="self_3d_full"),
    pytest.param((2, 3), (4, 5, 6), 2, id="self_2d_other_3d"),
    pytest.param((3,), (4, 5), 1, id="self_1d"),
    pytest.param((3,), (4, 5), 0, id="self_1d_N0"),
    pytest.param((), (4, 5), 0, id="self_0d"),
    pytest.param((), (), 0, id="both_0d"),
    pytest.param((2, 3, 4), (5,), 3, id="other_1d"),
    pytest.param((2, 3), (), 2, id="other_0d"),
    pytest.param((0, 3), (4, 5), 1, id="self_zero_dim"),
    pytest.param((2, 3), (0, 5), 1, id="other_zero_dim"),
]

_NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + tu.selected_cases([torch.int16])
    + [torch.bool]
    + utils.COMPLEX_DTYPES
)

_NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES = [
    pytest.param(torch.int16, torch.float16, id="self_int16_other_f16"),
    pytest.param(torch.float32, torch.bool, id="self_f32_other_bool"),
    pytest.param(torch.bool, torch.int32, id="self_bool_other_int32"),
    pytest.param(torch.int8, torch.float8_e4m3fn, id="self_int8_other_fp8"),
]

_VALUE_RANGE_CASES = [
    pytest.param((2, 3, 4), (5, 6), 1, id="self_3d_other_2d"),
    pytest.param((3,), (4, 5), 0, id="self_1d_N0"),
]

_MAIN_RANGE = ["-1", "1"]

_SHAPE_LEVEL_CASES = []
for shape in tu.selected_shapes():
    _SHAPE_LEVEL_CASES.append((shape, (4, 5), 0))
    if len(shape) >= 1:
        _SHAPE_LEVEL_CASES.append((shape, (4, 5), 1))
    if len(shape) >= 2:
        _SHAPE_LEVEL_CASES.append((shape, (2,), 2))
    _SHAPE_LEVEL_CASES.append(((2,), shape, 1))


def _special_tensor(shape, dtype, scenario):
    numel = math.prod(shape)
    values = tu.make_special_input(dtype, scenario)
    repeats = (numel + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:numel].reshape(shape)


def _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other):
    assert isinstance(res_out, torch.Tensor)
    assert res_out.device == other_t.device
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.requires_grad == ref_out.requires_grad
    tu.assert_result_equal(res_out, ref_out)
    # Default returns fresh storage. Explicit .out may alias an input, in
    # which case its writes and aliasing must agree with ATen.
    for inp, ref_inp in ((self_t, ref_self), (other_t, ref_other)):
        assert torch._C._is_alias_of(res_out, inp) == torch._C._is_alias_of(
            ref_out, ref_inp
        )
        tu.assert_result_equal(inp, ref_inp)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims",
    _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES,
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = tu.make_input(dtype, self_shape, _MAIN_RANGE)
    other_t = tu.make_input(dtype, other_shape, _MAIN_RANGE)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims",
    _NEW_ZEROS_WITH_SAME_FEATURE_META_CASES,
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_out(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = tu.make_input(dtype, self_shape, _MAIN_RANGE)
    other_t = tu.make_input(dtype, other_shape, _MAIN_RANGE)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    # Pre-sized out tensors with non-zero garbage values: the .out variant must
    # overwrite them in place with zeros and return the same object.
    expected_shape = self_shape[:self_num_batch_dims] + other_shape
    res_out = torch.full(expected_shape, 7, dtype=dtype, device=flag_gems.device)
    ref_out = torch.full(expected_shape, 7, dtype=dtype, device=ref_other.device)

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims, out=res_out
    )

    # The .out variant must write into and return the out tensor itself.
    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims", _SHAPE_LEVEL_CASES
)
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_shapes(
    self_shape, other_shape, self_num_batch_dims, dtype
):
    self_t = tu.make_input(dtype, self_shape, _MAIN_RANGE)
    other_t = tu.make_input(dtype, other_shape, _MAIN_RANGE)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize(
    "base_shape,stride,offset",
    [
        pytest.param((5, 4), (1, 4), 0, id="transposed"),
        pytest.param((4, 10), (10, 2), 0, id="strided"),
        pytest.param((5, 5), (5, 1), 5, id="offset"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES, quick=utils.FLOAT_DTYPES
    ),
)
def test__new_zeros_with_same_feature_meta_out_layouts(
    base_shape, stride, offset, dtype
):
    self_t = tu.make_input(dtype, (4, 10), _MAIN_RANGE)[:, ::2]
    other_t = tu.make_input(dtype, (8, 5), _MAIN_RANGE)[::2]
    ref_self, ref_other = tu.to_reference(self_t), tu.to_reference(other_t)
    res_base = torch.full(base_shape, 7, dtype=dtype, device=other_t.device)
    ref_base = torch.full(base_shape, 7, dtype=dtype, device=ref_other.device)
    res_out = res_base.as_strided((4, 5), stride, offset)
    ref_out = ref_base.as_strided((4, 5), stride, offset)

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=0, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=0, out=res_out
    )

    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)
    # The gaps and prefix outside the output view must remain untouched.
    tu.assert_result_equal(res_base, ref_base)


@pytest.mark._new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize("out_shape", [(0,), (1,)])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES, quick=utils.FLOAT_DTYPES
    ),
)
def test__new_zeros_with_same_feature_meta_out_resize(out_shape, dtype):
    self_t = tu.make_input(dtype, (2, 3), _MAIN_RANGE)
    other_t = tu.make_input(dtype, (4, 5), _MAIN_RANGE)
    ref_self, ref_other = tu.to_reference(self_t), tu.to_reference(other_t)
    res_out = torch.full(out_shape, 7, dtype=dtype, device=other_t.device)
    ref_out = torch.full(out_shape, 7, dtype=dtype, device=ref_other.device)

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=1, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1, out=res_out
    )

    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta_out
@pytest.mark.parametrize("alias_self", [True, False])
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES, quick=utils.FLOAT_DTYPES
    ),
)
def test__new_zeros_with_same_feature_meta_out_alias(alias_self, dtype):
    self_t = tu.make_input(dtype, (4, 5), _MAIN_RANGE)
    other_t = tu.make_input(dtype, (4, 5), _MAIN_RANGE)
    ref_self, ref_other = tu.to_reference(self_t), tu.to_reference(other_t)
    res_out = self_t if alias_self else other_t
    ref_out = ref_self if alias_self else ref_other

    torch.ops.aten._new_zeros_with_same_feature_meta.out(
        ref_self, ref_other, self_num_batch_dims=0, out=ref_out
    )
    res_ret = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=0, out=res_out
    )

    assert res_ret is res_out
    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_grad,other_grad", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize(
    "dtype",
    tu.selected_cases(
        [
            dtype
            for dtype in _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES
            if dtype.is_floating_point or dtype.is_complex
        ],
        quick=utils.FLOAT_DTYPES,
    ),
)
def test__new_zeros_with_same_feature_meta_no_autograd(self_grad, other_grad, dtype):
    self_t = tu.make_input(dtype, (2, 3), _MAIN_RANGE).requires_grad_(self_grad)
    other_t = tu.make_input(dtype, (4, 5), _MAIN_RANGE).requires_grad_(other_grad)
    ref_self, ref_other = tu.to_reference(self_t), tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_shape, other_shape, self_num_batch_dims", _VALUE_RANGE_CASES
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_value_ranges(
    self_shape, other_shape, self_num_batch_dims, value_range, dtype
):
    self_t = tu.make_input(dtype, self_shape, value_range)
    other_t = tu.make_input(dtype, other_shape, value_range)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=self_num_batch_dims
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=self_num_batch_dims
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_dtype, other_dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_MIXED_DTYPES
)
def test__new_zeros_with_same_feature_meta_other_dtype_wins(self_dtype, other_dtype):
    self_t = tu.make_input(self_dtype, (2, 3, 4), _MAIN_RANGE)
    other_t = tu.make_input(other_dtype, (7, 8), _MAIN_RANGE)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize("dtype", _NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)
def test__new_zeros_with_same_feature_meta_same_tensor(dtype):
    self_t = tu.make_input(dtype, (2, 3, 4), _MAIN_RANGE)
    other_t = self_t
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=1
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=1
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_NEW_ZEROS_WITH_SAME_FEATURE_META_DTYPES)),
)
def test__new_zeros_with_same_feature_meta_nan_inf_values(shape, dtype, scenario):
    self_t = _special_tensor(shape, dtype, scenario)
    other_t = _special_tensor((4, 5), dtype, scenario)
    ref_self = tu.to_reference(self_t)
    ref_other = tu.to_reference(other_t)

    ref_out = torch.ops.aten._new_zeros_with_same_feature_meta(
        ref_self, ref_other, self_num_batch_dims=0
    )
    res_out = flag_gems._new_zeros_with_same_feature_meta(
        self_t, other_t, self_num_batch_dims=0
    )

    _assert_zero_output(res_out, ref_out, self_t, other_t, ref_self, ref_other)


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test__new_zeros_with_same_feature_meta_negative_batch_dims_raises(dtype):
    self_t = tu.make_input(dtype, (2, 3), _MAIN_RANGE)
    other_t = tu.make_input(dtype, (4, 5), _MAIN_RANGE)

    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=-1
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=-1
        )


@pytest.mark._new_zeros_with_same_feature_meta_out
def test__new_zeros_with_same_feature_meta_out_wrong_dtype_raises():
    self_t = tu.make_input(torch.float32, (2, 3), _MAIN_RANGE)
    other_t = tu.make_input(torch.float32, (4, 5), _MAIN_RANGE)
    out_t = torch.full((2, 4, 5), 1, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta.out(
            self_t, other_t, self_num_batch_dims=1, out=out_t
        )
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._new_zeros_with_same_feature_meta(
            self_t, other_t, self_num_batch_dims=1, out=out_t
        )


@pytest.mark._new_zeros_with_same_feature_meta
@pytest.mark.parametrize(
    "self_arg,other_arg",
    [
        pytest.param((1, 2), None, id="tuple_self"),
        pytest.param(1, None, id="int_self"),
        pytest.param(3.14, None, id="float_self"),
        pytest.param(None, 1, id="none_self"),
        pytest.param(None, None, id="none_both"),
    ],
)
def test__new_zeros_with_same_feature_meta_rejects_non_tensor(self_arg, other_arg):
    with pytest.raises(RuntimeError):
        torch.ops.aten._new_zeros_with_same_feature_meta(self_arg, other_arg)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._new_zeros_with_same_feature_meta(self_arg, other_arg)

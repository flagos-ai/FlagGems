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

from . import test_utils as tu

# aten::is_inference(Tensor self) -> bool reads the inference flag of one
# TensorImpl and returns a Python bool. The flag is a per-tensor property, so the
# tested contract is only that flag_gems.is_inference reports the same state as
# torch.ops.aten.is_inference for the same kind of operand.

_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
    torch.complex64,
]

_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

_BOUNDARY_DTYPES = [torch.float32, torch.int64, torch.bool]

# Scalar and empty tensors retain their allocation-context state.
_BOUNDARY_SHAPES = [(), (0,), (2, 0)]

# Positive special-value workloads are default-only: quick selects none of them.
_SPECIAL_ROWS = tu.selected_cases(
    [
        (inference, dtype, scenario)
        for inference in (False, True)
        for dtype, scenario in tu.special_value_cases(_FLOAT_DTYPES)
    ],
    quick=[],
)

_VIEW_FORMS = ("view", "transpose", "slice", "expand", "detach")


def _state_pair(base, inference):
    """Two independent operands carrying the requested inference state.

    A clone created inside torch.inference_mode() is an inference tensor, while
    a clone, a device move or a storage copy outside the context (for example
    the one tu.to_reference builds) clears the flag. Both operands are therefore
    cloned under the same context.
    """
    if inference:
        with torch.inference_mode():
            return base.clone(), base.clone()
    return base.clone(), base.clone()


def _make_view(tensor, form):
    if form == "view":
        return tensor.view(2, 12)
    if form == "transpose":
        return tensor.t()
    if form == "slice":
        return tensor[1:3]
    if form == "expand":
        return tensor[0:1].expand(3, 6)
    return tensor.detach()


@pytest.mark.is_inference
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("inference", [False, True])
def test_is_inference_state(dtype, value_range, shape, inference):
    base = tu.make_input(dtype, shape, value_range)
    inp, ref_inp = _state_pair(base, inference)

    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("shape", _BOUNDARY_SHAPES)
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
@pytest.mark.parametrize("inference", [False, True])
def test_is_inference_scalar_and_empty(dtype, shape, inference):
    base = tu.make_input(dtype, shape, ["-1", "1"])
    inp, ref_inp = _state_pair(base, inference)

    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
def test_is_inference_context_is_not_tensor_state(dtype):
    plain = tu.make_input(dtype, (4, 6), ["-1", "1"])
    with torch.inference_mode():
        inferred = plain.clone()

    # The flag belongs to the tensor, not to the calling context: a plain tensor
    # queried from inside the context stays non-inference.
    with torch.inference_mode():
        ref_inside = torch.ops.aten.is_inference(plain)
        res_inside = flag_gems.is_inference(plain)
    assert isinstance(res_inside, bool)
    assert res_inside is ref_inside

    # ... and an inference tensor queried outside the context stays inference.
    ref_outside = torch.ops.aten.is_inference(inferred)
    res_outside = flag_gems.is_inference(inferred)
    assert isinstance(res_outside, bool)
    assert res_outside is ref_outside


@pytest.mark.is_inference
@pytest.mark.parametrize("form", _VIEW_FORMS)
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
def test_is_inference_derived_state(dtype, form):
    base = tu.make_input(dtype, (4, 6), ["-1", "1"])
    with torch.inference_mode():
        inferred = base.clone()
        ref_inferred = base.clone()

    # Views and detach share the source storage and inherit its flag ...
    inp = _make_view(inferred, form)
    ref_inp = _make_view(ref_inferred, form)
    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)
    assert isinstance(res_out, bool)
    assert res_out is ref_out

    # ... while a fresh clone outside the context starts non-inference.
    ref_clone = torch.ops.aten.is_inference(ref_inferred.clone())
    res_clone = flag_gems.is_inference(inferred.clone())
    assert isinstance(res_clone, bool)
    assert res_clone is ref_clone


@pytest.mark.is_inference
@pytest.mark.parametrize("dtype", _BOUNDARY_DTYPES)
def test_is_inference_no_grad_is_not_inference(dtype):
    with torch.no_grad():
        inp = tu.make_input(dtype, (4, 6), ["-1", "1"])
        ref_inp = tu.make_input(dtype, (4, 6), ["-1", "1"])

    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out
    assert res_out is False


@pytest.mark.is_inference
@pytest.mark.parametrize("inference", [False, True])
def test_is_inference_does_not_mutate_input(inference):
    base = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    inp, ref_inp = _state_pair(base, inference)
    inp = inp.t()
    ref_inp = ref_inp.t()
    storage_before = inp.untyped_storage().data_ptr()
    layout_before = (
        inp.data_ptr(),
        inp.storage_offset(),
        tuple(inp.shape),
        tuple(inp.stride()),
    )

    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out
    # The query is read-only: no storage swap, no metadata change, no materiali-
    # zation and no change to the reported state.
    assert inp.untyped_storage().data_ptr() == storage_before
    assert (
        inp.data_ptr(),
        inp.storage_offset(),
        tuple(inp.shape),
        tuple(inp.stride()),
    ) == layout_before
    assert torch.ops.aten.is_inference(inp) is ref_out
    tu.assert_result_equal(inp, ref_inp)


@pytest.mark.is_inference
@pytest.mark.parametrize("inference,dtype,scenario", _SPECIAL_ROWS)
def test_is_inference_special_values(inference, dtype, scenario):
    # NaN/Inf payloads must not change the reported state. e4m3fn only appears
    # with the nan scenario because it cannot represent infinity.
    inp, ref_inp = _state_pair(tu.make_special_input(dtype, scenario), inference)

    ref_out = torch.ops.aten.is_inference(ref_inp)
    res_out = flag_gems.is_inference(inp)

    assert isinstance(res_out, bool)
    assert res_out is ref_out


@pytest.mark.is_inference
@pytest.mark.parametrize("bad_input", [3.14, 1, True, [1, 2], "x"])
def test_is_inference_negative_non_tensor(bad_input):
    # None is deliberately absent: aten accepts it as an undefined tensor and
    # returns False instead of rejecting it, so it is not an invalid-input case.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference(bad_input)


@pytest.mark.is_inference
@pytest.mark.parametrize("second_arg", [1, 0, 2.0, True, (1,)])
def test_is_inference_negative_extra_positional(second_arg):
    inp = tu.make_input(torch.float32, (4, 6), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference(inp, second_arg)


@pytest.mark.is_inference
def test_is_inference_negative_missing_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference()


@pytest.mark.is_inference
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_is_inference_negative_out_kwarg(dtype):
    # aten::is_inference has no out= overload.
    inp = tu.make_input(dtype, (4, 6), ["-1", "1"])
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.is_inference(inp, out=inp)

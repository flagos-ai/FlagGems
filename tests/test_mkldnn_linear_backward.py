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

# aten::mkldnn_linear_backward is the oneDNN (MkldnnCPU) linear-backward
# primitive. self/grad_output must be opaque oneDNN tensors and weight must be a
# dense fp32 tensor, so the reference and the candidate both receive exactly
# those types and opaque results are compared by materializing them.
#
# Exemptions below were probed on this backend, not assumed:
#   * dtypes: only float32/bfloat16/float16 run (with a dense fp32 weight).
#     float64/int32/int64/bool/fp8 are rejected by to_mkldnn ("dense_to_mkldnn
#     expects float, bfloat16, half, uint8, int8 tensor input"), and int8/uint8
#     convert but fail with "could not create a primitive descriptor for the
#     inner product backward propagation primitive".
#   * shapes: the inner-product primitive is built from input.sizes(), so rank
#     must be >= 2 and the spec's (), (1,) and (256,) shapes fail with "could not
#     create a primitive descriptor for the inner product forward propagation
#     primitive".
#   * no broadcast operand and no scalar operand.
#   * autograd: this operator *is* the linear backward; native raises "derivative
#     for aten::mkldnn_linear_backward is not implemented", so the three gradient
#     components are covered by the eight-way output_mask sweep.
#   * output_mask has no schema default, so it is always passed explicitly.
_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
_WEIGHT_DTYPE = torch.float32
_DEFAULT_RANGE = ["-1", "1"]

# The quick row stays in the default list as well.
_SHAPE_CASES = tu.selected_cases(
    [
        ((2, 19, 7), 5),
        ((1024, 1024), 8),
        ((20, 320, 15), 16),
        ((16, 128, 64, 60), 4),
        ((16, 7, 57, 32, 29), 4),
    ],
    quick=[((2, 19, 7), 5)],
)

_MASK_SHAPE_CASES = tu.selected_cases(
    [
        ((2, 19, 7), 5),
        ((4, 8), 5),
        ((20, 320, 15), 16),
        ((16, 128, 64, 60), 4),
    ],
    quick=[((2, 19, 7), 5), ((4, 8), 5)],
)

# Every output_mask combination. A masked-off component is absent (None), except
# grad_bias: when only the weight is requested native leaves it as a
# default-constructed 0-dim tensor with undefined contents.
_OUTPUT_MASKS = [
    [False, False, False],
    [False, False, True],
    [False, True, False],
    [False, True, True],
    [True, False, False],
    [True, False, True],
    [True, True, False],
    [True, True, True],
]

_OUT_SHAPE_CASES = tu.selected_cases(
    [((2, 19, 7), 5), ((20, 320, 15), 16), ((16, 128, 64, 60), 4)],
    quick=[((2, 19, 7), 5)],
)

# The out overload is exercised with grad_input requested. A placeholder out0 is
# rejected by native (probed with mask [False, False, True]: "tried to directly
# modify sizes for customized tensor"), so the remaining combinations stay on
# the default overload in test_mkldnn_linear_backward_output_mask.
_OUT_MASKS = [
    [True, True, True],
    [True, True, False],
    [True, False, True],
]

_SPECIAL_SHAPE = (2, 19, 7)
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_DTYPES), quick=[])


def _to_mkldnn(tensor):
    # oneDNN operands are CPU-only; the shared generators build on
    # flag_gems.device, so this is the data movement to the operator's contract.
    return tensor.cpu().to_mkldnn()


def _rows(shape):
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows


def _make_operands(shape, out_features, dtype, value_range):
    """Build self/grad_output in oneDNN layout plus the dense fp32 weight."""
    self_mkldnn = _to_mkldnn(tu.make_input(dtype, shape, value_range))
    grad_out_mkldnn = _to_mkldnn(
        tu.make_input(dtype, (_rows(shape), out_features), value_range)
    )
    weight = tu.make_input(_WEIGHT_DTYPE, (out_features, shape[-1]), value_range).cpu()
    return self_mkldnn, grad_out_mkldnn, weight


def _reference_operands(self_mkldnn, grad_out_mkldnn, weight):
    """Independent snapshots for the reference run."""
    return (
        tu.to_reference(self_mkldnn),
        tu.to_reference(grad_out_mkldnn),
        tu.to_reference(weight),
    )


def _assert_same_component(res, ref):
    assert res is not None and ref is not None
    # Check metadata before the lossless dense materialization of the opaque
    # oneDNN tensor, so a wrong layout/dtype cannot hide behind it.
    assert res.layout == ref.layout
    assert res.dtype == ref.dtype
    assert res.shape == ref.shape
    if res.layout == torch._mkldnn:
        res, ref = res.to_dense(), ref.to_dense()
    tu.assert_result_close(res, ref)


def _assert_backward_outputs(res_out, ref_out, device):
    assert isinstance(res_out, (tuple, list))
    assert len(res_out) == 3
    for res, ref in zip(res_out, ref_out):
        if ref is None:
            # The component was not requested.
            assert res is None
            continue
        assert res is not None
        # The result must live on the same device as the oneDNN operands.
        assert res.device.type == device.type
        if ref.dim() == 0:
            # grad_bias is the default-constructed 0-dim tensor native returns
            # when only the weight is requested; its contents are undefined, so
            # only the rank is comparable.
            assert res.dim() == 0
            assert res.dtype == ref.dtype
            assert res.layout == ref.layout
            continue
        _assert_same_component(res, ref)


def _special_operand(shape, dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    count = 1
    for dim in shape:
        count *= dim
    repeats = (count + payload.numel() - 1) // payload.numel()
    return _to_mkldnn(payload.repeat(repeats)[:count].reshape(shape))


@pytest.mark.mkldnn_linear_backward
@pytest.mark.parametrize("shape,out_features", _SHAPE_CASES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_linear_backward(shape, out_features, value_range, dtype):
    self_mkldnn, grad_out_mkldnn, weight = _make_operands(
        shape, out_features, dtype, value_range
    )
    ref_self, ref_grad_out, ref_weight = _reference_operands(
        self_mkldnn, grad_out_mkldnn, weight
    )

    ref_out = torch.ops.aten.mkldnn_linear_backward(
        ref_self, ref_grad_out, ref_weight, [True, True, True]
    )
    res_out = flag_gems.mkldnn_linear_backward(
        self_mkldnn, grad_out_mkldnn, weight, [True, True, True]
    )

    _assert_backward_outputs(res_out, ref_out, self_mkldnn.device)
    tu.assert_result_equal(self_mkldnn.to_dense(), ref_self.to_dense())
    tu.assert_result_equal(grad_out_mkldnn.to_dense(), ref_grad_out.to_dense())
    tu.assert_result_equal(weight, ref_weight)


@pytest.mark.mkldnn_linear_backward
@pytest.mark.parametrize("shape,out_features", _MASK_SHAPE_CASES)
@pytest.mark.parametrize("output_mask", _OUTPUT_MASKS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_linear_backward_output_mask(shape, out_features, output_mask, dtype):
    self_mkldnn, grad_out_mkldnn, weight = _make_operands(
        shape, out_features, dtype, _DEFAULT_RANGE
    )
    ref_self, ref_grad_out, ref_weight = _reference_operands(
        self_mkldnn, grad_out_mkldnn, weight
    )

    ref_out = torch.ops.aten.mkldnn_linear_backward(
        ref_self, ref_grad_out, ref_weight, list(output_mask)
    )
    res_out = flag_gems.mkldnn_linear_backward(
        self_mkldnn, grad_out_mkldnn, weight, list(output_mask)
    )

    _assert_backward_outputs(res_out, ref_out, self_mkldnn.device)
    tu.assert_result_equal(self_mkldnn.to_dense(), ref_self.to_dense())
    tu.assert_result_equal(grad_out_mkldnn.to_dense(), ref_grad_out.to_dense())
    tu.assert_result_equal(weight, ref_weight)


@pytest.mark.mkldnn_linear_backward
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_mkldnn_linear_backward_nan_inf(dtype, scenario):
    out_features = 5
    # A regular weight keeps the special values coming from the operands.
    weight = tu.make_input(
        _WEIGHT_DTYPE, (out_features, _SPECIAL_SHAPE[-1]), _DEFAULT_RANGE
    ).cpu()
    self_mkldnn = _special_operand(_SPECIAL_SHAPE, dtype, scenario)
    grad_out_mkldnn = _special_operand(
        (_rows(_SPECIAL_SHAPE), out_features), dtype, scenario
    )
    ref_self, ref_grad_out, ref_weight = _reference_operands(
        self_mkldnn, grad_out_mkldnn, weight
    )

    ref_out = torch.ops.aten.mkldnn_linear_backward(
        ref_self, ref_grad_out, ref_weight, [True, True, True]
    )
    res_out = flag_gems.mkldnn_linear_backward(
        self_mkldnn, grad_out_mkldnn, weight, [True, True, True]
    )

    _assert_backward_outputs(res_out, ref_out, self_mkldnn.device)
    tu.assert_result_equal(self_mkldnn.to_dense(), ref_self.to_dense())
    tu.assert_result_equal(grad_out_mkldnn.to_dense(), ref_grad_out.to_dense())
    tu.assert_result_equal(weight, ref_weight)


@pytest.mark.mkldnn_linear_backward
@pytest.mark.parametrize("shape,out_features", _OUT_SHAPE_CASES)
@pytest.mark.parametrize("output_mask", _OUT_MASKS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_linear_backward_out(shape, out_features, output_mask, dtype):
    self_mkldnn, grad_out_mkldnn, weight = _make_operands(
        shape, out_features, dtype, _DEFAULT_RANGE
    )
    ref_self, ref_grad_out, ref_weight = _reference_operands(
        self_mkldnn, grad_out_mkldnn, weight
    )

    out0 = torch.empty(shape, dtype=dtype).to_mkldnn()
    out1 = torch.empty((out_features, shape[-1]), dtype=_WEIGHT_DTYPE)
    out2 = torch.empty((out_features,), dtype=_WEIGHT_DTYPE)
    ref0 = torch.empty(shape, dtype=dtype).to_mkldnn()
    ref1 = torch.empty_like(out1)
    ref2 = torch.empty_like(out2)

    ref_ret = torch.ops.aten.mkldnn_linear_backward.out(
        ref_self,
        ref_grad_out,
        ref_weight,
        list(output_mask),
        out0=ref0,
        out1=ref1,
        out2=ref2,
    )
    res_ret = flag_gems.mkldnn_linear_backward(
        self_mkldnn,
        grad_out_mkldnn,
        weight,
        list(output_mask),
        out0=out0,
        out1=out1,
        out2=out2,
    )

    assert isinstance(res_ret, (tuple, list))
    assert len(res_ret) == 3
    # The out overload writes into and returns the caller's buffers themselves.
    for res, buffer in zip(res_ret, (out0, out1, out2)):
        assert res is buffer
    for index, (res, ref) in enumerate(zip(res_ret, ref_ret)):
        if output_mask[index]:
            _assert_same_component(res, ref)
        # A masked-off buffer is left untouched; its contents are undefined and
        # therefore not compared.


@pytest.mark.mkldnn_linear_backward
def test_mkldnn_linear_backward_rejects_dense_input():
    # Native needs oneDNN tensors: a dense self raises "grad_output and input
    # needs to be mkldnn layout".
    _, grad_out_mkldnn, weight = _make_operands(
        (6, 8), 5, torch.float32, _DEFAULT_RANGE
    )
    dense_self = tu.make_input(torch.float32, (6, 8), _DEFAULT_RANGE).cpu()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear_backward(
            dense_self, grad_out_mkldnn, weight, [True, True, True]
        )


@pytest.mark.mkldnn_linear_backward
def test_mkldnn_linear_backward_rejects_non_fp32_weight():
    # Native raises "weight_t needs to be a dense tensor" for a non-fp32 weight.
    self_mkldnn, grad_out_mkldnn, _ = _make_operands(
        (6, 8), 5, torch.float32, _DEFAULT_RANGE
    )
    half_weight = tu.make_input(torch.float16, (5, 8), _DEFAULT_RANGE).cpu()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear_backward(
            self_mkldnn, grad_out_mkldnn, half_weight, [True, True, True]
        )


@pytest.mark.mkldnn_linear_backward
def test_mkldnn_linear_backward_rejects_wrong_mask_length():
    # output_mask is a fixed-size bool[3]; native raises "Tried to convert a List
    # with 2 elements to a fixed-size array of size 3".
    self_mkldnn, grad_out_mkldnn, weight = _make_operands(
        (6, 8), 5, torch.float32, _DEFAULT_RANGE
    )
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear_backward(
            self_mkldnn, grad_out_mkldnn, weight, [True, True]
        )


@pytest.mark.mkldnn_linear_backward
def test_mkldnn_linear_backward_rejects_rank1_input():
    # Rank < 2 cannot describe an inner-product primitive.
    self_mkldnn = _to_mkldnn(tu.make_input(torch.float32, (8,), _DEFAULT_RANGE))
    grad_out_mkldnn = _to_mkldnn(tu.make_input(torch.float32, (1, 5), _DEFAULT_RANGE))
    weight = tu.make_input(torch.float32, (5, 8), _DEFAULT_RANGE).cpu()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear_backward(
            self_mkldnn, grad_out_mkldnn, weight, [True, True, True]
        )


@pytest.mark.mkldnn_linear_backward
def test_mkldnn_linear_backward_rejects_int8_input():
    # int8 converts to oneDNN but has no inner-product backward kernel.
    self_mkldnn = _to_mkldnn(tu.make_input(torch.int8, (6, 8), _DEFAULT_RANGE))
    grad_out_mkldnn = _to_mkldnn(tu.make_input(torch.int8, (6, 5), _DEFAULT_RANGE))
    weight = tu.make_input(torch.float32, (5, 8), _DEFAULT_RANGE).cpu()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear_backward(
            self_mkldnn, grad_out_mkldnn, weight, [True, True, True]
        )

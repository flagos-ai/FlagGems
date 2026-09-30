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

# aten::mkldnn_linear is a CPU-only oneDNN inner product: `self` and the result
# are opaque mkldnn tensors while `weight`/`bias` are dense, so every operand is
# built on CPU and results are compared through their materialized values.
OUT_FEATURES = 8


def _dense(tensor):
    # Materialized values of an opaque mkldnn tensor.
    return tensor.to_dense() if tensor.is_mkldnn else tensor


def _operands(shape, dtype, value_range, *, with_bias=True):
    # (candidate, reference) operand triples with equal values but independent
    # storage: a candidate that writes into its operands cannot corrupt the
    # native reference or the values used for comparison.
    in_features = shape[-1]
    dense = tu.make_input(dtype, shape, value_range).to("cpu")
    weight = tu.make_input(dtype, (OUT_FEATURES, in_features), value_range).to("cpu")
    bias = None
    if with_bias:
        bias = tu.make_input(dtype, (OUT_FEATURES,), value_range).to("cpu")

    def build():
        return (
            dense.clone().to_mkldnn(),
            weight.clone(),
            None if bias is None else bias.clone(),
        )

    return build(), build()


def _snapshots(*tensors):
    return [None if t is None else _dense(t).clone() for t in tensors]


def _assert_operands_unchanged(tensors, snapshots):
    # The operands are inputs; a candidate must not write into them.
    for tensor, snapshot in zip(tensors, snapshots):
        if snapshot is not None:
            tu.assert_result_equal(_dense(tensor), snapshot)


def _assert_result(res_out, ref_out, inp):
    # The native result is an opaque mkldnn tensor; a strided candidate result is
    # a different layout contract and is rejected before any value comparison.
    assert res_out.is_mkldnn and ref_out.is_mkldnn
    assert res_out.device == inp.device
    tu.assert_result_close(_dense(res_out), _dense(ref_out))


# Dtypes the native op accepts: to_mkldnn() rejects float64/int32/int64/bool and
# fp8 (dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor input).
# The floating rows cover both bias modes; oneDNN has no int8/uint8 bias path
# (could not create a primitive descriptor for the inner product forward
# propagation primitive), so those two rows are bias-free. Their extreme ranges
# saturate exactly like the native op: a 4-term int8 product of 100s returns 127
# rather than 40000 mod 256.
VALUE_ROWS = [
    (torch.float32, True),
    (torch.float32, False),
    (torch.float16, True),
    (torch.float16, False),
    (torch.bfloat16, True),
    (torch.bfloat16, False),
    (torch.int8, False),
    (torch.uint8, False),
]

# Rank 0 is excluded: to_mkldnn() has no operand for a 0-dim tensor (could not
# create a primitive descriptor for the reorder primitive) and there is no
# in_features to contract. The remaining six spec shapes are all accepted.
VALUE_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) > 0]


@pytest.mark.mkldnn_linear
@pytest.mark.parametrize("shape", VALUE_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype,with_bias", VALUE_ROWS)
def test_mkldnn_linear(shape, value_range, dtype, with_bias):
    (res_inp, res_weight, res_bias), (ref_inp, ref_weight, ref_bias) = _operands(
        shape, dtype, value_range, with_bias=with_bias
    )
    snapshots = _snapshots(res_inp, res_weight, res_bias)

    ref_out = torch.ops.aten.mkldnn_linear(ref_inp, ref_weight, ref_bias)
    res_out = flag_gems.mkldnn_linear(res_inp, res_weight, res_bias)

    _assert_result(res_out, ref_out, res_inp)
    _assert_operands_unchanged((res_inp, res_weight, res_bias), snapshots)


# Schema-default call form: the bias argument is omitted entirely.
BIASLESS_CASES = [
    ((1024, 1024), torch.float32),
    ((2, 3, 5), torch.float32),
    ((256,), torch.float16),
]


@pytest.mark.mkldnn_linear
@pytest.mark.parametrize("shape,dtype", BIASLESS_CASES)
def test_mkldnn_linear_without_bias(shape, dtype):
    (res_inp, res_weight, _), (ref_inp, ref_weight, _) = _operands(
        shape, dtype, ["-1", "1"], with_bias=False
    )
    snapshots = _snapshots(res_inp, res_weight)

    ref_out = torch.ops.aten.mkldnn_linear(ref_inp, ref_weight)
    res_out = flag_gems.mkldnn_linear(res_inp, res_weight)

    _assert_result(res_out, ref_out, res_inp)
    _assert_operands_unchanged((res_inp, res_weight), snapshots)


OUT_CASES = [
    ((2, 3, 5), torch.float32, True),
    ((2, 3, 5), torch.float16, True),
    ((1, 1), torch.float32, False),
    ((4, 6), torch.bfloat16, True),
]


@pytest.mark.mkldnn_linear
@pytest.mark.parametrize("shape,dtype,with_bias", OUT_CASES)
def test_mkldnn_linear_out(shape, dtype, with_bias):
    (res_inp, res_weight, res_bias), (ref_inp, ref_weight, ref_bias) = _operands(
        shape, dtype, ["-1", "1"], with_bias=with_bias
    )
    snapshots = _snapshots(res_inp, res_weight, res_bias)
    out_shape = shape[:-1] + (OUT_FEATURES,)

    # The out buffer must be an mkldnn tensor: a dense one is refused by
    # copy_mkldnn_ (between mkldnn layout and dense Tensors is not implemented).
    buf = torch.empty(out_shape, dtype=dtype).to_mkldnn()
    ref_buf = torch.empty(out_shape, dtype=dtype).to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_linear.out(
        ref_inp, ref_weight, ref_bias, out=ref_buf
    )
    res_out = flag_gems.mkldnn_linear(res_inp, res_weight, res_bias, out=buf)

    _assert_result(res_out, ref_out, res_inp)
    assert res_out is buf
    _assert_operands_unchanged((res_inp, res_weight, res_bias), snapshots)


BACKWARD_SHAPES = tu.selected_cases([(4, 6), (2, 3, 5)], quick=[])


@pytest.mark.mkldnn_linear
@pytest.mark.parametrize("shape", BACKWARD_SHAPES)
def test_mkldnn_linear_backward(shape):
    # The original mkldnn activation is the differentiated leaf, alongside the
    # dense weight and bias.
    dense = tu.make_input(torch.float32, shape, ["-1", "1"]).to("cpu")
    weight = tu.make_input(torch.float32, (OUT_FEATURES, shape[-1]), ["-1", "1"]).to(
        "cpu"
    )
    bias = tu.make_input(torch.float32, (OUT_FEATURES,), ["-1", "1"]).to("cpu")

    res_inp = dense.to_mkldnn().requires_grad_(True)
    ref_inp = dense.clone().to_mkldnn().requires_grad_(True)
    res_weight = weight.clone().requires_grad_(True)
    ref_weight = weight.clone().requires_grad_(True)
    res_bias = bias.clone().requires_grad_(True)
    ref_bias = bias.clone().requires_grad_(True)

    # The native output is an opaque mkldnn tensor, so the non-uniform upstream
    # gradient must carry the same layout; a strided one raises invalid gradient
    # at index 0 - expected layout Mkldnn but got Strided.
    upstream = (
        tu.make_input(torch.float32, shape[:-1] + (OUT_FEATURES,), ["-1", "1"])
        .to("cpu")
        .to_mkldnn()
    )
    res_upstream = upstream.clone()
    ref_upstream = upstream.clone()

    ref_out = torch.ops.aten.mkldnn_linear(ref_inp, ref_weight, ref_bias)
    res_out = flag_gems.mkldnn_linear(res_inp, res_weight, res_bias)
    _assert_result(res_out, ref_out, res_inp)

    ref_grads = torch.autograd.grad(
        ref_out, [ref_inp, ref_weight, ref_bias], ref_upstream
    )
    res_grads = torch.autograd.grad(
        res_out, [res_inp, res_weight, res_bias], res_upstream
    )

    # The activation gradient keeps the opaque mkldnn layout; the weight and bias
    # gradients stay dense.
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        assert res_grad.is_mkldnn == ref_grad.is_mkldnn
        tu.assert_result_close(_dense(res_grad), _dense(ref_grad))


SPECIAL_CASES = tu.selected_cases(
    # Only the floating dtypes have a nan/inf matrix: to_mkldnn() rejects bool,
    # fp8 and the wider float/integer types.
    list(tu.special_value_cases([torch.float32, torch.float16, torch.bfloat16])),
    quick=[],
)


@pytest.mark.mkldnn_linear
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_mkldnn_linear_special_values(dtype, scenario):
    # The payload rides along the feature axis of a (5, 4) mkldnn activation and
    # an all-ones weight keeps every special value visible in the result.
    payload = tu.make_special_input(dtype, scenario).to("cpu")
    weight = torch.ones((OUT_FEATURES, 4), dtype=dtype)

    def activation():
        return payload.reshape(-1, 1).expand(-1, 4).contiguous().to_mkldnn()

    res_inp = activation()
    ref_inp = activation()
    res_weight = weight.clone()
    ref_weight = weight.clone()

    ref_out = torch.ops.aten.mkldnn_linear(ref_inp, ref_weight, None)
    res_out = flag_gems.mkldnn_linear(res_inp, res_weight, None)

    _assert_result(res_out, ref_out, res_inp)


NEGATIVE_CASES = [
    "in_features_mismatch",
    "weight_rank_1",
    "bias_length_mismatch",
]


@pytest.mark.mkldnn_linear
@pytest.mark.parametrize("case", NEGATIVE_CASES)
def test_mkldnn_linear_invalid_operands(case):
    (inp, weight, bias), _ = _operands((4, 8), torch.float32, ["-1", "1"])
    if case == "in_features_mismatch":
        weight = torch.ones((OUT_FEATURES, 7))
    elif case == "weight_rank_1":
        weight = torch.ones((OUT_FEATURES,))
    else:
        bias = torch.ones((OUT_FEATURES - 1,))

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear(inp, weight, bias)

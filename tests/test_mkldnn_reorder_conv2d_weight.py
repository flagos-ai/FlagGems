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

"""Correctness tests for ``aten::mkldnn_reorder_conv2d_weight``.

The operand and the result are opaque CPU ``torch._mkldnn`` tensors: every fixture is built
as a dense CPU tensor and converted with ``to_mkldnn()``, and the comparison is made on the
materialized ``to_dense()`` values, because ``torch.testing.assert_close`` has no kernel for
the mkldnn layout. The operator is a pure weight re-layout, so values must be preserved
exactly and every comparison below is exact. There is a single tensor operand, the
convolution attributes in its schema do not broadcast, and the result carries no autograd
history, so this file has no broadcast, tensor-vs-scalar or backward dimension.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Reorder support measured on this build with a valid rank-4 weight:
#   float32 / float16 / bfloat16 / int8 -> reordered, values bit-exact for all five ranges
#   uint8 -> ``to_mkldnn()`` accepts it, the convolution descriptor does not, so it is
#            reachable as a candidate call and appears as the dtype negative below
#   bool / int32 / int64 / float64 / fp8 -> ``to_mkldnn()`` itself refuses them
#            ("dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor input"),
#            so they cannot be expressed as a candidate call at all.
SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int8]

# Valid spec shapes for this operator: rank 3 (pass-through), rank 4 (explicit weight
# geometry) and rank 5, which the ``input_size=None`` branch collapses to
# ``(shape[0] * shape[1], *shape[2:])``. The other three spec shapes are not collectable:
#   ()              ``torch.zeros(()).to_mkldnn()`` raises "could not create a primitive
#                   descriptor for the reorder primitive" (RuntimeError, interpreter exit
#                   code 0), so the operand cannot be constructed.
#   (1,), (256,)    rank 1, and ``(1024, 1024)`` rank 2, build fine but the native reorder
#                   raises; they are asserted as negatives at the end of this file.
SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 3]

_WEIGHT_SHAPE = (32, 16, 3, 3)


def assert_reordered(res_out, ref_out, operand):
    """Layout and value checks the shared assertions cannot make on an opaque tensor."""
    assert res_out.is_mkldnn, "the reorder must return an opaque mkldnn tensor"
    assert res_out.device == operand.device
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_reorder_conv2d_weight(shape, value_range, dtype):
    dense = tu.make_input(dtype, shape, value_range).cpu()
    inp = dense.to_mkldnn()
    # The reference gets independently converted storage.
    ref_inp = dense.detach().clone().to_mkldnn()

    # Every convolution attribute keeps its schema default here; the explicit sweep is the
    # ``_params`` test below.
    ref_out = torch.ops.aten.mkldnn_reorder_conv2d_weight(ref_inp)
    res_out = flag_gems.mkldnn_reorder_conv2d_weight(inp)

    assert_reordered(res_out, ref_out, inp)
    # The reorder is out of place, so the operand's storage must stay untouched.
    tu.assert_result_equal(inp.to_dense(), dense)


# Rank-3 weights pass through and rank-5 weights collapse to
# ``(shape[0] * shape[1], *shape[2:])`` on the ``input_size=None`` size-inference branch. The
# resulting shape is asserted, not only the values, because that collapse is the observable
# behaviour of the branch.
_INFERRED_ROWS = [
    ((20, 320, 15), (20, 320, 15)),
    ((8, 4, 3), (8, 4, 3)),
    ((16, 7, 57, 32, 29), (112, 57, 32, 29)),
    ((8, 4, 3, 3, 2), (32, 3, 3, 2)),
]


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("shape,expected_shape", _INFERRED_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_reorder_conv2d_weight_size_inferred(shape, expected_shape, dtype):
    dense = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    inp = dense.to_mkldnn()
    ref_inp = dense.detach().clone().to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_reorder_conv2d_weight(ref_inp)
    res_out = flag_gems.mkldnn_reorder_conv2d_weight(inp)

    assert_reordered(res_out, ref_out, inp)
    assert tuple(res_out.shape) == expected_shape


# Explicit convolution attributes on a rank-4 weight. Each of padding/stride/dilation/
# groups/input_size has a schema default, which the grid above already covers by omission.
# ``input_size`` is never combined with ``groups=2``: with 16 input channels that geometry is
# invalid and the native operator rejects it.
_PARAM_ROWS = [
    {"padding": [1, 1]},
    {"padding": [2, 3]},
    {"padding": [5, 5]},
    {"padding": [0]},
    {"stride": [2, 2]},
    {"stride": [1, 2]},
    {"dilation": [2, 2]},
    {"dilation": [3, 3]},
    {"groups": 2},
    {"groups": 4},
    {"input_size": [1, 16, 32, 32]},
    {"padding": [1, 1], "stride": [2, 2], "dilation": [1, 1], "groups": 2},
    {"padding": [2, 2], "stride": [1, 1], "dilation": [2, 2], "groups": 1},
]


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("params", _PARAM_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_reorder_conv2d_weight_params(params, dtype):
    dense = tu.make_input(dtype, _WEIGHT_SHAPE, ["-1", "1"]).cpu()
    inp = dense.to_mkldnn()
    ref_inp = dense.detach().clone().to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_reorder_conv2d_weight(ref_inp, **params)
    res_out = flag_gems.mkldnn_reorder_conv2d_weight(inp, **params)

    assert_reordered(res_out, ref_out, inp)


# ``.out`` is genuinely callable and hands back the buffer it was given. The buffer has to be
# an mkldnn tensor of the result shape and dtype - the native kernel enforces both. Rank-5
# weights are not asserted here because the native ``.out`` kernel rejects them ("tried to
# directly modify sizes for customized tensor").
_OUT_ROWS = [
    (shape, dtype)
    for shape in [(32, 16, 3, 3), (8, 4, 3, 3)]
    for dtype in SUPPORTED_DTYPES
]


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("shape,dtype", _OUT_ROWS)
def test_mkldnn_reorder_conv2d_weight_out(shape, dtype):
    dense = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    inp = dense.to_mkldnn()
    ref_inp = dense.detach().clone().to_mkldnn()

    # A pre-filled buffer makes a partial write visible.
    res_buf = torch.full(shape, 7, dtype=dtype).to_mkldnn()
    ref_buf = torch.full(shape, 7, dtype=dtype).to_mkldnn()
    res_out = flag_gems.mkldnn_reorder_conv2d_weight(inp, out=res_buf)
    ref_out = torch.ops.aten.mkldnn_reorder_conv2d_weight.out(ref_inp, out=ref_buf)

    assert res_out is res_buf
    assert_reordered(res_out, ref_out, inp)


# NaN/Inf payloads survive the reorder unchanged for every supported floating dtype. fp8 is
# absent on purpose: neither float8_e4m3fn nor float8_e5m2 converts with ``to_mkldnn`` on
# this build, so no fp8 special-value scenario is expressible for this operator.
_SPECIAL_ROWS = tu.selected_cases(
    tu.special_value_cases([torch.float32, torch.float16, torch.bfloat16]), quick=[]
)


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_ROWS)
def test_mkldnn_reorder_conv2d_weight_special_values(dtype, scenario):
    # The shared generator's 5-element payload, shaped as a rank-4 weight.
    dense = tu.make_special_input(dtype, scenario).cpu().reshape(5, 1, 1, 1)
    inp = dense.to_mkldnn()
    ref_inp = dense.detach().clone().to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_reorder_conv2d_weight(ref_inp)
    res_out = flag_gems.mkldnn_reorder_conv2d_weight(inp)

    assert_reordered(res_out, ref_out, inp)


# Negative cases. Each invalid form was measured on the native operator first; only the
# candidate's exception is asserted here.

# Rank 1 and rank 2 weights and empty spatial/channel dimensions.
_BAD_SHAPES = [(1,), (256,), (1024, 1024), (0, 4, 3, 3), (4, 4, 0, 3)]


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("shape", _BAD_SHAPES)
def test_mkldnn_reorder_conv2d_weight_bad_shape(shape):
    inp = torch.zeros(shape, dtype=torch.float32).cpu().to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv2d_weight(inp)


# uint8 reaches the operator (``to_mkldnn`` accepts it) but has no reorder kernel. The other
# rejected dtypes (bool/int32/int64/float64/fp8) cannot be turned into the operator's operand
# at all, so their rejection is a conversion limit rather than a callable negative.
@pytest.mark.mkldnn_reorder_conv2d_weight
def test_mkldnn_reorder_conv2d_weight_unsupported_dtype():
    inp = torch.zeros(_WEIGHT_SHAPE, dtype=torch.uint8).cpu().to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv2d_weight(inp)


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("padding", [2, [0, 0, 0, 0], [-1, -1]])
def test_mkldnn_reorder_conv2d_weight_bad_padding(padding):
    inp = torch.zeros(_WEIGHT_SHAPE, dtype=torch.float32).cpu().to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv2d_weight(inp, padding=padding)


# ``stride=[0, 0]`` kills the test process with SIGFPE (measured exit code 136, core dump),
# so it is asserted nowhere; ``[-1, -1]`` is the measured invalid-stride form.
@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("stride", [[-1, -1]])
def test_mkldnn_reorder_conv2d_weight_bad_stride(stride):
    inp = torch.zeros(_WEIGHT_SHAPE, dtype=torch.float32).cpu().to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv2d_weight(inp, stride=stride)


@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("groups", [0, -1])
def test_mkldnn_reorder_conv2d_weight_bad_groups(groups):
    inp = torch.zeros(_WEIGHT_SHAPE, dtype=torch.float32).cpu().to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv2d_weight(inp, groups=groups)


# ``input_size`` only reaches the memory descriptor that is built from it, so a short list is
# accepted and degenerates to the same reorder. The measured invalid forms are a non-positive
# extent and a non-sequence.
@pytest.mark.mkldnn_reorder_conv2d_weight
@pytest.mark.parametrize("input_size", [[1, 16, 32, -1], "x"])
def test_mkldnn_reorder_conv2d_weight_bad_input_size(input_size):
    inp = torch.zeros(_WEIGHT_SHAPE, dtype=torch.float32).cpu().to_mkldnn()
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv2d_weight(inp, input_size=input_size)


# A non-tensor operand.
@pytest.mark.mkldnn_reorder_conv2d_weight
def test_mkldnn_reorder_conv2d_weight_none_operand():
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.mkldnn_reorder_conv2d_weight(None)


# A dense tensor where the operator requires the opaque mkldnn layout.
@pytest.mark.mkldnn_reorder_conv2d_weight
def test_mkldnn_reorder_conv2d_weight_dense_operand():
    with pytest.raises((RuntimeError, TypeError, NotImplementedError)):
        flag_gems.mkldnn_reorder_conv2d_weight(torch.zeros(_WEIGHT_SHAPE))

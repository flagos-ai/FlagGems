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

# aten::mkldnn_reorder_conv3d_weight dispatches to MkldnnCPU only: the operand and
# the .out buffer are CPU torch._mkldnn tensors that cannot live on
# flag_gems.device. Values are generated on flag_gems.device and moved with
# .cpu().to_mkldnn(). The op only relocates existing values, so results are
# compared exactly; the shared helpers reject the torch._mkldnn layout, so both
# sides are materialised losslessly with to_dense() after the candidate result
# layout has been asserted.
_PADDING = [0, 0, 0]
_STRIDE = [1, 1, 1]
_DILATION = [1, 1, 1]

# Probed on this backend: to_mkldnn accepts float32/bfloat16/float16/uint8/int8;
# uint8 is then rejected by the oneDNN conv3d weight descriptor, and
# float64/int32/int64/float8* raise inside the conversion ("dense_to_mkldnn
# expects float, bfloat16, half, uint8, int8 tensor input"), so four of the nine
# required dtypes are representable for this op.
_DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.int8]
_FLOAT_DTYPES = [dtype for dtype in _DTYPES if dtype.is_floating_point]

# The oneDNN conv3d weight descriptor needs three spatial dims, so ranks 0-2 are
# not representable and rank-1/2 operands are negative cases below. The quick
# rank-3 geometry is added to the default list so the default level covers the
# quick cases.
_SPEC_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 3]
_SHAPES = _SPEC_SHAPES if (2, 19, 7) in _SPEC_SHAPES else _SPEC_SHAPES + [(2, 19, 7)]

_DEFAULT_CALL_SHAPES = tu.selected_cases(
    [(20, 320, 15), (2, 19, 7)], quick=[(2, 19, 7)]
)


def _mkldnn_weight(dtype, shape, value_range=("-1", "1")):
    """Generate on flag_gems.device, then build the MkldnnCPU operand."""
    return tu.make_input(dtype, shape, list(value_range)).cpu().to_mkldnn()


def _assert_reorder_equal(res, ref):
    # The shared assertions only see the dense materialisation, so the blocked
    # layout of the candidate result is checked before it.
    assert res.layout == torch._mkldnn
    tu.assert_result_equal(res.to_dense(), ref.to_dense())


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_reorder_conv3d_weight_value_grid(dtype, shape, value_range):
    inp = _mkldnn_weight(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight(
        ref_inp, _PADDING, _STRIDE, _DILATION, 1
    )
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(
        inp, _PADDING, _STRIDE, _DILATION, 1
    )

    _assert_reorder_equal(res_out, ref_out)
    # Only the .out overload writes a tensor, so the operand keeps its values.
    tu.assert_result_equal(inp.to_dense(), ref_inp.to_dense())


# (shape, padding, stride, dilation, groups, input_size). The cheap rows come
# first so the quick level keeps every small padding/dilation/input_size case;
# only the large (16, 128, 64, 60) rows are default-only. The oneDNN descriptor
# requires group counts dividing both channel dims and kernel dims matching
# input_size.
_SMALL_PARAM_ROWS = [
    ((8, 4, 3, 3, 3), (1, 2, 3), (2, 2, 2), (1, 1, 1), 1, None),
    ((8, 4, 3, 3, 3), (2, 2, 2), (3, 3, 3), (2, 2, 2), 1, None),
    ((2, 19, 7), (0, 0, 0), (1, 1, 1), (1, 1, 1), 1, None),
    ((2, 19, 7), (1, 1, 1), (1, 1, 1), (0, 0, 0), 1, None),
    ((8, 3, 3, 3, 3), (0, 0, 0), (1, 1, 1), (1, 1, 1), 1, (1, 3, 16, 16, 16)),
    ((8, 3, 3, 3, 3), (0, 0, 0), (1, 1, 1), (1, 1, 1), 1, (2, 3, 8, 8, 8)),
]
_LARGE_PARAM_ROWS = [
    ((16, 128, 64, 60), (0, 0, 0), (1, 1, 1), (1, 1, 1), 1, None),
    ((16, 128, 64, 60), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, None),
    ((16, 128, 64, 60), (1, 2, 3), (2, 2, 2), (1, 1, 1), 1, None),
    ((16, 128, 64, 60), (2, 2, 2), (3, 3, 3), (2, 2, 2), 1, None),
    ((16, 128, 64, 60), (1, 1, 1), (1, 1, 1), (0, 0, 0), 1, None),
    ((16, 128, 64, 60), (0, 0, 0), (1, 1, 1), (1, 1, 1), 2, None),
    ((16, 128, 64, 60), (0, 0, 0), (1, 1, 1), (1, 1, 1), 4, None),
]
_PARAM_ROWS = _SMALL_PARAM_ROWS + _LARGE_PARAM_ROWS
_PARAM_QUICK_ROWS = _SMALL_PARAM_ROWS


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("row", tu.selected_cases(_PARAM_ROWS, quick=_PARAM_QUICK_ROWS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_mkldnn_reorder_conv3d_weight_params(dtype, row):
    shape, padding, stride, dilation, groups, input_size = row
    inp = _mkldnn_weight(dtype, shape)
    ref_inp = tu.to_reference(inp)

    args = (list(padding), list(stride), list(dilation), groups)
    if input_size is not None:
        args += (list(input_size),)

    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight(ref_inp, *args)
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(inp, *args)

    _assert_reorder_equal(res_out, ref_out)


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("shape", _DEFAULT_CALL_SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_reorder_conv3d_weight_schema_defaults(dtype, shape):
    # padding/stride/dilation/groups/input_size all carry schema defaults; the
    # candidate's public signature must accept the operand alone.
    inp = _mkldnn_weight(dtype, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight(ref_inp)
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(inp)

    _assert_reorder_equal(res_out, ref_out)


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
def test_mkldnn_reorder_conv3d_weight_out(dtype, shape):
    # Callable on this backend with an mkldnn buffer of the result shape, which it
    # also returns.
    inp = _mkldnn_weight(dtype, shape)
    ref_inp = tu.to_reference(inp)
    ref_buf = torch.empty(shape, dtype=dtype, device=inp.device).to_mkldnn()
    res_buf = torch.empty(shape, dtype=dtype, device=inp.device).to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight.out(
        ref_inp, _PADDING, _STRIDE, _DILATION, 1, out=ref_buf
    )
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(
        inp, _PADDING, _STRIDE, _DILATION, 1, out=res_buf
    )

    assert res_out is res_buf
    _assert_reorder_equal(res_out, ref_out)


# groups > 1 selects the oneDNN grouped weight layout, which needs both channel
# dims divisible by the group count (the spec grid shapes are only valid with
# groups=1); (2, 19, 7) covers the quick level. Every row is small, so quick
# keeps all of them.
_GROUPED_ROWS = [
    ((2, 19, 7), 2),
    ((6, 3, 3, 3, 3), 3),
    ((8, 4, 3, 3, 3), 2),
    ((4, 4, 3, 3, 3), 4),
    ((12, 6, 3, 3, 3), 6),
]


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("shape,groups", _GROUPED_ROWS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mkldnn_reorder_conv3d_weight_grouped(dtype, shape, groups):
    inp = _mkldnn_weight(dtype, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight(
        ref_inp, _PADDING, _STRIDE, _DILATION, groups
    )
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(
        inp, _PADDING, _STRIDE, _DILATION, groups
    )

    _assert_reorder_equal(res_out, ref_out)


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])
)
def test_mkldnn_reorder_conv3d_weight_special_values(dtype, scenario):
    # Rank 3 is the smallest mkldnn operand, so the 5-value payload sits in the
    # leading (channel) dimension; the reorder must carry NaN/Inf through.
    inp = tu.make_special_input(dtype, scenario).cpu().reshape(5, 1, 1).to_mkldnn()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight(
        ref_inp, _PADDING, _STRIDE, _DILATION, 1
    )
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(
        inp, _PADDING, _STRIDE, _DILATION, 1
    )

    _assert_reorder_equal(res_out, ref_out)


# A stride containing 0 is not listed: the native call aborts the interpreter with
# SIGFPE during primitive creation, so the failure cannot be observed in-process.
_NEGATIVE_PARAMS = [
    ((0, 0), (1, 1, 1), (1, 1, 1), 1),
    ((0, 0, 0), (1, 1), (1, 1, 1), 1),
    ((0, 0, 0), (1, 1, 1), (1, 1), 1),
    ((-1, -1, -1), (1, 1, 1), (1, 1, 1), 1),
    ((0, 0, 0), (1, 1, 1), (1, 1, 1), 0),
    ((0, 0, 0), (1, 1, 1), (1, 1, 1), -1),
]


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("padding,stride,dilation,groups", _NEGATIVE_PARAMS)
def test_mkldnn_reorder_conv3d_weight_invalid_params(padding, stride, dilation, groups):
    inp = _mkldnn_weight(torch.float32, (8, 3, 3, 3, 3))

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv3d_weight(
            inp, list(padding), list(stride), list(dilation), groups
        )


# Inputs outside the operator's domain. The 0-dim shape is not listed because
# to_mkldnn() already rejects it, leaving no candidate call to make.
_NEGATIVE_INPUTS = [
    ((1024, 1024), torch.float32),
    ((256,), torch.float32),
    ((8, 3, 0, 3, 3), torch.float32),
    ((8, 3, 3, 3, 3), torch.uint8),
]


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("shape,dtype", _NEGATIVE_INPUTS)
def test_mkldnn_reorder_conv3d_weight_invalid_input(dtype, shape):
    inp = _mkldnn_weight(dtype, shape)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv3d_weight(inp, _PADDING, _STRIDE, _DILATION, 1)


@pytest.mark.mkldnn_reorder_conv3d_weight
def test_mkldnn_reorder_conv3d_weight_dense_input_rejected():
    # The aten kernel is registered for MkldnnCPU only, so a dense operand with
    # otherwise valid values and shape must still fail.
    inp = tu.make_input(torch.float32, (8, 3, 3, 3, 3), ["-1", "1"]).cpu()

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_reorder_conv3d_weight(inp, _PADDING, _STRIDE, _DILATION, 1)


@pytest.mark.mkldnn_reorder_conv3d_weight
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mkldnn_reorder_conv3d_weight_rejects_backward(dtype):
    inp = (
        tu.make_input(dtype, (8, 4, 3, 3, 3), ["-1", "1"])
        .cpu()
        .to_mkldnn()
        .requires_grad_()
    )
    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.mkldnn_reorder_conv3d_weight(ref_inp)
    res_out = flag_gems.mkldnn_reorder_conv3d_weight(inp)

    assert res_out.is_mkldnn
    assert res_out.requires_grad
    tu.assert_result_equal(res_out.to_dense(), ref_out.to_dense())
    upstream = tu.make_input(dtype, res_out.shape, ["-1", "1"]).cpu().to_mkldnn()
    with pytest.raises(
        RuntimeError,
        match="derivative for aten::mkldnn_reorder_conv3d_weight is not implemented",
    ):
        torch.autograd.grad(res_out, inp, grad_outputs=upstream)

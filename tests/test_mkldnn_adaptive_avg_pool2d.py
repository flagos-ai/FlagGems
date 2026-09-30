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

# Host-only op: the operand must carry the mkldnn layout and the conversion has
# no accelerator path, so a value-range tensor is built on the selected device
# and converted on the host. Other dtypes cannot form an operand at all
# ('dense_to_mkldnn expects float, bfloat16, half, uint8, int8 tensor input').
_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int8,
    torch.uint8,
]

_RANGE = tu.selected_ranges()[0]

# 4-D operands only; every output_size divides both spatial extents. Quick keeps
# an identity, a small real reduction, a global pool and two asymmetric kernels
# for every supported dtype; default keeps those plus bigger workloads.
_SHAPE_ROWS = [
    ((2, 19, 7, 7), (7, 7)),
    ((2, 3, 8, 8), (2, 2)),
    ((2, 3, 8, 8), (1, 1)),
    ((2, 3, 8, 8), (4, 8)),
    ((2, 3, 8, 8), (8, 2)),
    ((4, 5, 15, 15), (3, 5)),
    ((1, 1, 256, 256), (16, 16)),
    ((16, 128, 64, 60), (2, 2)),
    ((20, 320, 15, 16), (5, 8)),
    ((16, 7, 57, 32), (3, 4)),
    ((1, 64, 224, 224), (7, 7)),
    ((4, 128, 112, 112), (14, 14)),
    ((1, 3, 1024, 1024), (32, 32)),
]
_QUICK_SHAPE_ROWS = [
    ((2, 19, 7, 7), (7, 7)),
    ((2, 3, 8, 8), (2, 2)),
    ((2, 3, 8, 8), (1, 1)),
    ((2, 3, 8, 8), (4, 8)),
    ((2, 3, 8, 8), (8, 2)),
]
_SIZE_ROWS = tu.selected_cases(_SHAPE_ROWS, quick=_QUICK_SHAPE_ROWS)

# Boundary sweep of the required output_size argument on the spec 4-D shape.
_PARAM_SHAPE = (16, 128, 64, 60)
_OUTPUT_SIZES = [(1, 1), (2, 2), (4, 4), (8, 4), (16, 15), (64, 60), (2, 5), (64, 1)]
_SMALL_PARAM_SHAPE = (1, 3, 64, 60)
_OUTPUT_SIZE_CASES = tu.selected_cases(
    [
        (shape, size)
        for shape in (_PARAM_SHAPE, _SMALL_PARAM_SHAPE)
        for size in _OUTPUT_SIZES
    ],
    quick=[(_SMALL_PARAM_SHAPE, size) for size in _OUTPUT_SIZES],
)


_SMALL_SHAPE = (2, 3, 8, 8)
_SMALL_OUTPUT_SIZE = (2, 2)
_STRIDED_ROWS = [
    ((2, 3, 8, 16), (slice(None), slice(None), slice(None), slice(None, None, 2))),
    ((2, 3, 8, 20), (slice(None), slice(None), slice(None), slice(2, 18, 2))),
]
_BACKWARD_CASES = tu.selected_cases(
    [torch.float32, torch.bfloat16, torch.float16], quick=[]
)
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases([torch.float32, torch.bfloat16, torch.float16]), quick=[]
)


def _mkldnn_operand(dtype, shape, value_range):
    return tu.make_input(dtype, shape, value_range).cpu().to_mkldnn()


def _pair_inputs(dtype, shape, value_range):
    """Two independent mkldnn operands sharing one value-range sample."""
    dense = tu.make_input(dtype, shape, value_range).cpu()
    return dense.to_mkldnn(), dense.to_mkldnn()


def _assert_mkldnn_result(result, reference, operand, *, exact=False):
    """Layout and device first, then the values stored behind the opaque layout."""
    assert result.layout == operand.layout == reference.layout == torch._mkldnn
    assert result.device == operand.device
    if exact:
        tu.assert_result_equal(result.to_dense(), reference.to_dense())
    else:
        tu.assert_result_close(result.to_dense(), reference.to_dense())


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape,output_size", _SIZE_ROWS)
def test_mkldnn_adaptive_avg_pool2d(dtype, value_range, shape, output_size):
    inp, ref_inp = _pair_inputs(dtype, shape, value_range)

    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d(ref_inp, list(output_size))
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d(inp, list(output_size))

    _assert_mkldnn_result(res_out, ref_out, inp)
    tu.assert_result_equal(inp.to_dense(), ref_inp.to_dense())


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("shape,output_size", _OUTPUT_SIZE_CASES)
def test_mkldnn_adaptive_avg_pool2d_output_size(dtype, shape, output_size):
    inp, ref_inp = _pair_inputs(dtype, shape, _RANGE)

    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d(ref_inp, list(output_size))
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d(inp, list(output_size))

    _assert_mkldnn_result(res_out, ref_out, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mkldnn_adaptive_avg_pool2d_out(dtype):
    inp, ref_inp = _pair_inputs(dtype, _SMALL_SHAPE, _RANGE)
    out_shape = (_SMALL_SHAPE[0], _SMALL_SHAPE[1], *_SMALL_OUTPUT_SIZE)
    # .out needs an mkldnn buffer; a dense one fails inside copy_mkldnn_.
    out = tu.make_input(dtype, out_shape, _RANGE).cpu().to_mkldnn()
    ref_out = tu.make_input(dtype, out_shape, _RANGE).cpu().to_mkldnn()

    torch.ops.aten.mkldnn_adaptive_avg_pool2d.out(
        ref_inp, list(_SMALL_OUTPUT_SIZE), out=ref_out
    )
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d(
        inp, list(_SMALL_OUTPUT_SIZE), out=out
    )

    assert res_out is out
    _assert_mkldnn_result(res_out, ref_out, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("full_shape,view", _STRIDED_ROWS)
def test_mkldnn_adaptive_avg_pool2d_strided_host_view(dtype, full_shape, view):
    dense = tu.make_input(dtype, full_shape, _RANGE).cpu()[view]
    assert not dense.is_contiguous()
    inp = dense.to_mkldnn()
    ref_inp = dense.to_mkldnn()

    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d(
        ref_inp, list(_SMALL_OUTPUT_SIZE)
    )
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d(inp, list(_SMALL_OUTPUT_SIZE))

    _assert_mkldnn_result(res_out, ref_out, inp)


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", _BACKWARD_CASES)
def test_mkldnn_adaptive_avg_pool2d_backward(dtype):
    # Differentiate the operator output through the original leaf operand.
    dense = tu.make_input(dtype, _SMALL_SHAPE, _RANGE).cpu().requires_grad_(True)
    ref_dense = tu.to_reference(dense).requires_grad_(True)
    out_shape = (_SMALL_SHAPE[0], _SMALL_SHAPE[1], *_SMALL_OUTPUT_SIZE)
    upstream = tu.make_input(dtype, out_shape, _RANGE).cpu()
    ref_upstream = tu.to_reference(upstream)

    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d(
        ref_dense.to_mkldnn(), list(_SMALL_OUTPUT_SIZE)
    )
    (ref_grad,) = torch.autograd.grad(
        ref_out.to_dense(), ref_dense, grad_outputs=ref_upstream
    )

    res_out = flag_gems.mkldnn_adaptive_avg_pool2d(
        dense.to_mkldnn(), list(_SMALL_OUTPUT_SIZE)
    )
    assert res_out.layout == torch._mkldnn
    tu.assert_result_close(res_out.to_dense(), ref_out.to_dense())
    (res_grad,) = torch.autograd.grad(res_out.to_dense(), dense, grad_outputs=upstream)

    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_mkldnn_adaptive_avg_pool2d_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario).cpu()
    width = payload.numel()
    inp = payload.reshape(1, 1, 1, width).to_mkldnn()
    ref_inp = payload.reshape(1, 1, 1, width).to_mkldnn()

    # A 1x1 window copies each value, so NaN / Inf must survive exactly.
    ref_out = torch.ops.aten.mkldnn_adaptive_avg_pool2d(ref_inp, [1, width])
    res_out = flag_gems.mkldnn_adaptive_avg_pool2d(inp, [1, width])

    _assert_mkldnn_result(res_out, ref_out, inp, exact=True)


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", [torch.float32, torch.int8])
@pytest.mark.parametrize(
    "output_size",
    [
        pytest.param([0, 2], id="zero-extent"),
        pytest.param([2, 0], id="zero-other-extent"),
        pytest.param([3, 3], id="non-divisible"),
        pytest.param([-1, 2], id="negative-extent"),
    ],
)
def test_mkldnn_adaptive_avg_pool2d_rejects_invalid_output_size(dtype, output_size):
    inp = _mkldnn_operand(dtype, _SMALL_SHAPE, _RANGE)
    with pytest.raises(RuntimeError):
        flag_gems.mkldnn_adaptive_avg_pool2d(inp, output_size)


@pytest.mark.mkldnn_adaptive_avg_pool2d
@pytest.mark.parametrize("shape", [(2, 3, 8), (2, 3, 8, 8, 2)])
def test_mkldnn_adaptive_avg_pool2d_rejects_non_4d(shape):
    inp = _mkldnn_operand(torch.float32, shape, _RANGE)
    with pytest.raises(RuntimeError):
        flag_gems.mkldnn_adaptive_avg_pool2d(inp, [1, 1])

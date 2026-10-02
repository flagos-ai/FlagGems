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

# aten::mkldnn_max_pool3d is registered for MkldnnCPU only, so reference and candidate
# both take rank-5 (N, C, D, H, W) blocking-layout tensors in a dtype that layout can
# store: float32/float16/bfloat16/int8/uint8 (dense_to_mkldnn refuses float64, fp8 and the
# wide integer types before any operator call). kernel_size = 0 and stride = 0 are absent
# from the negative rows because the native primitive aborts the process with SIGFPE
# instead of raising. Pooling copies stored values, so comparisons are exact; the
# torch._mkldnn layout is materialized with to_dense() for the comparison only.

SUPPORTED_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
]

_FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

_SPEC_5D_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) == 5]
_EXTRA_5D_SHAPES = [
    (2, 3, 5, 6, 7),
    (1, 1, 9, 9, 9),
    (4, 8, 16, 28, 28),
    (2, 16, 8, 32, 24),
]
# Zero-extent N and W volumes are valid natively: the primitive still computes output
# extents ((0,1,2,2,2) -> (0,1,1,1,1), (1,1,2,2,0) -> (1,1,1,1,0)).
_EMPTY_5D_SHAPES = [(0, 1, 2, 2, 2), (1, 1, 2, 2, 0)]
# The shared quick shape (2, 19, 7) has rank 3, so quick mode uses rank-5 volumes.
SHAPES = tu.selected_cases(
    _SPEC_5D_SHAPES + _EXTRA_5D_SHAPES + _EMPTY_5D_SHAPES + [(2, 3, 19, 7, 5)],
    quick=[(2, 3, 19, 7, 5), (2, 3, 5, 6, 7), (1, 1, 9, 9, 9)] + _EMPTY_5D_SHAPES,
)

# Bounds come from the shared selector, whose entries carry the helper's symbol form.
DEFAULT_RANGE = tu.selected_ranges()[0]

KERNEL_SIZE = [2, 2, 2]
STRIDE = [2, 2, 2]
PADDING = [0, 0, 0]
DILATION = [1, 1, 1]
CEIL_MODE = False


def to_mkldnn(dense):
    # MkldnnCPU storage is host-only; the value helpers build on the configured device.
    return dense.cpu().to_mkldnn()


def dense_values(tensor):
    return tensor.to_dense() if tensor.is_mkldnn else tensor


def assert_pooled(res, ref):
    """Compare pooling results, rejecting a strided candidate result."""
    assert res.is_mkldnn == ref.is_mkldnn
    tu.assert_result_equal(dense_values(res), dense_values(ref))


@pytest.mark.mkldnn_max_pool3d
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_max_pool3d(shape, value_range, dtype):
    dense = tu.make_input(dtype, shape, value_range)
    inp = to_mkldnn(dense)
    ref_inp = to_mkldnn(tu.to_reference(dense))

    ref_out = torch.ops.aten.mkldnn_max_pool3d(
        ref_inp, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE
    )
    res_out = flag_gems.mkldnn_max_pool3d(
        inp, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE
    )

    assert_pooled(res_out, ref_out)


# Rows are (volume, kernel_size, keyword arguments). The small volume carries every cheap
# semantic form (full explicit call, all-default call, empty stride list, scalar
# padding/dilation, padding beyond the half kernel, kernel 1 with stride > kernel,
# ceil_mode True, asymmetric kernel/padding/stride) and is therefore also the quick
# subset; the large volume keeps the original heavy rows in the default suite only.
_SMALL_VOLUME = (2, 3, 6, 8, 10)
_LARGE_VOLUME = (16, 7, 57, 32, 29)
_PARAM_ROWS = [
    (
        _SMALL_VOLUME,
        [2, 2, 2],
        {
            "stride": [2, 2, 2],
            "padding": [0, 0, 0],
            "dilation": [1, 1, 1],
            "ceil_mode": False,
        },
    ),
    (_SMALL_VOLUME, [2, 2, 2], {}),
    (_SMALL_VOLUME, [2, 2, 2], {"stride": []}),
    (_SMALL_VOLUME, [2, 2, 2], {"padding": 0, "dilation": 1}),
    (_SMALL_VOLUME, [2, 2, 2], {"stride": [3, 3, 3], "padding": [3, 3, 3]}),
    (_SMALL_VOLUME, [3, 3, 3], {"stride": [1, 1, 1], "padding": [1, 1, 1]}),
    (_SMALL_VOLUME, [3, 3, 3], {"padding": [1, 1, 1], "ceil_mode": True}),
    (_SMALL_VOLUME, [1, 1, 1], {"stride": [3, 3, 3], "padding": [1, 1, 1]}),
    (_SMALL_VOLUME, [2, 3, 2], {"stride": [1, 2, 2], "padding": [0, 1, 0]}),
    (_SMALL_VOLUME, [1, 1, 2], {"stride": [1, 1, 2]}),
    (_LARGE_VOLUME, [2, 2, 2], {}),
    (_LARGE_VOLUME, [57, 3, 3], {"stride": [1, 1, 1], "padding": [0, 0, 0]}),
]
PARAM_CASES = tu.selected_cases(
    _PARAM_ROWS, quick=[row for row in _PARAM_ROWS if row[0] == _SMALL_VOLUME]
)


@pytest.mark.mkldnn_max_pool3d
@pytest.mark.parametrize("volume,kernel_size,kwargs", PARAM_CASES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_max_pool3d_pool_params(volume, kernel_size, kwargs, dtype):
    dense = tu.make_input(dtype, volume, DEFAULT_RANGE)
    inp = to_mkldnn(dense)
    ref_inp = to_mkldnn(tu.to_reference(dense))

    ref_out = torch.ops.aten.mkldnn_max_pool3d(ref_inp, kernel_size, **kwargs)
    res_out = flag_gems.mkldnn_max_pool3d(inp, kernel_size, **kwargs)

    assert_pooled(res_out, ref_out)


# Sentinel written into every out buffer; it cannot be produced by pooling a
# [-1, 1] input, so an element left unwritten by the candidate is caught.
_OUT_SENTINEL = {
    torch.float32: -7.0,
    torch.float16: -7.0,
    torch.bfloat16: -7.0,
    torch.int8: -7,
    torch.uint8: 200,
}


@pytest.mark.mkldnn_max_pool3d
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_max_pool3d_out(dtype):
    shape = (2, 3, 8, 8, 8)
    dense = tu.make_input(dtype, shape, ["-1", "1"])
    inp = to_mkldnn(dense)
    ref_inp = to_mkldnn(tu.to_reference(dense))
    snapshot = dense_values(inp).clone()

    out_shape = torch.ops.aten.mkldnn_max_pool3d(
        ref_inp, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE
    ).size()
    res_buf = torch.full(out_shape, _OUT_SENTINEL[dtype], dtype=dtype).to_mkldnn()
    ref_buf = torch.full(out_shape, _OUT_SENTINEL[dtype], dtype=dtype).to_mkldnn()

    returned = flag_gems.mkldnn_max_pool3d(
        inp, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE, out=res_buf
    )
    torch.ops.aten.mkldnn_max_pool3d(
        ref_inp, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE, out=ref_buf
    )

    # aten out= returns the buffer it was handed.
    assert returned is res_buf
    assert_pooled(res_buf, ref_buf)
    tu.assert_result_equal(dense_values(inp), snapshot)


# Positive nan/inf coverage is default-only. tu.special_value_cases supplies the shared
# scenario names, and its dtype list is limited to the float types MkldnnCPU can store.
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_FLOAT_DTYPES), quick=[])


@pytest.mark.mkldnn_max_pool3d
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_mkldnn_max_pool3d_special_values(dtype, scenario):
    # The shared payload is five values in row-major order; a (1,1,1,1,5) volume with
    # kernel [1,1,2] and stride [1,1,1] puts each of them into a pooling window.
    payload = tu.make_special_input(dtype, scenario).reshape(1, 1, 1, 1, 5)
    inp = to_mkldnn(payload)
    ref_inp = to_mkldnn(tu.to_reference(payload))

    ref_out = torch.ops.aten.mkldnn_max_pool3d(
        ref_inp, [1, 1, 2], [1, 1, 1], PADDING, DILATION, CEIL_MODE
    )
    res_out = flag_gems.mkldnn_max_pool3d(
        inp, [1, 1, 2], [1, 1, 1], PADDING, DILATION, CEIL_MODE
    )

    assert_pooled(res_out, ref_out)


# Backward is default-only and only defined for the differentiable float dtypes.
BACKWARD_DTYPES = tu.selected_cases(_FLOAT_DTYPES, quick=[])


# Deterministic, positive and non-uniform: a constant upstream gradient would hide a
# wrong arg-max scatter.
def upstream_gradient(out_shape, dtype):
    count = 1
    for extent in out_shape:
        count *= extent
    ramp = torch.arange(count, dtype=torch.float32) % 8
    return ((ramp + 1) / 8).reshape(out_shape).to(dtype)


@pytest.mark.mkldnn_max_pool3d
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_mkldnn_max_pool3d_backward(dtype):
    shape = (2, 3, 6, 8, 10)
    dense = tu.make_input(dtype, shape, DEFAULT_RANGE)
    res_in = dense.detach().cpu().clone().to_mkldnn().requires_grad_(True)
    ref_in = (
        tu.to_reference(dense).detach().cpu().clone().to_mkldnn().requires_grad_(True)
    )

    res_out = flag_gems.mkldnn_max_pool3d(
        res_in, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE
    )
    ref_out = torch.ops.aten.mkldnn_max_pool3d(
        ref_in, KERNEL_SIZE, STRIDE, PADDING, DILATION, CEIL_MODE
    )
    assert_pooled(res_out, ref_out)

    # The MkldnnCPU backward keeps the blocking layout on grad_outputs ('invalid gradient
    # at index 0 - expected layout Mkldnn but got Strided' for a strided one), so each
    # graph gets its own MkldnnCPU upstream gradient carrying the same values.
    out_shape = tuple(ref_out.size())
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_in, grad_outputs=upstream_gradient(out_shape, dtype).to_mkldnn()
    )
    (res_grad,) = torch.autograd.grad(
        res_out, res_in, grad_outputs=upstream_gradient(out_shape, dtype).to_mkldnn()
    )

    assert res_grad.is_mkldnn == ref_grad.is_mkldnn
    tu.assert_result_equal(dense_values(res_grad), dense_values(ref_grad))


# Rows are (volume, kernel_size, keyword arguments); each one was observed at this exact
# call form. The int[3] parameters must expand to the rank (rank-4 with a 2-element list
# and rank-5 with a 2-element kernel list are both refused) and empty padding/dilation
# lists are rejected, while an empty stride list is the documented default. Unsupported
# dtypes are not negative rows: MkldnnCPU storage cannot hold float64/fp8/int32/int64/bool,
# so dense_to_mkldnn refuses them before any candidate call can be reached.
_INVALID_CALLS = [
    ((2, 3, 5, 6), [2, 2], {"stride": [2, 2], "padding": [0, 0, 0]}),
    ((2, 3, 5, 6, 7), [2, 2], {"stride": [2, 2, 2], "padding": [0, 0, 0]}),
    ((2, 3, 5, 6, 7), [2, 2, 2], {"stride": [2, 2, 2], "padding": [-1, -1, -1]}),
    ((2, 3, 5, 6, 7), [2, 2, 2], {"stride": [-1, -1, -1], "padding": [0, 0, 0]}),
    ((2, 3, 5, 6, 7), [2, 2, 2], {"stride": [2, 2, 2], "padding": []}),
    ((2, 3, 5, 6, 7), [2, 2, 2], {"stride": [2, 2, 2], "dilation": []}),
    ((2, 3, 5, 6, 7), [2, 2, 2], {"dilation": [2, 2, 2]}),
    ((2, 3, 5, 6, 7), [9, 9, 9], {}),
]


@pytest.mark.mkldnn_max_pool3d
@pytest.mark.parametrize("shape,kernel_size,kwargs", _INVALID_CALLS)
def test_mkldnn_max_pool3d_invalid_arguments(shape, kernel_size, kwargs):
    inp = to_mkldnn(tu.make_input(torch.float32, shape, DEFAULT_RANGE))
    with pytest.raises(RuntimeError):
        flag_gems.mkldnn_max_pool3d(inp, kernel_size, **kwargs)

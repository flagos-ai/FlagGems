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

# aten::glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim) -> Tensor
#
# Forward-mode helper of glu: grad_x / dx are shaped like x, grad_glu /
# dgrad_glu keep the rank of x with extent 1 or exactly x.shape[dim] // 2 along
# dim, and the result truncates dim to 2 * (x.shape[dim] // 2), so an odd split
# axis loses its last element.

DTYPES = [torch.float32, torch.float16, torch.int8, torch.uint8, torch.int32]
if utils.bf16_is_supported:
    DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    DTYPES.append(torch.float64)
if utils.int64_is_supported:
    DTYPES.append(torch.int64)

# Measured on the CUDA ("nvidia") vendor: sigmoid_cuda has no float8 kernel
# there, so that expectation is scoped to the vendor instead of a device name.
FP8_REJECTED = utils.fp8_is_supported and flag_gems.vendor_name == "nvidia"

# The operator is itself the gradient-level operation and has no autograd
# formula, so the value grid below is its gradient coverage. It also has no
# scalar-operand form: all five inputs are tensors and dim is a required int
# without a schema default.


def _half_shape(shape, dim):
    half = list(shape)
    half[dim] //= 2
    return tuple(half)


def _out_shape(shape, dim):
    # The result truncates the split axis to an even size.
    out = list(shape)
    out[dim] = 2 * (out[dim] // 2)
    return tuple(out)


def _out_dtype(dtype):
    # Measured: integer inputs are promoted to float32, floating ones are kept.
    return dtype if dtype.is_floating_point else torch.float32


def _tensors(dtype, shapes, value_range):
    return tuple(tu.make_input(dtype, shape, value_range) for shape in shapes)


def _operands(dtype, shape, dim, value_range):
    # grad_x, grad_glu, x, dgrad_glu, dx on the FlagGems device.
    half = _half_shape(shape, dim)
    return _tensors(dtype, (shape, half, shape, half, shape), value_range)


def _viewed_input(dtype, shape, value_range, view):
    # A native-valid view: a step on the last axis, a transposed buffer, or a
    # contiguous slice with a nonzero storage offset.
    if view == "strided":
        padded = tu.make_input(dtype, shape[:-1] + (2 * shape[-1],), value_range)
        return padded[..., ::2]
    if view == "transposed":
        padded = tu.make_input(
            dtype, (shape[1], shape[0]) + tuple(shape[2:]), value_range
        )
        return padded.transpose(0, 1)
    padded = tu.make_input(dtype, (shape[0] + 2,) + tuple(shape[1:]), value_range)
    return padded[1 : 1 + shape[0]]


# One (shape, dim) row per spec shape. The rank-0 shape cannot form a positive
# workload because dim is required, so it only appears as a negative case.
SHAPE_DIM_CASES = tu.selected_cases(
    [
        ((1,), 0),
        ((256,), 0),
        ((1024, 1024), 1),
        ((20, 320, 15), 2),
        ((16, 128, 64, 60), 3),
        ((16, 7, 57, 32, 29), 1),
    ],
    quick=[((2, 19, 7), 1)],
)


# dim sweep on the parameter-coverage dtype (int32 promotes to float32 on both
# sides, so the extreme ranges stay representable).
DIM_CASES = tu.selected_cases(
    [
        ((1024, 1024), 0, torch.int32, ["-1", "1"]),
        ((1024, 1024), 1, torch.int32, ["-1", "1"]),
        ((1024, 1024), -1, torch.int32, ["-1", "1"]),
        ((1024, 1024), -2, torch.int32, ["-1", "1"]),
    ],
    quick=[],
)


VALUE_ROWS = [
    (shape, dim, dtype, value_range)
    for shape, dim in SHAPE_DIM_CASES
    for dtype in DTYPES
    for value_range in tu.selected_ranges()
] + DIM_CASES


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("shape,dim,dtype,value_range", VALUE_ROWS)
def test_glu_backward_jvp_value_range(shape, dim, dtype, value_range):
    grad_x, grad_glu, x, dgrad_glu, dx = _operands(dtype, shape, dim, value_range)

    ref_out = torch.ops.aten.glu_backward_jvp(
        tu.to_reference(grad_x),
        tu.to_reference(grad_glu),
        tu.to_reference(x),
        tu.to_reference(dgrad_glu),
        tu.to_reference(dx),
        dim,
    )
    res_out = flag_gems.glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim)

    tu.assert_result_close(res_out, ref_out)


# Out-of-range dims and the required-dim / rank-0 combination are rejected with
# IndexError; the expectation is the precise class, which a missing candidate
# (AttributeError) cannot satisfy.
INVALID_DIM_CASES = [
    ((1024, 1024), 2),
    ((1024, 1024), -3),
    ((), 0),
]


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("shape,dim", INVALID_DIM_CASES)
def test_glu_backward_jvp_invalid_dim(shape, dim):
    # dim is validated before the operands are indexed, so the halved operand
    # only needs an axis that is valid for the rank.
    half = () if not shape else (shape[0] // 2,) + tuple(shape[1:])
    operands = _tensors(torch.float32, (shape, half, shape, half, shape), ["-1", "1"])

    with pytest.raises(IndexError):
        flag_gems.glu_backward_jvp(*operands, dim)


OUT_CASES = tu.selected_cases(
    [
        ((1024, 1024), 1, torch.float32),
        ((20, 320, 15), 2, torch.float16),
        ((16, 128, 64, 60), 3, torch.int32),
    ],
    quick=[],
)


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("shape,dim,dtype", OUT_CASES)
def test_glu_backward_jvp_out(shape, dim, dtype):
    grad_x, grad_glu, x, dgrad_glu, dx = _operands(dtype, shape, dim, ["-1", "1"])
    ref_ops = [tu.to_reference(t) for t in (grad_x, grad_glu, x, dgrad_glu, dx)]
    out_shape = _out_shape(shape, dim)

    ref_buf = torch.empty(out_shape, dtype=_out_dtype(dtype), device=ref_ops[0].device)
    ref_out = torch.ops.aten.glu_backward_jvp.out(*ref_ops, dim, out=ref_buf)

    res_buf = torch.empty(out_shape, dtype=_out_dtype(dtype), device=flag_gems.device)
    res_out = flag_gems.glu_backward_jvp(
        grad_x, grad_glu, x, dgrad_glu, dx, dim, out=res_buf
    )

    # The out overload writes into the caller's buffer and hands it back.
    assert res_out is res_buf
    assert res_out.device == res_buf.device
    tu.assert_result_close(res_out, ref_out)


# Two broadcast patterns: a halved axis alone, and a halved axis together with a
# singleton outer axis. grad_glu / dgrad_glu keep the rank of x, so they
# broadcast with 1-sized axes only.
BROADCAST_DTYPE = torch.bfloat16 if utils.bf16_is_supported else torch.float16

BROADCAST_CASES = tu.selected_cases(
    [
        ((1024, 1024), 1, (1, 512)),
        ((20, 320, 15), 1, (1, 160, 1)),
    ],
    quick=[],
)


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("shape,dim,grad_shape", BROADCAST_CASES)
def test_glu_backward_jvp_broadcast(shape, dim, grad_shape):
    value_range = ["-1", "1"]
    grad_x, x, dx = _tensors(BROADCAST_DTYPE, (shape, shape, shape), value_range)
    grad_glu, dgrad_glu = _tensors(
        BROADCAST_DTYPE, (grad_shape, grad_shape), value_range
    )

    ref_out = torch.ops.aten.glu_backward_jvp(
        tu.to_reference(grad_x),
        tu.to_reference(grad_glu),
        tu.to_reference(x),
        tu.to_reference(dgrad_glu),
        tu.to_reference(dx),
        dim,
    )
    res_out = flag_gems.glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim)

    tu.assert_result_close(res_out, ref_out)


LAYOUT_CASES = tu.selected_cases(
    [
        ((4, 3, 16), 2, "strided"),
        ((3, 4, 8), 2, "transposed"),
        ((4, 8), 1, "offset"),
        ((20, 320, 8), 2, "offset"),
    ],
    quick=[],
)


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("shape,dim,view", LAYOUT_CASES)
def test_glu_backward_jvp_view_operands(shape, dim, view):
    value_range = ["-1", "1"]
    dtype = torch.float32
    half = _half_shape(shape, dim)
    grad_x = _viewed_input(dtype, shape, value_range, view)
    grad_glu = _viewed_input(dtype, half, value_range, view)
    x = _viewed_input(dtype, shape, value_range, view)
    dgrad_glu = _viewed_input(dtype, half, value_range, view)
    dx = _viewed_input(dtype, shape, value_range, view)

    ref_out = torch.ops.aten.glu_backward_jvp(
        tu.to_reference(grad_x),
        tu.to_reference(grad_glu),
        tu.to_reference(x),
        tu.to_reference(dgrad_glu),
        tu.to_reference(dx),
        dim,
    )
    res_out = flag_gems.glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim)

    tu.assert_result_close(res_out, ref_out)


SPECIAL_DTYPES = [torch.float32, torch.float16]
if utils.bf16_is_supported:
    SPECIAL_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    SPECIAL_DTYPES.append(torch.float64)

SPECIAL_CASES = tu.selected_cases(
    list(tu.special_value_cases(SPECIAL_DTYPES)), quick=[]
)


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_glu_backward_jvp_special_values(dtype, scenario):
    # The shared payload is as long as the halved split axis it feeds, so it is
    # used directly for grad_glu / dgrad_glu and repeated for grad_x / x / dx.
    payload = tu.make_special_input(dtype, scenario)
    x = payload.repeat(2)
    grad_x = x.clone()
    grad_glu = payload.clone()
    dgrad_glu = payload.clone()
    dx = x.clone()

    ref_out = torch.ops.aten.glu_backward_jvp(
        tu.to_reference(grad_x),
        tu.to_reference(grad_glu),
        tu.to_reference(x),
        tu.to_reference(dgrad_glu),
        tu.to_reference(dx),
        0,
    )
    res_out = flag_gems.glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, 0)

    tu.assert_result_close(res_out, ref_out)


INVALID_DTYPE_CASES = [(torch.bool, RuntimeError)]
if FP8_REJECTED:
    INVALID_DTYPE_CASES += [
        (torch.float8_e4m3fn, RuntimeError),
        (torch.float8_e5m2, RuntimeError),
    ]


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("dtype,error", INVALID_DTYPE_CASES)
def test_glu_backward_jvp_invalid_dtype(dtype, error):
    shape, dim = (1024, 1024), 1
    half = _half_shape(shape, dim)
    operands = [
        torch.zeros(size, dtype=dtype, device=flag_gems.device)
        for size in (shape, half, shape, half, shape)
    ]

    with pytest.raises(error):
        flag_gems.glu_backward_jvp(*operands, dim)


INVALID_OPERAND_CASES = [
    ((2, 3, 8), 1, (1, 2, 1)),
    ((4, 3, 8), 1, (4, 2, 8)),
]


@pytest.mark.glu_backward_jvp
@pytest.mark.parametrize("shape,dim,grad_shape", INVALID_OPERAND_CASES)
def test_glu_backward_jvp_invalid_operand(shape, dim, grad_shape):
    # grad_glu / dgrad_glu must be 1 or exactly half the split axis wide; the
    # native operator rejects any other extent in the slice it takes on x.
    dtype = torch.float32
    grad_x, x, dx = _tensors(dtype, (shape, shape, shape), ["-1", "1"])
    grad_glu, dgrad_glu = _tensors(dtype, (grad_shape, grad_shape), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems.glu_backward_jvp(grad_x, grad_glu, x, dgrad_glu, dx, dim)

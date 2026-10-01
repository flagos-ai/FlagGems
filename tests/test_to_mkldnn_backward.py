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

# aten::to_mkldnn_backward converts grad to dense with input.dtype. A dense
# same-dtype grad is returned by identity; an opaque oneDNN grad is materialized
# on CPU. Only input.dtype/layout matter, not its shape or device. The input
# must be strided, and opaque grad conversion follows oneDNN dtype restrictions.

_DEVICE_DTYPE_FLAG = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
    torch.int64: "support_int64",
}


def _dtype_supported(dtype):
    """Static backend capability gate; no operator call is made at collection."""
    flag = _DEVICE_DTYPE_FLAG.get(dtype)
    return True if flag is None else bool(getattr(flag_gems.runtime.device, flag))


# The nine required dtypes plus bool/float64/complex where the backend has them.
GRID_DTYPES = [
    dtype
    for dtype in list(tu.REQUIRED_DTYPES)
    + [torch.bool, torch.float64, torch.complex64, torch.complex128]
    if _dtype_supported(dtype)
]

# Every pair differs, so each row exercises the fresh-cast branch.
CONVERSION_PAIRS = [
    (torch.float32, torch.float16),
    (torch.float32, torch.bfloat16),
    (torch.float32, torch.float64),
    (torch.float32, torch.int32),
    (torch.float32, torch.int64),
    (torch.float32, torch.int8),
    (torch.float32, torch.uint8),
    (torch.float32, torch.bool),
    (torch.float32, torch.complex64),
    (torch.float32, torch.float8_e4m3fn),
    (torch.float32, torch.float8_e5m2),
    (torch.float16, torch.float32),
    (torch.float16, torch.int32),
    (torch.bfloat16, torch.float32),
    (torch.int32, torch.int64),
    (torch.int64, torch.int8),
    (torch.int8, torch.uint8),
    (torch.complex64, torch.complex128),
    (torch.complex64, torch.float32),
    (torch.float64, torch.float32),
]
CONVERSION_PAIRS = [
    pair for pair in CONVERSION_PAIRS if all(_dtype_supported(dt) for dt in pair)
]

CAST_SHAPE = (20, 320, 15)

LAYOUTS = [
    "contiguous",
    "scalar",
    "empty",
    "storage offset",
    "stepped",
    "transposed",
    "channels last",
    "expanded",
]

# `input` shapes are unrelated to `grad` shapes: only input.dtype is read.
RANK_SIZE_CASES = [
    ((2, 3), (5,)),
    ((3,), (2, 5)),
    ((1024, 1024), (16,)),
    ((7,), (7, 5, 3)),
    ((), (4,)),
    ((2, 3), ()),
]

CROSS_DEVICE_SAME_DTYPE = [torch.float32, torch.int32]
CROSS_DEVICE_CAST = [(torch.float32, torch.float16), (torch.int32, torch.int64)]

# CPU fixtures are allocated on the CPU directly and are not gated by
# accelerator dtype flags, because the operator is a pure dtype read.
CPU_CAST_PAIRS = [
    (torch.float32, torch.float16),
    (torch.float32, torch.float64),
    (torch.int32, torch.int64),
]

SPECIAL_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _dtype_supported(dtype)
]
SPECIAL_SCENARIOS = ["nan", "inf", "mixed"]
# e4m3fn has no inf, so its inf/mixed fp32 sources become NaN on both sides;
# e5m2 keeps inf.
SPECIAL_CAST_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _dtype_supported(dtype)
]

BACKWARD_PAIRS = [
    pair
    for pair in [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.float32, torch.float8_e4m3fn),
        (torch.float32, torch.float8_e5m2),
        (torch.complex64, torch.complex128),
    ]
    if all(_dtype_supported(dt) for dt in pair)
]

NEGATIVE_FORMS = ["non-tensor grad", "non-tensor input", "missing input"]


def _layout_grad(layout):
    """Grad tensors with non-trivial layout; the identity branch must not touch them."""
    dtype = torch.float32
    if layout == "contiguous":
        return tu.make_input(dtype, (8, 16, 4), ["-1", "1"])
    if layout == "scalar":
        return tu.make_input(dtype, (), ["-1", "1"])
    if layout == "empty":
        return tu.make_input(dtype, (4, 0), ["-1", "1"])
    if layout == "storage offset":
        return tu.make_input(dtype, (8, 32), ["-1", "1"])[:, 4:20]
    if layout == "stepped":
        return tu.make_input(dtype, (8, 64), ["-1", "1"])[:, ::3]
    if layout == "transposed":
        return tu.make_input(dtype, (16, 8), ["-1", "1"]).t()
    if layout == "channels last":
        return tu.make_input(dtype, (2, 8, 4, 4), ["-1", "1"]).to(
            memory_format=torch.channels_last
        )
    if layout == "expanded":
        return tu.make_input(dtype, (1, 6), ["-1", "1"]).expand(8, 6)
    raise ValueError(f"unknown layout {layout}")


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", GRID_DTYPES)
def test_to_mkldnn_backward_matching_dtype_returns_grad(shape, value_range, dtype):
    grad = tu.make_input(dtype, shape, value_range)
    inp = tu.make_input(dtype, shape, value_range)
    grad_before = tu.to_reference(grad)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out is grad
    tu.assert_result_equal(res_out, ref_out)
    # The operator only reads input.dtype, so neither operand may be written.
    tu.assert_result_equal(grad, grad_before)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("grad_dtype,input_dtype", CONVERSION_PAIRS)
def test_to_mkldnn_backward_dtype_mismatch_follows_input(grad_dtype, input_dtype):
    grad = tu.make_input(grad_dtype, CAST_SHAPE, ["-1", "1"])
    inp = tu.make_input(input_dtype, CAST_SHAPE, ["0", "1"])
    grad_before = tu.to_reference(grad)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.dtype == input_dtype
    assert res_out.shape == grad.shape
    assert res_out.device == grad.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, grad_before)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("layout", LAYOUTS)
def test_to_mkldnn_backward_returns_grad_layout_unchanged(layout):
    grad = _layout_grad(layout)
    inp = tu.make_input(torch.float32, (4, 4), ["0", "1"])

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out is grad
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("grad_shape,input_shape", RANK_SIZE_CASES)
def test_to_mkldnn_backward_ignores_operand_shape(grad_shape, input_shape):
    grad = tu.make_input(torch.float32, grad_shape, ["-1", "1"])
    inp = tu.make_input(torch.float32, input_shape, ["0", "1"])
    grad_before = tu.to_reference(grad)
    inp_before = tu.to_reference(inp)

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out is grad
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, grad_before)
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize(
    "grad_shape,input_shape",
    [((2, 3), (5,)), ((3,), (2, 5)), ((1024, 1024), (16,)), ((2, 3), ())],
)
def test_to_mkldnn_backward_shape_mismatch_with_dtype_cast(grad_shape, input_shape):
    grad = tu.make_input(torch.float32, grad_shape, ["-1", "1"])
    inp = tu.make_input(torch.float16, input_shape, ["0", "1"])
    grad_before = tu.to_reference(grad)

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.dtype == torch.float16
    assert res_out.shape == grad.shape
    assert res_out.device == grad.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad, grad_before)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("dtype", CROSS_DEVICE_SAME_DTYPE)
def test_to_mkldnn_backward_cpu_input_matching_dtype(dtype):
    # `input` lives on the CPU while `grad` stays on the device.
    grad = tu.make_input(dtype, CAST_SHAPE, ["-1", "1"])
    inp = tu.make_input(dtype, CAST_SHAPE, ["0", "1"]).cpu()

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out is grad
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("grad_dtype,input_dtype", CROSS_DEVICE_CAST)
def test_to_mkldnn_backward_cpu_input_casts_on_grad_device(grad_dtype, input_dtype):
    grad = tu.make_input(grad_dtype, CAST_SHAPE, ["-1", "1"])
    inp = tu.make_input(input_dtype, CAST_SHAPE, ["0", "1"]).cpu()

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.dtype == input_dtype
    assert res_out.shape == grad.shape
    assert res_out.device == grad.device
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
def test_to_mkldnn_backward_cpu_operands_matching_dtype():
    grad = torch.testing.make_tensor(
        CAST_SHAPE, dtype=torch.float32, device="cpu", low=-1.0, high=1.0
    )
    inp = torch.testing.make_tensor(
        CAST_SHAPE, dtype=torch.float32, device="cpu", low=0.0, high=1.0
    )

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out is grad
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("grad_dtype,input_dtype", CPU_CAST_PAIRS)
def test_to_mkldnn_backward_cpu_operands_cast(grad_dtype, input_dtype):
    grad = torch.testing.make_tensor(
        CAST_SHAPE, dtype=grad_dtype, device="cpu", low=-1.0, high=1.0
    )
    inp = torch.testing.make_tensor(
        CAST_SHAPE, dtype=input_dtype, device="cpu", low=0.0, high=1.0
    )

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.dtype == input_dtype
    assert res_out.device == grad.device
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
def test_to_mkldnn_backward_result_is_grad_storage():
    grad = tu.make_input(torch.float32, (4, 8), ["-1", "1"])
    inp = tu.make_input(torch.float32, (4, 8), ["0", "1"])

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)
    tu.assert_result_equal(res_out, ref_out)

    written = torch.full((4, 8), -2.5, dtype=torch.float32, device=grad.device)
    res_out.copy_(written)
    tu.assert_result_equal(grad, written)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(SPECIAL_DTYPES), quick=[]),
)
def test_to_mkldnn_backward_special_values_pass_through(dtype, scenario):
    grad = tu.make_special_input(dtype, scenario)
    inp = tu.make_special_input(dtype, scenario)

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out is grad
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize(
    "scenario,input_dtype",
    tu.selected_cases(
        [
            (scenario, dtype)
            for dtype in SPECIAL_CAST_DTYPES
            for scenario in SPECIAL_SCENARIOS
        ],
        quick=[],
    ),
)
def test_to_mkldnn_backward_special_values_follow_input_dtype(scenario, input_dtype):
    grad = tu.make_special_input(torch.float32, scenario)
    inp = tu.make_input(input_dtype, (4, 4), ["-1", "1"])

    ref_out = torch.ops.aten.to_mkldnn_backward(
        tu.to_reference(grad), tu.to_reference(inp)
    )
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.dtype == input_dtype
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize(
    "grad_dtype,input_dtype",
    tu.selected_cases(BACKWARD_PAIRS, quick=[]),
)
def test_to_mkldnn_backward_backward(grad_dtype, input_dtype):
    shape = (16, 128)
    grad = tu.make_input(grad_dtype, shape, ["-1", "1"]).requires_grad_(True)
    inp = tu.make_input(input_dtype, shape, ["0", "1"])

    res_out = flag_gems.to_mkldnn_backward(grad, inp)
    assert res_out.requires_grad

    # A non-uniform upstream gradient so neither a constant nor a broadcast can
    # pass by accident.
    upstream = tu.make_input(res_out.dtype, shape, ["-1", "1"])
    res_grad = torch.autograd.grad(res_out, grad, grad_outputs=upstream)[0]

    ref_leaf = tu.to_reference(grad.detach()).requires_grad_(True)
    ref_out = torch.ops.aten.to_mkldnn_backward(ref_leaf, tu.to_reference(inp))
    tu.assert_result_equal(res_out, ref_out)
    ref_grad = torch.autograd.grad(
        ref_out, ref_leaf, grad_outputs=tu.to_reference(upstream)
    )[0]

    assert res_grad.dtype == grad.dtype
    assert res_grad.shape == grad.shape
    tu.assert_result_equal(res_grad, ref_grad)
    # A pure dtype cast differentiates to the upstream gradient cast back.
    tu.assert_result_equal(res_grad, upstream.to(grad.dtype))


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("form", NEGATIVE_FORMS)
def test_to_mkldnn_backward_rejects_invalid_arguments(form):
    grad = tu.make_input(torch.float32, (4, 8), ["-1", "1"])

    if form == "non-tensor grad":
        with pytest.raises((RuntimeError, TypeError)):
            flag_gems.to_mkldnn_backward([1.0, 2.0], grad)
    elif form == "non-tensor input":
        with pytest.raises((RuntimeError, TypeError)):
            flag_gems.to_mkldnn_backward(grad, "input")
    else:
        with pytest.raises((RuntimeError, TypeError)):
            flag_gems.to_mkldnn_backward(grad)


MKLDNN_DTYPES = [torch.float32, torch.float16, torch.bfloat16, torch.int8, torch.uint8]
MKLDNN_CAST_PAIRS = [
    (source, target)
    for source in MKLDNN_DTYPES[:3]
    for target in MKLDNN_DTYPES
    if source != target
]


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("dtype", MKLDNN_DTYPES)
@pytest.mark.parametrize(
    "shape", [s for s in tu.selected_shapes() if s] + [(0,), (0, 3)]
)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_to_mkldnn_backward_opaque_grad(dtype, shape, value_range):
    dense = tu.make_input(dtype, shape, value_range).cpu()
    grad = dense.to_mkldnn()
    ref_grad = dense.clone().to_mkldnn()
    inp = torch.zeros((7,), dtype=dtype, device="cpu")

    ref_out = torch.ops.aten.to_mkldnn_backward(ref_grad, inp)
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.layout == torch.strided
    assert res_out.device == grad.device
    assert res_out is not grad
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad.to_dense(), dense)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize("source,target", MKLDNN_CAST_PAIRS)
def test_to_mkldnn_backward_opaque_dtype_conversion(source, target):
    dense = tu.make_input(source, (2, 3), ["-1", "1"]).cpu()
    grad = dense.to_mkldnn()
    ref_grad = dense.clone().to_mkldnn()
    inp = torch.zeros((7,), dtype=target, device="cpu")

    ref_out = torch.ops.aten.to_mkldnn_backward(ref_grad, inp)
    res_out = flag_gems.to_mkldnn_backward(grad, inp)

    assert res_out.layout == torch.strided
    assert res_out.device == grad.device
    tu.assert_result_equal(res_out, ref_out)
    tu.assert_result_equal(grad.to_dense(), dense)


@pytest.mark.to_mkldnn_backward
@pytest.mark.parametrize(
    "source,target",
    [
        (torch.float32, torch.float64),
        (torch.float32, torch.int32),
        (torch.float32, torch.int64),
        (torch.float32, torch.bool),
        (torch.float32, torch.complex64),
        (torch.float32, torch.float8_e4m3fn),
        (torch.float32, torch.float8_e5m2),
        (torch.int8, torch.float32),
        (torch.uint8, torch.float32),
        (torch.int8, torch.uint8),
    ],
)
def test_to_mkldnn_backward_rejects_opaque_dtype_conversion(source, target):
    grad = torch.ones(2, 3, dtype=source, device="cpu").to_mkldnn()
    inp = torch.empty(7, dtype=target, device="cpu")
    with pytest.raises(RuntimeError):
        flag_gems.to_mkldnn_backward(grad, inp)


@pytest.mark.to_mkldnn_backward
def test_to_mkldnn_backward_rejects_opaque_input():
    grad = torch.ones(2, 3, device="cpu").to_mkldnn()
    inp = torch.ones(7, device="cpu").to_mkldnn()
    with pytest.raises(RuntimeError):
        flag_gems.to_mkldnn_backward(grad, inp)

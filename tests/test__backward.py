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

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# aten::_backward is the AutogradBackward executor: it runs the autograd engine
# from an existing graph root, fills `.grad` on the listed inputs and returns
# None. Every workload builds a real forward graph from a leaf, so the operator
# is never asked to differentiate a root with respect to itself.
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}

# Dtypes an engine leaf can carry here. Two native limits shape the graphs: an
# fp8 leaf joins through a differentiable float32 cast because "sin_cuda" is not
# implemented for Float8 (its accumulated gradient is still fp8), and a complex
# graph needs a real-valued root, so complex roots are reduced with .real.
_GRAD_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float64,
    torch.complex64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_GRAD_DTYPES = [dtype for dtype in _GRAD_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Cannot require grad; covered by the negative rows below.
_NON_GRAD_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.int64, torch.bool]

_NON_GRAD_DTYPES = [
    dtype for dtype in _NON_GRAD_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]

# Replaying a graph adds the same gradient again. Float8 gradients have no add
# kernel on this backend ("ufunc_add_CUDA" not implemented for Float8).
_ACCUM_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
]

_ACCUM_DTYPES = [dtype for dtype in _ACCUM_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# Executing second order reduces over the first gradient, and float8 has no
# reduction here ("Promotion for Float8 Types is not supported"). The
# create_graph flag itself stays covered for every dtype by the grad-state row.
_SECOND_ORDER_DTYPES = tu.selected_cases(
    [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.complex64],
    quick=[],
)

# One 1-D operand pair and one multi-dim operand pair at two large spec shapes.
_BROADCAST_CASES = [
    ((20, 320, 15), (15,)),
    ((16, 128, 64, 60), (1, 128, 1, 60)),
]


def _cast(leaf):
    """Values the elementwise graph sees for a dtype without kernels of its own."""
    return leaf.float() if leaf.dtype in _FP8_DTYPES else leaf


def _leaf(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)
    return inp, ref_inp


def _vector_output(leaf):
    """Vector-valued graph; a complex output keeps its dtype so an explicit
    complex cotangent stays schema-compatible."""
    return torch.sin(_cast(leaf)) * 2.0


def _scalar_root(leaf):
    """Scalar root of the same graph; a complex output needs a real root."""
    out = _vector_output(leaf)
    return (out.real if out.is_complex() else out).sum()


def _product_root(a, b):
    """Scalar root of sin(a) * b, used by the graphs over two operands."""
    out = torch.sin(_cast(a)) * _cast(b)
    return (out.real if out.is_complex() else out).sum()


def _real(grad):
    """Real-valued observer of a possibly complex gradient."""
    return grad.real if grad.is_complex() else grad


@pytest.mark.backward
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_leaf_grad(shape, value_range, dtype):
    inp, ref_inp = _leaf(dtype, shape, value_range)
    ref_out = _scalar_root(ref_inp)
    res_out = _scalar_root(inp)

    torch.ops.aten._backward(ref_out, [ref_inp])
    res_ret = flag_gems._backward(res_out, [inp])

    assert res_ret is None
    tu.assert_result_close(inp.grad, ref_inp.grad)


@pytest.mark.backward
@pytest.mark.parametrize("shape", [(2, 3, 5), (16, 32)])
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_explicit_gradient(shape, dtype):
    # `gradient` supplies the cotangent of the non-scalar root and must match
    # its dtype: float32 for the fp8 cast, complex for a complex root.
    inp, ref_inp = _leaf(dtype, shape, ["-1", "1"])
    ref_out = _vector_output(ref_inp)
    res_out = _vector_output(inp)
    grad_dtype = torch.float32 if dtype in _FP8_DTYPES else dtype
    grad = tu.make_input(grad_dtype, shape, ["-1", "1"])
    ref_grad = tu.to_reference(grad)

    torch.ops.aten._backward(ref_out, [ref_inp], ref_grad)
    res_ret = flag_gems._backward(res_out, [inp], grad)

    assert res_ret is None
    tu.assert_result_close(inp.grad, ref_inp.grad)


@pytest.mark.backward
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_keyword_arguments(dtype):
    inp, ref_inp = _leaf(dtype, (256,), ["-1", "1"])
    ref_out = _scalar_root(ref_inp)
    res_out = _scalar_root(inp)

    torch.ops.aten._backward(
        self=ref_out,
        inputs=[ref_inp],
        gradient=None,
        retain_graph=True,
        create_graph=False,
    )
    flag_gems._backward(
        self=res_out,
        inputs=[inp],
        gradient=None,
        retain_graph=True,
        create_graph=False,
    )

    tu.assert_result_close(inp.grad, ref_inp.grad)


@pytest.mark.backward
@pytest.mark.parametrize("shape", [(256,), (16, 32)])
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_multiple_inputs(shape, dtype):
    inp_a, ref_a = _leaf(dtype, shape, ["-1", "1"])
    inp_b, ref_b = _leaf(dtype, shape, ["0", "1"])
    ref_out = _product_root(ref_a, ref_b)
    res_out = _product_root(inp_a, inp_b)

    torch.ops.aten._backward(ref_out, [ref_a, ref_b])
    flag_gems._backward(res_out, [inp_a, inp_b])

    tu.assert_result_close(inp_a.grad, ref_a.grad)
    tu.assert_result_close(inp_b.grad, ref_b.grad)


@pytest.mark.backward
@pytest.mark.parametrize("shape", [(256,), (16, 32)])
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_non_leaf_input(shape, dtype):
    inp, ref_inp = _leaf(dtype, shape, ["-1", "1"])
    mid = _cast(inp) * 3.0
    ref_mid = _cast(ref_inp) * 3.0
    ref_out = _scalar_root(ref_mid)
    res_out = _scalar_root(mid)

    torch.ops.aten._backward(ref_out, [ref_mid])
    flag_gems._backward(res_out, [mid])

    tu.assert_result_close(mid.grad, ref_mid.grad)
    # The engine stops at the requested input and does not reach the leaf.
    assert inp.grad is None


@pytest.mark.backward
@pytest.mark.parametrize(
    "shape, broadcast_shape",
    tu.selected_cases(_BROADCAST_CASES, quick=[_BROADCAST_CASES[0]]),
)
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_broadcast_gradients(shape, broadcast_shape, dtype):
    inp, ref_inp = _leaf(dtype, shape, ["-1", "1"])
    other, ref_other = _leaf(dtype, broadcast_shape, ["-1", "1"])
    ref_out = _product_root(ref_inp, ref_other)
    res_out = _product_root(inp, other)

    torch.ops.aten._backward(ref_out, [ref_inp, ref_other])
    flag_gems._backward(res_out, [inp, other])

    # The broadcast operand must receive a gradient reduced back to its shape.
    assert other.grad.shape == other.shape
    tu.assert_result_close(inp.grad, ref_inp.grad)
    tu.assert_result_close(other.grad, ref_other.grad)


@pytest.mark.backward
@pytest.mark.parametrize("dtype", _GRAD_DTYPES)
def test__backward_create_graph_grad_state(dtype):
    # create_graph=True must leave a differentiable `.grad`; executing the
    # second-order pass is the default-only workload below.
    inp, ref_inp = _leaf(dtype, (256,), ["-1", "1"])
    ref_out = _scalar_root(ref_inp)
    res_out = _scalar_root(inp)

    torch.ops.aten._backward(ref_out, [ref_inp], None, True, True)
    flag_gems._backward(res_out, [inp], None, True, True)

    assert inp.grad.grad_fn is not None
    tu.assert_result_close(inp.grad, ref_inp.grad)


@pytest.mark.backward
@pytest.mark.parametrize("dtype", _SECOND_ORDER_DTYPES)
def test__backward_second_order_execution(dtype):
    inp, ref_inp = _leaf(dtype, (256,), ["-1", "1"])
    ref_out = _product_root(ref_inp, ref_inp)
    res_out = _product_root(inp, inp)

    torch.ops.aten._backward(ref_out, [ref_inp], None, True, True)
    flag_gems._backward(res_out, [inp], None, True, True)

    assert inp.grad.grad_fn is not None
    res_second = torch.autograd.grad(_real(inp.grad).sum(), inp)[0]
    ref_second = torch.autograd.grad(_real(ref_inp.grad).sum(), ref_inp)[0]
    tu.assert_result_close(res_second, ref_second)


@pytest.mark.backward
@pytest.mark.parametrize("dtype", _ACCUM_DTYPES)
def test__backward_accumulates_on_repeat(dtype):
    # retain_graph replays the same graph; every pass adds to `.grad` again.
    inp, ref_inp = _leaf(dtype, (64,), ["-1", "1"])
    ref_out = _scalar_root(ref_inp)
    res_out = _scalar_root(inp)

    for _ in range(2):
        torch.ops.aten._backward(ref_out, [ref_inp], None, True)
    for _ in range(2):
        flag_gems._backward(res_out, [inp], None, True)

    tu.assert_result_close(inp.grad, ref_inp.grad)


@pytest.mark.backward
@pytest.mark.parametrize(
    "dtype, scenario", tu.selected_cases(tu.special_value_cases(_GRAD_DTYPES), quick=[])
)
def test__backward_special_values(dtype, scenario):
    values = tu.make_special_input(dtype, scenario)
    ref_values = tu.to_reference(values.detach()).requires_grad_(True)
    values = values.requires_grad_(True)
    ref_out = _scalar_root(ref_values)
    res_out = _scalar_root(values)

    torch.ops.aten._backward(ref_out, [ref_values])
    flag_gems._backward(res_out, [values])

    tu.assert_result_close(values.grad, ref_values.grad)


@pytest.mark.backward
def test__backward_empty_inputs_runs_full_pass():
    # Native probe: an empty `inputs` list still runs the whole pass and
    # accumulates on the graph leaf; the list only names the intermediates that
    # receive retain_grad.
    inp, ref_inp = _leaf(torch.float32, (256,), ["-1", "1"])
    ref_out = _scalar_root(ref_inp)
    res_out = _scalar_root(inp)

    torch.ops.aten._backward(ref_out, [])
    res_ret = flag_gems._backward(res_out, [])

    assert res_ret is None
    tu.assert_result_close(inp.grad, ref_inp.grad)


@pytest.mark.backward
def test__backward_unrelated_inputs_leave_leaf_untouched():
    # Native probe: naming a tensor that is not part of the graph populates no
    # gradient, not even on the graph's own leaf.
    inp, ref_inp = _leaf(torch.float32, (256,), ["-1", "1"])
    ref_out = _scalar_root(ref_inp)
    res_out = _scalar_root(inp)
    other = tu.make_input(torch.float32, (256,), ["-1", "1"]).requires_grad_(True)

    torch.ops.aten._backward(ref_out, [other])
    flag_gems._backward(res_out, [other])

    assert inp.grad is None
    assert other.grad is None


@pytest.mark.backward
@pytest.mark.parametrize("dtype", _NON_GRAD_DTYPES)
def test__backward_rejects_input_without_grad(dtype):
    inp = tu.make_input(torch.float32, (256,), ["-1", "1"]).requires_grad_(True)
    root = (torch.sin(inp) * 2.0).sum()
    other = tu.make_input(dtype, (256,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._backward(root, [other])


@pytest.mark.backward
def test__backward_rejects_root_without_graph():
    inp = tu.make_input(torch.float32, (256,), ["-1", "1"]).requires_grad_(True)
    root = tu.make_input(torch.float32, (256,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._backward(root, [inp])


@pytest.mark.backward
def test__backward_rejects_non_scalar_root_without_gradient():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"]).requires_grad_(True)
    root = torch.sin(inp) * 2.0

    with pytest.raises(RuntimeError):
        flag_gems._backward(root, [inp])


@pytest.mark.backward
def test__backward_rejects_mismatched_gradient_shape():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"]).requires_grad_(True)
    root = torch.sin(inp) * 2.0
    grad = tu.make_input(torch.float32, (3,), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._backward(root, [inp], grad)


@pytest.mark.backward
def test__backward_rejects_non_tensor_gradient():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"]).requires_grad_(True)
    root = (torch.sin(inp) * 2.0).sum()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._backward(root, [inp], [1.0])


@pytest.mark.backward
def test__backward_rejects_plain_tensor_inputs():
    inp = tu.make_input(torch.float32, (4,), ["-1", "1"]).requires_grad_(True)
    root = (torch.sin(inp) * 2.0).sum()

    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._backward(root, inp)


@pytest.mark.backward
def test__backward_rejects_freed_graph_reuse():
    inp = tu.make_input(torch.float32, (256,), ["-1", "1"]).requires_grad_(True)
    root = (torch.sin(inp) * 2.0).sum()

    flag_gems._backward(root, [inp], None, False)
    with pytest.raises(RuntimeError):
        flag_gems._backward(root, [inp], None, False)

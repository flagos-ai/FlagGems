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

_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.complex128: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}

# aten::is_leaf(Tensor self) -> bool reports the autograd state of the exact
# tensor object it is handed: True when the tensor carries no grad_fn (a fresh
# tensor, a clone, a detach result, or a view taken while grad tracking is off --
# a view shares storage and is still a leaf as long as no graph records it),
# False for the result of a recorded differentiable operation. No element value
# is read and nothing is written back, so the value grid only shows the answer is
# independent of shape, dtype and stored range; the state rows carry the autograd
# coverage.
#
# The oracle runs on the very same tensor the candidate receives: a
# tu.to_reference copy clones storage and drops the graph state under test.
# Because the read is non-destructive, each row snapshots the input's identity,
# layout and autograd state and requires them unchanged - a candidate that
# rebuilt or re-graphed its input could still return the same bool.
#
# Broadcast, tensor-vs-scalar, dim/param and output-shape dimensions do not
# apply: one tensor in, one Python bool out.

_STATE_SHAPE = (4, 6)

_EXTRA_DTYPES = [torch.bool, torch.complex64]
if utils.fp64_is_supported:
    _EXTRA_DTYPES += [torch.float64, torch.complex128]
_IS_LEAF_DTYPES = tu.REQUIRED_DTYPES + _EXTRA_DTYPES
_IS_LEAF_DTYPES = [dtype for dtype in _IS_LEAF_DTYPES if _DTYPE_FLAGS.get(dtype, True)]

# PyTorch accepts requires_grad only on floating/complex tensors (int32 and bool
# raise "only Tensors of floating point dtype can require gradients"), so the
# grad_fn rows below run for those dtypes; int/bool keep the leaf rows.
_TRACKED_DTYPES = [
    dtype for dtype in _IS_LEAF_DTYPES if dtype.is_floating_point or dtype.is_complex
]

# The arithmetic rows need an elementwise or reduction kernel: on CUDA aten::mul
# raises 'mul_cuda' not implemented for Float8_e4m3fn / Float8_e5m2 and aten::sum
# raises 'sum_cuda' not implemented for Float8_e4m3fn / Float8_e5m2. FP8 keeps
# every metadata-only state and is excluded only from these arithmetic rows.
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_ARITH_DTYPES = [dtype for dtype in _TRACKED_DTYPES if dtype not in _FP8_DTYPES]


def _snapshot(tensor):
    # Identity, layout and autograd state the operator may observe or disturb.
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.requires_grad,
        tensor.grad_fn is not None,
        tensor.is_leaf,
    )


def _plain(dtype):
    return tu.make_input(dtype, _STATE_SHAPE, ["-1", "1"])


def _view(dtype):
    return _plain(dtype).view(24)


def _transpose(dtype):
    return _plain(dtype).t()


def _narrow(dtype):
    return _plain(dtype).narrow(0, 1, 2)


def _expand(dtype):
    return _plain(dtype)[:1].expand(3, 6)


def _clone(dtype):
    return _plain(dtype).clone()


def _cat(dtype):
    operand = _plain(dtype)
    return torch.cat([operand, operand], dim=0)


def _chunk(dtype):
    return _plain(dtype).chunk(2, dim=0)[1]


def _detach(dtype):
    return _plain(dtype).detach()


def _to_contiguous(dtype):
    return _plain(dtype).t().contiguous()


# A view, a clone or a detach result is a leaf whenever no graph records it,
# whatever the layout and dtype.
_LEAF_STATES = [
    pytest.param(_plain, id="plain"),
    pytest.param(_view, id="view"),
    pytest.param(_transpose, id="transpose"),
    pytest.param(_narrow, id="narrow"),
    pytest.param(_expand, id="expand"),
    pytest.param(_clone, id="clone"),
    pytest.param(_cat, id="cat"),
    pytest.param(_chunk, id="chunk"),
    pytest.param(_detach, id="detach"),
    pytest.param(_to_contiguous, id="to_contiguous"),
]


def _grad_source(dtype):
    return _plain(dtype).requires_grad_(True)


def _grad_view(dtype):
    return _grad_source(dtype).view(24)


def _grad_transpose(dtype):
    return _grad_source(dtype).t()


def _grad_expand(dtype):
    return _grad_source(dtype)[:1].expand(3, 6)


def _grad_clone(dtype):
    return _grad_source(dtype).clone()


def _out_of_place_detach(dtype):
    return _grad_view(dtype).detach()


def _in_place_detach(dtype):
    # detach_() is rejected on a view ("Can't detach views in-place"), so the
    # in-place row detaches a clone, a non-view grad_fn result.
    operand = _grad_clone(dtype)
    operand.detach_()
    return operand


def _data_alias(dtype):
    return _grad_view(dtype).data


def _detach_then_requires_grad(dtype):
    return _grad_view(dtype).detach().requires_grad_(True)


def _parameter(dtype):
    return torch.nn.Parameter(_grad_source(dtype))


def _no_grad_view(dtype):
    operand = _grad_source(dtype)
    with torch.no_grad():
        return operand.view(24)


def _inference_mode_view(dtype):
    operand = _grad_source(dtype)
    with torch.inference_mode():
        return operand.view(24)


# Autograd states that need no elementwise kernel: tracking leaves, view nodes,
# and the operations that strip the graph again (detach/.data/no_grad).
_GRAPH_STATES = [
    pytest.param(_grad_source, True, id="leaf_requires_grad"),
    pytest.param(_grad_view, False, id="nonleaf_view"),
    pytest.param(_grad_transpose, False, id="nonleaf_transpose"),
    pytest.param(_grad_expand, False, id="nonleaf_expand"),
    pytest.param(_grad_clone, False, id="nonleaf_clone"),
    pytest.param(_out_of_place_detach, True, id="detach_out_of_place"),
    pytest.param(_in_place_detach, True, id="detach_in_place"),
    pytest.param(_data_alias, True, id="data_alias"),
    pytest.param(_detach_then_requires_grad, True, id="detach_then_requires_grad"),
    pytest.param(_parameter, True, id="parameter"),
    pytest.param(_no_grad_view, True, id="no_grad_view"),
    pytest.param(_inference_mode_view, True, id="inference_mode_view"),
]


def _mul(dtype):
    return _grad_source(dtype) * 2


def _sum(dtype):
    return _grad_source(dtype).sum()


def _grad_cumsum(dtype):
    return _grad_source(dtype).cumsum(0)


def _grad_pow(dtype):
    return _grad_source(dtype) ** 2


def _no_grad_mul(dtype):
    operand = _grad_source(dtype)
    with torch.no_grad():
        return operand * 2


def _inference_mode_mul(dtype):
    operand = _plain(dtype)
    with torch.inference_mode():
        return operand * 2


_ARITH_STATES = [
    pytest.param(_mul, False, id="nonleaf_mul"),
    pytest.param(_sum, False, id="nonleaf_sum"),
    pytest.param(_grad_cumsum, False, id="nonleaf_cumsum"),
    pytest.param(_grad_pow, False, id="nonleaf_pow"),
    pytest.param(_no_grad_mul, True, id="no_grad_mul"),
    pytest.param(_inference_mode_mul, True, id="inference_mode_mul"),
]


@pytest.mark.is_leaf
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _IS_LEAF_DTYPES)
def test_is_leaf(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    before = _snapshot(inp)

    ref = torch.ops.aten.is_leaf(inp)
    res = flag_gems.is_leaf(inp)

    assert type(res) is bool
    assert res is True
    assert res == ref
    assert _snapshot(inp) == before


@pytest.mark.is_leaf
@pytest.mark.parametrize("factory", _LEAF_STATES)
@pytest.mark.parametrize("dtype", _IS_LEAF_DTYPES)
def test_is_leaf_leaf_state(factory, dtype):
    inp = factory(dtype)
    before = _snapshot(inp)

    ref = torch.ops.aten.is_leaf(inp)
    res = flag_gems.is_leaf(inp)

    assert type(res) is bool
    assert res is True
    assert res == ref
    assert _snapshot(inp) == before


@pytest.mark.is_leaf
@pytest.mark.parametrize("factory,expected", _GRAPH_STATES)
@pytest.mark.parametrize("dtype", _TRACKED_DTYPES)
def test_is_leaf_graph_state(factory, expected, dtype):
    inp = factory(dtype)
    before = _snapshot(inp)

    ref = torch.ops.aten.is_leaf(inp)
    res = flag_gems.is_leaf(inp)

    assert type(res) is bool
    assert res is expected
    assert res == ref
    assert _snapshot(inp) == before


@pytest.mark.is_leaf
@pytest.mark.parametrize("factory,expected", _ARITH_STATES)
@pytest.mark.parametrize("dtype", _ARITH_DTYPES)
def test_is_leaf_arithmetic_state(factory, expected, dtype):
    inp = factory(dtype)
    before = _snapshot(inp)

    ref = torch.ops.aten.is_leaf(inp)
    res = flag_gems.is_leaf(inp)

    assert type(res) is bool
    assert res is expected
    assert res == ref
    assert _snapshot(inp) == before


@pytest.mark.is_leaf
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_IS_LEAF_DTYPES), quick=[]),
)
def test_is_leaf_special_values(dtype, scenario):
    # No kernel reads these payloads; they show the reported state survives
    # unusual stored values.
    inp = tu.make_special_input(dtype, scenario)
    before = _snapshot(inp)

    ref = torch.ops.aten.is_leaf(inp)
    res = flag_gems.is_leaf(inp)

    assert type(res) is bool
    assert res is True
    assert res == ref
    assert _snapshot(inp) == before


@pytest.mark.is_leaf
@pytest.mark.parametrize("dtype", tu.selected_cases(_ARITH_DTYPES, quick=[]))
def test_is_leaf_backward(dtype):
    # is_leaf has no derivative of its own (its output is a Python bool). Its
    # backward contract is that the reported state marks the object autograd
    # accumulates into, so the loss is differentiated with respect to the
    # original leaf, never with respect to the tensor whose state is read.
    inp = tu.make_input(dtype, _STATE_SHAPE, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp.detach()).requires_grad_(True)

    ref_leaf = torch.ops.aten.is_leaf(inp)
    res_leaf = flag_gems.is_leaf(inp)
    assert type(res_leaf) is bool
    assert res_leaf is True
    assert res_leaf == ref_leaf

    loss = (inp * inp).sum()
    ref_loss = (ref_inp * ref_inp).sum()
    ref_loss_leaf = torch.ops.aten.is_leaf(loss)
    res_loss_leaf = flag_gems.is_leaf(loss)
    assert type(res_loss_leaf) is bool
    assert res_loss_leaf is False
    assert res_loss_leaf == ref_loss_leaf

    upstream = torch.ones_like(loss)
    ref_upstream = tu.to_reference(upstream)
    (grad,) = torch.autograd.grad(loss, inp, grad_outputs=upstream)
    (ref_grad,) = torch.autograd.grad(ref_loss, ref_inp, grad_outputs=ref_upstream)
    assert grad.shape == inp.shape
    tu.assert_result_close(grad, ref_grad)

    # Reading the state neither consumed nor rebuilt the graph: the input is
    # still the leaf the gradient accumulated into.
    assert flag_gems.is_leaf(inp) is True


_NON_TENSOR_INPUTS = [3.14, [1.0, 2.0], "not a tensor", None]


@pytest.mark.is_leaf
@pytest.mark.parametrize("bad_input", _NON_TENSOR_INPUTS)
def test_is_leaf_rejects_non_tensor(bad_input):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.is_leaf(bad_input)

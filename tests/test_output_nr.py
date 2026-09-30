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

# aten::output_nr(Tensor self) -> int reports the position of self in the output
# list of its autograd node: 0 for every tensor without a grad_fn (leaves,
# no_grad results, single-output nodes) and the slot index for a differentiable
# output of a multi-output node. It reads autograd metadata only, so the value,
# dtype, layout and device of the queried tensor never reach the result.
#
# output_nr takes exactly one tensor, so there is no broadcast geometry and no
# scalar-operand form. It returns a Python int, which cannot be differentiated,
# so no autograd.grad workload exists; the graph tests below cover that dimension
# by querying genuine outputs of recorded nodes and checking the reported slot.

_SUPPORTED_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.bool, torch.complex64]
)

# Dtypes that can record an autograd node at all.
_GRAPH_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.complex64,
] + ([torch.float64] if utils.fp64_is_supported else [])

# Recording an arithmetic graph needs a real kernel, and this backend has no CUDA
# arithmetic for fp8 (float8_e4m3fn * 2 raises "'mul_cuda' not implemented"), so
# fp8 leaves are exercised through metadata-only graphs, which record a node
# without computing on any element.
_ARITHMETIC_DTYPES = [torch.float16, torch.float32, torch.bfloat16] + (
    [torch.float64] if utils.fp64_is_supported else []
)


def _graph_tensor(dtype, shape, requires_grad=False, empty=False):
    # output_nr never reads element values, so the fixture only has to be a valid
    # tensor. Fixtures that a kernel computes on are initialized, so no
    # uninitialized memory is ever fed to a backend operation; pure metadata
    # fixtures (views, lazy flags, detach) stay uninitialized.
    if empty:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    if requires_grad:
        inp.requires_grad_(True)
    return inp


def _twin(inp):
    # Independent tensor with the same metadata and autograd state; the native
    # query on it is the oracle for the candidate query on inp.
    return torch.zeros_like(inp).requires_grad_(inp.requires_grad)


def _split_parts(t):
    # torch.split sizes every part max(n // 2, 1), so the part count is
    # ceil(n / max(n // 2, 1)).
    return tuple(torch.split(t, max(t.shape[0] // 2, 1), dim=0))


def _view_of_split_slot(t):
    # A view built on top of one slot starts a fresh node, so the split slot
    # index no longer applies to the derived tensor.
    return (_split_parts(t)[1].reshape(-1),)


def _no_grad_view(t):
    with torch.no_grad():
        return (t.reshape(-1),)


def _inplace_rebase(t):
    # In-place work on a non-view output rebases the tensor onto a fresh node
    # whose output number is 0.
    y = t * 2
    y.mul_(3)
    return (y,)


_GRAPH_BUILDERS = {
    "detach": lambda t: (t.detach(),),
    "no_grad": _no_grad_view,
    "mul": lambda t: (t * 2,),
    "neg": lambda t: (torch.neg(t),),
    "view": lambda t: (t.reshape(-1),),
    "unsqueeze": lambda t: (t.unsqueeze(0),),
    "transpose": lambda t: (t.transpose(0, -1),),
    "select": lambda t: (t[0],),
    "narrow": lambda t: (t.narrow(0, 0, 1),),
    "inplace_rebase": _inplace_rebase,
    "view_of_split_slot": _view_of_split_slot,
    "split": _split_parts,
    "chunk": lambda t: tuple(torch.chunk(t, 3, dim=0)),
    "unbind": lambda t: tuple(torch.unbind(t, dim=0)),
    "var_mean": lambda t: tuple(torch.var_mean(t, dim=0)),
    "aminmax": lambda t: tuple(torch.aminmax(t, dim=0)),
    "frexp": lambda t: tuple(torch.frexp(t)),
    "sort": lambda t: tuple(torch.sort(t, dim=0)),
    "topk": lambda t: tuple(torch.topk(t, 2, dim=0)),
    "max_dim": lambda t: tuple(torch.max(t, dim=0)),
    "median_dim": lambda t: tuple(torch.median(t, dim=0)),
    "conj": lambda t: (t.conj(),),
    "resolve_conj": lambda t: (t.conj().resolve_conj(),),
    "neg_view": lambda t: (torch.ops.aten._neg_view(t),),
}

# Graphs that only rearrange metadata or set a lazy flag: they read and compute on
# no element, so their fixtures may stay uninitialized. Note that split, chunk and
# unbind are metadata-only on their inputs even though they build a multi-output
# node, and that a metadata view on an fp8 leaf records a single-output node
# without needing an fp8 arithmetic kernel.
_METADATA_ONLY_GRAPHS = frozenset(
    {
        "detach",
        "no_grad",
        "view",
        "unsqueeze",
        "transpose",
        "select",
        "narrow",
        "view_of_split_slot",
        "split",
        "chunk",
        "unbind",
        "conj",
        "resolve_conj",
        "neg_view",
    }
)

# Graphs whose queried tensor has no grad_fn at all.
_GRAD_FREE_GRAPHS = ["detach", "no_grad"]

# Metadata views that record a genuine single-output node for every graph-capable
# dtype, including the fp8 types that have no arithmetic kernel on this backend.
_METADATA_VIEW_GRAPHS = ["view", "unsqueeze"]

# Multi-output nodes whose slot count depends on the input shape.
_SLOT_SHAPES = {
    "split": [(6,), (4, 8), (3, 7)],
    "chunk": [(9,), (6, 4)],
    "unbind": [(3,), (3, 4)],
    "var_mean": [(8,), (4, 6)],
    "aminmax": [(8,), (4, 6)],
    "frexp": [(8,), (4, 6)],
    "sort": [(8,), (4, 6)],
    "topk": [(8,), (4, 6)],
    "max_dim": [(8,), (4, 6)],
    "median_dim": [(8,), (4, 6)],
}

_DIM_SHAPES = tu.selected_shapes()

# Single-output arithmetic graphs over the spec shape grid. transpose, select and
# narrow need rank >= 1, and the split-slot view needs a first dimension that
# yields at least two parts.
_SINGLE_OUTPUT_SHAPES = {
    "mul": _DIM_SHAPES,
    "neg": _DIM_SHAPES,
    "inplace_rebase": _DIM_SHAPES,
    "transpose": [s for s in _DIM_SHAPES if len(s) >= 1],
    "select": [s for s in _DIM_SHAPES if len(s) >= 1],
    "narrow": [s for s in _DIM_SHAPES if len(s) >= 1],
    "view_of_split_slot": [s for s in _DIM_SHAPES if len(s) >= 1 and s[0] >= 2],
}

# Every output of these nodes is differentiable, so each slot reports its own
# index. Nodes that also return integer metadata (indices, exponents) report 0 on
# that slot instead, and a fresh node always reports 0.
_SLOT_INDEX_GRAPHS = frozenset({"split", "chunk", "unbind", "var_mean", "aminmax"})
_TWO_SLOT_GRAPHS = frozenset(
    {"var_mean", "aminmax", "frexp", "sort", "topk", "max_dim", "median_dim"}
)


def _slot_count(graph, shape):
    if graph == "split":
        return -(-shape[0] // max(shape[0] // 2, 1))
    if graph == "chunk":
        # torch.chunk returns min(chunks, size) parts.
        return min(3, shape[0])
    if graph == "unbind":
        return shape[0]
    return 2 if graph in _TWO_SLOT_GRAPHS else 1


def _slot_rows(graphs):
    rows = []
    for graph, shapes in graphs.items():
        for shape in shapes:
            for slot in range(_slot_count(graph, shape)):
                rows.append(
                    (graph, shape, slot, slot if graph in _SLOT_INDEX_GRAPHS else 0)
                )
    return rows


_GRAD_ROWS = _slot_rows(_SINGLE_OUTPUT_SHAPES) + _slot_rows(_SLOT_SHAPES)

# Graphs whose node identity, storage and view metadata must survive the query.
_GRAPH_STATE_CASES = [
    ("mul", (4, 6), 0, 0),
    ("transpose", (4, 6), 0, 0),
    ("select", (4, 6), 0, 0),
    ("inplace_rebase", (4, 6), 0, 0),
    ("view_of_split_slot", (6,), 0, 0),
    ("split", (6,), 1, 1),
    ("var_mean", (8,), 1, 1),
]

# Views whose alias, shape, stride and offset must be unchanged by the query.
_VIEW_CASES = [
    ("transpose", (4, 6)),
    ("select", (4, 6)),
    ("narrow", (4, 6)),
    ("view", (4, 6)),
    ("view_of_split_slot", (6,)),
]

# Empty inputs still build grad nodes, which is the boundary worth checking for a
# shape-driven metadata query.
_EMPTY_CASES = [
    (shape, graph) for shape in [(0,), (0, 3), (2, 0, 4)] for graph in ("leaf", "mul")
]

# graph, shape, requires_grad, is_conj, is_neg: conj() and _neg_view() are lazy
# flags, so the query must read the autograd metadata without materializing the
# conjugate or the negation.
_LAZY_CASES = [
    ("conj", (4, 4), False, True, False),
    ("conj", (3,), False, True, False),
    ("conj", (4, 4), True, True, False),
    ("resolve_conj", (4, 4), False, False, False),
    ("neg_view", (4, 4), False, False, True),
    ("neg_view", (2, 3, 4), False, False, True),
    ("neg_view", (4,), True, False, True),
]

_LAZY_DTYPES = {
    "conj": torch.complex64,
    "resolve_conj": torch.complex64,
    "neg_view": torch.float32,
}

# nan / inf / mixed rows for every supported floating dtype; the shared helper
# omits the inf scenarios for float8_e4m3fn, which cannot represent infinity.
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SUPPORTED_DTYPES), quick=[])


def _assert_output_nr(res, ref, expected):
    # The schema returns a Python int: check the type, the native value and the
    # slot this graph implies, all directly rather than through a tensor buffer.
    assert type(res) is int, type(res)
    assert res == ref, (res, ref)
    assert res == expected, (res, expected)


@pytest.mark.output_nr
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_output_nr_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.output_nr(ref_inp)
    res_out = flag_gems.output_nr(inp)

    _assert_output_nr(res_out, ref_out, 0)


@pytest.mark.output_nr
@pytest.mark.parametrize("graph", _GRAD_FREE_GRAPHS)
@pytest.mark.parametrize("dtype", _SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_output_nr_without_grad_fn(graph, shape, dtype):
    inp = _graph_tensor(dtype, shape, empty=graph in _METADATA_ONLY_GRAPHS)
    ref_inp = _twin(inp)

    ref_out = torch.ops.aten.output_nr(_GRAPH_BUILDERS[graph](ref_inp)[0])
    res_out = flag_gems.output_nr(_GRAPH_BUILDERS[graph](inp)[0])

    _assert_output_nr(res_out, ref_out, 0)


@pytest.mark.output_nr
@pytest.mark.parametrize("graph", _METADATA_VIEW_GRAPHS)
@pytest.mark.parametrize("dtype", _GRAPH_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
def test_output_nr_metadata_view_of_leaf(graph, shape, dtype):
    # A metadata view over a differentiable leaf is a genuine single-output
    # autograd node, so fp8 and complex leaves are covered without an arithmetic
    # kernel; only the queried view's metadata is read, never any element.
    leaf = _graph_tensor(dtype, shape, requires_grad=True, empty=True)
    ref_leaf = _twin(leaf)

    queried = _GRAPH_BUILDERS[graph](leaf)[0]
    ref_queried = _GRAPH_BUILDERS[graph](ref_leaf)[0]
    grad_fn = queried.grad_fn

    ref_out = torch.ops.aten.output_nr(ref_queried)
    res_out = flag_gems.output_nr(queried)

    _assert_output_nr(res_out, ref_out, 0)
    assert queried.grad_fn is grad_fn
    assert queried.untyped_storage().data_ptr() == leaf.untyped_storage().data_ptr()


@pytest.mark.output_nr
@pytest.mark.parametrize("dtype", _ARITHMETIC_DTYPES)
@pytest.mark.parametrize("graph,shape,slot,expected", _GRAD_ROWS)
def test_output_nr_grad_graph_slots(graph, shape, slot, expected, dtype):
    leaf = _graph_tensor(
        dtype, shape, requires_grad=True, empty=graph in _METADATA_ONLY_GRAPHS
    )
    ref_leaf = _twin(leaf)

    ref_out = torch.ops.aten.output_nr(_GRAPH_BUILDERS[graph](ref_leaf)[slot])
    res_out = flag_gems.output_nr(_GRAPH_BUILDERS[graph](leaf)[slot])

    _assert_output_nr(res_out, ref_out, expected)


@pytest.mark.output_nr
@pytest.mark.parametrize("shape,graph", _EMPTY_CASES)
def test_output_nr_empty_tensor(shape, graph):
    leaf = _graph_tensor(torch.float32, shape, requires_grad=True, empty=True)
    ref_leaf = _twin(leaf)

    queried = leaf if graph == "leaf" else leaf * 2
    ref_queried = ref_leaf if graph == "leaf" else ref_leaf * 2

    ref_out = torch.ops.aten.output_nr(ref_queried)
    res_out = flag_gems.output_nr(queried)

    _assert_output_nr(res_out, ref_out, 0)


@pytest.mark.output_nr
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_output_nr_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.output_nr(ref_inp)
    res_out = flag_gems.output_nr(inp)

    _assert_output_nr(res_out, ref_out, 0)


@pytest.mark.output_nr
@pytest.mark.parametrize("graph,shape,requires_grad,is_conj,is_neg", _LAZY_CASES)
def test_output_nr_lazy_conjugate_and_negation(
    graph, shape, requires_grad, is_conj, is_neg
):
    dtype = _LAZY_DTYPES[graph]
    leaf = _graph_tensor(dtype, shape, requires_grad=requires_grad, empty=True)
    ref_leaf = _twin(leaf)

    queried = _GRAPH_BUILDERS[graph](leaf)[0]
    ref_queried = _GRAPH_BUILDERS[graph](ref_leaf)[0]
    grad_fn = queried.grad_fn
    ptr = queried.data_ptr()
    storage_ptr = queried.untyped_storage().data_ptr()

    ref_out = torch.ops.aten.output_nr(ref_queried)
    res_out = flag_gems.output_nr(queried)

    _assert_output_nr(res_out, ref_out, 0)
    # The lazy flags must survive the query unmaterialized, together with the
    # node and the storage the queried tensor already had.
    assert queried.is_conj() is is_conj
    assert queried.is_neg() is is_neg
    assert queried.grad_fn is grad_fn
    assert queried.data_ptr() == ptr
    assert queried.untyped_storage().data_ptr() == storage_ptr


@pytest.mark.output_nr
@pytest.mark.parametrize("graph,shape", _VIEW_CASES)
def test_output_nr_view_metadata_preserved(graph, shape):
    leaf = _graph_tensor(torch.float32, shape, requires_grad=True, empty=True)
    ref_leaf = _twin(leaf)

    queried = _GRAPH_BUILDERS[graph](leaf)[0]
    ref_queried = _GRAPH_BUILDERS[graph](ref_leaf)[0]
    grad_fn = queried.grad_fn
    size_before = queried.shape
    stride_before = queried.stride()
    offset_before = queried.storage_offset()
    storage_ptr = queried.untyped_storage().data_ptr()

    ref_out = torch.ops.aten.output_nr(ref_queried)
    res_out = flag_gems.output_nr(queried)

    _assert_output_nr(res_out, ref_out, 0)
    assert queried.grad_fn is grad_fn
    assert queried.shape == size_before
    assert queried.stride() == stride_before
    assert queried.storage_offset() == offset_before
    # The query must not rebase the view onto fresh storage.
    assert queried.untyped_storage().data_ptr() == storage_ptr
    assert queried.untyped_storage().data_ptr() == leaf.untyped_storage().data_ptr()


@pytest.mark.output_nr
@pytest.mark.parametrize("graph,shape,slot,expected", _GRAPH_STATE_CASES)
def test_output_nr_preserves_graph_and_storage(graph, shape, slot, expected):
    leaf = _graph_tensor(
        torch.float32, shape, requires_grad=True, empty=graph in _METADATA_ONLY_GRAPHS
    )
    ref_leaf = _twin(leaf)

    queried = _GRAPH_BUILDERS[graph](leaf)[slot]
    ref_queried = _GRAPH_BUILDERS[graph](ref_leaf)[slot]
    grad_fn = queried.grad_fn
    ptr = queried.data_ptr()
    storage_ptr = queried.untyped_storage().data_ptr()
    size_before = queried.shape
    stride_before = queried.stride()
    offset_before = queried.storage_offset()

    ref_out = torch.ops.aten.output_nr(ref_queried)
    res_out = flag_gems.output_nr(queried)

    _assert_output_nr(res_out, ref_out, expected)
    assert queried.grad_fn is grad_fn
    assert queried.data_ptr() == ptr
    assert queried.untyped_storage().data_ptr() == storage_ptr
    assert queried.shape == size_before
    assert queried.stride() == stride_before
    assert queried.storage_offset() == offset_before


@pytest.mark.output_nr
def test_output_nr_missing_self_raises():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.output_nr()


@pytest.mark.output_nr
def test_output_nr_too_many_positional_arguments_raises():
    inp = torch.empty(4, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.output_nr(inp, inp)


@pytest.mark.output_nr
def test_output_nr_unexpected_keyword_argument_raises():
    inp = torch.empty(4, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.output_nr(inp, dim=0)


@pytest.mark.output_nr
def test_output_nr_non_tensor_operand_raises():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.output_nr(3.14)


@pytest.mark.output_nr
def test_output_nr_undefined_tensor_raises():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.output_nr(None)

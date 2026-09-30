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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# aten::numel(Tensor self) -> int is a host-side metadata query: it has a single
# ``default`` overload (no ``.out``), one tensor operand (no broadcast) and an
# int result (no backward), so those spec dimensions do not apply. Shape,
# layout, offset and allocation-state semantics are covered instead; the value
# range never changes the result but keeps the shared grid.
_NUMEL_DTYPES = (
    tu.REQUIRED_DTYPES
    + ([torch.float64] if utils.fp64_is_supported else [])
    + [torch.int16, torch.bool, torch.complex64]
)
_NUMEL_SPECIAL_DTYPES = [dtype for dtype in _NUMEL_DTYPES if dtype.is_floating_point]

# Strided states that separate the logical size from the storage layout.
_STRIDED_LAYOUT_KINDS = [
    "dense",
    "transpose",
    "step_slice",
    "narrowed",
    "offset_view",
    "zero_stride",
    "empty_1d",
    "empty_3d",
    "scalar",
    "conj_view",
]

_STRUCTURED_LAYOUT_KINDS = [
    "sparse_coo",
    "sparse_coo_hybrid",
    "sparse_coo_uncoalesced",
    "sparse_csr",
    "nested_strided",
    "nested_jagged",
    "quantized_per_tensor",
    "quantized_per_channel",
    "meta",
]

# None is not listed: aten::numel(None) returns 0 instead of raising, so it is a
# positive case below rather than a rejected argument.
_NON_TENSOR_ARG_CASES = [
    pytest.param(3.14, id="float"),
    pytest.param(1, id="int"),
    pytest.param(True, id="bool"),
    pytest.param("abc", id="str"),
    pytest.param((1, 2), id="tuple"),
    pytest.param([1, 2], id="list"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(object(), id="object"),
]


def _strided_tensor(kind, dtype, device):
    if kind == "dense":
        return torch.zeros((4, 6), dtype=dtype, device=device)
    if kind == "transpose":
        return torch.zeros((6, 4), dtype=dtype, device=device).t()
    if kind == "step_slice":
        return torch.zeros((8, 8), dtype=dtype, device=device)[::2]
    if kind == "narrowed":
        return torch.zeros((8, 8), dtype=dtype, device=device).narrow(0, 2, 3)
    if kind == "offset_view":
        base = torch.zeros((6, 6), dtype=dtype, device=device)
        return torch.empty(0, dtype=dtype, device=device).set_(
            base.untyped_storage(), 10, (5,), (2,)
        )
    if kind == "zero_stride":
        return torch.zeros((4, 1), dtype=dtype, device=device).expand(4, 6)
    if kind == "empty_1d":
        return torch.empty(0, dtype=dtype, device=device)
    if kind == "empty_3d":
        return torch.empty((0, 3, 4), dtype=dtype, device=device)
    if kind == "scalar":
        return torch.zeros((), dtype=dtype, device=device)
    if kind == "conj_view":
        return torch.zeros((4, 4), dtype=dtype, device=device).conj()
    raise ValueError(f"Unknown strided layout kind: {kind!r}")


def _quantize(kind, device):
    source = torch.ones((4, 5), device=device)
    if kind == "quantized_per_tensor":
        return torch.quantize_per_tensor(source, 0.1, 3, torch.quint8)
    return torch.quantize_per_channel(
        source,
        torch.tensor([0.1, 0.2, 0.3, 0.4], device=device),
        torch.tensor([3, 4, 5, 6], device=device),
        0,
        torch.quint8,
    )


def _structured_tensor(kind, device):
    if kind == "sparse_coo":
        indices = torch.tensor([[0, 1, 2], [1, 0, 3]], device=device)
        values = torch.ones(3, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 6), device=device)
    if kind == "sparse_coo_hybrid":
        indices = torch.tensor([[0, 1], [1, 0]], device=device)
        values = torch.ones(2, 2, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 6, 2), device=device)
    if kind == "sparse_coo_uncoalesced":
        indices = torch.tensor([[0, 0], [1, 1]], device=device)
        values = torch.ones(2, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 6), device=device)
    if kind == "sparse_csr":
        crow = torch.tensor([0, 1, 2, 3], device=device)
        col = torch.tensor([0, 1, 2], device=device)
        values = torch.ones(3, device=device)
        return torch.sparse_csr_tensor(crow, col, values, size=(3, 6), device=device)
    if kind == "nested_strided":
        parts = [
            torch.ones(2, 3, device=device),
            torch.ones(5, 3, device=device),
        ]
        return torch.nested.nested_tensor(parts)
    if kind == "nested_jagged":
        parts = [
            torch.ones(2, 3, device=device),
            torch.ones(5, 3, device=device),
        ]
        return torch.nested.nested_tensor(parts, layout=torch.jagged)
    if kind in ("quantized_per_tensor", "quantized_per_channel"):
        return _quantize(kind, device)
    if kind == "meta":
        return torch.zeros(3, 4, dtype=torch.float32, device="meta")
    raise ValueError(f"Unknown structured layout kind: {kind!r}")


def _strided_state(tensor):
    """Host-side allocation metadata that a metadata query must not touch."""
    return (
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.data_ptr(),
        tensor.untyped_storage().nbytes(),
        tensor._version,
    )


def _special_tensor(shape, dtype, scenario):
    count = math.prod(shape)
    values = tu.make_special_input(dtype, scenario)
    repeats = (count + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:count].reshape(shape)


@pytest.mark.numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", _NUMEL_DTYPES)
def test_numel_value_ranges(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.numel
@pytest.mark.parametrize("kind", _STRIDED_LAYOUT_KINDS)
@pytest.mark.parametrize("dtype", _NUMEL_DTYPES)
def test_numel_strided_layouts(kind, dtype):
    inp = _strided_tensor(kind, dtype, flag_gems.device)
    # The reference is rebuilt on the reference device: moving a view would
    # compact its storage and change the offset/stride being tested.
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _strided_tensor(kind, dtype, ref_device)
    before = _strided_state(inp)

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out
    # A shape-only query must leave storage, stride and offset untouched.
    assert _strided_state(inp) == before


@pytest.mark.numel
@pytest.mark.parametrize("kind", _STRUCTURED_LAYOUT_KINDS)
def test_numel_structured_layouts(kind):
    inp = _structured_tensor(kind, flag_gems.device)
    ref_device = "cpu" if utils.TO_CPU else flag_gems.device
    ref_inp = _structured_tensor(kind, ref_device)
    version_before = inp._version

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out
    assert inp._version == version_before
    if kind.startswith("sparse_coo"):
        # The dense element count, not the number of stored entries.
        assert res_out != inp._nnz()
    if kind == "sparse_csr":
        assert res_out != inp.values().numel()


@pytest.mark.numel
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(_NUMEL_SPECIAL_DTYPES), quick=[]),
)
def test_numel_special_values(shape, dtype, scenario):
    inp = _special_tensor(shape, dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.numel
def test_numel_ignores_autograd():
    inp = torch.zeros((4, 6), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    version_before = inp._version

    ref_out = torch.ops.aten.numel(ref_inp)
    res_out = flag_gems.numel(inp)

    assert type(res_out) is int
    assert res_out == ref_out
    # The count carries no grad_fn and does not bump the version counter.
    assert inp._version == version_before
    assert inp.grad is None


@pytest.mark.numel
def test_numel_none_returns_zero():
    # aten::numel(None) is accepted and returns 0, so this is a positive case
    # and None stays out of the rejected-argument list.
    ref_out = torch.ops.aten.numel(None)
    res_out = flag_gems.numel(None)

    assert type(res_out) is int
    assert res_out == ref_out


@pytest.mark.numel
@pytest.mark.parametrize("arg", _NON_TENSOR_ARG_CASES)
def test_numel_rejects_non_tensor(arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.numel(arg)


@pytest.mark.numel
def test_numel_rejects_missing_argument():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.numel()

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

# aten::is_distributed(Tensor self) -> bool reports TensorImpl state and never
# reads data, so its workloads are tensor states and dtypes, not values.

_DTYPE_CANDIDATES = [
    torch.int8,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.int32,
    torch.int64,
    torch.float64,
    torch.bool,
]

# Static capability flags from the shared test helpers; they gate whether this
# backend can allocate the dtype, not whether the operator accepts it.
_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}


def _supported(dtype):
    return _DTYPE_FLAGS.get(dtype, True)


SUPPORTED_DTYPES = [dtype for dtype in _DTYPE_CANDIDATES if _supported(dtype)]

_SUPPORTED_FLOAT_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _supported(dtype)
]

# Positive NaN/Inf and autograd workloads are default-only.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_SUPPORTED_FLOAT_DTYPES), quick=[]
)
_AUTOGRAD_CASES = tu.selected_cases([torch.float32], quick=[])


@pytest.mark.is_distributed
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_is_distributed(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_distributed(ref_inp)
    res_out = flag_gems.is_distributed(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


_STATE_KINDS = [
    "transpose",
    "slice",
    "expand",
    "step",
    "conj",
    "neg",
    "zero_dim",
    "empty",
    "meta",
    "sparse_coo",
    "sparse_csr",
]

# These rows vary state, not values; the five value ranges are exercised by the
# numeric grid above.
_STATE_DTYPES = [torch.float32, torch.complex64]


def _state_input(kind, dtype):
    """Build one non-trivial tensor state for the state-independence test."""
    device = flag_gems.device
    if kind == "meta":
        # A device-less tensor state: the native op accepts it and reads no data.
        return torch.empty((12, 10), dtype=dtype, device="meta")
    if kind == "sparse_coo":
        return torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]], device=device),
            torch.ones(2, dtype=dtype, device=device),
            (12, 10),
        )
    if kind == "sparse_csr":
        return torch.sparse_csr_tensor(
            torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
            torch.tensor([1, 0], dtype=torch.int32, device=device),
            torch.ones(2, dtype=dtype, device=device),
            (2, 10),
        )
    base = tu.make_input(dtype, (12, 10), ["-1", "1"])
    if kind == "transpose":
        return base.t()
    if kind == "slice":
        return base[1:5, ::2]
    if kind == "expand":
        return base[:1, :1].expand(4, 6)
    if kind == "step":
        return base[::3, ::4]
    if kind == "conj":
        return base.conj()
    if kind == "neg":
        return torch._neg_view(base)
    if kind == "zero_dim":
        return tu.make_input(dtype, (), ["-1", "1"])
    if kind == "empty":
        # Zero elements and no value is read, so only the sizes matter here.
        return torch.empty((0, 7), dtype=dtype, device=device)
    raise AssertionError(f"unknown state kind {kind}")


@pytest.mark.is_distributed
@pytest.mark.parametrize("kind", _STATE_KINDS)
@pytest.mark.parametrize("dtype", _STATE_DTYPES)
def test_is_distributed_tensor_state(kind, dtype):
    inp = _state_input(kind, dtype)
    # A meta tensor cannot be moved to the reference device and no data is read,
    # so its reference is built independently with the same state.
    ref_inp = inp.detach().clone() if kind == "meta" else tu.to_reference(inp)
    shape = tuple(inp.shape)
    flags = (inp.is_conj(), inp.is_neg())

    ref_out = torch.ops.aten.is_distributed(ref_inp)
    res_out = flag_gems.is_distributed(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    # Reading the state must not disturb the input's sizes or its lazy flags.
    assert (inp.is_conj(), inp.is_neg()) == flags
    assert tuple(inp.shape) == shape


@pytest.mark.is_distributed
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_is_distributed_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_distributed(ref_inp)
    res_out = flag_gems.is_distributed(inp)

    assert type(res_out) is bool
    assert res_out == ref_out


@pytest.mark.is_distributed
@pytest.mark.parametrize("dtype", _AUTOGRAD_CASES)
def test_is_distributed_leaves_autograd_state(dtype):
    # The operator returns a Python bool, so there is no output graph to
    # differentiate and no gradient to check: the applicable contract is that
    # reading the flag neither builds a graph through the leaf nor deposits a
    # gradient on it.
    inp = tu.make_input(dtype, (8, 8), ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.is_distributed(ref_inp)
    res_out = flag_gems.is_distributed(inp)

    assert type(res_out) is bool
    assert res_out == ref_out
    assert inp.requires_grad and inp.is_leaf
    assert inp.grad is None and inp.grad_fn is None


@pytest.mark.is_distributed
@pytest.mark.parametrize("bad", [1.0, [1, 2], "tensor"])
def test_is_distributed_rejects_non_tensor(bad):
    # The schema takes a Tensor; the operator performs no dtype or dim dispatch,
    # so a non-tensor argument is the only invalid input form.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.is_distributed(bad)

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

from typing import NamedTuple

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import test_utils as tu

# ``pytest.mark`` refuses to create an attribute for a name starting with an
# underscore, so the marker is registered on the MarkGenerator directly.
setattr(
    pytest.mark,
    "_sparse_softmax_backward_data",
    MarkDecorator(
        Mark("_sparse_softmax_backward_data", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# Only SparseCPU / SparseCUDA kernels are registered and the CUDA kernel
# dispatches on AT_DISPATCH_FLOATING_TYPES, so the supported set is the static
# device capability: float32 plus float64 when the device reports fp64 support.
_FP64_SUPPORTED = flag_gems.runtime.device.support_fp64
SUPPORTED_DTYPES = [torch.float32] + ([torch.float64] if _FP64_SUPPORTED else [])

# Measured on the active NVIDIA CUDA sparse backend and therefore vendor-scoped:
# the float/integer/bool types stop in the softmax_backward dispatch, while the
# two fp8 types stop one step earlier in the sparse-coalesce metadata step. Both
# surface as RuntimeError. Other vendors have no matching measurement, so they
# assert no dtype rejection: the parameter set is then empty and pytest collects
# its single empty-parameter-set placeholder for that vendor. This static
# selection does not execute the operator.
UNSUPPORTED_DTYPES = (
    [
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bool,
    ]
    if flag_gems.vendor_name == "nvidia"
    else []
)


class Row(NamedTuple):
    # One sparse workload: an explicit coords pair fixes the stored supports,
    # otherwise distinct random coordinates are drawn.
    shape: tuple
    dim: int
    sparse_dim: int
    value_range: list
    grad_coords: list = None
    output_coords: list = None
    nnz: int = 6


def _wrapped_dims(shape):
    # Every dim the native maybe_wrap_dim accepts for this shape; the 0-dim
    # sparse operand has the single valid dim -1.
    return (-1,) if not shape else tuple(range(-len(shape), len(shape)))


def _indices(shape, sparse_dim, nnz, coords):
    # COO index tensor of shape (sparse_dim, nnz). sparse_dim == 0 is a
    # native-valid boundary whose index tensor has no rows.
    if coords is not None:
        return torch.tensor(coords, dtype=torch.long, device=flag_gems.device)
    if sparse_dim == 0:
        return torch.empty(0, nnz, dtype=torch.long, device=flag_gems.device)
    extents = shape[:sparse_dim]
    numel = 1
    for extent in extents:
        numel *= extent
    flat = torch.unique(
        torch.randint(0, numel, (4 * min(nnz, numel),), device=flag_gems.device)
    )[: min(nnz, numel)]
    rows = []
    for extent in reversed(extents):
        rows.append(flat % extent)
        flat = flat // extent
    return torch.stack(list(reversed(rows)))


def _make_inputs(row, dtype):
    # grad_output and output are independent tensors and may carry different
    # supports; the native preprocessing compares only their size and sparse_dim.
    grad_indices = _indices(row.shape, row.sparse_dim, row.nnz, row.grad_coords)
    output_indices = _indices(row.shape, row.sparse_dim, row.nnz, row.output_coords)
    value_tail = tuple(row.shape[row.sparse_dim :])
    grad = torch.sparse_coo_tensor(
        grad_indices,
        tu.make_input(dtype, (grad_indices.shape[1],) + value_tail, row.value_range),
        row.shape,
        device=flag_gems.device,
    )
    output = torch.sparse_coo_tensor(
        output_indices,
        tu.make_input(dtype, (output_indices.shape[1],) + value_tail, row.value_range),
        row.shape,
        device=flag_gems.device,
    )
    self_tensor = tu.make_input(dtype, (1,), row.value_range)
    return grad, output, self_tensor


def _zeroed_buffer(tensor):
    # Caller-owned sparse COO buffer for the out overload. The indices are cloned
    # because torch.sparse_coo_tensor keeps the passed index storage, and the
    # native out variant writes the coalesced result into the buffer in place.
    return torch.sparse_coo_tensor(
        tensor._indices().clone(),
        torch.zeros_like(tensor._values()),
        tensor.shape,
        device=flag_gems.device,
    )


# Each row is one Workload: all valid wrapped dims of every spec shape at every
# spec value range. Quick keeps a single representative dim per shape.
GRID_ROWS = [
    Row(shape, dim, len(shape), value_range)
    for shape in tu.selected_shapes()
    for dim in tu.selected_cases(_wrapped_dims(shape), quick=_wrapped_dims(shape)[-1:])
    for value_range in tu.selected_ranges()
]

# Default-only rows the plain shape grid cannot express. Hybrid inputs
# (sparse_dim < rank) are the only form with dense value blocks: dim >= sparse_dim
# sends the value blocks to a dense softmax backward, while dim < sparse_dim still
# reduces the sparse coordinates, so both kernel paths appear with hybrid inputs
# of both kinds.
EXTRA_ROWS = [
    Row((8, 5), 0, 1, ["-1", "1"]),
    Row((8, 5), 1, 1, ["-1", "1"]),
    Row((4, 3, 5), 1, 2, ["-1", "1"]),
    Row((4, 3, 5), 2, 2, ["-1", "1"]),
    # Overlapping, disjoint and deterministically duplicated supports, plus
    # entries grouped along the reduced dim.
    Row(
        (4, 6),
        -1,
        2,
        ["-1", "1"],
        grad_coords=[[0, 0, 1, 1, 2], [0, 1, 0, 1, 3]],
        output_coords=[[1, 1, 2, 2, 3], [0, 1, 0, 1, 3]],
    ),
    Row(
        (4, 6),
        -1,
        2,
        ["-1", "1"],
        grad_coords=[[0, 0], [0, 1]],
        output_coords=[[3, 3], [4, 5]],
    ),
    Row(
        (4, 6),
        -1,
        2,
        ["-1", "1"],
        grad_coords=[[0, 0, 1, 1], [0, 0, 2, 2]],
        output_coords=[[0, 0, 1, 1], [0, 0, 2, 2]],
    ),
    Row(
        (5, 7), -1, 2, ["-1", "1"], grad_coords=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 5]]
    ),
    Row(
        (5, 7), 0, 2, ["-1", "1"], grad_coords=[[0, 1, 2, 3, 4, 0], [0, 0, 1, 1, 2, 2]]
    ),
    # No stored entries: empty support and a zero-sized dense dimension, which
    # are different structural cases.
    Row((4, 6), -1, 2, ["-1", "1"], grad_coords=[[], []]),
    Row((4, 0), -1, 1, ["-1", "1"], grad_coords=[[0, 1, 2]]),
    # sparse_dim == 0 boundary: every dimension is dense.
    Row((5,), 0, 0, ["-1", "1"], nnz=3),
]

CASE_ROWS = [
    pytest.param(row, id="{}d{}".format(list(row.shape), row.dim))
    for row in GRID_ROWS + tu.selected_cases(EXTRA_ROWS, quick=[])
]

# Deterministic grouped supports shared by grad_output and output: 6 entries in
# lexicographic order, three per (i1, i2) group along the reduced dim 0, so every
# one of the 5 special values written below lands in a reduced group.
SPECIAL_ROW = Row(
    (20, 320, 15),
    0,
    3,
    ["-1", "1"],
    grad_coords=[[0, 0, 1, 1, 2, 2], [3, 4, 3, 4, 3, 4], [5, 6, 5, 6, 5, 6]],
    output_coords=[[0, 0, 1, 1, 2, 2], [3, 4, 3, 4, 3, 4], [5, 6, 5, 6, 5, 6]],
)


@pytest.mark._sparse_softmax_backward_data
@pytest.mark.parametrize("case", CASE_ROWS)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_softmax_backward_data(case, dtype):
    grad, output, self_tensor = _make_inputs(case, dtype)

    ref = torch.ops.aten._sparse_softmax_backward_data(
        tu.to_reference(grad),
        tu.to_reference(output),
        case.dim,
        tu.to_reference(self_tensor),
    )
    res = flag_gems._sparse_softmax_backward_data(grad, output, case.dim, self_tensor)

    # The value comparison does not cover the coalesced flag the kernel must set.
    assert res.is_coalesced()
    tu.assert_result_close(res, ref)


@pytest.mark._sparse_softmax_backward_data
@pytest.mark.parametrize("shape", tu.selected_cases([(5, 7)], quick=[]))
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test__sparse_softmax_backward_data_out(shape, dtype):
    grad, output, self_tensor = _make_inputs(Row(shape, -1, 2, ["-1", "1"]), dtype)

    # The sparse out overload is callable on the active backend and is invoked
    # directly on both sides. The reference buffer follows the configured
    # reference device; the two buffers own their storage.
    res_buffer = _zeroed_buffer(grad)
    ref_buffer = tu.to_reference(_zeroed_buffer(grad))

    ref = torch.ops.aten._sparse_softmax_backward_data.out(
        tu.to_reference(grad),
        tu.to_reference(output),
        -1,
        tu.to_reference(self_tensor),
        out=ref_buffer,
    )
    res = flag_gems._sparse_softmax_backward_data(
        grad, output, -1, self_tensor, out=res_buffer
    )

    # The native out variant writes into the caller buffer and returns that same
    # tensor, so identity with the buffer and the mutated buffer are checked.
    assert res is res_buffer
    assert res_buffer.is_coalesced()
    tu.assert_result_close(res, ref)


def _make_special_inputs(dtype, scenario):
    grad, output, self_tensor = _make_inputs(SPECIAL_ROW, dtype)
    payload = tu.make_special_input(dtype, scenario)
    grad._values()[: payload.numel()].copy_(payload)
    output._values()[: payload.numel()].copy_(payload)
    return grad, output, self_tensor


@pytest.mark._sparse_softmax_backward_data
@pytest.mark.parametrize(
    "dtype,scenario",
    tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[]),
)
def test__sparse_softmax_backward_data_special_values(dtype, scenario):
    grad, output, self_tensor = _make_special_inputs(dtype, scenario)

    ref = torch.ops.aten._sparse_softmax_backward_data(
        tu.to_reference(grad),
        tu.to_reference(output),
        SPECIAL_ROW.dim,
        tu.to_reference(self_tensor),
    )
    res = flag_gems._sparse_softmax_backward_data(
        grad, output, SPECIAL_ROW.dim, self_tensor
    )

    # Stored NaN / Inf entries flow through unchanged, so matching NaNs are part
    # of the expected result.
    assert res.is_coalesced()
    tu.assert_result_close(res, ref)


# dim is the only non-tensor parameter and ATen accepts it only as a Python int,
# so the tensor-operand and scalar-operand form of this signature coincide and
# there is no separate scalar workload.
#
# There is no broadcast workload: the native preprocessing rejects operands of
# different sizes with checkSameSize instead of broadcasting them.
#
# There is no backward workload: this operator is itself an autograd primitive
# and autograd.grad through it reports "derivative for
# aten::_sparse_softmax_backward_data is not implemented".


@pytest.mark._sparse_softmax_backward_data
@pytest.mark.parametrize("dtype", UNSUPPORTED_DTYPES)
def test__sparse_softmax_backward_data_unsupported_dtype(dtype):
    grad, output, self_tensor = _make_inputs(Row((5, 7), -1, 2, ["-1", "1"]), dtype)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_softmax_backward_data(grad, output, -1, self_tensor)


@pytest.mark._sparse_softmax_backward_data
@pytest.mark.parametrize(
    "shape,dim,sparse_dim",
    [
        ((5, 7), 2, 2),
        ((5, 7), -3, 2),
        ((256,), 1, 1),
        ((256,), -2, 1),
        # The 0-dim operand is accepted with dim -1 and rejects dim 0 with
        # IndexError; dim 1 and -2 are likewise rejected.
        ((), 0, 0),
    ],
)
def test__sparse_softmax_backward_data_invalid_dim(shape, dim, sparse_dim):
    grad, output, self_tensor = _make_inputs(
        Row(shape, dim, sparse_dim, ["-1", "1"]), torch.float32
    )

    # maybe_wrap_dim reports an out-of-range dim as IndexError; a candidate may
    # surface the same invalid argument as a RuntimeError.
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems._sparse_softmax_backward_data(grad, output, dim, self_tensor)


def _incompatible_output(kind):
    # Native-invalid output for a fully sparse (5, 7) grad_output.
    if kind == "size":
        # Different size: rejected by checkSameSize.
        return _make_inputs(Row((5, 8), -1, 2, ["-1", "1"]), torch.float32)[1]
    # Same size but sparse_dim 1: checkSameSize passes and the sparse_dim
    # equality check then rejects it.
    indices = _indices((5,), 1, 6, None)
    return torch.sparse_coo_tensor(
        indices,
        tu.make_input(torch.float32, (indices.shape[1], 7), ["-1", "1"]),
        (5, 7),
        device=flag_gems.device,
    )


@pytest.mark._sparse_softmax_backward_data
@pytest.mark.parametrize("kind", ["size", "sparse_dim"])
def test__sparse_softmax_backward_data_incompatible_output(kind):
    grad, _, self_tensor = _make_inputs(Row((5, 7), -1, 2, ["-1", "1"]), torch.float32)

    with pytest.raises(RuntimeError):
        flag_gems._sparse_softmax_backward_data(
            grad, _incompatible_output(kind), -1, self_tensor
        )


@pytest.mark._sparse_softmax_backward_data
def test__sparse_softmax_backward_data_dense_inputs_rejected():
    # Only SparseCPU / SparseCUDA kernels are registered, so dense operands raise
    # NotImplementedError (a RuntimeError subclass).
    dense = tu.make_input(torch.float32, (5, 7), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_softmax_backward_data(dense, dense, -1, dense)

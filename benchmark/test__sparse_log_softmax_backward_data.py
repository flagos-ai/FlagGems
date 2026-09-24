# Copyright 2026 FlagOS Contributors. All rights reserved.
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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

setattr(
    pytest.mark,
    "_sparse_log_softmax_backward_data",
    MarkDecorator(
        Mark("_sparse_log_softmax_backward_data", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# The native operator carries float32/float64 kernels; float64 stays behind the static
# capability flag. Listing and execution share this list.
_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)

# A case is (shape, nnz[, sparse_dim, dim]); sparse_dim defaults to the full rank and
# dim to the last dimension. A shape file may use any of these spellings.
_DEFAULT_CASES = [
    ((1 << 20,), 1 << 20),
    ((1024, 1024), 1 << 20),
    ((2048, 2048), 1 << 20),
    ((4096, 4096), 1 << 20),
    ((256, 256, 256), 1 << 20),
    ((1024, 64), 1024, 1, 1),
    # Middle sparse reduction axis with a trailing sparse axis: the group is identified
    # by axes 0 and 2, so a prefix-only group id would collapse every entry into one
    # group. Kept in the core level on purpose.
    ((1024, 64, 16), 8192, 3, 1),
    # Duplicate coordinates are valid uncoalesced COO input, so a requested nnz above
    # the number of distinct positions is kept as asked instead of being clamped.
    ((4096,), 8192),
    # Native-valid scalar COO (dim=-1 is its only in-range dim) and empty operands;
    # the operator accepts nnz=0.
    ((), 1, 0, -1),
    ((), 0, 0, -1),
    ((1 << 20,), 0),
]


def _plan_fields(entry):
    """Validate a (shape, nnz[, sparse_dim, dim]) descriptor without altering it."""
    entry = tuple(entry)
    if entry and isinstance(entry[0], (tuple, list)):
        shape = tuple(entry[0])
        nnz = entry[1] if len(entry) > 1 else math.prod(shape)
        sparse_dim = entry[2] if len(entry) > 2 else len(shape)
        dim = entry[3] if len(entry) > 3 else len(shape) - 1
    else:
        shape = tuple(entry)
        nnz, sparse_dim, dim = math.prod(shape), len(shape), len(shape) - 1
    nnz, sparse_dim, dim = int(nnz), int(sparse_dim), int(dim)
    if not 0 <= sparse_dim <= len(shape):
        raise ValueError(
            f"invalid case {entry!r}: sparse_dim must be in [0, {len(shape)}]"
        )
    if nnz < 0:
        raise ValueError(f"invalid case {entry!r}: nnz must not be negative")
    if len(shape) == 0:
        # A 0-dim COO tensor is only valid at dim=-1; 0 and -2 are out of range.
        if dim != -1:
            raise ValueError(
                f"invalid case {entry!r}: a scalar shape only accepts dim=-1"
            )
    elif not -len(shape) <= dim < len(shape):
        raise ValueError(
            f"invalid case {entry!r}: dim must be in [{-len(shape)}, {len(shape) - 1}]"
        )
    return shape, nnz, sparse_dim, dim


def _coords(shape, sparse_dim, dim, nnz, device):
    """Support coordinates built on `device` in O(nnz).

    A normalization group is identified by every sparse coordinate except the one along
    `dim`, so the group ids enumerate those axes and the `dim` coordinate is inserted
    back at its own position. Several entries are placed in each group so the reduction
    has a within-group sum; the three operands of a case share this support and differ
    only in their values, otherwise a group of a single random entry sums to zero about
    half the time and the workload degrades to the trivial result.
    """
    shape = tuple(shape)
    if sparse_dim == 0:
        # A 0-dim COO tensor carries indices of shape (0, nnz).
        return torch.empty((0, nnz), dtype=torch.long, device=device)
    if nnz == 0:
        return torch.empty((sparse_dim, 0), dtype=torch.long, device=device)
    if dim < 0:
        dim += len(shape)
    if dim < sparse_dim:
        dim_size = shape[dim]
        group_shape = tuple(shape[a] for a in range(sparse_dim) if a != dim)
        n_groups = math.prod(group_shape) if group_shape else 1
        cols = max(1, min(dim_size, nnz))
        index = torch.arange(nnz, device=device)
        # The used groups are scattered over the other axes and their count is rounded
        # up: with a floor count the trailing partial group can wrap back onto group 0
        # and repeat coordinates that were meant to span the index space.
        used = max(1, -(-nnz // cols))
        step = max(1, n_groups // used)
        group_id = ((index // cols) * step) % n_groups
        dim_coord = index % cols
        parts = torch.unravel_index(group_id, group_shape) if group_shape else ()
        axes = iter(parts)
        flat = torch.stack(
            [dim_coord if a == dim else next(axes) for a in range(sparse_dim)]
        )
    else:
        # `dim` reduces over a dense dim: no sparse coordinate positions form the group,
        # so the entries are spread evenly over the sparse index space.
        spread = math.prod(shape[:sparse_dim])
        step = max(1, spread // nnz)
        flat = torch.stack(
            torch.unravel_index(
                (torch.arange(nnz, device=device) * step) % spread, shape[:sparse_dim]
            )
        )
    return flat.to(torch.long)


def _coo(shape, sparse_dim, dim, nnz, dtype, device):
    # Values are generated directly: benchmark.utils.generate_tensor_input returns None
    # for float64 because consts.FLOAT_DTYPES omits it.
    size = tuple(shape)
    values = torch.randn((nnz,) + size[sparse_dim:], dtype=dtype, device=device)
    indices = _coords(shape, sparse_dim, dim, nnz, device)
    return torch.sparse_coo_tensor(indices, values, torch.Size(size), device=device)


def _case_fn(shape, dtype):
    del dtype
    sparse_shape, nnz, sparse_dim, dim = _plan_fields(shape)
    yield base.BenchmarkCasePlan(
        shape={"sparse_shape": list(sparse_shape), "nnz": nnz},
        params={
            "sparse_dim": sparse_dim,
            "dense_dim": len(sparse_shape) - sparse_dim,
            "dim": dim,
        },
        builder_args=(sparse_shape, nnz, sparse_dim, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, nnz, sparse_dim, dim = plan.builder_args
    grad_output = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    output = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    self_ = _coo(sparse_shape, sparse_dim, dim, nnz, dtype, device)
    # Flat tuple of positional arguments, then a trailing empty keyword mapping so the
    # value of dim is never mistaken for it.
    return grad_output, output, dim, self_, {}


class SparseLogSoftmaxBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_CASES)


@pytest.mark._sparse_log_softmax_backward_data
def test__sparse_log_softmax_backward_data_benchmark():
    bench = SparseLogSoftmaxBackwardBenchmark(
        op_name="_sparse_log_softmax_backward_data",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_log_softmax_backward_data,
        gems_op=getattr(flag_gems, "_sparse_log_softmax_backward_data", None),
        dtypes=_DTYPES,
    )
    bench.run()

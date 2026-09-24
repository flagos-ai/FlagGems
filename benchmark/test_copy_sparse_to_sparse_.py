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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False)
# -> Tensor(a!)
#
# Performance-relevant sparse COO layouts: (shape, sparse_dim, nnz). ``nnz`` is
# always <= prod(shape[:sparse_dim]) so the requested number of stored entries
# actually fits the sparse dimensions. Both a small and a large nnz (relative to
# the sparse extent) are included because the copy cost tracks the stored-entry
# count, not the logical shape.

_COPY_SPARSE_SHAPES = [
    ((65536,), 1, 32768),  # 1-D all-sparse
    ((1024, 1024), 2, 262144),  # 2-D COO, 25% density
    ((4096, 4096), 2, 1048576),  # 2-D COO, large sparse extent
    ((16, 1024, 1024), 2, 8192),  # 3-D hybrid, low density
    ((16, 1024, 1024), 2, 65536),  # 3-D hybrid, medium density
    ((16, 1024, 1024), 2, 524288),  # 3-D hybrid, high density
    ((16, 1024, 1024), 3, 8192),  # 3-D all-sparse
    ((256, 2048, 128), 2, 524288),  # 3-D hybrid, dense extent 128
    ((8, 16, 64, 64), 4, 262144),  # 4-D all-sparse
]


def _make_sparse_input(shape, sparse_dim, nnz, dtype, device):
    """Deterministic coalesced sparse COO input on ``device``."""
    gen = torch.Generator("cpu").manual_seed(0)
    indices = torch.stack(
        [
            torch.randint(0, shape[dim], (nnz,), generator=gen)
            for dim in range(sparse_dim)
        ]
    )
    dense_shape = tuple(shape[sparse_dim:])
    values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=gen)
    return torch.sparse_coo_tensor(
        indices.to(device), values.to(device), tuple(shape), device=device
    )


def _case_fn(shape, dtype):
    # ``shape`` is the (shape, sparse_dim, nnz) descriptor from
    # _COPY_SPARSE_SHAPES; the plan carries it through builder_args so the
    # input builder can reconstruct both tensors without a legacy input_fn.
    del dtype
    shape, sparse_dim, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape, "sparse_dim": sparse_dim, "nnz": nnz},
        params={"non_blocking": False},
        builder_args=(shape, sparse_dim, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim, nnz = plan.builder_args
    # The destination is deliberately a small empty tensor: copy_sparse_to_sparse_
    # resizes it to the source structure, so the measured work is index/value
    # materialization rather than any destination allocation done by the caller.
    dst = torch.sparse_coo_tensor(
        torch.empty((sparse_dim, 0), dtype=torch.long, device=device),
        torch.empty((0,) + tuple(shape[sparse_dim:]), dtype=dtype, device=device),
        tuple(shape),
        device=device,
    )
    src = _make_sparse_input(shape, sparse_dim, nnz, dtype, device)
    return dst, src, {"non_blocking": plan.params["non_blocking"]}


class CopySparseToSparseBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with the dedicated sparse layout set.

    ``core_shapes.yaml`` has no entry for this op, so the shape list is supplied
    as defaults; operator entries in a shape file can override them.
    """

    DEFAULT_SHAPE_DESC = "sparse_coo(shape, sparse_dim, nnz)"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_COPY_SPARSE_SHAPES)


@pytest.mark.copy_sparse_to_sparse_
def test_copy_sparse_to_sparse_():
    # Override first, then the direct flag_gems callable if this checkout ships
    # one; ``None`` lets the harness fall back to its configured gem-routing.
    # The override injected by KernelGen is still honored either way.
    gems_op = getattr(flag_gems, "copy_sparse_to_sparse_", None)
    bench = CopySparseToSparseBenchmark(
        op_name="copy_sparse_to_sparse_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.copy_sparse_to_sparse_,
        gems_op=gems_op,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()

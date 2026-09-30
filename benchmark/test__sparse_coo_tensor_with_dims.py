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
#
# Benchmark for aten::_sparse_coo_tensor_with_dims. The measured work is an
# empty-COO allocation and its metadata: a zero-nnz tensor stores nothing per
# element, so a large extent costs metadata only and no requested shape is
# dropped or capped. No public benchmark family covers an allocation-only
# operator, hence the two-phase GenericBenchmark with an explicit
# case_fn/build_inputs_fn pair.

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Allocation descriptors spanning ranks 1-5 and dense tails. They are unioned
# with the shapes resolved from core_shapes.yaml (or the shared default grid)
# instead of replacing them.
SPARSE_COO_SHAPES = [
    (16384,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# Native-valid (sparse_dim, dense_dim) pairs per rank; the operator requires
# sparse_dim + dense_dim == len(size). A rank outside this table (e.g. a 0-D or
# >5-D entry from a custom shape file) keeps one valid split rather than being
# dropped.
_SPLITS = {
    1: [(1, 0)],
    2: [(2, 0), (1, 1)],
    3: [(3, 0), (2, 1)],
    4: [(4, 0), (2, 2)],
    5: [(5, 0), (3, 2)],
}

_BENCH_DTYPES = consts.FLOAT_DTYPES + [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
]


def _case_fn(shape, dtype):
    del dtype
    for sparse_dim, dense_dim in _SPLITS.get(len(shape), [(len(shape), 0)]):
        yield base.BenchmarkCasePlan(
            shape={"size": list(shape)},
            params={"sparse_dim": sparse_dim, "dense_dim": dense_dim},
            builder_args=(sparse_dim, dense_dim, list(shape)),
        )


def _build_inputs_fn(plan, dtype, device):
    # Flat positional arguments plus a trailing kwargs dict: the form
    # unpack_to_args_kwargs expects.
    sparse_dim, dense_dim, size = plan.builder_args
    return (
        sparse_dim,
        dense_dim,
        size,
        {"dtype": dtype, "layout": torch.sparse_coo, "device": device},
    )


def _build_inputs_fn_out(plan, dtype, device):
    # The out buffer must already match the requested size: resizing a sparse
    # output is not implemented, so a mismatched buffer is rejected natively.
    sparse_dim, dense_dim, size = plan.builder_args
    out = torch.sparse_coo_tensor(
        torch.empty((sparse_dim, 0), dtype=torch.int64, device=device),
        torch.empty((0,) + tuple(size[sparse_dim:]), dtype=dtype, device=device),
        size=tuple(size),
        device=device,
    )
    return (sparse_dim, dense_dim, size, {"out": out})


class SparseCooTensorWithDimsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None, **kwargs):
        # Normal resolution (--shape_file, core_shapes.yaml, then the shared
        # default grid) unioned with the allocation descriptors, so a requested
        # extent is never narrowed.
        super().set_shapes(shape_file_path or self.DEFAULT_SHAPE_FILES, **kwargs)
        self.shapes = list(dict.fromkeys(self.shapes + SPARSE_COO_SHAPES))


# getattr(..., None) keeps the module importable (and --list-cases working)
# before a candidate exists; the KernelGen override is picked up at runtime.
@pytest.mark.sparse_coo_tensor_with_dims
def test_sparse_coo_tensor_with_dims():
    bench = SparseCooTensorWithDimsBenchmark(
        op_name="_sparse_coo_tensor_with_dims",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_coo_tensor_with_dims,
        gems_op=getattr(flag_gems, "_sparse_coo_tensor_with_dims", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()


@pytest.mark.sparse_coo_tensor_with_dims
def test_sparse_coo_tensor_with_dims_out():
    bench = SparseCooTensorWithDimsBenchmark(
        op_name="_sparse_coo_tensor_with_dims",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._sparse_coo_tensor_with_dims.out,
        gems_op=getattr(flag_gems, "_sparse_coo_tensor_with_dims", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()

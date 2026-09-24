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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# The operator name starts with an underscore, which pytest's MarkGenerator cannot
# synthesise, so the marker is registered explicitly.
setattr(
    pytest.mark,
    "_sparse_softmax",
    MarkDecorator(Mark("_sparse_softmax", (), {}, _ispytest=True), _ispytest=True),
)

# The default half_to_float=False form is native for float32/float64 only, so the
# fp16/bf16 part of consts.FLOAT_DTYPES is not measurable for this operator; fp64 is
# gated on the target's capability flag.
_BENCH_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)

# Work per case scales with nnz and with the normalization group count, the product
# of every axis other than dim, so both are varied across the descriptors. The nnz=0
# descriptor keeps the empty-reduction workload measurable.
_SPARSE_SOFTMAX_SHAPES = [
    ((1024, 1024), 0, 1),
    ((1024, 1024), 262144, 1),
    ((1024, 1024), 1048576, 1),
    ((4096, 4096), 1048576, 1),
    ((2048, 2048, 64), 1048576, 2),
    ((256, 256, 256), 1048576, 2),
]


def _case_fn(shape, dtype):
    # Listing metadata only: one plan per descriptor keeps nnz and dim in the case
    # id/params and defers construction to _build_inputs_fn.
    del dtype
    sparse_shape, nnz, softmax_dim = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz, "dim": softmax_dim},
        builder_args=(sparse_shape, nnz, softmax_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    # Coordinates are drawn with replacement, as in real sparse data, but the tensor
    # is coalesced before timing, so duplicate merging is outside the measurement and
    # both callables receive the same coalesced input. nnz is used as requested;
    # nnz=0 builds a valid empty COO. half_to_float is passed by name because a
    # positional bool binds ScalarType on this Torch.
    sparse_shape, nnz, softmax_dim = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, extent, (nnz,), dtype=torch.long, device=device)
            for extent in sparse_shape
        ]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(
        indices, values, sparse_shape, dtype=dtype, device=device
    )
    return inp.coalesce(), {"dim": softmax_dim, "half_to_float": False}


class SparseSoftmaxBenchmark(OperatorBenchmark):
    # core_shapes.yaml holds no meaningful dense shapes for a sparse operator, so the
    # (shape, nnz, dim) descriptors above are the default shape set; an explicitly
    # supplied default set or shape file still wins.
    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        if default_shapes is None:
            default_shapes = _SPARSE_SOFTMAX_SHAPES
        super().set_shapes(shape_file_path, default_shapes=default_shapes)


@pytest.mark._sparse_softmax
def test__sparse_softmax():
    bench = SparseSoftmaxBenchmark(
        op_name="_sparse_softmax",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_softmax.default,
        gems_op=getattr(flag_gems, "_sparse_softmax", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()

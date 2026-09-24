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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_values`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly
# on the MarkGenerator so ``@pytest.mark._values`` and ``-m _values`` both
# work.
setattr(
    pytest.mark,
    "_values",
    MarkDecorator(Mark("_values", (), {}, _ispytest=True), _ispytest=True),
)

# (sparse_shape, dense_shape, nnz). _values returns the (nnz,) + dense_shape
# values tensor of a sparse COO tensor — a metadata accessor whose result is an
# alias of the input's internal values storage, so its cost is proportional to
# the size of the returned values tensor (nnz * prod(dense_shape)) and is
# independent of the logical (sparse) extent. Benchmark a spread of sparse
# ranks, dense ranks and nnz values. nnz * prod(dense_shape) is capped at ~8.4M
# elements so the float32/float16/bfloat16 inputs stay small on device (the
# sparse tensor only stores nnz entries, so the logical size can be much
# larger).
_VALUES_SHAPES = [
    ((1024, 1024), (), 65536),
    ((1024, 1024), (), 1048576),
    ((1024, 1024), (16,), 262144),
    ((256, 256, 256), (), 1048576),
    ((128, 128, 128, 128), (8,), 1048576),
    ((4096, 4096), (4,), 1048576),
]


def _case_fn(shape, dtype):
    # One BenchmarkCasePlan per (sparse_shape, dense_shape, nnz) triple; the
    # plan carries the builder args so build_inputs_fn materializes the sparse
    # input lazily for the selected dtype.
    del dtype
    sparse_shape, dense_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape + dense_shape},
        params={"nnz": nnz},
        builder_args=(sparse_shape, dense_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, dense_shape, nnz = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in sparse_shape
        ]
    )
    values = utils.generate_tensor_input((nnz,) + tuple(dense_shape), dtype, device)
    size = tuple(sparse_shape) + tuple(dense_shape)
    inp = torch.sparse_coo_tensor(indices, values, size, device=device)
    return inp, {}


class ValuesBenchmark(OperatorBenchmark):
    # _values is a sparse metadata accessor; there are no meaningful dense
    # shapes in core_shapes.yaml, so benchmark dedicated (sparse_shape,
    # dense_shape, nnz) triples instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_VALUES_SHAPES)


@pytest.mark._values
def test__values():
    bench = ValuesBenchmark(
        op_name="_values",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._values,
        gems_op=getattr(flag_gems, "_values", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

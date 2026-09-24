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

# ``_dimV`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimV`` and ``-m _dimV`` both work.
setattr(
    pytest.mark,
    "_dimV",
    MarkDecorator(Mark("_dimV", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimV(Tensor self) -> int reports the dense dimension count of a sparse
# tensor. It is a pure metadata query (the measured work is dispatch and layout
# introspection, never data movement), and dense / SparseCsr tensors raise
# NotImplementedError for it, so every benchmark input is a sparse COO tensor.
#
# Case descriptors: (shape, dense_dim), the exact complement of the _dimI
# benchmark's (shape, sparse_dim) pairs, covering
#   * all-sparse layouts (dense_dim == 0), ranks 1-5;
#   * hybrid layouts (0 < dense_dim < ndim) of rank 2-5.
# The logical shapes are performance-relevant while the actual allocation stays
# tiny, because nnz is fixed and small.
_BENCH_CASES = [
    ((256,), 0),
    ((64, 64), 0),
    ((1024, 1024), 0),
    ((1024, 1024), 1),
    ((20, 320, 15), 0),
    ((20, 320, 15), 1),
    ((64, 512, 512), 0),
    ((64, 512, 512), 1),
    ((16, 1024, 1024, 16), 1),
    ((8, 16, 16, 16, 16), 0),
]

# Number of stored entries for every benchmark case: the op is O(1), so nnz only
# affects input allocation, not the measured call.
_DIMV_NNZ = 1024


def _make_sparse_input(shape, dense_dim, dtype, device, nnz=_DIMV_NNZ, seed=0):
    """Sparse COO input whose reported ``dense_dim`` is ``dense_dim``.

    Indices are generated deterministically on the CPU and the values use the
    shared benchmark input generator on the test device; the exact numbers do
    not affect the measured op (it only reads layout metadata).
    """
    shape = tuple(shape)
    sparse_dim = len(shape) - dense_dim
    if sparse_dim < 1:
        raise ValueError("sparse COO tensors need at least one sparse dimension")
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    gen = torch.Generator("cpu").manual_seed(seed)
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = utils.generate_tensor_input((nnz,) + dense_shape, dtype, device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _case_fn(case, dtype):
    # ``set_shapes`` feeds each (shape, dense_dim) pair from _BENCH_CASES
    # through case_fn as one case descriptor.
    del dtype
    shape, dense_dim = case
    yield base.BenchmarkCasePlan(
        shape={"input": tuple(shape)},
        params={"dense_dim": dense_dim},
        builder_args=(tuple(shape), dense_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dense_dim = plan.builder_args
    inp = _make_sparse_input(shape, dense_dim, dtype, device)
    return inp, {}


class DimVBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark whose inputs are sparse COO tensors covering
    all-sparse and hybrid sparse+dense layouts."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark._dimV
def test__dimV():
    bench = DimVBenchmark(
        op_name="_dimV",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._dimV,
        gems_op=getattr(flag_gems, "_dimV", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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
from . import base, consts

# ``_dimI`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._dimI`` and ``-m _dimI`` both work.
setattr(
    pytest.mark,
    "_dimI",
    MarkDecorator(Mark("_dimI", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_dimI(Tensor self) -> int reports the sparse dimension count of a sparse
# tensor. It is a pure metadata query (the measured work is dispatch and layout
# introspection, never data movement), and dense / SparseCsr tensors raise
# NotImplementedError for it, so every benchmark input is a sparse COO tensor.
#
# Case descriptors: (shape, sparse_dim), covering
#   * all-sparse layouts (sparse_dim == ndim, dense_dim == 0), ranks 1-5;
#   * hybrid layouts (0 < sparse_dim < ndim) of rank 2-5.
# The logical shapes are performance-relevant while the actual allocation stays
# tiny, because nnz is fixed and small.
_BENCH_CASES = [
    ((256,), 1),
    ((64, 64), 2),
    ((1024, 1024), 2),
    ((1024, 1024), 1),
    ((20, 320, 15), 3),
    ((20, 320, 15), 2),
    ((64, 512, 512), 3),
    ((64, 512, 512), 2),
    ((16, 1024, 1024, 16), 3),
    ((8, 16, 16, 16, 16), 5),
]

# Number of stored entries for every benchmark case: the op is O(1), so nnz only
# affects input allocation, not the measured call.
_DIMI_NNZ = 1024


def _make_sparse_values(shape, dtype, device):
    """Values for the sparse payload; the exact numbers do not affect the
    measured op (it only reads layout metadata)."""
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype.is_floating_point:
        try:
            return torch.randn(shape, dtype=dtype, device=device)
        except (RuntimeError, TypeError):
            # e.g. float8_* has no native randn: build in fp32 then cast.
            return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    return torch.randint(0, 8, shape, dtype=dtype, device=device)


def _make_sparse_input(shape, sparse_dim, dtype, device, nnz=_DIMI_NNZ, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = tuple(shape[:sparse_dim])
    dense_shape = tuple(shape[sparse_dim:])
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = _make_sparse_values((nnz,) + dense_shape, dtype, device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _case_fn(case, dtype):
    # ``set_shapes`` feeds each (shape, sparse_dim) pair from _BENCH_CASES
    # through case_fn as one case descriptor.
    del dtype
    shape, sparse_dim = case
    yield base.BenchmarkCasePlan(
        shape={"input": tuple(shape)},
        params={"sparse_dim": sparse_dim},
        builder_args=(tuple(shape), sparse_dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim = plan.builder_args
    inp = _make_sparse_input(shape, sparse_dim, dtype, device)
    return inp, {}


class DimIBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark whose inputs are sparse COO tensors covering
    all-sparse and hybrid sparse+dense layouts."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_CASES)


@pytest.mark._dimI
def test__dimI():
    bench = DimIBenchmark(
        op_name="_dimI",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._dimI,
        gems_op=getattr(flag_gems, "_dimI", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

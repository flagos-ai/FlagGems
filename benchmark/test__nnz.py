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

# ``_nnz`` starts with an underscore, and ``pytest.mark`` refuses to generate a
# marker via attribute access for such names. Register it directly on the
# MarkGenerator so ``@pytest.mark._nnz`` and ``-m _nnz`` both work.
setattr(
    pytest.mark,
    "_nnz",
    MarkDecorator(Mark("_nnz", (), {}, _ispytest=True), _ispytest=True),
)

# aten::_nnz(Tensor self) -> int reports the number of stored entries of a
# sparse tensor. It is a pure metadata query (the measured work is dispatch and
# layout introspection, never data movement), and dense tensors raise
# NotImplementedError for it, so every benchmark input is a sparse tensor. The
# logical shapes below cover ranks 2-4 at representative sizes; the actual
# device allocation stays modest because nnz is fixed and small.
_NNZ_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
    (16, 1024, 1024, 16),
]

# Number of stored entries for every benchmark case: the op is O(1), so nnz
# only affects input allocation, not the measured call.
_NNZ = 1024


def _make_sparse_coo_input(shape, sparse_dim, dtype, device, nnz=_NNZ, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:sparse_dim]
    dense_shape = shape[sparse_dim:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    )
    values = utils.generate_tensor_input((nnz,) + dense_shape, dtype, device)
    return torch.sparse_coo_tensor(indices, values, shape, device=device)


def _make_sparse_csr_input(shape, dtype, device, nnz=_NNZ):
    # 2-D (rows, cols) or batched 3-D (batch, rows, cols); every batch stores
    # the same nnz entries (shared crow/col pattern).
    if len(shape) == 2:
        rows, cols = shape
    else:
        _, rows, cols = shape
    assert 0 <= nnz <= rows * cols
    counts = torch.full((rows,), nnz // rows, dtype=torch.long)
    counts[: nnz % rows] += 1
    crow_indices = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    col_indices = torch.arange(nnz) - torch.repeat_interleave(crow_indices[:-1], counts)
    if len(shape) == 3:
        crow_indices = crow_indices.expand(shape[0], -1).contiguous()
        col_indices = col_indices.expand(shape[0], -1).contiguous()
        values = utils.generate_tensor_input((shape[0], nnz), dtype, device)
    else:
        values = utils.generate_tensor_input((nnz,), dtype, device)
    return torch.sparse_csr_tensor(
        crow_indices, col_indices, values, shape, device=device
    )


def _case_fn(shape, dtype):
    del dtype
    # Cover all-sparse (2-D) and mixed sparse+dense layouts (3-D/4-D); every
    # derived sparse_dim stays within [1, ndim] so additional shapes merged in
    # by the comprehensive bench level remain valid.
    sparse_dim = len(shape) if len(shape) <= 2 else len(shape) - 1
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"sparse_dim": sparse_dim, "nnz": _NNZ, "layout": "coo"},
        builder_args=(shape, sparse_dim, "coo"),
    )
    # Also exercise the SparseCsr dispatch path for the 2-D / batched 3-D
    # shapes where a CSR layout exists.
    if len(shape) in (2, 3):
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"sparse_dim": 0, "nnz": _NNZ, "layout": "csr"},
            builder_args=(shape, None, "csr"),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, sparse_dim, layout = plan.builder_args
    if layout == "csr":
        inp = _make_sparse_csr_input(shape, dtype, device)
    else:
        inp = _make_sparse_coo_input(shape, sparse_dim, dtype, device)
    return inp, {}


class NnzBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark whose inputs are sparse COO/CSR tensors."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_NNZ_SHAPES)


@pytest.mark._nnz
def test__nnz():
    bench = NnzBenchmark(
        op_name="_nnz",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._nnz,
        # flag_gems has no public ``_nnz`` direct callable; the candidate is
        # supplied by the KernelGen process-local override keyed on the public
        # operator name "_nnz" (resolved via Benchmark._candidate_call
        # inside the benchmark runner).
        gems_op=getattr(flag_gems, "_nnz", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

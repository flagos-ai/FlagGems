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

"""Benchmark for ``aten::_sparse_mm_reduce_impl_backward``.

Torch registers the operator for SparseCsrCPU only, so ``self`` (CSR),
``grad_out``, ``weight`` and ``arg_out`` are CPU operands and the input builder
deliberately ignores the configured device: reference and candidate receive the
same CPU tensors.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# (m, k, n, nnz): ``self`` is an (m, k) CSR matrix with nnz stored entries per
# row, ``grad_out`` is (m, n) and ``weight`` is (k, n). nnz drives how much of
# the sparse matrix each reduce touches, so the descriptors span tiny to large.
_SHAPES = [
    (1, 1, 1, 1),
    (0, 4, 3, 0),
    (4, 0, 3, 0),
    (4, 3, 0, 1),
    (4, 3, 2, 0),
    (64, 64, 64, 4),
    (256, 512, 128, 8),
    (320, 480, 15, 8),
    (1024, 1024, 1024, 4),
    (4096, 4096, 512, 8),
]
_REDUCES = ["sum", "mean", "amax", "amin"]
_MASK = [True, True]


def _descriptor(shape):
    # A short entry cannot describe a sparse operand with a fixed nnz, so it is
    # rejected instead of being silently replaced or padded with invented
    # extents.
    sizes = tuple(int(size) for size in shape)
    if len(sizes) != 4:
        raise ValueError(f"expected an (m, k, n, nnz) descriptor, got {sizes}")
    return sizes


def _stored_columns(m, k, nnz):
    cols = torch.arange(nnz) + torch.arange(m).unsqueeze(1) * 2
    return (cols % k if nnz else cols).sort(dim=1).values


def _case_fn(shape, dtype):
    # Plans hold metadata only, so --list-cases allocates no tensor; each
    # descriptor carries every reduce that the operator supports.
    del dtype
    m, k, n, nnz = _descriptor(shape)
    for reduce in _REDUCES:
        yield base.BenchmarkCasePlan(
            shape={"self": [m, k], "grad_out": [m, n], "weight": [k, n]},
            params={"reduce": reduce, "nnz": nnz, "output_mask": list(_MASK)},
            builder_args=((m, k, n, nnz), reduce),
        )


def _build_inputs_fn(plan, dtype, device):
    del device  # SparseCsrCPU-only operator: the operands stay on CPU.
    (m, k, n, nnz), reduce = plan.builder_args
    cols = _stored_columns(m, k, nnz)
    crow = torch.arange(m + 1, dtype=torch.int64) * nnz
    values = torch.empty((m, nnz), dtype=dtype).uniform_(-1, 1)
    grad_out = torch.empty((m, n), dtype=dtype).uniform_(-1, 1)
    weight = torch.empty((k, n), dtype=dtype).uniform_(-1, 1)
    self_csr = torch.sparse_csr_tensor(
        crow, cols.reshape(-1), values.reshape(-1), size=(m, k)
    )
    # The native forward supplies absolute CSR storage indices for amax/amin.
    _, arg_out = torch.ops.aten._sparse_mm_reduce_impl(
        self_csr.detach().requires_grad_(True), weight, reduce
    )
    return self_csr, grad_out, weight, reduce, arg_out, list(_MASK)


class SparseMmReduceBackwardBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_DESC = "M, K, N, NNZ"

    def set_shapes(self, shape_file_path=None):
        # Four extents describe a CSR workload including its row density. Keep
        # every compatible shared descriptor and all explicit workloads.
        super().set_shapes(shape_file_path)
        loaded = [tuple(shape) for shape in self.shapes if len(shape) == 4]
        self.shapes = list(dict.fromkeys([*loaded, *_SHAPES]))


@pytest.mark.sparse_mm_reduce_impl_backward
def test__sparse_mm_reduce_impl_backward():
    bench = SparseMmReduceBackwardBenchmark(
        op_name="_sparse_mm_reduce_impl_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_mm_reduce_impl_backward,
        gems_op=getattr(flag_gems, "_sparse_mm_reduce_impl_backward", None),
        dtypes=[*consts.FLOAT_DTYPES, torch.float64],
    )
    bench.run()

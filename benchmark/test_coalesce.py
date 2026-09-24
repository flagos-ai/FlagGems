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

# (sparse shape, nnz) workload descriptors. Coalescing work scales with nnz, so
# the tensors are sized for a meaningful measurement while staying small enough
# to build repeatedly; nnz is always > numel(shape), which guarantees duplicate
# coordinates and therefore real sort/merge work.
_COALESCE_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 262144),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((2048, 2048), 2097152),
    ((256, 256, 256), 1048576),
]


def _case_fn(shape, dtype):
    # Two-phase GenericBenchmark: the shape entries are the (shape, nnz)
    # workload descriptors; emitting one BenchmarkCasePlan per descriptor keeps
    # nnz in the case id/params and defers tensor construction.
    del dtype
    sparse_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz},
        builder_args=(sparse_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    # Draw coordinates with replacement so the input is uncoalesced (the CUDA
    # reference asserts on that) and coalescing has to sort/merge.
    sparse_shape, nnz = plan.builder_args
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
            for dim in sparse_shape
        ]
    )
    values = torch.randn(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(indices, values, sparse_shape, device=device)
    return inp, {}


class CoalesceBenchmark(OperatorBenchmark):
    # coalesce is a sparse op, so there are no meaningful dense shapes in
    # core_shapes.yaml; benchmark the dedicated (shape, nnz) descriptors above.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_COALESCE_SHAPES)


@pytest.mark.coalesce
def test_coalesce():
    bench = CoalesceBenchmark(
        op_name="coalesce",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        # flag_gems has no public ``coalesce`` direct callable in this checkout;
        # the candidate is supplied by the KernelGen process-local override keyed
        # on the public operator name "coalesce" (resolved via
        # Benchmark._candidate_call inside the benchmark).
        torch_op=torch.ops.aten.coalesce,
        gems_op=getattr(flag_gems, "coalesce", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

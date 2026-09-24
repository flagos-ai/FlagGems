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

import math

import pytest
import torch

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# sparse_mask gathers values from a dense ``self`` at a sparse mask's index
# positions and returns a sparse COO tensor. There is no public Benchmark family
# for sparse gather ops, so the benchmark uses the two-phase GenericBenchmark
# (case_fn + build_inputs_fn): case_fn yields one BenchmarkCasePlan per
# (shape, nnz_ratio) workload and build_inputs_fn materializes the dense self
# and the coalesced sparse mask lazily. The candidate is resolved at run time
# from the process-local override (Benchmark._candidate_call) via
# GenericBenchmark's _candidate_call; flag_gems.sparse_mask does not exist
# yet as an attribute, so the direct-callable default is fetched with getattr and
# may be None. The perf reference is the aten op itself
# (torch.ops.aten.sparse_mask) and both are called with the same (self, mask)
# signature.
#
# The benchmark work scales with the number of mask nonzeros and the size of the
# dense self, so the shapes pair moderate square layouts with a fixed nnz ratio;
# the ratio drops for the largest shapes to keep the mask/build cost bounded.
_SPARSE_MASK_SHAPES = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
]


def _nnz_ratio(shape):
    # 10% nonzeros on the small layouts, 2% on the large ones.
    return 0.1 if math.prod(shape) <= 1024 * 1024 else 0.02


def _case_fn(shape, dtype):
    # yield generates one BenchmarkCasePlan per (shape, dtype) parametrization.
    del dtype
    ratio = _nnz_ratio(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": shape, "mask": shape},
        params={"nnz_ratio": ratio},
        builder_args=(shape, ratio),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, nnz_ratio = plan.builder_args
    self_t = torch.randn(shape, dtype=dtype, device=device)
    mask_dense = torch.rand(shape, device=device) > (1.0 - nnz_ratio)
    mask = mask_dense.to_sparse()
    return self_t, mask, {}


class SparseMaskBenchmark(OperatorBenchmark):
    # sparse_mask has no meaningful dense shapes in core_shapes.yaml, so
    # benchmark the dedicated square layouts above.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SPARSE_MASK_SHAPES)


@pytest.mark.sparse_mask
def test_sparse_mask():
    bench = SparseMaskBenchmark(
        op_name="sparse_mask",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_mask,
        gems_op=getattr(flag_gems, "sparse_mask", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

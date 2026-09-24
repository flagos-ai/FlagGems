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
from . import base, consts, utils

# aten::sparse_coo_tensor (the explicit ``indices_size`` overload) builds a
# sparse COO tensor from raw (indices, values, size) components. Construction
# work scales with the stored nnz and the number of sparse dims, not with the
# dense logical size, so each case pairs a large logical tensor shape with a
# bounded nnz. Every shape keeps a 2-D sparse grid (sparse_dim == 2) with the
# remaining dims dense, matching the dominant COO usage; the last case carries
# one dense dim. Each pair is a (tensor_shape, nnz) case.
_BENCH_SHAPES = [
    ((4096, 4096), 10000),
    ((4096, 4096), 100000),
    ((1024, 8192), 50000),
    ((8192, 1024), 50000),
    ((2048, 2048, 64), 20000),
]


def _make_coo_inputs(shape, nnz, dtype, device, seed=0):
    # Deterministic CPU-side generation of a valid (indices, values) pair for
    # the explicit-size overload: sparse_dim == 2 with the trailing dims dense,
    # every stored coordinate in range and the values on the benchmark device.
    gen = torch.Generator("cpu").manual_seed(seed)
    sparse_shape = shape[:2]
    dense_shape = shape[2:]
    indices = torch.stack(
        [
            torch.randint(0, dim, (nnz,), dtype=torch.long, generator=gen)
            for dim in sparse_shape
        ]
    ).to(device)
    values = utils.generate_tensor_input((nnz,) + tuple(dense_shape), dtype, device)
    return indices, values


def _case_fn(shape, dtype):
    del dtype
    tensor_shape, nnz = shape
    yield base.BenchmarkCasePlan(
        shape={"input": tensor_shape},
        params={"nnz": nnz},
        builder_args=(tensor_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, nnz = plan.builder_args
    indices, values = _make_coo_inputs(tensor_shape, nnz, dtype, device)
    # The trailing dict is unpacked into kwargs by the benchmark runner, so
    # torch_op and gems_op both receive (indices, values, size, dtype=...,
    # device=...). dtype is passed explicitly: without it the sparse tensor
    # defaults to Float regardless of the values dtype.
    return indices, values, list(tensor_shape), {"dtype": dtype, "device": device}


class SparseCooTensorBenchmark(OperatorBenchmark):
    # sparse_coo_tensor is a sparse construction op; there are no meaningful
    # dense shapes in core_shapes.yaml, so benchmark dedicated
    # (tensor_shape, nnz) pairs instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_BENCH_SHAPES)

    def set_more_shapes(self):
        return []


@pytest.mark.sparse_coo_tensor
def test_sparse_coo_tensor():
    bench = SparseCooTensorBenchmark(
        op_name="sparse_coo_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.sparse_coo_tensor,
        # KernelGen injects the candidate via --override; the
        # direct module callable may not exist until the op is merged, in which
        # case the benchmark falls back to the dispatcher reference.
        gems_op=getattr(flag_gems, "sparse_coo_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

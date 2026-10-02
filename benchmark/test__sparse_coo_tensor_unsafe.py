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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# Construction cost scales with the stored nnz and the sparse rank, not with the
# logical dense size, so each descriptor pairs a large logical shape with a
# bounded nnz. A descriptor is (tensor_shape, nnz, sparse_dim); the dims after
# sparse_dim stay dense, covering the fully sparse layout as well as the
# leading-dims sparse layout with a dense tail.
_BENCH_CASES = [
    ((256,), 256, 1),
    ((1024, 1024), 10000, 2),
    ((20, 320, 15), 20000, 2),
    ((1024, 1024, 8), 20000, 2),
    ((16, 128, 64, 60), 20000, 4),
    ((16, 7, 57, 32, 29), 20000, 5),
]

_BENCH_DTYPES = consts.FLOAT_DTYPES


def _is_descriptor(spec):
    # A descriptor starts with a shape sequence; a plain extent list does not.
    return bool(spec) and isinstance(spec[0], (list, tuple))


def _plan(tensor_shape, nnz, sparse_dim):
    # Private builder_args keep the torch objects; the public metadata stays
    # JSON-compatible so case listing allocates nothing and runs no operator.
    return base.BenchmarkCasePlan(
        shape={"input": list(tensor_shape), "sparse_dim": sparse_dim},
        params={"nnz": nnz},
        builder_args=(tuple(tensor_shape), nnz, sparse_dim),
    )


def _case_fn(shape, dtype):
    del dtype
    if _is_descriptor(shape):
        yield _plan(*shape)
        return
    # A shape requested through --shape_file is the logical dense size of the
    # sparse tensor: every requested extent is kept and one value is stored per
    # element, i.e. the fully sparse reading of that extent list.
    tensor_shape = tuple(shape)
    yield _plan(tensor_shape, math.prod(tensor_shape), len(tensor_shape))


def _build_inputs_fn(plan, dtype, device):
    tensor_shape, nnz, sparse_dim = plan.builder_args
    if sparse_dim and nnz:
        indices = torch.stack(
            [
                torch.randint(0, int(dim), (nnz,), dtype=torch.int64, device=device)
                for dim in tensor_shape[:sparse_dim]
            ],
            0,
        )
    else:
        indices = torch.empty((sparse_dim, nnz), dtype=torch.int64, device=device)
    values = utils.generate_tensor_input(
        (nnz,) + tuple(tensor_shape[sparse_dim:]), dtype, device
    )
    # The result dtype and device come from values; the native dtype/device
    # keywords are ignored, so they are not passed here.
    return indices, values, list(tensor_shape)


class SparseCooTensorUnsafeBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over the (shape, nnz, sparse_dim) descriptors."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        for descriptor in _BENCH_CASES:
            if descriptor not in self.shapes:
                self.shapes.append(descriptor)


@pytest.mark.sparse_coo_tensor_unsafe
def test_sparse_coo_tensor_unsafe():
    bench = SparseCooTensorUnsafeBenchmark(
        op_name="_sparse_coo_tensor_unsafe",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_coo_tensor_unsafe,
        # KernelGen injects the candidate through --override; getattr keeps
        # listing and import working while the op is not merged yet.
        gems_op=getattr(flag_gems, "_sparse_coo_tensor_unsafe", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()

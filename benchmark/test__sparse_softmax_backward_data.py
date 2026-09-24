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

# The public operator name starts with an underscore, which pytest does not
# expose as a marker attribute, so register it on the MarkGenerator directly.
setattr(
    pytest.mark,
    "_sparse_softmax_backward_data",
    MarkDecorator(
        Mark("_sparse_softmax_backward_data", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# (sparse_shape, nnz, sparse_dim, dim) workload descriptors. sparse_dim == rank
# keeps every dim on the sparse-coordinate kernel; sparse_dim < rank stores dense
# value blocks and switches to the dense-values kernel as soon as dim >=
# sparse_dim, so the default rows cover both kernels.
_SSBD_SHAPES = [
    ((1024, 1024), 65536, 2, 0),
    ((2048, 2048), 1048576, 2, 1),
    ((256, 256, 256), 262144, 3, 0),
    ((4096, 4096, 64), 65536, 2, 2),
    ((1048576, 128), 65536, 1, 1),
    ((16, 7, 57, 32, 29), 6, 5, 4),
]

# Only SparseCPU / SparseCUDA kernels are registered and the CUDA kernel
# dispatches on AT_DISPATCH_FLOATING_TYPES, so the plan is the static device
# capability and is identical for --list-cases and for execution.
_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


def _case_fn(shape, dtype):
    # Two-phase benchmarking: one BenchmarkCasePlan per descriptor keeps nnz,
    # sparse_dim and dim in the case id/params and defers tensor construction.
    # The framework reads a trailing dict as kwargs and the elements before it as
    # positional arguments.
    del dtype
    sparse_shape, nnz, sparse_dim, dim = shape
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz, "sparse_dim": sparse_dim, "dim": dim},
        builder_args=(sparse_shape, nnz, sparse_dim, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    sparse_shape, nnz, sparse_dim, dim = plan.builder_args
    # sparse_dim == 0 is a valid descriptor whose index tensor has no rows.
    if sparse_dim == 0:
        indices = torch.empty(0, nnz, dtype=torch.long, device=device)
    else:
        indices = torch.stack(
            [
                torch.randint(0, extent, (nnz,), dtype=torch.long, device=device)
                for extent in sparse_shape[:sparse_dim]
            ]
        )
    value_shape = (nnz,) + tuple(sparse_shape[sparse_dim:])
    # grad_output and output are independent tensors: the index storage is cloned
    # because torch.sparse_coo_tensor keeps the tensor it is given.
    grad = torch.sparse_coo_tensor(
        indices,
        torch.randn(value_shape, dtype=dtype, device=device),
        sparse_shape,
        device=device,
    )
    output = torch.sparse_coo_tensor(
        indices.clone(),
        torch.randn(value_shape, dtype=dtype, device=device),
        sparse_shape,
        device=device,
    )
    self_tensor = torch.randn(1, dtype=dtype, device=device)
    return grad, output, dim, self_tensor, {}


class SparseSoftmaxBackwardDataBenchmark(OperatorBenchmark):
    # _sparse_softmax_backward_data takes sparse operands, so the dense shapes in
    # core_shapes.yaml do not apply; benchmark the (shape, nnz, sparse_dim, dim)
    # descriptors above unless the caller supplies a shape file entry.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SSBD_SHAPES)


@pytest.mark._sparse_softmax_backward_data
def test__sparse_softmax_backward_data():
    bench = SparseSoftmaxBackwardDataBenchmark(
        op_name="_sparse_softmax_backward_data",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_softmax_backward_data,
        gems_op=getattr(flag_gems, "_sparse_softmax_backward_data", None),
        dtypes=_DTYPES,
    )
    bench.run()

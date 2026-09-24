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

# aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor flattens every input
# to a contiguous 1-D tensor and concatenates them into one 1-D result. It is a
# bandwidth-bound data-movement op (copy + cat), so each case is a list of
# tensor shapes whose total element count is performance-relevant
# (1M - 12.6M elements).
FLATTEN_DENSE_TENSORS_SHAPES = [
    [(1024, 1024)],
    [(4096, 4096)],
    [(1024, 1024), (1024, 1024), (1024, 1024)],
    [(64, 512, 512)],
    [(2048, 2048), (2048, 2048)],
    [(16384, 256), (16384, 256), (16384, 256)],
]


def _case_fn(shape, dtype):
    del dtype
    numel = 0
    for s in shape:
        numel += torch.Size(s).numel()
    yield base.BenchmarkCasePlan(
        shape={"tensors": shape},
        params={"numel": numel},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (tensor_shapes,) = plan.builder_args
    tensors = [
        utils.generate_tensor_input(shape, dtype, device) for shape in tensor_shapes
    ]
    return tensors, {}


class FlattenDenseTensorsBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to list-of-shapes cases."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=FLATTEN_DENSE_TENSORS_SHAPES)

    def set_more_shapes(self):
        # Every case consumes a list of tensors, so the framework's bare-tuple
        # comprehensive shapes do not apply.
        return []


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors():
    bench = FlattenDenseTensorsBenchmark(
        op_name="flatten_dense_tensors",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.flatten_dense_tensors,
        gems_op=getattr(flag_gems, "flatten_dense_tensors", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

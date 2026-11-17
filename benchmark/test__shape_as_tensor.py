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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::_shape_as_tensor(self) materializes the logical shape as a fresh 1-D
# int64 CPU tensor. It is a pure metadata query: the measured work is the input
# allocation plus dispatch/shape introspection, never data movement, and the
# output always has one element per dimension. The shapes below cover the
# allocation cost across ranks 1-5, mirroring the _dim_arange benchmark.
_SHAPE_AS_TENSOR_SHAPES = [
    (2**20,),
    (2**24,),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (16, 1024, 1024, 16),
    (16, 7, 57, 32, 29),
]
_MASTER_SHAPES = [(16,), (1024,), (65536,), (1048576,)]


def _case_fn(shape, dtype):
    if dtype not in consts.FLOAT_DTYPES and shape not in _MASTER_SHAPES:
        return
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return (inp,)


class ShapeAsTensorBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with shapes tuned for _shape_as_tensor."""

    def set_shapes(self, shape_file_path=None):
        shapes = list(dict.fromkeys(_SHAPE_AS_TENSOR_SHAPES + _MASTER_SHAPES))
        super().set_shapes(shape_file_path, default_shapes=shapes)


@pytest.mark.underscore_shape_as_tensor
def test__shape_as_tensor():
    bench = ShapeAsTensorBenchmark(
        op_name="_shape_as_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._shape_as_tensor,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()

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

# ``_shape_as_tensor`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._shape_as_tensor`` and ``-m
# _shape_as_tensor`` both work.
setattr(
    pytest.mark,
    "_shape_as_tensor",
    MarkDecorator(Mark("_shape_as_tensor", (), {}, _ispytest=True), _ispytest=True),
)

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


def _case_fn(shape, dtype):
    del dtype
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
        super().set_shapes(shape_file_path, default_shapes=_SHAPE_AS_TENSOR_SHAPES)


@pytest.mark._shape_as_tensor
def test__shape_as_tensor():
    bench = ShapeAsTensorBenchmark(
        op_name="_shape_as_tensor",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._shape_as_tensor,
        # flag_gems has no public ``_shape_as_tensor`` direct callable yet; the
        # candidate is supplied by the KernelGen process-local override keyed on
        # the public operator name (resolved via
        # Benchmark._candidate_call inside the benchmark runner).
        gems_op=getattr(flag_gems, "_shape_as_tensor", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

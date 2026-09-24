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

# aten::data(Tensor self) -> Tensor returns a storage-sharing shallow copy of
# the input; it performs no arithmetic, so the benchmark measures dispatch plus
# view-materialization overhead across sizes. Moderate shapes are used instead
# of the generic 1G-element default since the op is zero-copy.
DATA_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (16, 128, 64, 60),
    (1024, 512, 256),
]


def _case_fn(shape, dtype):
    # One BenchmarkCasePlan per tensor shape; the plan carries the builder args
    # so build_inputs_fn materializes the input lazily for the selected dtype.
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {}


class DataBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark for the aten::data shallow-copy view op."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=DATA_SHAPES)

    def set_more_shapes(self):
        # The inherited 2**28 / 10000x65536 generic shapes would allocate
        # hundreds of MB extra for a zero-copy op; the dedicated shape list
        # above already covers small, medium and large tensors.
        return []


@pytest.mark.data
def test_data():
    bench = DataBenchmark(
        op_name="data",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.data,
        gems_op=getattr(flag_gems, "data", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

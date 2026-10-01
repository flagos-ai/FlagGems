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

"""Benchmark for ``aten::permute`` (a metadata-only view)."""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

PERMUTE_SHAPES = [
    (),
    (256,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


def _case_fn(shape, dtype):
    del dtype
    # Only JSON-safe metadata here; the tensor itself stays in builder_args.
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"dims": list(range(len(shape) - 1, -1, -1))},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    inp = utils.generate_tensor_input(plan.builder_args[0], dtype, device)
    return inp, {"dims": plan.params["dims"]}


class PermuteBenchmark(OperatorBenchmark):
    """Case-based permute benchmark; dims follow each shape's rank."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=PERMUTE_SHAPES)


@pytest.mark.permute
def test_permute():
    bench = PermuteBenchmark(
        op_name="permute",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.permute,
        gems_op=getattr(flag_gems, "permute", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

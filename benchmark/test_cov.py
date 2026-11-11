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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


_COV_SHAPES = [
    (32, 256),
    (64, 1024),
    (128, 1024),
    (256, 1024),
    (256, 4096),
    (512, 2048),
    (512, 4096),
]


class CovBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = _COV_SHAPES


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={}, builder_args=(dims,)
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.cov
def test_cov():
    bench = CovBenchmark(
        op_name="cov",
        torch_op=torch.cov,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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

from . import base
from .generated_operator_utils import OperatorBenchmark
from .test_linalg_cond import _MusaLinalgStaticBaseline

SHAPES = [(8, 8), (16, 16), (32, 32), (64, 64)]


class LinalgSvdBenchmark(_MusaLinalgStaticBaseline, OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(*B), M, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"full_matrices": base.vendor_name != "mthreads"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    if base.vendor_name == "mthreads":
        torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return inp, dict(plan.params)


@pytest.mark.linalg_svd
def test_linalg_svd():
    LinalgSvdBenchmark(
        op_name="linalg_svd",
        torch_op=torch.linalg.svd,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()



class LinalgSvdPrivateBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "(*B), M, N"


@pytest.mark.underscore_linalg_svd
def test__linalg_svd():
    LinalgSvdPrivateBenchmark(
        op_name="_linalg_svd",
        torch_op=torch.ops.aten._linalg_svd.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()

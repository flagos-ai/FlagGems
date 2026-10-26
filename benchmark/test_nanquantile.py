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

_DEFAULT_SHAPES = [
    (1,),
    (17,),
    (1024,),
    (1025,),
    (128, 128),
    (512, 512),
    (1024, 1024),
]


class NanquantileBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [(33, 65), (4096,), (256, 1024), (4096, 1024)]


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "q": [3]},
        params={"dim": -1, "interpolation": "linear"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp.reshape(-1)[::5] = float("nan")
    q = torch.tensor([0.1, 0.5, 0.9], dtype=dtype, device=device)
    return inp, q, -1


@pytest.mark.nanquantile
def test_nanquantile():
    NanquantileBenchmark(
        op_name="nanquantile",
        torch_op=torch.ops.aten.nanquantile.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()

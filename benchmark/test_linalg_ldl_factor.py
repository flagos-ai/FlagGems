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

LDL_FACTOR_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64)]


class LdlFactorBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LDL_FACTOR_SHAPES
        self.shape_desc = "N, N"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"hermitian": False},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    if base.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double values")
    (shape,) = plan.builder_args
    if base.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin cannot preserve float64 input storage")
    if base.vendor_name == "iluvatar" and dtype == torch.float64:
        pytest.skip("Iluvatar native ldl_factor does not support float64")
    torch.manual_seed(0)
    n = shape[-1]
    factory_device = (
        "cpu" if base.vendor_name == "ascend" and dtype == torch.float64 else device
    )
    inp = torch.randn(shape, dtype=dtype, device=factory_device)
    inp = inp @ inp.transpose(-2, -1) + n * torch.eye(
        n, dtype=dtype, device=factory_device
    )
    return inp.to(device), {"hermitian": False}


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
def test_linalg_ldl_factor():
    LdlFactorBenchmark(
        op_name="ldl_factor",
        torch_op=torch.linalg.ldl_factor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32, torch.float64],
    ).run()

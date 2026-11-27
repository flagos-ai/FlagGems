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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

CASES = [((4096, 256), 0), ((4096, 256), 1), ((4, 256, 3), 0), ((4, 256, 3), 2)]


class WeightNormBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CASES)
        self.shape_desc = "shape, dim"

    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype == torch.float16:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlunxin native _weight_norm does not support Float16")
        return super().build_inputs(case)


def _case_fn(shape, dtype):
    del dtype
    tensor_shape, dim = shape
    tensor_shape = tuple(tensor_shape)
    yield base.BenchmarkCasePlan(
        shape={"v": list(tensor_shape)},
        params={"dim": dim},
        builder_args=(tensor_shape, dim),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim = plan.builder_args
    torch.manual_seed(0)
    v = torch.randn(shape, dtype=dtype, device=device)
    g_shape = tuple(shape[index] if index == dim else 1 for index in range(len(shape)))
    g = torch.randn(g_shape, dtype=dtype, device=device)
    return v, g, dim


@pytest.mark.underscore_weight_norm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_underscore_weight_norm():
    WeightNormBenchmark(
        op_name="_weight_norm",
        torch_op=torch.ops.aten._weight_norm.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()

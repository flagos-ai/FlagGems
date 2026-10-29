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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(1024, 1024), (2048, 2048), (4096, 4096)]


class AmpForeachBenchmark(OperatorBenchmark):
    def build_inputs(self, case):
        if flag_gems.vendor_name == "kunlunxin" and case.dtype == torch.float16:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlunxin native Half AMP unscale returns incorrect values")
        return super().build_inputs(case)

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "primary gradient shape"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    second = (max(1, dims[0] // 2), *dims[1:])
    yield base.BenchmarkCasePlan(
        shape={"grads": [list(dims), list(second)]},
        params={"inv_scale": 0.5},
        builder_args=(dims, second),
    )


def _build_inputs_fn(plan, dtype, device):
    first_shape, second_shape = plan.builder_args
    torch.manual_seed(0)
    tensors = [
        torch.randn(first_shape, dtype=dtype, device=device),
        torch.randn(second_shape, dtype=dtype, device=device),
    ]
    found_inf = torch.tensor(0.0, dtype=torch.float32, device=device)
    inv_scale = torch.tensor(0.5, dtype=torch.float32, device=device)
    return tensors, found_inf, inv_scale


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
def test_amp_foreach_non_finite_check_and_unscale_():
    AmpForeachBenchmark(
        op_name="_amp_foreach_non_finite_check_and_unscale_",
        torch_op=torch._amp_foreach_non_finite_check_and_unscale_,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float16, torch.float32],
        fresh_inputs=True,
    ).run()

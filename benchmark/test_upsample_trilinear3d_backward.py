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


def _case_fn(shape, dtype, align_corners=False):
    del dtype
    batch, channel, height, width = shape
    depth = 4
    width = width // 4
    new_height = height // depth
    input_size = (batch, channel, depth, new_height, width)
    output_size = (depth * 2, new_height * 2, width * 2)
    yield base.BenchmarkCasePlan(
        shape={"grad_output": [batch, channel, *output_size]},
        params={
            "output_size": list(output_size),
            "input_size": list(input_size),
            "align_corners": align_corners,
        },
        builder_args=(input_size, output_size, align_corners),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    input_size, output_size, align_corners = plan.builder_args
    grad_output = torch.randn(
        (input_size[0], input_size[1], *output_size), device=device, dtype=dtype
    )
    return grad_output, output_size, input_size, align_corners, None, None, None


class UpsampleBenchmark(OperatorBenchmark):
    def build_inputs(self, case):
        needs_native = getattr(base.Config, "reference_only", False) or (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if (
            flag_gems.vendor_name == "kunlunxin"
            and case.dtype in (torch.float16, torch.bfloat16)
            and needs_native
        ):
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlunxin native trilinear3d backward supports float32 only")
        return super().build_inputs(case)

    def set_more_shapes(self):
        return None


@pytest.mark.upsample_trilinear3d_backward
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_upsample_trilinear3d_backward(align_corners):
    def case_fn(shape, dtype):
        yield from _case_fn(shape, dtype, align_corners)

    UpsampleBenchmark(
        op_name="upsample_trilinear3d_backward",
        torch_op=torch.ops.aten.upsample_trilinear3d_backward.default,
        case_fn=case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()

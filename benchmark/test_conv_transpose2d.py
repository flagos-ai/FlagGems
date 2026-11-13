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


class ConvTranspose2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (1, 64, 128, 128, 64, 3, 3, 1, 1, 1),
            (1, 64, 64, 64, 32, 3, 3, 2, 1, 1),
            (4, 32, 32, 32, 32, 3, 3, 2, 1, 1),
            (8, 16, 64, 64, 16, 5, 5, 2, 2, 1),
            (16, 32, 16, 16, 64, 3, 3, 2, 1, 1),
            (16, 32, 32, 32, 64, 3, 3, 2, 1, 1),
            (32, 64, 32, 32, 32, 3, 3, 1, 0, 1),
            (32, 64, 16, 16, 32, 3, 3, 2, 1, 1),
        ]


def _case_fn(shape, dtype):
    del dtype
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
    yield base.BenchmarkCasePlan(
        shape={"input": input_shape, "weight": weight_shape},
        params={"bias": None, "groups": groups, "stride": stride, "padding": padding},
        builder_args=(input_shape, weight_shape),
    )


def _build_inputs(plan, dtype, device):
    input_shape, weight_shape = plan.builder_args
    return ({
        "input": torch.randn(input_shape, device=device, dtype=dtype),
        "weight": torch.randn(weight_shape, device=device, dtype=dtype),
        **plan.params,
    },)


@pytest.mark.conv_transpose2d
def test_perf_conv_transpose2d(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

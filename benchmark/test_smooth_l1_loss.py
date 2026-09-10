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


class SmoothL1LossBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (262144,),
        (1024, 1024),
        (4096, 4096),
        (64, 512, 512),
    ]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            target = torch.randn(shape, device=self.device, dtype=dtype)
            yield inp, target, 1, 1.0


@pytest.mark.smooth_l1_loss
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss():
    bench = SmoothL1LossBenchmark(
        op_name="smooth_l1_loss",
        torch_op=torch.ops.aten.smooth_l1_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class SmoothL1LossBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES = [
        (262144,),
        (1024, 1024),
        (4096, 4096),
        (64, 512, 512),
    ]
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, device=self.device, dtype=dtype)
            target = torch.randn(shape, device=self.device, dtype=dtype)
            grad_output = torch.randn((), device=self.device, dtype=dtype)
            yield grad_output, inp, target, 1, 1.0


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_smooth_l1_loss_backward():
    bench = SmoothL1LossBackwardBenchmark(
        op_name="smooth_l1_loss_backward",
        torch_op=torch.ops.aten.smooth_l1_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

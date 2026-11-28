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


@pytest.mark.skip_native(
    vendors=["enflame"],
    reason=(
        "torch_gcu has no PrivateUse1 kernel for aten::angle; its CPU fallback "
        "aborts the process, so there is no native baseline"
    ),
)
@pytest.mark.angle
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_angle():
    bench = base.UnaryPointwiseBenchmark(
        op_name="angle",
        torch_op=torch.angle,
        dtypes=consts.COMPLEX_DTYPES
        + [torch.float32]
        + consts.INT_DTYPES
        + consts.BOOL_DTYPES,
    )
    bench.run()


def angle_backward_reference(inp):
    # CUDA angle has no fp16/bf16 forward, while its real-input adjoint is zero.
    # Multiplication by zero provides the equivalent native backward path.
    return torch.angle(inp) if inp.is_complex() else inp * 0.0


class AngleBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def unpack_to_args_kwargs(self, input_tuple):
        args, kwargs = super().unpack_to_args_kwargs(input_tuple)
        for arg in args:
            if torch.is_tensor(arg) and arg.is_complex():
                arg.requires_grad_(True)
        return args, kwargs


@pytest.mark.angle_backward
def test_angle_backward():
    bench = AngleBackwardBenchmark(
        op_name="angle_backward",
        torch_op=angle_backward_reference,
        gems_op=flag_gems.angle,
        # The real-input adjoint is identically zero, and CUDA does not provide
        # angle forward for fp16/bf16. Benchmark the non-trivial native dtype;
        # accuracy tests still cover every FLOAT_DTYPE zero-gradient path.
        dtypes=consts.COMPLEX_DTYPES,
        is_backward=True,
    )
    bench.run()

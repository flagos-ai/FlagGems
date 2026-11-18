# Copyright 2026, The FlagOS Contributors.
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

from flag_gems.ops.div_backward import DivFunction

from . import base, consts


def div_backward_input_fn(shape, dtype, device):
    # ``div`` has two differentiable inputs; keeping ``other`` away from zero
    # keeps the fp16/bf16 intermediates finite so both sides stay comparable.
    a = torch.rand(shape, dtype=dtype, device=device) + 0.5
    b = torch.rand(shape, dtype=dtype, device=device) + 1.5
    yield a, b


@pytest.mark.div_backward
def test_div_backward():
    # ``aten::div`` keeps its own ``DivBackward0`` node, so drive the gradient
    # through the FlagGems autograd bridge explicitly -- otherwise this would
    # time ``div_tensor_*_backward``, not this kernel.
    bench = base.BinaryPointwiseBenchmark(
        op_name="div_backward",
        torch_op=torch.div,
        gems_op=DivFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()

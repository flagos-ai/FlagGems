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

from flag_gems.ops.dot_backward import DotFunction

from . import base, consts


def dot_backward_input_fn(shape, dtype, device):
    # ``dot`` is a 1-D reduction; the harness hands out multi-dimensional shapes
    # from ``core_shapes.yaml`` only for the shapes registered there, so flatten
    # defensively to keep the operand a vector.
    n = 1
    for dim in shape:
        n *= dim
    x = torch.randn((n,), dtype=dtype, device=device)
    y = torch.randn((n,), dtype=dtype, device=device)
    yield x, y


@pytest.mark.dot_backward
def test_dot_backward():
    # ``aten::dot`` keeps its own ``DotBackward0`` node, so drive the gradient
    # through the FlagGems autograd bridge explicitly -- otherwise this would
    # time ATen's internal backward, not this kernel.
    bench = base.GenericBenchmark(
        input_fn=dot_backward_input_fn,
        op_name="dot_backward",
        torch_op=torch.dot,
        gems_op=DotFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()

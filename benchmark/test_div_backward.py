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


class DivBackwardBenchmark(base.BinaryPointwiseBenchmark):
    # The backward pass keeps both operands plus their ``requires_grad`` copies,
    # the forward output, the upstream gradient and both input gradients live at
    # once, and the ATen reference materialises each intermediate at full size.
    # Measured peak for one case in fp32: 36 GiB at 2**30 elements, 18 GiB at
    # 2**29, 9 GiB at 2**28. The two 2**30-element "from perf" shapes in
    # ``consts.DEFAULT_SHAPES`` / ``core_shapes.yaml`` therefore cannot fit the
    # ~39 GiB of a shared CI GPU (the run OOMs allocating the 4 GiB fp32
    # grad_output). Keep the same layout coverage with both large shapes capped
    # at 2**28 elements, which leaves roughly 4x headroom and still exercises
    # the memory-bound path (2.3-2.8x speedup, unlike the launch-bound 1 MiB
    # flat shape used by ``gelu_backward``).
    SHAPES = [
        (268435456,),
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (256, 1024, 1024),
    ]

    def set_shapes(self, shape_file_path=None):
        # Bypass the shared shape YAML: ``set_shapes`` walks the class MRO and
        # would otherwise pick up the catch-all ``Benchmark:`` entry, which
        # carries the billion-element shapes this op cannot afford.
        self.shapes = list(self.SHAPES)


@pytest.mark.div_backward
def test_div_backward():
    # ``aten::div`` keeps its own ``DivBackward0`` node, so drive the gradient
    # through the FlagGems autograd bridge explicitly -- otherwise this would
    # time ``div_tensor_*_backward``, not this kernel.
    bench = DivBackwardBenchmark(
        op_name="div_backward",
        torch_op=torch.div,
        gems_op=DivFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()

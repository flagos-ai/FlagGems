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

import flag_gems

from . import base, consts


class SafeSoftmaxBackwardBenchmark(base.Benchmark):
    # The backward keeps the forward output, the upstream gradient and the
    # input gradient live at once (the ATen reference materialises the same
    # set). The 1B-element "from perf" default shapes would overflow a shared
    # CI GPU (3 tensors of 2 GiB fp32 each plus ATen autograd bookkeeping), so
    # the sweep uses shapes that stay in the few-GiB range while covering the
    # narrow-row, wide-row and 3-D non-inner reduction layouts.
    SHAPES = [
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
    ]

    def set_shapes(self, shape_file_path=None):
        # Bypass the shared shape YAML: ``set_shapes`` walks the class MRO and
        # would otherwise pick up the catch-all ``Benchmark:`` entry, which
        # carries the billion-element shapes this op cannot afford.
        self.shapes = list(self.SHAPES)

    def get_input_iter(self, dtype):
        """Yield ``(grad_output, output, dim, input_dtype)`` for a real
        ``safe_softmax`` forward; rows of all -inf are part of the workload so
        the "safe" zero rows appear in the timed backward too."""
        for shape in self.shapes:
            inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 12.0 - 6.0
            flat = inp.reshape(-1, shape[-1])
            if flat.shape[0] >= 8:
                flat[0, :] = float("-inf")
            inp = flat.reshape(shape).contiguous()
            output = torch.ops.aten._safe_softmax(inp, -1)
            grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
            yield grad_output, output, -1, dtype


@pytest.mark.safe_softmax_backward
def test_safe_softmax_backward():
    bench = SafeSoftmaxBackwardBenchmark(
        op_name="safe_softmax_backward",
        torch_op=torch.ops.aten._softmax_backward_data,
        gems_op=flag_gems.safe_softmax_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

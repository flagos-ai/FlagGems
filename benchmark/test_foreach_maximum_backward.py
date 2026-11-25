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

from flag_gems.ops.foreach_maximum_backward import ForeachMaximumFunction

from . import base, consts


class _ForeachMaximumBackwardBenchmark(base.Benchmark):
    """A single tensor vs a shared scalar, gradients through the entry point.

    ``is_backward=True`` makes the harness wrap each call in
    ``torch.autograd.grad`` against the tensor arguments, which is exactly the
    comparison wanted here: the torch side times ATen's native
    ``ForeachMaximumBackward0`` and the gems side times the Triton kernel
    entry, both including their own backward dispatch.
    """

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device) * 2.0
            yield x, 0.5

    def get_tflops(self, op, *args, **kwargs):
        return args[0].numel()


def _torch_backward(x, scalar):
    return torch.ops.aten._foreach_maximum.Scalar([x], scalar)[0]


@pytest.mark.foreach_maximum_backward
def test_foreach_maximum_backward():
    bench = _ForeachMaximumBackwardBenchmark(
        op_name="foreach_maximum_backward",
        torch_op=_torch_backward,
        gems_op=ForeachMaximumFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()

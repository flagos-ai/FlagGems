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

from typing import Callable, Generator, Tuple

import pytest
import torch

from flag_gems.ops.foreach_div_backward import foreach_div_backward

from . import base, consts


def _aten_div_backward(grad_list, self_list, other_list):
    """The ATen baseline with the same ``(grad, self, other)`` signature.

    ATen has no standalone ``_foreach_div_backward`` schema; the autograd
    engine derives these gradients through the ``ForeachDivBackward0`` node,
    so the reference drives the same chain ``torch.autograd.grad`` records for
    the forward op.  The graph is rebuilt per call (the closure runs under the
    repeat loop), matching what the FlagGems entry point does: consume three
    lists of already-materialised tensors, return two lists of gradients.
    """
    self_in = [t.detach().requires_grad_(True) for t in self_list]
    other_in = [t.detach().requires_grad_(True) for t in other_list]
    grad_in = [t.detach() for t in grad_list]
    out = torch._foreach_div(self_in, other_in)
    inputs = [t for pair in zip(self_in, other_in) for t in pair]
    grads = torch.autograd.grad(out, inputs, grad_in)
    return grads[0::2], grads[1::2]


class ForeachDivBackwardBenchmark(base.Benchmark):
    """One-position paired-list benchmark for ``foreach_div_backward``.

    The ``.List`` overload takes three tensor lists and returns a pair of
    gradient lists, so the base ``is_backward`` machinery (which assumes a
    single-tensor output) does not fit; this subclass supplies the
    ``(grad, self, other)`` triple directly and lets each side carry its own
    callable.  Keeping ``other`` away from zero keeps the fp16/bf16
    intermediates finite so both sides stay comparable.
    """

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 20, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            self_list = [torch.rand(shape, dtype=dtype, device=self.device) + 0.5]
            other_list = [torch.rand(shape, dtype=dtype, device=self.device) + 1.5]
            grad_list = [torch.randn(shape, dtype=dtype, device=self.device)]
            yield grad_list, self_list, other_list

    def _benchmark_callable(
        self, op: Callable, *args, **kwargs
    ) -> Tuple[Callable, None]:
        grad_list, self_list, other_list = args
        fn = lambda: op(grad_list, self_list, other_list)
        return fn, None


@pytest.mark.foreach_div_backward
def test_foreach_div_backward():
    bench = ForeachDivBackwardBenchmark(
        op_name="foreach_div_backward",
        torch_op=_aten_div_backward,
        gems_op=foreach_div_backward,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.set_gems(foreach_div_backward)
    bench.run()

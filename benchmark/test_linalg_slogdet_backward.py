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

"""Backward benchmark for ``_linalg_slogdet``.

The forward class ``SlogdetBenchmark`` in ``test__linalg_slogdet.py`` pins the
five default shapes this operator ships with: one batched 2x3x3 case plus
4x4/8x8/16x16/32x32, all float32. Those are the exact cases measured here; the
harness runs the forward once per case outside the timed closure and times only
``torch.autograd.grad`` of ``logabsdet``.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base

# ``linalg_slogdet_backward`` does not start with an underscore, so pytest can
# create the marker by attribute access; register it explicitly anyway so
# ``-m linalg_slogdet_backward`` works without relying on attribute magic.
setattr(
    pytest.mark,
    "linalg_slogdet_backward",
    MarkDecorator(
        Mark("linalg_slogdet_backward", (), {}, _ispytest=True), _ispytest=True
    ),
)

SLOGDET_SHAPES = [
    (2, 3, 3),
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
]


class SlogdetBackwardBenchmark(base.Benchmark):
    """Backward of ``_linalg_slogdet`` for the operator's default shapes.

    ``_linalg_slogdet`` returns ``(sign, logabsdet, LU, pivots)``, so the
    generic single-output backward closure cannot be used: it would call
    ``randn_like`` on the tuple. The forward is built once per case here and
    the timed closure is ``torch.autograd.grad(logabsdet, A, grad_logabsdet,
    retain_graph=True)``; construction, the first forward and the creation of
    the upstream gradient all stay outside the timer. ``sign`` is not
    differentiable for real input, so only ``logabsdet`` is differentiated --
    the same term the native backward computes.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = SLOGDET_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[-1]
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            A = A + torch.eye(n, dtype=cur_dtype, device=self.device) * n
            yield (A,)

    def _benchmark_callable(self, op, *args, **kwargs):
        A = args[0]
        # Build the graph and the upstream gradient once, outside the timer.
        _sign, logabsdet, _LU, _pivots = op(*args, **kwargs)
        batch_shape = A.shape[:-2]
        if batch_shape:
            grad_logabsdet = torch.randn(batch_shape, dtype=A.dtype, device=A.device)
        else:
            grad_logabsdet = torch.randn((), dtype=A.dtype, device=A.device)
        fn = lambda: torch.autograd.grad(
            (logabsdet,), (A,), (grad_logabsdet,), retain_graph=True
        )
        return fn, [A]


@pytest.mark.linalg_slogdet_backward
def test_linalg_slogdet_backward():
    bench = SlogdetBackwardBenchmark(
        op_name="_linalg_slogdet",
        torch_op=torch.ops.aten._linalg_slogdet,
        # The generated slogdet kernel supports float32 on CUDA only.
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.set_gems(flag_gems._linalg_slogdet)
    bench.run()

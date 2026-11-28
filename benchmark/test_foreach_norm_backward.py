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

# Workload descriptors mirroring the KernelGen acceptance benchmark: the
# backward costs one norm reduction plus one elementwise pass over every
# element, so the sweep is memory-bound and driven by the largest tensor.
_NORM_BACKWARD_SHAPES = [
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 4096),
]


def _norm_backward_input_fn(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device) * 2.0
    grad = torch.randn((), dtype=dtype, device=device)
    # The foreach entry points take *lists* of tensors: one per position.
    yield [grad], [x], 2.0


@pytest.mark.foreach_norm_backward
def test_perf_foreach_norm_backward():
    # The native reference for this workload is the autograd chain of
    # ``_foreach_norm.Scalar`` (a norm reduction plus the elementwise
    # norm_backward).  ``latency_base`` times the reference callable directly,
    # so ``_reference_autograd`` below *is* that chain; the candidate is the
    # fused FlagGems entry point over the same (grad, self, ord) tuple.
    bench = base.GenericBenchmark(
        op_name="foreach_norm_backward",
        torch_op=_reference_autograd,
        input_fn=_norm_backward_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.foreach_norm_backward)
    bench.run()


def _reference_autograd(grad_list, x_list, ord_=2.0):
    """The native backward of ``_foreach_norm.Scalar`` for one tensor."""
    x = x_list[0]
    grad = grad_list[0]
    x_ref = x.detach().requires_grad_(True)
    out = torch.ops.aten._foreach_norm.Scalar([x_ref], ord=ord_)[0]
    (grad_self,) = torch.autograd.grad(out, [x_ref], grad_outputs=[grad])
    return [grad_self]

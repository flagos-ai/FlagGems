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

import math
from typing import Callable, Tuple

import pytest
import torch

from flag_gems.ops.foreach_pow_backward import (
    foreach_pow_backward,
    foreach_pow_scalar_and_tensor_backward,
    foreach_pow_scalar_list_backward,
)

from . import base, consts


def _aten_pow_list_backward(grad_list, self_list, exponent_list, result_list):
    """The ATen baseline with the same ``(grad, self, exponent, result)`` shape.

    ATen has no standalone ``_foreach_pow_backward`` schema; the autograd
    engine derives these gradients through the ``ForeachPowBackward0`` node,
    so the reference drives the same chain ``torch.autograd.grad`` records for
    the forward op.  The graph is rebuilt per call (the closure runs under the
    repeat loop), matching what the FlagGems entry point does.  ``result`` is
    accepted for signature parity with the FlagGems entry point (the autograd
    engine saves the forward output itself).
    """
    self_in = [t.detach().requires_grad_(True) for t in self_list]
    exp_in = [t.detach().requires_grad_(True) for t in exponent_list]
    grad_in = [t.detach() for t in grad_list]
    out = torch.ops.aten._foreach_pow.List(self_in, exp_in)
    grads = torch.autograd.grad(out, self_in + exp_in, grad_in + grad_in)
    return grads[: len(self_in)], grads[len(self_in) :]


def _aten_pow_scalarlist_backward(grad_list, self_list, exponents):
    self_in = [t.detach().requires_grad_(True) for t in self_list]
    grad_in = [t.detach() for t in grad_list]
    out = torch.ops.aten._foreach_pow.ScalarList(self_in, list(exponents))
    return torch.autograd.grad(out, self_in, grad_in)


def _aten_pow_scalarandtensor_backward(grad_list, exponent_list, base):
    exp_in = [t.detach().requires_grad_(True) for t in exponent_list]
    grad_in = [t.detach() for t in grad_list]
    out = torch.ops.aten._foreach_pow.ScalarAndTensor(base, exp_in)
    return torch.autograd.grad(out, exp_in, grad_in)


class ForeachPowBackwardBenchmark(base.Benchmark):
    """One-position benchmark for ``foreach_pow_backward``.

    The ``.List`` overload takes four tensor lists (grad, self, exponent,
    forward result) and returns a pair of gradient lists, so this subclass
    supplies the lists directly and lets each side carry its own callable,
    like the ``foreach_div_backward`` benchmark.
    """

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def init_user_config(self):
        super().init_user_config()
        # The ATen baseline rebuilds the autograd graph per call (four tensors
        # plus the forward result), and the shared DEFAULT_SHAPES carry
        # 2**30-element entries -- trim to the 2**24 tier before COMPREHENSIVE
        # merges set_more_shapes.
        self.shapes = [s for s in self.shapes if math.prod(s) <= 2**24]

    def set_more_shapes(self):
        # The ATen baseline rebuilds the autograd graph per call (four tensors
        # plus the forward result), so large shapes OOM the CI GPU: cap at
        # 2**24 elements (~192 MiB per bf16 tensor, x4 tensors x2 sides).
        special_shapes_2d = [(1024, 2**i) for i in range(0, 15, 4)]
        shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + shapes_3d

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self_list = [torch.rand(shape, dtype=dtype, device=self.device) + 0.5]
            exp_list = [torch.rand(shape, dtype=dtype, device=self.device) * 1.5 + 0.25]
            grad_list = [torch.randn(shape, dtype=dtype, device=self.device)]
            with torch.no_grad():
                result = torch.ops.aten._foreach_pow.List(self_list, exp_list)
            yield grad_list, self_list, exp_list, [result[0].detach()]

    def _benchmark_callable(
        self, op: Callable, *args, **kwargs
    ) -> Tuple[Callable, None]:
        grad_list, self_list, exp_list, result_list = args
        fn = lambda: op(grad_list, self_list, exp_list, result_list)
        return fn, None


class ForeachPowScalarBackwardBenchmark(base.Benchmark):
    """Scalar-overload benchmarks: ``.ScalarList`` and ``.ScalarAndTensor``."""

    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def init_user_config(self):
        super().init_user_config()
        # The ATen baseline rebuilds the autograd graph per call, and the
        # shared DEFAULT_SHAPES carry 2**30-element entries (2 GiB per fp16
        # tensor before the graph's saved activations).  Trim to the
        # 2**24-element tier -- the same budget as the .List benchmark.
        self.shapes = [s for s in self.shapes if math.prod(s) <= 2**24]

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            self_list = [torch.rand(shape, dtype=dtype, device=self.device) + 0.5]
            grad_list = [torch.randn(shape, dtype=dtype, device=self.device)]
            yield grad_list, self_list


@pytest.mark.foreach_pow_backward
def test_foreach_pow_backward():
    bench = ForeachPowBackwardBenchmark(
        op_name="foreach_pow_backward",
        torch_op=_aten_pow_list_backward,
        gems_op=foreach_pow_backward,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.set_gems(foreach_pow_backward)
    bench.run()


@pytest.mark.foreach_pow_backward
def test_foreach_pow_scalar_list_backward():
    def _gems(grad_list, self_list):
        return foreach_pow_scalar_list_backward(grad_list, self_list, [0.5])

    bench = ForeachPowScalarBackwardBenchmark(
        op_name="foreach_pow_scalar_list_backward",
        torch_op=lambda g, s: _aten_pow_scalarlist_backward(g, s, [0.5]),
        gems_op=_gems,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.set_gems(_gems)
    bench.run()


@pytest.mark.foreach_pow_backward
def test_foreach_pow_scalar_and_tensor_backward():
    def _gems(grad_list, exp_list):
        return foreach_pow_scalar_and_tensor_backward(
            grad_list, exp_list, exp_list, 2.0
        )

    bench = ForeachPowScalarBackwardBenchmark(
        op_name="foreach_pow_scalar_and_tensor_backward",
        torch_op=lambda g, e: _aten_pow_scalarandtensor_backward(g, e, 2.0),
        gems_op=_gems,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.set_gems(_gems)
    bench.run()

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

from . import base, consts, utils


def _input_fn_scalar(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, 0


def _case_fn_scalar(shape, dtype):
    # Listing metadata only; construction is deferred to _build_inputs_fn so that
    # ``--list-cases`` never allocates a tensor.
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"scalar": 0},
        builder_args=(shape,),
    )


def _build_inputs_fn_scalar(plan, dtype, device):
    shape = plan.builder_args[0]
    return utils.generate_tensor_input(shape, dtype, device), 0


def _to_working_copies(args):
    """Replace differentiable inputs with non-leaf working copies.

    ``lt_`` writes into its receiver, so ATen rejects a leaf that requires grad.
    The harness marks the argument tensors themselves as leaves for backward
    cases; clone those once so the benchmark differentiates a non-leaf copy, as
    a real training graph does.
    """
    return [
        arg.clone() if torch.is_tensor(arg) and arg.requires_grad else arg
        for arg in args
    ]


class _InplaceBackwardMixin:
    """Backward timing for an in-place op.

    The base ``Benchmark._benchmark_callable`` feeds the *same* tensors that the
    forward mutated back into ``torch.autograd.grad`` as differentiable inputs.
    For an in-place operator that leaves a degenerate graph (the receiver was
    overwritten by its own output), so the measured "backward" is not the
    operator's backward.  Build the forward once on non-leaf working copies and
    then time ``autograd.grad`` with respect to the untouched leaves, which is
    the graph a training step actually backpropagates through.
    """

    def _benchmark_callable(self, op, *args, **kwargs):
        if not self.is_backward:
            return super()._benchmark_callable(op, *args, **kwargs)
        leaves = [a for a in args if torch.is_tensor(a) and a.requires_grad]
        working = _to_working_copies(args)
        out = op(*working, **kwargs)
        dout = torch.randn_like(out)

        def backward():
            return torch.autograd.grad(
                (out,), leaves, grad_outputs=(dout,), retain_graph=True
            )

        return backward, leaves


class InplaceBackwardBenchmark(_InplaceBackwardMixin, base.BinaryPointwiseBenchmark):
    """Backward benchmark for an in-place op whose receiver must not be a leaf."""


class InplaceGenericBackwardBenchmark(_InplaceBackwardMixin, base.GenericBenchmark):
    """Same as :class:`InplaceBackwardBenchmark` for a custom input builder."""


@pytest.mark.lt_
def test_lt_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt_",
        torch_op=lambda a, b: torch.ops.aten.lt_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.lt_scalar_
def test_lt_scalar_():
    bench = base.GenericBenchmark(
        op_name="lt_scalar_",
        input_fn=_input_fn_scalar,
        torch_op=lambda a, b: torch.ops.aten.lt_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.lt_backward_
def test_lt_tensor_backward():
    """Time the autograd path of ``lt_.Tensor`` against native PyTorch.

    ``is_backward=True`` makes the harness build the forward graph once and then
    time ``torch.autograd.grad`` over the same graph.  ``op_name`` is ``lt`` so
    the base class records the case as ``lt_backward`` instead of the doubled
    ``lt__backward`` spelling.
    """
    bench = InplaceBackwardBenchmark(
        op_name="lt",
        torch_op=lambda a, b: torch.ops.aten.lt_.Tensor(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
        is_inplace=True,
    )
    bench.set_gems(flag_gems.lt_)
    bench.run()


@pytest.mark.lt_backward_
def test_lt_scalar_backward():
    """Time the autograd path of ``lt_.Scalar`` against native PyTorch."""
    bench = InplaceGenericBackwardBenchmark(
        op_name="lt_scalar",
        case_fn=_case_fn_scalar,
        build_inputs_fn=_build_inputs_fn_scalar,
        torch_op=lambda a, b: torch.ops.aten.lt_.Scalar(a, b),
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
        is_inplace=True,
    )
    bench.set_gems(flag_gems.lt_scalar_)
    bench.run()

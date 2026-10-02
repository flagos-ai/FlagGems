# Copyright 2026, The FlagGems Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark for ``torch.ops.aten._use_cudnn_rnn_flatten_weight``.

``aten::_use_cudnn_rnn_flatten_weight() -> bool`` takes no operand, so one case
is one ambient configuration (cuDNN switches plus grad mode) under which the
query is invoked, and there is no dtype axis. The timing override only applies
that configuration and then delegates to ``Benchmark.get_latency``, so the
invoked and measured callable is still the exact ``torch_op``;
``@base.reference_uses_torch_op`` declares that contract for ``--reference-only``.
"""

import contextlib

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_GRAD_CONTEXTS = {
    "grad": torch.enable_grad,
    "no_grad": torch.no_grad,
    "inference_mode": torch.inference_mode,
}

# case = (grad mode, cudnn enabled, cudnn deterministic, cudnn benchmark,
#         cudnn allow_tf32)
DEFAULT_CONTEXTS = [
    ("grad", True, False, False, True),
    ("grad", False, False, False, False),
    ("no_grad", True, False, False, True),
    ("no_grad", False, False, False, False),
    ("inference_mode", True, True, False, True),
    ("inference_mode", False, False, True, False),
]

# The query has no operand, so there is no dtype axis; the harness still keys
# case ids by dtype, so a single placeholder keeps the ids unique.
_DTYPES = [torch.float32]


def _as_context(case):
    """Validate one case row before anything is timed or allocated."""
    if not isinstance(case, (tuple, list)) or len(case) != 5:
        raise ValueError(f"a case must be a 5-field sequence, got {case!r}")
    grad_mode, enabled, deterministic, benchmark, allow_tf32 = case
    if grad_mode not in _GRAD_CONTEXTS:
        raise ValueError(f"unsupported grad mode {grad_mode!r}")
    for name, value in (
        ("cudnn_enabled", enabled),
        ("cudnn_deterministic", deterministic),
        ("cudnn_benchmark", benchmark),
        ("cudnn_allow_tf32", allow_tf32),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {value!r}")
    return (grad_mode, enabled, deterministic, benchmark, allow_tf32)


@contextlib.contextmanager
def _ambient_context(case):
    grad_mode, enabled, deterministic, benchmark, allow_tf32 = case
    saved = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
    )
    torch.backends.cudnn.enabled = enabled
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        with _GRAD_CONTEXTS[grad_mode]():
            yield
    finally:
        (
            torch.backends.cudnn.enabled,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.allow_tf32,
        ) = saved


def _case_fn(shape, dtype):
    del dtype
    case = _as_context(shape)
    grad_mode, enabled, deterministic, benchmark, allow_tf32 = case
    yield base.BenchmarkCasePlan(
        shape={"config": str(case), "operands": 0},
        params={
            "grad_mode": grad_mode,
            "cudnn_enabled": enabled,
            "cudnn_deterministic": deterministic,
            "cudnn_benchmark": benchmark,
            "cudnn_allow_tf32": allow_tf32,
        },
        builder_args=(case,),
    )


class _ContextBuilder:
    """Builds the operand-free input and remembers its case's ambient context.

    The schema has no argument that could carry the case into ``get_latency``,
    so the builder records it on itself for the timing wrapper to read back.
    """

    def __init__(self):
        self.context = None

    def __call__(self, plan, dtype, device):
        del dtype, device
        self.context = plan.builder_args[0]
        # No operand: the query must be called with exactly zero arguments, so
        # this yields neither positional args nor kwargs.
        return ()


class UseCudnnRnnFlattenWeightBenchmark(OperatorBenchmark):
    """One case per ambient configuration of the operand-free query."""

    def set_shapes(self, shape_file_path=None):
        # A --shape-file still wins; without one these configurations are used,
        # which also bypasses the generic COMPREHENSIVE shapes (up to 2**28
        # elements) that an operand-free query cannot use.
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_CONTEXTS)

    def set_more_shapes(self):
        # Nullary query configurations have no tensor-shape dimension.
        return []

    @base.reference_uses_torch_op
    def get_latency(self, op, *args, **kwargs):
        # Timing wrapper only: the ambient configuration is applied around the
        # standard measurement, which still invokes the exact op.
        context = getattr(self.build_inputs_fn, "context", None)
        scope = (
            _ambient_context(context)
            if context is not None
            else contextlib.nullcontext()
        )
        with scope:
            return super().get_latency(op, *args, **kwargs)


@pytest.mark.use_cudnn_rnn_flatten_weight
def test__use_cudnn_rnn_flatten_weight():
    bench = UseCudnnRnnFlattenWeightBenchmark(
        op_name="_use_cudnn_rnn_flatten_weight",
        case_fn=_case_fn,
        build_inputs_fn=_ContextBuilder(),
        torch_op=torch.ops.aten._use_cudnn_rnn_flatten_weight,
        gems_op=getattr(flag_gems, "_use_cudnn_rnn_flatten_weight", None),
        dtypes=_DTYPES,
    )
    bench.run()

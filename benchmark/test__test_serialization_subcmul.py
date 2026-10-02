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

"""Benchmark for ``aten::_test_serialization_subcmul`` (``self - alpha*other``).

``torch_op`` is the perf baseline and ``gems_op`` the candidate injected through
``--override``; both are called positionally as ``op(input, other, alpha)``. Plans
carry metadata only, so ``--list-cases`` allocates no tensor and calls no operator.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# Unioned with the shared core/comprehensive grid rather than replacing it: the op
# has no core_shapes.yaml entry, so the base loader falls back to
# consts.DEFAULT_SHAPES and these allocation-friendly shapes are added on top.
_EXTRA_SHAPES = [
    (256,),
    (1_048_576,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_DTYPES = (
    list(consts.FLOAT_DTYPES)
    + list(consts.INT_DTYPES)
    + list(consts.EXTRA_INT_DTYPES)
    + list(consts.COMPLEX_DTYPES)
)
# Static device capability flag, read once at import; no runtime probe.
if flag_gems.runtime.device.support_fp64:
    _DTYPES += [torch.float64, torch.complex128]


def _case_fn(shape, dtype):
    del dtype  # plans depend on the shape only; the harness supplies the dtype
    for alpha in (1.0, 2.5, 3):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape), "other": list(shape)},
            params={"alpha": alpha},
            builder_args=(shape, shape, alpha),
        )
    # Trailing-dimension broadcast. It differs from the same-shape pair only for
    # rank >= 2; a rank-1 trailing dim would repeat the pair above.
    if len(shape) > 1:
        other_shape = (shape[-1],)
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape), "other": list(other_shape)},
            params={"alpha": 2.5},
            builder_args=(shape, other_shape, 2.5),
        )


def _build_inputs_fn(plan, dtype, device):
    inp_shape, other_shape, alpha = plan.builder_args
    inp = torch.empty(inp_shape, dtype=dtype, device=device)
    other = torch.empty(other_shape, dtype=dtype, device=device)
    # unpack_to_args_kwargs takes flat positional items plus a trailing kwargs dict,
    # so the operands and alpha are returned flat rather than nested.
    return inp, other, alpha, {}


class SerializationSubcmulBenchmark(OperatorBenchmark):
    """Shared shape grid plus the allocation-friendly extras of this operator."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path or self.DEFAULT_SHAPE_FILES)
        merged = [tuple(shape) for shape in list(self.shapes) + _EXTRA_SHAPES]
        self.shapes = list(dict.fromkeys(merged))


@pytest.mark.test_serialization_subcmul
def test__test_serialization_subcmul():
    bench = SerializationSubcmulBenchmark(
        op_name="_test_serialization_subcmul",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_serialization_subcmul,
        gems_op=getattr(flag_gems, "_test_serialization_subcmul", None),
        dtypes=_DTYPES,
    )
    bench.run()

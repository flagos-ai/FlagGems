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

"""Benchmark for ``aten::ge_`` (in-place ``self >= other``).

Both overloads write into the receiver, so the receiver is always the first
argument. BinaryPointwiseBenchmark only builds equal-shaped operands, so the
Tensor overload adds a right-broadcast operand pair; the Scalar overload needs
receiver-first builders because that family yields the scalar first.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils

BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

SCALAR_OTHER = 0.0  # splits the comparison on both sides of zero


class _GeBenchmark(base.BinaryPointwiseBenchmark):
    def _shape_pairs(self):
        for shape in self.shapes:
            yield shape, shape
            if len(shape) > 1:
                yield shape, (1,) * (len(shape) - 1) + (shape[-1],)

    def get_case_iter(self, dtype):
        for ordinal, (receiver, other) in enumerate(self._shape_pairs()):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"receiver": list(receiver), "other": list(other)},
                    builder_args=(receiver, other),
                ),
            )

    def build_inputs(self, case):
        receiver_shape, other_shape = case.builder_args[0].builder_args
        receiver = utils.generate_tensor_input(receiver_shape, case.dtype, self.device)
        other = utils.generate_tensor_input(other_shape, case.dtype, self.device)
        return receiver, other


class _GeScalarBenchmark(base.ScalarBinaryPointwiseBenchmark):
    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"receiver": list(shape)},
                    params={"other": SCALAR_OTHER},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        shape = case.builder_args[0].builder_args[0]
        receiver = utils.generate_tensor_input(shape, case.dtype, self.device)
        return receiver, SCALAR_OTHER

    def get_tflops(self, op, *args, **kwargs):
        # Receiver-first call tuple, unlike the family's scalar-first one.
        return args[0].numel()


@pytest.mark.ge_
def test_ge_():
    bench = _GeBenchmark(
        op_name="ge_",
        torch_op=torch.ops.aten.ge_.Tensor,
        gems_op=getattr(flag_gems, "ge_", None),
        dtypes=BENCH_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.ge_
def test_ge__scalar():
    bench = _GeScalarBenchmark(
        op_name="ge_",
        torch_op=torch.ops.aten.ge_.Scalar,
        gems_op=getattr(flag_gems, "ge_", None),
        dtypes=BENCH_DTYPES,
        is_inplace=True,
    )
    bench.run()

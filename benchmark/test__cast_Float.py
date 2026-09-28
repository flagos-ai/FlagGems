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

# The shared input builder covers float16/float32/bfloat16/int16/int32 for this
# operator; bfloat16 is gated by the static backend flag so a backend without
# bf16 support neither lists nor times it. float32 rows are the native no-op
# whose result shares the input storage.
_BENCH_DTYPES = [d for d in consts.FLOAT_DTYPES if d is not torch.bfloat16]
if flag_gems.runtime.device.support_bf16:
    _BENCH_DTYPES.append(torch.bfloat16)
_BENCH_DTYPES += list(consts.INT_DTYPES)

# The optional `non_blocking` flag in its three same-device call forms: the
# schema default (argument omitted, so the plan carries no keyword arguments),
# False and True. Each plan's `params` is exactly the keyword set handed to the
# operator, so listing, execution and --case-id replay share one metadata form.
_NON_BLOCKING_PLANS = ({}, {"non_blocking": False}, {"non_blocking": True})


def _validate_extents(shape):
    # Tensor sizes are non-negative integers. Bools and floats are not sizes and
    # a negative extent cannot be allocated, so an invalid descriptor is an
    # error in the supplied workload list rather than a row to skip silently.
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(
                f"Invalid tensor extent {extent!r} in shape {tuple(shape)}: "
                "tensor sizes must be integers (bools and floats are rejected)."
            )
        if extent < 0:
            raise ValueError(
                f"Invalid tensor extent {extent!r} in shape {tuple(shape)}: "
                "tensor sizes must be non-negative."
            )


# UnaryPointwiseBenchmark, whose class shape file and core/comprehensive shape
# scales stay authoritative, with the `non_blocking` call forms as extra cases.
class _CastFloatBenchmark(base.UnaryPointwiseBenchmark):
    def get_case_iter(self, dtype):
        # Shapes come from the operator/class shape files, so every descriptor
        # in the supplied workload list is validated up front: () and zero
        # extents are valid, an invalid descriptor raises here instead of being
        # dropped from listing, execution and replay.
        for shape in self.shapes:
            _validate_extents(shape)
        return self._iter_cases(dtype)

    def _iter_cases(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            for params in _NON_BLOCKING_PLANS:
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape)},
                        params=dict(params),
                        builder_args=(shape,),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        inp = utils.generate_tensor_input(plan.builder_args[0], case.dtype, self.device)
        return (inp, dict(plan.params))


@pytest.mark.cast_Float
def test__cast_Float():
    bench = _CastFloatBenchmark(
        op_name="_cast_Float",
        torch_op=torch.ops.aten._cast_Float,
        gems_op=getattr(flag_gems, "_cast_Float", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()

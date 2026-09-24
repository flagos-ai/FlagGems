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

from .generated_operator_utils import OperatorBenchmark
from . import base

# aten::can_cast(ScalarType from_, ScalarType to) -> bool is a pure dtype-
# metadata query: it allocates nothing and never touches the device, so the
# benchmark measures pure dispatch/function-call overhead. There are no tensor
# inputs to materialize; the only meaningful case dimension is the (from_, to)
# dtype pair. The representative pairs below cover every scalar-type family in
# both directions, including both True (same-family widening, bool -> float,
# fp8 <-> float, complex widening) and False (float -> int, float -> bool,
# int -> float16, complex -> float) outcomes. No public Benchmark family covers
# a tensor-free dtype query, so the two-phase GenericBenchmark drives the case
# list directly from _case_fn / _build_inputs_fn; the candidate is passed
# explicitly via gems_op and the perf reference is the aten op itself
# (torch.ops.aten.can_cast), both called with the same two-dtype signature.
_CAN_CAST_BENCH_PAIRS = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float16),
    (torch.float32, torch.float64),
    (torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16),
    (torch.int16, torch.float16),
    (torch.int32, torch.int64),
    (torch.int64, torch.int32),
    (torch.int32, torch.float32),
    (torch.float32, torch.int32),
    (torch.float64, torch.int64),
    (torch.uint8, torch.int32),
    (torch.bool, torch.float32),
    (torch.float32, torch.bool),
    (torch.complex64, torch.complex128),
    (torch.complex64, torch.float32),
]

# fp8 pairs are only added when the running PyTorch exposes the ScalarTypes.
for _fp8_name in ("float8_e4m3fn", "float8_e5m2"):
    _fp8 = getattr(torch, _fp8_name, None)
    if isinstance(_fp8, torch.dtype):
        _CAN_CAST_BENCH_PAIRS.append((_fp8, torch.float32))
        _CAN_CAST_BENCH_PAIRS.append((torch.float32, _fp8))

# The op never creates tensors, so the benchmark dtype bucket is irrelevant; a
# single bucket keeps the case list free of redundant per-dtype repeats.
_CAN_CAST_BENCH_DTYPES = [torch.float32]


def _case_fn(shape, dtype):
    del shape, dtype
    for from_dtype, to_dtype in _CAN_CAST_BENCH_PAIRS:
        yield base.BenchmarkCasePlan(
            shape={"from_": str(from_dtype), "to": str(to_dtype)},
            params={"from_": str(from_dtype), "to": str(to_dtype)},
            builder_args=(from_dtype, to_dtype),
        )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    from_dtype, to_dtype = plan.builder_args
    return from_dtype, to_dtype


class CanCastBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark for the tensor-free dtype-pair op can_cast."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=[(0,)])

    def set_more_shapes(self):
        # Additional tensor shapes are meaningless for a dtype-metadata query.
        return []


@pytest.mark.can_cast
def test_can_cast():
    bench = CanCastBenchmark(
        op_name="can_cast",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.can_cast,
        gems_op=getattr(flag_gems, "can_cast", None),
        dtypes=_CAN_CAST_BENCH_DTYPES,
    )
    bench.run()

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

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType has no
# tensor operands: it allocates nothing and never touches the device, so the
# benchmark measures pure dispatch/host-call overhead and the only case dimension
# is the dtype pair. Every pair below is accepted by the native op and is covered
# against it in tests/test_promote_types.py. No public Benchmark family models a
# tensor-free dtype query, so the two-phase GenericBenchmark drives the case list
# from _case_fn / _build_inputs_fn.
_PROMOTE_BENCH_PAIRS = [
    (torch.float16, torch.float32),
    (torch.float32, torch.float64),
    (torch.bfloat16, torch.float32),
    (torch.int8, torch.int32),
    (torch.uint8, torch.int64),
    (torch.int32, torch.int64),
    (torch.int32, torch.float32),
    (torch.bool, torch.float32),
    (torch.float32, torch.complex64),
    (torch.float64, torch.complex128),
    (torch.complex64, torch.complex128),
]

_FP8_BENCH_DTYPES = [
    dtype
    for dtype in (
        getattr(torch, name, None) for name in ("float8_e4m3fn", "float8_e5m2")
    )
    if isinstance(dtype, torch.dtype)
]
_WIDE_UINT_BENCH_DTYPES = [
    dtype
    for dtype in (getattr(torch, name, None) for name in ("uint16", "uint32", "uint64"))
    if isinstance(dtype, torch.dtype)
]
# Float8 types promote with themselves only; the wide unsigned types promote with
# themselves and with floating types.
_PROMOTE_BENCH_PAIRS += [(dtype, dtype) for dtype in _FP8_BENCH_DTYPES]
_PROMOTE_BENCH_PAIRS += [(dtype, dtype) for dtype in _WIDE_UINT_BENCH_DTYPES]
_PROMOTE_BENCH_PAIRS += [(dtype, torch.float32) for dtype in _WIDE_UINT_BENCH_DTYPES]


def _case_fn(shape, dtype):
    del shape, dtype
    for type1, type2 in _PROMOTE_BENCH_PAIRS:
        yield base.BenchmarkCasePlan(
            shape={"type1": str(type1), "type2": str(type2)},
            params={"type1": str(type1), "type2": str(type2)},
            builder_args=(type1, type2),
        )


def _build_inputs_fn(plan, dtype, device):
    # No tensor payload exists; the two dtype arguments are passed positionally,
    # matching unpack_to_args_kwargs on the flat tuple.
    del dtype, device
    type1, type2 = plan.builder_args
    return type1, type2


class PromoteTypesBenchmark(OperatorBenchmark):
    """Two-phase benchmark for the tensor-free dtype-pair op promote_types."""

    def set_shapes(self, shape_file_path=None):
        # An empty buffer is the only placeholder this operator needs; a caller
        # supplied shape file still wins inside the base helper.
        super().set_shapes(shape_file_path, default_shapes=[(0,)])

    def set_more_shapes(self):
        # Additional tensor shapes are meaningless for a dtype query.
        return []


@pytest.mark.promote_types
def test_promote_types():
    bench = PromoteTypesBenchmark(
        op_name="promote_types",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.promote_types,
        gems_op=getattr(flag_gems, "promote_types", None),
        # The actual dtype pair is recorded by each case plan.
        dtypes=[torch.float32],
    )
    bench.run()

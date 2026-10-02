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

# aten::_test_optional_intlist is CPU-registered, so the reference and the
# candidate both take CPU operands. Its two natively valid forms are benchmarked
# as two plans per shape: the zero-copy identity (addends=None, valid for every
# supported CPU dtype, over the shared default/comprehensive shape grid, with a
# caller shape file flattened to the 1-D operand the schema requires) and the
# int32 1-D elementwise add (an int32 operand with one addends entry per element).
_CPU = torch.device("cpu")

BENCH_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

_ADDEND = 3
_addends_cache = {}


def _flat_length(shape):
    length = 1
    for extent in shape:
        length *= extent
    return length


def _case_fn(shape, dtype):
    length = _flat_length(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"mode": "identity"},
        builder_args=(tuple(shape), "identity"),
    )
    if dtype == torch.int32:
        yield base.BenchmarkCasePlan(
            shape={"input": [length]},
            params={"mode": "addends"},
            builder_args=(length, "addends"),
        )


def _operand(length, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, (length,), device=_CPU).bool()
    if dtype.is_complex:
        real = torch.randn(length, device=_CPU)
        return torch.complex(real, real.flip(0)).to(dtype)
    if dtype.is_floating_point:
        # randn has no fp8 overload, so the cast follows generation.
        return torch.randn(length, device=_CPU).to(dtype)
    return torch.randint(0, 8, (length,), dtype=dtype, device=_CPU)


def _constant_addends(length):
    # One entry per operand element, all reusing a single int object; the list is
    # materialized once and reused across the measurement iterations.
    addends = _addends_cache.get(length)
    if addends is None:
        addends = [_ADDEND] * length
        _addends_cache[length] = addends
    return addends


def _build_inputs_fn(plan, dtype, device):
    del device
    size, mode = plan.builder_args
    if mode == "identity":
        return torch.empty(size, dtype=dtype, device=_CPU), {"addends": None}
    length = size
    values = _operand(length, dtype)
    return values, {"addends": _constant_addends(length)}


class OptionalIntlistBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape)
                for shape in list(self.shapes) + [(256,), (4096,), (65536,), (1000003,)]
            )
        )


@pytest.mark.test_optional_intlist
def test__test_optional_intlist():
    bench = OptionalIntlistBenchmark(
        op_name="_test_optional_intlist",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_optional_intlist,
        gems_op=getattr(flag_gems, "_test_optional_intlist", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()

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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# squeeze_ only rebinds metadata, so the timed region never reads element values
# and every case can be built with torch.empty. The shared default (and
# comprehensive) shape sets, plus any --shape-file entry, are kept as-is; the
# boundaries below add the unit-dimension workloads those sets do not contain,
# since squeeze_ is a no-op for all of them. Each shape is measured with all
# three argument forms.
SQUEEZE_BOUNDARY_SHAPES = [
    (1,),
    (1, 1024, 1),
    (1024, 1, 1024, 1),
    (1, 1, 1, 1, 1),
    (1, 20, 320, 1, 15),
    (1, 16, 128, 1, 64),
    (1, 16, 7, 57, 32),
    (2, 19, 7),
]

# Metadata reshaping is dtype independent: cover the required integer, FP8,
# bool, complex and floating dtypes on the same cases.
BENCH_DTYPES = consts.FLOAT_DTYPES + [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.complex64,
]
if getattr(flag_gems.runtime.device, "support_fp64", False):
    BENCH_DTYPES.append(torch.float64)


def _case_fn(shape, dtype):
    del dtype
    unit_dims = [index for index, size in enumerate(shape) if size == 1]
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"form": "default"},
        builder_args=(shape, ()),
    )
    # Without a unit dimension both the dim and the dims form are valid no-ops.
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"form": "dim", "dim": unit_dims[0] if unit_dims else 0},
        builder_args=(shape, ((unit_dims[0] if unit_dims else 0),)),
    )
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"form": "dims", "dims": list(unit_dims)},
        builder_args=(shape, (list(unit_dims),)),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, extra = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return (inp, *extra)


class SqueezeBenchmark(OperatorBenchmark):
    """In-place metadata benchmark: every timed call gets a fresh operand."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # A shape file may deliver lists; normalise before de-duplicating so the
        # shared default/comprehensive grids and the boundary shapes all survive.
        shapes = (tuple(shape) for shape in (*self.shapes, *SQUEEZE_BOUNDARY_SHAPES))
        self.shapes = list(dict.fromkeys(shapes))


@pytest.mark.squeeze_
def test_squeeze_():
    bench = SqueezeBenchmark(
        op_name="squeeze_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.squeeze_,
        gems_op=getattr(flag_gems, "squeeze_", None),
        dtypes=[dtype for dtype in BENCH_DTYPES if _DTYPE_FLAGS.get(dtype, True)],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()

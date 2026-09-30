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

# aten::mH builds a conjugate-transpose view, so the measured work is view
# construction (dispatch, stride swap, lazy conj bit) and nothing reads tensor
# values. The shared default and COMPREHENSIVE shape grids are kept and only
# filtered to the ranks the operator's schema accepts (aten::mH raises on every
# 1-D input, including the empty (0,) tensor); the boundary shapes below are
# added on top, and a user shape file still replaces the grids.
_MH_BOUNDARY_SHAPES = [(), (0, 3), (1, 1), (2, 1), (1, 2), (3, 5, 7), (4, 1, 2, 1)]

_MH_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.COMPLEX_DTYPES
    + consts.BOOL_DTYPES
    + consts.INT_DTYPES
    + consts.EXTRA_INT_DTYPES
    + [torch.float8_e4m3fn, torch.float8_e5m2]
)


def _matrix_shapes(shapes):
    return [tuple(shape) for shape in shapes if len(shape) != 1]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    # mH only reads shape/stride/dtype metadata, so uninitialized storage is
    # enough and the fixture does not pay for a value fill.
    return torch.empty(shape, dtype=dtype, device=device), {}


class MHBenchmark(base.GenericBenchmark):
    """Shared shape grids, filtered to the ranks aten::mH accepts."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = _matrix_shapes(self.shapes) + _MH_BOUNDARY_SHAPES


@pytest.mark.mH
def test_mH():
    bench = MHBenchmark(
        op_name="mH",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mH,
        gems_op=getattr(flag_gems, "mH", None),
        dtypes=_MH_DTYPES,
    )
    bench.run()

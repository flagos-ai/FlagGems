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
"""Benchmark for ``aten::_linalg_check_errors``.

The operator scans an int32 info tensor for the first non-zero status code, so
the measured work is that scan. All-zero codes keep every case on the successful
path; a non-zero code would raise instead of being timed.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# Default shapes in int32 info elements: 1, 1024, 65536, 2 ** 20, 4096 x 256 and
# 128 x 256 x 256, covering the scalar-status shape and the flat and
# multi-dimensional scans.
_LINALG_CHECK_ERRORS_SHAPES = [
    (1,),
    (1024,),
    (65536,),
    (1 << 20,),
    (4096, 256),
    (128, 256, 256),
]

# api_name and is_matrix are the only parameters of the operator. All-zero info is
# valid for either value of is_matrix at any size, so both rows are measured for
# every shape.
_INFO_PARAMS = (
    {"api_name": "torch.linalg.inv", "is_matrix": False},
    {"api_name": "torch.linalg.solve", "is_matrix": True},
)


def _normalize_shape(descriptor):
    """Return a shape descriptor as a tuple of integer extents.

    A shape file may list a one-dimensional shape as a bare integer, where 8
    means (8,) and the empty tuple is the scalar shape. Booleans, fractional,
    string and negative extents cannot describe a tensor and are rejected before
    any allocation instead of being reinterpreted.
    """
    extents = descriptor if isinstance(descriptor, (list, tuple)) else (descriptor,)
    shape = []
    for extent in extents:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"invalid shape extent {extent!r}")
        if extent < 0:
            raise ValueError(f"negative shape extent {extent}")
        shape.append(extent)
    return tuple(shape)


def _case_fn(shape, dtype):
    del dtype
    shape = _normalize_shape(shape)
    for params in _INFO_PARAMS:
        yield base.BenchmarkCasePlan(
            shape={"info": list(shape)},
            params=dict(params),
            builder_args=(shape, params["api_name"], params["is_matrix"]),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, api_name, is_matrix = plan.builder_args
    info = torch.zeros(shape, dtype=dtype, device=device)
    return info, {"api_name": api_name, "is_matrix": is_matrix}


class LinalgCheckErrorsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_LINALG_CHECK_ERRORS_SHAPES)
        # The persisted shapes feed the inherited case iteration as well, so a
        # bare integer from a shape file is normalized here too.
        self.shapes = [_normalize_shape(shape) for shape in self.shapes]


@pytest.mark.linalg_check_errors
def test__linalg_check_errors():
    bench = LinalgCheckErrorsBenchmark(
        op_name="_linalg_check_errors",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._linalg_check_errors,
        gems_op=getattr(flag_gems, "_linalg_check_errors", None),
        dtypes=[torch.int32],
    )
    bench.run()

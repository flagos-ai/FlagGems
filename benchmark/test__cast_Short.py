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
from .generated_operator_utils import OperatorBenchmark

# Default workload set; a caller-supplied shape file for op_name "_cast_Short"
# (or for this class) replaces it without any size cap or filtering.
CAST_SHAPES = [
    (65536,),
    (1048576,),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (16, 128, 64, 60),
]

# ``None`` is a builder-only sentinel meaning "call without the argument"; the
# case metadata for it carries an empty ``params`` so listing, execution and
# --case-id replay all describe the same no-argument call, while True/False
# carry the actual bool.
NON_BLOCKING_VALUES = [None, False, True]

# Static device capability gate for the optional BF16 measurements (the same
# flag as tests/accuracy_utils.bf16_is_supported); no dtype is probed here.
BENCH_DTYPES = (
    consts.FLOAT_DTYPES
    if flag_gems.runtime.device.support_bf16
    else [dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16]
)


def _case_fn(shape, dtype):
    del dtype
    for non_blocking in NON_BLOCKING_VALUES:
        params = {} if non_blocking is None else {"non_blocking": non_blocking}
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params=params,
            builder_args=(shape, non_blocking),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if non_blocking is None:
        return inp, {}
    return inp, {"non_blocking": non_blocking}


class CastShortBenchmark(OperatorBenchmark):
    """Benchmark harness for the int16 cast with caller-supplied shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=CAST_SHAPES)
        # Shapes may come from a caller's shape file: validate the extents
        # before any allocation, without capping what was requested.  A scalar
        # entry (empty shape) and zero extents are valid inputs here.
        for shape in self.shapes:
            if not isinstance(shape, (tuple, list)):
                raise ValueError(
                    f"Shape entry {shape!r} must be a sequence of extents."
                )
            for extent in shape:
                if (
                    isinstance(extent, bool)
                    or not isinstance(extent, int)
                    or extent < 0
                ):
                    raise ValueError(
                        f"Shape {tuple(shape)} must contain non-negative integers."
                    )


@pytest.mark.cast_Short
def test__cast_Short():
    bench = CastShortBenchmark(
        op_name="_cast_Short",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cast_Short,
        gems_op=getattr(flag_gems, "_cast_Short", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()

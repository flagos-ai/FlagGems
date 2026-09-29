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

# aten::_autocast_to_full_precision(Tensor self, bool cuda_enabled, bool
# cpu_enabled) -> Tensor: fp16/bf16 are promoted to fp32 when the flag of their
# own device is set, every other combination is a pass-through identity. No
# public Benchmark family models a dtype-conversion/dispatch op, so this uses the
# two-phase GenericBenchmark (case_fn + build_inputs_fn). Every flag pair is
# timed for every (shape, dtype), which keeps the cheap pass-through cost visible
# next to the promotion copy.

_FLAG_PAIRS = [(True, True), (True, False), (False, True), (False, False)]

# The framework scales used for every other operator, plus the two moderate spec
# shapes. Every extent is kept exactly as declared, with no cap and no rounding.
SHAPES = list(consts.DEFAULT_SHAPES) + [(20, 320, 15), (16, 7, 57, 32, 29)]

# BF16 is the only dtype of the shared float set a backend can lack; the flag is
# the same static device property the correctness suite reads, so the dtype list
# is built once at import time and nothing is probed or skipped at run time.
_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _validated_shape(shape):
    """Reject boolean, non-integer or negative extents before allocation.

    A 0-dim shape and 0-extent dimensions are valid input geometry.
    """
    dims = tuple(shape)
    for extent in dims:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(
                f"invalid shape metadata {dims!r}: extents must be non-negative ints"
            )
    return dims


class AutocastToFullPrecisionBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_DESC = "shape"

    def set_shapes(self, shape_file_path=None):
        # Delegate to the shared resolver: an explicit caller file (including a
        # nonexistent path, which must raise) still wins whenever it carries an
        # entry for this operator or for this benchmark class; otherwise these
        # defaults apply.
        super().set_shapes(shape_file_path, default_shapes=list(SHAPES))


def _case_fn(shape, dtype):
    # One Workload per (shape, flag pair). The extents are validated here, before
    # a plan is emitted, so --list-cases rejects malformed metadata exactly like a
    # run would; the same plans are used for listing and execution.
    del dtype
    dims = _validated_shape(shape)
    for cuda_enabled, cpu_enabled in _FLAG_PAIRS:
        yield base.BenchmarkCasePlan(
            shape={"input": dims},
            params={"cuda_enabled": cuda_enabled, "cpu_enabled": cpu_enabled},
            builder_args=(dims,),
        )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, dict(plan.params)


@pytest.mark.autocast_to_full_precision
def test__autocast_to_full_precision():
    # gems_op is resolved here, never at import time, so a process-local override
    # installed for this run wins while listing still works without a candidate.
    bench = AutocastToFullPrecisionBenchmark(
        op_name="_autocast_to_full_precision",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._autocast_to_full_precision,
        gems_op=getattr(flag_gems, "_autocast_to_full_precision", None),
        dtypes=_DTYPES,
    )
    bench.run()

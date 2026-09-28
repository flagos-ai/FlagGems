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

# Allocation-friendly defaults: the largest entry (16, 128, 64, 60 = 7.9M
# elements) needs ~32 MiB for the input plus ~32 MiB for the int32 result. A
# shape file that names ``_cast_Int`` (or this benchmark class) supplies its own
# shapes unfiltered through OperatorBenchmark.set_shapes, so caller workloads
# are neither capped nor silently replaced.
_CAST_INT_SHAPES = [
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

# Static backend capability (same flag source as tests/accuracy_utils.py);
# bf16 is the only bench dtype whose availability is not guaranteed.
_DTYPE_GATES = {torch.bfloat16: flag_gems.runtime.device.support_bf16}
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES + consts.INT_DTYPES
    if _DTYPE_GATES.get(dtype, True)
]

# ``None`` is the omitted sentinel for the only schema parameter (``bool
# non_blocking=False``): that form leaves ``params`` empty and calls with no
# argument, the other two pass the real booleans. Each form is a distinct case
# ID whose listing metadata equals the executed plan.
_CALL_FORMS = (None, False, True)


def _checked_shape(shape):
    """Reject unusable shape metadata while listing, without allocating."""
    dims = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"shape dimensions must be ints, got {dim!r}")
        if dim < 0:
            raise ValueError(f"shape dimensions must be non-negative, got {dim!r}")
        dims.append(dim)
    return tuple(dims)


def _case_fn(shape, dtype):
    del dtype
    dims = _checked_shape(shape)
    for non_blocking in _CALL_FORMS:
        yield base.BenchmarkCasePlan(
            shape={"input": dims},
            params={} if non_blocking is None else {"non_blocking": non_blocking},
            builder_args=(dims, non_blocking),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if non_blocking is None:
        return (inp,)
    return inp, {"non_blocking": non_blocking}


class CastIntBenchmark(OperatorBenchmark):
    """Listing and execution share these plans; listing allocates no tensors."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CAST_INT_SHAPES)


@pytest.mark.cast_Int
def test__cast_Int():
    bench = CastIntBenchmark(
        op_name="_cast_Int",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cast_Int,
        gems_op=getattr(flag_gems, "_cast_Int", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()

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

"""Benchmark for ``aten::_cast_Long`` (``Tensor self, bool non_blocking=False``).

``torch_op`` is the performance baseline and ``gems_op`` the FlagGems candidate;
both receive the same arguments, and the case that omits ``non_blocking``
measures the schema default. Capability flags come from
``flag_gems.runtime.device``; aten::_cast_Long only produces int64, so the
benchmark is ineligible on a backend without int64 support.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# Performance-relevant shapes for a memory-bound cast.
CAST_LONG_SHAPES = [(1024, 1024), (20, 320, 15), (16, 128, 64, 60)]

# The "omitted" sentinel distinguishes a call with no keyword from an explicit
# boolean; only False and True are passed through as non_blocking flags.
CAST_LONG_PARAM_CALLS = ("omitted", False, True)


def _dtype_eligible(dtype):
    """Static dtype eligibility taken from the runtime capability flags."""
    if dtype == torch.bfloat16:
        return flag_gems.runtime.device.support_bf16
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return flag_gems.runtime.device.support_fp8
    if dtype == torch.int64:
        return flag_gems.runtime.device.support_int64
    return True


_CAST_INPUT_DTYPES = (
    consts.FLOAT_DTYPES
    + consts.INT_DTYPES
    + consts.BOOL_DTYPES
    + [
        torch.int8,
        torch.uint8,
        torch.int64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
)
# The cast produces int64 whatever the input dtype is, so every case needs int64
# on the backend; the remaining eligibility follows the input dtype.
CAST_LONG_DTYPES = (
    [dtype for dtype in _CAST_INPUT_DTYPES if _dtype_eligible(dtype)]
    if flag_gems.runtime.device.support_int64
    else []
)


def _make_input(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES:
        return utils.generate_tensor_input(shape, dtype, device)
    if dtype.is_floating_point:
        # FP8 has no dense random sampler; the target dtype rounds the values,
        # which is what the cast then reads.
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    info = torch.iinfo(dtype)
    return torch.randint(
        int(info.min), int(info.max), shape, dtype=dtype, device=device
    )


def _case_fn(shape, dtype):
    del dtype
    for call in CAST_LONG_PARAM_CALLS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"non_blocking": call},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = _make_input(shape, dtype, device)
    non_blocking = plan.params["non_blocking"]
    if non_blocking == "omitted":
        return inp, {}
    return inp, {"non_blocking": non_blocking}


def _validated_shape(shape):
    """Keep every shape a caller supplies, including scalars and zero extents."""
    if not isinstance(shape, tuple):
        shape = (shape,)
    dims = []
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"shape extents must be ints, got {extent!r}")
        if extent < 0:
            raise ValueError(f"shape extents must not be negative, got {extent!r}")
        dims.append(extent)
    return tuple(dims)


class CastLongBenchmark(OperatorBenchmark):
    """Two-phase benchmark with an explicit, validated default shape set."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=CAST_LONG_SHAPES)
        self.shapes = [_validated_shape(shape) for shape in self.shapes]


@pytest.mark.cast_Long
def test__cast_Long():
    bench = CastLongBenchmark(
        op_name="_cast_Long",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cast_Long,
        gems_op=getattr(flag_gems, "_cast_Long", None),
        dtypes=CAST_LONG_DTYPES,
    )
    bench.run()

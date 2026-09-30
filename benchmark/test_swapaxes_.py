# Copyright 2026, The FlagOS Contributors.
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

# aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
# Stride-only view: the measured call mutates the tensor it is handed, so the
# benchmark keeps fresh_inputs=True and rebuilds one input per sample. The op is
# dtype- and rank-agnostic, so the shared default/comprehensive shape families
# stay in place and only tiny, natively valid rank boundaries are appended.
_BOUNDARY_SHAPES = [(), (0,), (1,), (2, 3), (0, 3, 4), (2, 3, 4)]


def _bench_dtypes():
    # Every dtype the native stride swap accepts: the float/integer families,
    # bool and complex, the device fp8 entry and both fp8 types the torch build
    # exposes. None of them is dropped for the quick suite.
    fp8 = [
        getattr(torch, name)
        for name in ("float8_e4m3fn", "float8_e5m2")
        if hasattr(torch, name)
    ]
    return list(
        dict.fromkeys(
            consts.FLOAT_DTYPES
            + consts.INT_DTYPES
            + consts.EXTRA_INT_DTYPES
            + consts.BOOL_DTYPES
            + consts.COMPLEX_DTYPES
            + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
            + fp8
        )
    )


def _axis_pairs(shape):
    # In-range pairs for the rank: outer pair, last pair, the equal-axis no-op
    # and, from rank 3 on, the first pair. A 0-D or 1-D tensor accepts (0, 0)
    # only; every other pair raises IndexError natively.
    rank = len(shape)
    if rank < 2:
        return ((0, 0),)
    pairs = {(0, rank - 1), (rank - 2, rank - 1), (0, 0)}
    if rank >= 3:
        pairs.add((0, 1))
    return tuple(sorted(pairs))


def _case_fn(shape, dtype):
    # Listing stays tensor-free: this describes the plan, it allocates nothing
    # and calls no operator. The same plans feed execution.
    del dtype
    extents = tuple(int(extent) for extent in shape)
    return [
        base.BenchmarkCasePlan(
            shape={"input": list(extents)},
            params={"axis0": axis0, "axis1": axis1},
            builder_args=(extents, axis0, axis1),
        )
        for axis0, axis1 in _axis_pairs(extents)
    ]


def _build_inputs_fn(plan, dtype, device):
    # Only stride metadata is swapped, so the element values are never read and
    # an uninitialized buffer is enough; the axes are passed positionally, the
    # call form the correctness suite already exercises.
    shape, axis0, axis1 = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, axis0, axis1


class SwapaxesBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Keep the shared default/comprehensive families and any core_shapes.yaml
        # entry for this operator or class; append the rank boundaries on top.
        super().set_shapes(shape_file_path)
        existing = [tuple(shape) for shape in self.shapes]
        self.shapes = list(dict.fromkeys(existing + _BOUNDARY_SHAPES))


@pytest.mark.swapaxes_
def test_swapaxes_():
    bench = SwapaxesBenchmark(
        op_name="swapaxes_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.swapaxes_,
        gems_op=getattr(flag_gems, "swapaxes_", None),
        dtypes=[
            dtype
            for dtype in _bench_dtypes() + [torch.float64]
            if _DTYPE_FLAGS.get(dtype, True)
        ],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()

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

# aten::view_as_real_copy is complex-only (real and integer inputs raise), so the
# real dtype matrix of the generic benchmarks does not apply. complex32 is ATen's
# ComplexHalf alias over float16; complex128 needs the device fp64 capability.
VIEW_AS_REAL_COPY_DTYPES = [torch.complex32, torch.complex64] + (
    [torch.complex128] if flag_gems.runtime.device.support_fp64 else []
)

# The seven reference scales.
VIEW_AS_REAL_COPY_SHAPES = [
    (),
    (1,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


def _case_fn(shape, dtype):
    del dtype
    # Shapes may also arrive from a user shape file: a negative or non-integer
    # extent must fail the listing, while scalar and empty shapes stay valid.
    extents = tuple(shape)
    if not all(
        isinstance(e, int) and not isinstance(e, bool) and e >= 0 for e in extents
    ):
        raise ValueError(f"view_as_real_copy: invalid shape {extents!r}")
    yield base.BenchmarkCasePlan(
        shape={"input": list(extents)},
        params={},
        builder_args=(extents,),
    )


def _build_inputs_fn(plan, dtype, device):
    # benchmark.utils.generate_tensor_input returns None for complex dtypes other
    # than complex64, so the input is built from the requested complex dtype.
    return torch.randn(plan.builder_args[0], dtype=dtype, device=device), {}


class ViewAsRealCopyBenchmark(OperatorBenchmark):
    """Two-phase benchmark over complex inputs.

    The inherited shape defaults target element-wise float operators, so this
    operator declares its own; a shape file entry keyed by the operator or class
    name still takes precedence.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=VIEW_AS_REAL_COPY_SHAPES)


@pytest.mark.view_as_real_copy
@pytest.mark.parametrize("dtype", VIEW_AS_REAL_COPY_DTYPES)
def test_view_as_real_copy_bench(dtype):
    bench = ViewAsRealCopyBenchmark(
        op_name="view_as_real_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.view_as_real_copy,
        gems_op=getattr(flag_gems, "view_as_real_copy", None),
        dtypes=[dtype],
    )
    bench.run()

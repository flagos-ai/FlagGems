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

# _debug_has_internal_overlap only classifies the layout of its single input,
# so the meaningful workloads are layouts of the shared shape grid rather than
# extra shapes. The input payload stays empty: the schema takes one tensor and
# no extra arguments.
_LAYOUTS = ("contiguous", "transpose", "expand", "step_slice")
# Original layout-focused shapes, merged in additively at comprehensive level.
_LAYOUT_SHAPES = [(64, 64), (1024, 1024), (64, 512, 512)]


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for layout in _LAYOUTS if shape else ("contiguous",):
        if layout == "transpose" and len(shape) < 2:
            # Transposing needs two axes; the other layouts still cover it.
            continue
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape), "layout": layout},
            params={"layout": layout},
            builder_args=(shape, layout),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    if layout == "expand":
        # Broadcasting a size-1 leading axis produces the stride-0 layout that
        # the operator classifies as overlapping.
        base_inp = torch.empty((1,) + tuple(shape[1:]), dtype=dtype, device=device)
        inp = base_inp.expand(tuple(shape))
    elif layout == "transpose":
        inp = torch.empty(shape, dtype=dtype, device=device).transpose(-1, -2)
    elif layout == "step_slice":
        inp = torch.empty(shape, dtype=dtype, device=device)[..., ::2]
    else:
        inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, {}


class DebugHasInternalOverlapBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # Additive only: the shared core shape grid stays the default shape set
        # and a caller-supplied --shape-file still wins; these layout shapes are
        # merged in at comprehensive level.
        return super().set_more_shapes() + _LAYOUT_SHAPES


@pytest.mark.debug_has_internal_overlap
def test_debug_has_internal_overlap():
    bench = DebugHasInternalOverlapBenchmark(
        op_name="_debug_has_internal_overlap",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._debug_has_internal_overlap,
        gems_op=getattr(flag_gems, "_debug_has_internal_overlap", None),
        dtypes=(
            consts.FLOAT_DTYPES
            + consts.INT_DTYPES
            + consts.EXTRA_INT_DTYPES
            + consts.BOOL_DTYPES
            + consts.COMPLEX_DTYPES
        ),
    )
    bench.run()

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

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# aten::gradient returns Tensor[] (one tensor per differentiated dimension), so
# no single-output public benchmark family matches it: two-phase GenericBenchmark
# with an explicit case_fn + build_inputs_fn.
#
# A shape may carry several parameter variants (for example dim=0 and dim=[0,1]),
# so the table is grouped per shape and only JSON-compatible metadata is
# published; torch objects stay in the private builder_args. edge_order=2 appears
# only where every differentiated dimension has size >= 3.
_DEFAULT_PARAMS = {"dim": None, "spacing": None, "edge_order": 1}

_BENCH_CASES = [
    ((1024, 1024), {"dim": None}),
    ((2048, 1024), {"dim": 0}),
    ((2048, 1024), {"dim": [0, 1], "spacing": 2.0}),
    ((20, 320, 15), {"dim": None}),
    ((20, 160, 15), {"dim": [1, 2]}),
    ((20, 320, 30), {"dim": [2, 0], "spacing": 0.5, "edge_order": 2}),
    ((16, 128, 64, 60), {"dim": None}),
    ((8, 128, 64, 60), {"dim": [0, 2], "edge_order": 2}),
    ((16, 7, 57, 32, 29), {"dim": None}),
    ((16, 7, 57, 16, 29), {"dim": [4, 3]}),
    ((4096,), {"dim": None}),
    ((65536,), {"dim": 0, "edge_order": 2}),
]

# Listing and execution share this table: _case_fn reads it for every shape it is
# asked about, so a shape file publishes exactly the cases the run benchmarks and
# no separate hand-maintained list exists. Each shape is enumerated once even
# when it carries several parameter variants.
_PARAMS_BY_SHAPE = {}
for _shape, _params in _BENCH_CASES:
    _PARAMS_BY_SHAPE.setdefault(_shape, []).append(_params)
_BENCH_SHAPES = list(_PARAMS_BY_SHAPE)

_BENCH_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_bf16:
    _BENCH_DTYPES.append(torch.bfloat16)


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for variant in _PARAMS_BY_SHAPE.get(shape, [{}]):
        params = dict(_DEFAULT_PARAMS)
        params.update(variant)
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params=dict(params),
            builder_args=(
                shape,
                params["dim"],
                params["spacing"],
                params["edge_order"],
            ),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, dim, spacing, edge_order = plan.builder_args
    inp = utils.generate_tensor_input(tuple(shape), dtype, device)
    gradient_kwargs = {"edge_order": edge_order}
    if dim is not None:
        gradient_kwargs["dim"] = dim
    if spacing is not None:
        gradient_kwargs["spacing"] = spacing
    return inp, gradient_kwargs


class GradientBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = _BENCH_SHAPES
    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        # Delegates to the shared resolver: an operator-key entry wins, then this
        # exact class name, then _BENCH_SHAPES.
        super().set_shapes(shape_file_path, default_shapes=_BENCH_SHAPES)

    def set_more_shapes(self):
        # The inherited comprehensive extras contain size-1 dimensions, which
        # gradient rejects; _BENCH_SHAPES already spans 1-D to 5-D.
        return []


@pytest.mark.gradient
def test_gradient():
    bench = GradientBenchmark(
        op_name="gradient",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.gradient,
        gems_op=getattr(flag_gems, "gradient", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()

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

# The executor's per-element cost is a traversal plus one accumulation, so the
# elementwise core shape set is this operator's own workload list.
_BACKWARD_SHAPES = [
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"retain_graph": True, "create_graph": False},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    leaf = utils.generate_tensor_input(shape, dtype, device).requires_grad_(True)
    # Positional arguments followed by a trailing kwargs dict; the executor
    # returns None and refuses to differentiate a root with respect to itself.
    return (
        (torch.sin(leaf) * 2.0).sum(),
        [leaf],
        {
            "retain_graph": plan.params["retain_graph"],
            "create_graph": plan.params["create_graph"],
        },
    )


class BackwardBenchmark(OperatorBenchmark):
    """Two-phase benchmark for the AutogradBackward executor.

    Each case builds its own scalar graph from a requires-grad leaf and replays
    it with retain_graph=True, so no sample measures an already-freed graph.
    fresh_inputs stays off: that path detaches and clones the tensor arguments,
    which turns the graph root into a plain leaf, so the timed samples would
    only run an empty engine pass (the listed leaf's .grad stays None).
    """

    def set_shapes(self, shape_file_path=None):
        # Shared loader: honours --shape-file and keeps the core_shapes.yaml
        # default and COMPREHENSIVE grids. The workloads above are appended,
        # never capped.
        super().set_shapes(shape_file_path)
        self.shapes = list(dict.fromkeys([*map(tuple, self.shapes), *_BACKWARD_SHAPES]))


@pytest.mark.backward
def test__backward():
    bench = BackwardBenchmark(
        op_name="_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._backward,
        gems_op=getattr(flag_gems, "_backward", None),
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.run()

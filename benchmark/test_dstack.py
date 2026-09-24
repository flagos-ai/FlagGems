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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# aten::dstack(Tensor[] tensors) -> Tensor views every input as 3-D (atleast_3d)
# and concatenates along the new depth axis (dim 2). No public Benchmark family
# models a TensorList depth-concatenation, so the benchmark uses the two-phase
# GenericBenchmark (case_fn + build_inputs_fn) rather than a bare legacy
# input_fn.
#
# The benchmark concats three tensors: the equal-depth case (three identical
# shapes) makes the raw copy bandwidth the bottleneck, and for tensors with
# ndim >= 3 a second depth-varying case (only dim 2 differs between inputs,
# which is legal for dstack) exercises the candidate's per-input depth
# bookkeeping.
#
# The shape cap keeps allocations reasonable: dstack writes ~3x the input
# elements and the generic DEFAULT_SHAPES include 1G-element tensors, so any
# shape whose 3-tensor list exceeds MAX_ELEMENTS is dropped.
#
# gems_op is resolved inside the test function (never at import time) so the
# process-local override installed by KernelGen wins. flag_gems.dstack is not
# registered as a direct callable yet, so getattr(..., None) keeps the file
# importable/runnable before an implementation exists; either way
# torch_op=torch.ops.aten.dstack stays the perf comparison reference and
# gems_op, when resolved, is the candidate timed against it.


def _case_fn(shape, dtype):
    # One Workload per (shape, depth layout): a 3-element TensorList.
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"inputs": [shape, shape, shape]},
        params={"num_tensors": 3},
        builder_args=((shape, shape, shape),),
    )

    # Depth may vary per input (dim 2 is the concat axis); add one such case for
    # every tensor that has a depth axis of its own.
    if len(shape) >= 3 and shape[2] >= 2:
        step = max(1, shape[2] // 2)
        depths = (shape[2], shape[2] + step, max(1, shape[2] - step))
        if len(set(depths)) > 1:
            varying = tuple(
                tuple(depths[j] if i == 2 else dim for i, dim in enumerate(shape))
                for j in range(3)
            )
            yield base.BenchmarkCasePlan(
                shape={"inputs": list(varying)},
                params={"num_tensors": 3},
                builder_args=(varying,),
            )


def _build_inputs_fn(plan, dtype, device):
    # builder_args[0] is the tuple of per-input shapes; the TensorList is passed
    # positionally, matching aten::dstack(Tensor[])'s call semantics.
    shapes = plan.builder_args[0]
    inp = [
        utils.generate_tensor_input(shape, dtype, device)
        for shape in shapes[: plan.params["num_tensors"]]
    ]
    return inp, {}


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


class DstackBenchmark(OperatorBenchmark):
    # A 3-tensor dstack allocates 3 inputs plus a ~3x input output; cap the
    # total to avoid multi-GB cases (the generic DEFAULT_SHAPES reach 2**30
    # elements) that carry no extra signal.
    MAX_ELEMENTS = 3 * 2**26

    def set_more_shapes(self):
        # Depth-axis performance-relevant shapes: long 1-D rows, 2-D rows of
        # width 2**i, and 3-D volumes whose concat dim (dim 2) is the axis of
        # interest. Each stays well inside MAX_ELEMENTS for a 3-tensor list.
        return [
            (2**20,),
            (1024, 2**0),
            (1024, 2**8),
            (1024, 2**12),
            (64, 2**0, 64),
            (64, 2**4, 64),
            (64, 2**8, 64),
        ]

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            shape for shape in self.shapes if _numel(shape) * 3 <= self.MAX_ELEMENTS
        ]


@pytest.mark.dstack
@pytest.mark.dstack_benchmark
def test_dstack():
    bench = DstackBenchmark(
        op_name="dstack",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.dstack,
        gems_op=getattr(flag_gems, "dstack", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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

# aten::atleast_3d is a pure view/identity op (0-dim -> (1, 1, 1), 1-dim ->
# (1, N, 1), 2-dim -> (M, N, 1); ndim >= 3 returned unchanged) with two
# overloads (Tensor and Tensor[]). No public Benchmark family models a view
# op, so both overloads use the two-phase GenericBenchmark (case_fn +
# build_inputs_fn), never a bare legacy input_fn.
#
# A view's latency is dominated by dispatch/call overhead rather than tensor
# size, so the shape set is curated (one case per rank 0..4 plus a large 2-D
# and 3-D case) and stays small: the generic DEFAULT_SHAPES include 1G-element
# tensors that would only burn memory for no signal.
#
# gems_op is resolved inside each test function (never at import time) so the
# process-local override installed by KernelGen for this run wins; when no
# override is registered, the benchmark retains master's normal Gems dispatch.

_CURATED_SHAPES = [
    (),  # 0-dim scalar -> (1, 1, 1)
    (1,),  # single-element 1-dim -> (1, 1, 1)
    (256,),  # regular 1-dim -> (1, 256, 1)
    (1024, 1024),  # large 2-dim -> (1024, 1024, 1)
    (20, 320, 15),  # 3-dim identity
    (16, 128, 64),  # 3-dim identity
    (8, 16, 32, 4),  # 4-dim identity
]


class Atleast3DBenchmark(OperatorBenchmark):
    # Curated, size-bounded shape set: a view op has no data-dependent work, so
    # the larger generic shapes add allocation time without changing the
    # measured dispatch cost.
    DEFAULT_SHAPE_DESC = "rank-complete view shapes"

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=list(_CURATED_SHAPES))


def _case_fn(shape, dtype):
    # One Workload per shape for the Tensor overload.
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {}


def _sequence_case_fn(shape, dtype):
    # One Workload per shape for the Tensor[] overload, mixing a 0-dim scalar,
    # a 1-dim tensor, a 2-dim tensor and the current shape so the scalar ->
    # (1,1,1), 1-dim -> (1,N,1), 2-dim -> (M,N,1) and >= 3-dim identity paths
    # are all timed.
    del dtype
    seq_shapes = [(), (3,), (4, 5), shape]
    yield base.BenchmarkCasePlan(
        shape={"input": seq_shapes},
        params={},
        builder_args=(seq_shapes,),
    )


def _sequence_build_inputs_fn(plan, dtype, device):
    seq_shapes = plan.builder_args[0]
    inp = [utils.generate_tensor_input(s, dtype, device) for s in seq_shapes]
    return inp, {}


@pytest.mark.atleast_3d
@pytest.mark.atleast_3d_benchmark
def test_atleast_3d():
    bench = Atleast3DBenchmark(
        op_name="atleast_3d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_3d,
        gems_op=getattr(flag_gems, "atleast_3d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.atleast_3d_sequence
@pytest.mark.atleast_3d_benchmark
def test_atleast_3d_sequence():
    bench = Atleast3DBenchmark(
        op_name="atleast_3d",
        case_fn=_sequence_case_fn,
        build_inputs_fn=_sequence_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_3d.Sequence,
        gems_op=getattr(flag_gems, "atleast_3d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

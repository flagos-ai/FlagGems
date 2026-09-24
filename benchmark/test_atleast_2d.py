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

"""Benchmark for ``aten::atleast_2d`` (``default`` and ``Sequence`` overloads).

The op is a view/identity op, so the benchmark shapes are the dim boundary
cases that define it (0-dim -> (1, 1), 1-dim -> (1, N)) plus regular higher-rank
inputs. Both overloads use the two-phase ``GenericBenchmark`` API
(``case_fn`` + ``build_inputs_fn``).
"""

import pytest
import torch

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# View op: the benchmark allocates the full input, so keep shapes bounded while
# still covering the dim boundary (0-dim / 1-dim) and regular ranks.
_ATLEAST_2D_SHAPES = [
    (),
    (3,),
    (256,),
    (16, 256),
    (1024, 1024),
    (2, 19, 7),
    (20, 320, 15),
]
_MAX_BENCH_NUMEL = 8 * 1024 * 1024


class Atleast2DBenchmark(OperatorBenchmark):
    """``GenericBenchmark`` that always includes the 0-dim / 1-dim cases."""

    DEFAULT_SHAPES = _ATLEAST_2D_SHAPES

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Never benchmark the multi-GB default shapes: the op only returns a
        # view, the input allocation would dominate and OOM.
        self.shapes = [
            s for s in self.shapes if torch.Size(s).numel() <= _MAX_BENCH_NUMEL
        ]
        for extra in ((), (3,)):
            if extra not in self.shapes:
                self.shapes = [extra] + list(self.shapes)


def _case_fn(shape, dtype):
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
    del dtype
    # Mix a 0-dim scalar, a 1-dim tensor and the current shape so the sequence
    # overload exercises scalar -> (1, 1), 1-dim -> (1, N) and the >= 2-dim
    # identity paths.
    seq_shapes = [(), (3,), shape]
    yield base.BenchmarkCasePlan(
        shape={"input": seq_shapes},
        params={},
        builder_args=(seq_shapes,),
    )


def _sequence_build_inputs_fn(plan, dtype, device):
    seq_shapes = plan.builder_args[0]
    inp = [utils.generate_tensor_input(s, dtype, device) for s in seq_shapes]
    return inp, {}


@pytest.mark.atleast_2d
def test_atleast_2d():
    bench = Atleast2DBenchmark(
        op_name="atleast_2d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_2d,
        gems_op=getattr(flag_gems, "atleast_2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.atleast_2d_sequence
def test_atleast_2d_sequence():
    bench = Atleast2DBenchmark(
        op_name="atleast_2d",
        case_fn=_sequence_case_fn,
        build_inputs_fn=_sequence_build_inputs_fn,
        torch_op=torch.ops.aten.atleast_2d.Sequence,
        gems_op=getattr(flag_gems, "atleast_2d", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

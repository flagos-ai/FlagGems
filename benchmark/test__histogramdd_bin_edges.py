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

"""Benchmark for ``aten::_histogramdd_bin_edges``.

The native operator only has a CPU kernel, so the perf baseline
(``torch.ops.aten._histogramdd_bin_edges``) and the candidate receive the same
CPU tensors: `--list-cases` metadata stays device independent and the input
builder allocates on CPU inside the timing phase. `torch_op` is only the perf
reference; both paths share the call form `op(input, bins)`, with `bins`
carried through the case params and unpacked into the operator call.

There is no ``core_shapes.yaml`` entry for this operator, so the class supplies
the shape set as a default and a custom ``--shape_file`` still overrides it.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# Native support is float32/float64 only (consts.FLOAT_DTYPES would add
# float16/bfloat16, which the operator rejects).
HDBE_DTYPES = [torch.float32, torch.float64]

# The innermost dimension is the histogram dimension, so a shape yields
# len(shape[-1]) edge tensors of HDBE_BINS_PER_DIM + 1 entries.
HDBE_SHAPES = [
    (256, 3),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

HDBE_BINS_PER_DIM = 4


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"bins": [HDBE_BINS_PER_DIM] * shape[-1]},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    del device  # CPU-only native kernel: both paths are timed on CPU tensors.
    shape = plan.builder_args[0]
    inp = torch.randn(shape, dtype=dtype, device="cpu")
    return inp, {"bins": plan.params["bins"]}


class HistogramDDBinEdgesBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # A one-dimensional sample buffer represents one histogram dimension.
        self.shapes = [
            tuple(shape) if len(shape) >= 2 else ((shape[0], 1) if shape else (1, 1))
            for shape in self.shapes
        ]
        for shape in HDBE_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)


@pytest.mark.histogramdd_bin_edges
def test__histogramdd_bin_edges():
    bench = HistogramDDBinEdgesBenchmark(
        op_name="_histogramdd_bin_edges",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._histogramdd_bin_edges,
        gems_op=getattr(flag_gems, "_histogramdd_bin_edges", None),
        dtypes=HDBE_DTYPES,
    )
    bench.run()

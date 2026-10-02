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

# aten::_mkldnn_transpose runs on CPU only, over opaque oneDNN operands that
# dense_to_mkldnn builds from float32/float16/bfloat16/uint8/int8 alone, so
# those five dtypes are the complete supported set and no device capability
# flag applies.
MKLDNN_TRANSPOSE_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.float32,
    torch.bfloat16,
    torch.float16,
]

# Reordering-friendly shapes added to whatever the shape file provides for this
# benchmark class.
MKLDNN_TRANSPOSE_EXTRA_SHAPES = [
    (256,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]


def _dim_pairs(rank):
    # A rank-1 operand has one dimension, so only the dim0 == dim1 relayout is
    # valid; higher ranks add the last-two pair, the reversed (negative) pair
    # and that relayout.
    if rank == 1:
        return ((0, 0),)
    return ((0, 1), (-1, -2), (0, 0))


def _case_fn(shape, dtype):
    del dtype
    for dim0, dim1 in _dim_pairs(len(shape)):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"dim0": dim0, "dim1": dim1},
            builder_args=(shape, dim0, dim1),
        )


def _build_inputs_fn(plan, dtype, device):
    del device  # the operand lives in the CPU-only mkldnn layout
    shape, dim0, dim1 = plan.builder_args
    # Input allocation happens outside timing. The output is not compared,
    # and utils.generate_tensor_input has no branch for int8/uint8, so the
    # payload is allocated directly in the required layout. dim0/dim1 are
    # positional arguments of the aten schema.
    inp = torch.empty(shape, dtype=dtype, device="cpu").to_mkldnn()
    return inp, dim0, dim1


class MkldnnTransposeBenchmark(OperatorBenchmark):
    """Two-phase benchmark over CPU opaque mkldnn operands."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        merged = [tuple(shape) for shape in self.shapes]
        merged += MKLDNN_TRANSPOSE_EXTRA_SHAPES
        # The layout has no 0-dim form, so a rank-0 shape supplied by a shape
        # file cannot be built; every other requested shape is kept.
        self.shapes = [shape for shape in dict.fromkeys(merged) if len(shape) >= 1]


@pytest.mark.mkldnn_transpose
def test__mkldnn_transpose():
    bench = MkldnnTransposeBenchmark(
        op_name="_mkldnn_transpose",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._mkldnn_transpose,
        gems_op=getattr(flag_gems, "_mkldnn_transpose", None),
        dtypes=MKLDNN_TRANSPOSE_DTYPES,
    )
    bench.run()

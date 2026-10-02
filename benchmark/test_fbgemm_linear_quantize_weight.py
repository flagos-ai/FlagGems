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

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

OP_NAME = "fbgemm_linear_quantize_weight"

# Per-row weight blocks added on top of the shared shape loader. The native
# kernel needs rank >= 2 (rank 1/0 raise IndexError inside the kernel) and treats
# everything after dim 0 as one row block.
BENCH_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2048, 2048),
    (4096, 4096),
    (4096, 11008),
    (20, 320, 15),
    (16, 128, 64, 60),
]


def _weight_shape(shape):
    """Return a rank >= 2 geometry for ``shape`` without changing its numel.

    A rank < 2 entry (the shared loader leads with the 1-D [1073741824], and the
    comprehensive set with 1-D [268435456]) is refactored into the most balanced
    rank-2 geometry with the same element count, because the kernel indexes
    dim 1 unconditionally: no extent is shrunk and no case is dropped. Extents
    stay exactly as requested; a missing shape file still raises
    FileNotFoundError from the shared loader and a malformed extent raises here.
    """
    try:
        extents = tuple(shape)
    except TypeError:
        raise ValueError("a shape must be a sequence of extents") from None
    for extent in extents:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError("invalid shape extent " + repr(extent))
    if len(extents) >= 2:
        return extents
    numel = math.prod(extents) if extents else 1
    if numel == 0:
        return (0, 1)
    side = math.isqrt(numel)
    while side > 1 and numel % side:
        side -= 1
    return (side, numel // side)


def _case_fn(shape, dtype):
    del dtype
    shape = _weight_shape(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # The kernel is CPU-only (a CUDA operand faults inside the native op), so the
    # weight tensor is built on the CPU that both the reference and the injected
    # candidate receive. Allocation is outside the timed region.
    inp = torch.randn(shape, dtype=dtype, device="cpu")
    return inp, {}


class FbgemmLinearQuantizeWeightBenchmark(OperatorBenchmark):
    # Probed: every other dtype raises "expected scalar type Float but found
    # <dtype>", so float32 is the supported set.
    DEFAULT_DTYPES = [torch.float32]
    DEFAULT_SHAPE_DESC = "N, K"

    def set_shapes(self, shape_file_path=None):
        # The shared loader stays authoritative; the declared weight blocks are
        # added afterwards and every entry is refactored to the kernel's rank >= 2
        # contract, which also covers the comprehensive merge driven by
        # set_more_shapes() below.
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys([_weight_shape(s) for s in list(self.shapes) + BENCH_SHAPES])
        )

    def set_more_shapes(self):
        return [_weight_shape(shape) for shape in super().set_more_shapes()]


@pytest.mark.fbgemm_linear_quantize_weight
def test_fbgemm_linear_quantize_weight():
    bench = FbgemmLinearQuantizeWeightBenchmark(
        op_name=OP_NAME,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_linear_quantize_weight,
        gems_op=getattr(flag_gems, OP_NAME, None),
        dtypes=[torch.float32],
    )
    bench.run()

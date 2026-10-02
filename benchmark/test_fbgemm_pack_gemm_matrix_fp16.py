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

"""Benchmark for ``aten::fbgemm_pack_gemm_matrix_fp16``.

CPU-only FBGEMM packing of an ``(N, K)`` weight: the cost is a saturation pass
over ``N * K`` elements, so the geometry below keeps realistic matrix shapes and
avoids arbitrarily large allocations. The runtime device argument is ignored --
the native kernel has no CUDA build and crashes the process on a CUDA weight --
and the weight is built on CPU instead. The shared shape grid is resolved
normally and then unioned with native-valid extras; a rank < 2 entry cannot be
packed, so it is mapped to numel-preserving factor-pair geometry (see
``_matrix_geometry``) instead of being dropped.
"""

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

OP_NAME = "fbgemm_pack_gemm_matrix_fp16"

# float32 is the only dtype the native kernel accepts; any other dtype raises
# "expected scalar type Float but found <X>", so the listed dtype set is exact.
BENCH_DTYPES = [torch.float32]

# (N, K) weight geometry: the small/odd sizes cover the per-element saturation
# pass and the larger ones the FBGEMM micro-kernel; the rank >= 3 entries exercise
# the flat-prefix packing path.
PACK_SHAPES = [
    (0, 7),
    (1, 1),
    (7, 17),
    (33, 65),
    (128, 256),
    (256, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 1),
    (1, 4096),
    (2, 19, 7),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]

_CPU = torch.device("cpu")


def _matrix_geometry(shape):
    """Map a requested shape to native-valid, numel-preserving (N, K) geometry.

    The operator reads ``size(0)``/``size(1)``, so rank < 2 cannot be packed: a
    scalar or 1-D request keeps its element count as the balanced factor pair
    rather than being dropped. Rank >= 2 requests are used unchanged.
    """
    shape = tuple(shape)
    if len(shape) >= 2:
        return shape
    numel = math.prod(shape)
    rows = math.gcd(numel, max(1, math.isqrt(numel)))
    return (rows, numel // rows)


def _case_fn(shape, dtype):
    del dtype
    geometry = _matrix_geometry(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(geometry)},
        params={},
        builder_args=(geometry,),
    )


def _build_inputs_fn(plan, dtype, device):
    # The runtime device is ignored on purpose: this op is CPU-only FBGEMM, so the
    # weight has to be a CPU tensor. Allocation stays outside the measured call.
    del device
    (shape,) = plan.builder_args
    inp = torch.rand(shape, dtype=dtype, device=_CPU)
    return inp, {}


class FbgemmPackGemmMatrixFp16Benchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over the shared grid plus native-valid extras."""

    def set_shapes(self, shape_file_path=None):
        # Ordinary resolution keeps a caller --shape_file authoritative (it wins
        # over the shared grid); the native-valid extras are unioned in after it,
        # normalized to the same geometry the case builder executes.
        super().set_shapes(shape_file_path)
        merged = []
        for shape in list(self.shapes) + PACK_SHAPES:
            geometry = _matrix_geometry(shape)
            if geometry not in merged:
                merged.append(geometry)
        self.shapes = merged


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
def test_fbgemm_pack_gemm_matrix_fp16():
    bench = FbgemmPackGemmMatrixFp16Benchmark(
        op_name=OP_NAME,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_pack_gemm_matrix_fp16,
        gems_op=getattr(flag_gems, OP_NAME, None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()

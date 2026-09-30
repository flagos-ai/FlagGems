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

"""Benchmark for ``aten::fbgemm_pack_quantized_matrix``.

CPU-only FBGEMM int8 host packer, so builders allocate int8 CPU tensors directly
and the native rank >= 2 contract is enforced by folding the shared 1-D extras
to rank 2 with an identical element count rather than dropping them.
"""

import math

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_SHAPES = [
    [1, 4],
    [8, 16],
    [33, 17],
    [128, 256],
    [1024, 1024],
    [20, 320, 15],
    [16, 128, 64, 60],
    [16, 7, 57, 32, 29],
]

# None exercises the public ``.default`` form; the pairs exercise ``.KN``.
_KN_VALUES = [None, [4, 4], [8, 16], [2, 19]]


def _packable_shape(shape):
    """Rank >= 2 geometry with an identical element count.

    The native packer reads dimensions 0 and 1, so a rank-1 shape is
    native-invalid. Folding it into a near-square matrix keeps the whole numel
    and therefore the workload, e.g. (2**28,) -> (2**14, 2**14) and
    (10000,) -> (100, 100); rank >= 2 extents are returned unchanged.
    """
    if len(shape) >= 2:
        return tuple(shape)
    if not shape:
        return (1, 1)
    numel = shape[0]
    if numel < 4:
        return (1, numel)
    factor = math.isqrt(numel)
    while numel % factor:
        factor -= 1
    return (factor, numel // factor)


def _case_fn(shape, dtype):
    del dtype
    for kn in _KN_VALUES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"kn": kn},
            builder_args=(shape, kn),
        )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device  # the native packer accepts int8 on the CPU only
    shape, kn = plan.builder_args
    inp = torch.randint(-128, 128, tuple(shape), dtype=torch.int8)
    if kn is None:
        return (inp,)
    return (inp, int(kn[0]), int(kn[1]))


class FbgemmPackQuantizedMatrixBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                _packable_shape(tuple(shape)) for shape in list(self.shapes) + _SHAPES
            )
        )


@pytest.mark.fbgemm_pack_quantized_matrix
def test_fbgemm_pack_quantized_matrix():
    bench = FbgemmPackQuantizedMatrixBenchmark(
        op_name="fbgemm_pack_quantized_matrix",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.fbgemm_pack_quantized_matrix,
        gems_op=getattr(flag_gems, "fbgemm_pack_quantized_matrix", None),
        dtypes=[torch.int8],
    )
    bench.run()

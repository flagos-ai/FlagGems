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

from . import base, consts

# (input_shape, row_strided) tuples. Every shape ends with the required size-2
# last dimension; `row_strided=True` feeds a non-contiguous batch view (even
# rows of a larger pool) so the strided gather path is exercised.
VIEW_AS_COMPLEX_COPY_SHAPES = [
    ((2048, 1024, 2), False),
    ((4096, 4096, 2), False),
    ((8, 4096, 4096, 2), False),
    ((4096, 8192, 2), False),
    ((16, 4096, 2048, 2), False),
    ((4096, 8192, 2), True),
]

# float16 -> complex32, float32 -> complex64, float64 -> complex128. The
# complex32 output is unsupported on many comparison/math paths and bfloat16
# is rejected outright by view_as_complex_copy, so bfloat16 is dropped from
# the standard float dtype set and float64 is added in its place. The line
# below documents the deliberate restriction; the base class validates it.
BENCH_DTYPES = [d for d in consts.FLOAT_DTYPES if d != torch.bfloat16]
if torch.float64 not in BENCH_DTYPES:
    BENCH_DTYPES.append(torch.float64)


class ViewAsComplexCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = VIEW_AS_COMPLEX_COPY_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape, row_strided in self.shapes:
            if row_strided:
                pool = torch.randn(
                    (shape[0] * 2,) + tuple(shape[1:]),
                    dtype=cur_dtype,
                    device=self.device,
                )
                yield (pool[::2],)
            else:
                yield (torch.randn(shape, dtype=cur_dtype, device=self.device),)


@pytest.mark.view_as_complex_copy
def test_view_as_complex_copy():
    bench = ViewAsComplexCopyBenchmark(
        op_name="view_as_complex_copy",
        torch_op=torch.ops.aten.view_as_complex_copy,
        dtypes=BENCH_DTYPES,
    )
    bench.run()

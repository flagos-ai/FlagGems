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

# (input_shape, row_strided) tuples of complex tensors. `row_strided=True`
# feeds a non-contiguous view (even rows of a larger pool) so the strided
# gather path is exercised.
VIEW_AS_REAL_COPY_SHAPES = [
    ((2048, 1024), False),
    ((4096, 4096), False),
    ((8, 4096, 4096), False),
    ((4096, 8192), False),
    ((16, 4096, 2048), False),
    ((4096, 8192), True),
]

# view_as_real_copy consumes a complex tensor and produces the real tensor one
# size down, so the benchmark domains are the complex counterparts of the
# shared float dtype set: the precision choices come from `consts.FLOAT_DTYPES`
# and are mapped to complex32/64/128, with the float64 entry added explicitly
# because the shared set stops at float32. bfloat16 is rejected outright (it has
# no complex form) so it is dropped from the standard set. The line below
# documents the deliberate restriction; the base class validates it.
_FLOAT_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
BENCH_DTYPES = [
    _FLOAT_TO_COMPLEX[d] for d in consts.FLOAT_DTYPES if d in _FLOAT_TO_COMPLEX
]
if torch.complex128 not in BENCH_DTYPES:
    BENCH_DTYPES.append(torch.complex128)


class ViewAsRealCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = VIEW_AS_REAL_COPY_SHAPES

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


@pytest.mark.view_as_real_copy
def test_view_as_real_copy():
    bench = ViewAsRealCopyBenchmark(
        op_name="view_as_real_copy",
        torch_op=torch.ops.aten.view_as_real_copy,
        dtypes=BENCH_DTYPES,
    )
    bench.run()

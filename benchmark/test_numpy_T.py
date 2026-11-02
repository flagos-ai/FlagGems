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

from . import base, consts

# Ranks 2 through 4. numpy_T is a zero-copy view operator, so the benchmark
# measures dispatch + as_strided overhead rather than data movement; the
# extents span the small (launch-bound) and large (metadata-heavy) regimes.
NUMPY_T_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 64),
    (8, 16, 32, 64),
]


class NumpyTBenchmark(base.Benchmark):
    """Benchmark for ``aten::numpy_T`` (zero-copy full-dimension reversal)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = NUMPY_T_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (inp,)


@pytest.mark.numpy_T
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_numpy_T(dtype):
    bench = NumpyTBenchmark(
        op_name="numpy_T",
        torch_op=torch.ops.aten.numpy_T,
        gems_op=flag_gems.numpy_T,
        dtypes=[dtype],
    )
    bench.run()

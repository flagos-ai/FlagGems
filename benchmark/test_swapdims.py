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

# 2D / 3D / 4D shapes for benchmarking swapdims. swapdims is a zero-copy view
# op, so the benchmark measures dispatch + as_strided overhead rather than
# data movement; the sizes still exercise both small and large stride products.
SWAPDIMS_SHAPES = [
    (64, 64),
    (256, 512),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 64),
    (8, 16, 32, 64),
]


class SwapdimsBenchmark(base.Benchmark):
    """Benchmark for ``aten::swapdims`` (zero-copy dim swap view)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = SWAPDIMS_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            # Swap the first and last dims for every shape.
            ndim = inp.dim()
            yield inp, 0, ndim - 1


@pytest.mark.swapdims
def test_swapdims():
    bench = SwapdimsBenchmark(
        op_name="swapdims",
        torch_op=torch.swapdims,
        gems_op=flag_gems.swapdims,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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


class LinalgPowsumBenchmark(base.Benchmark):
    CORE_SHAPES = [(1024,), (256, 256), (1024, 1024), (64, 512, 256)]
    COMPREHENSIVE_SHAPES = [(2048, 2048), (128, 1024, 512)]

    def __init__(self, *args, ord, **kwargs):
        super().__init__(*args, **kwargs)
        self.ord = ord

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += self.COMPREHENSIVE_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=dtype, device=self.device)
            dim = None if len(shape) == 1 else [-1]
            yield inp, self.ord, dim, False


@pytest.mark.linalg__powsum
@pytest.mark.parametrize("ord", [1, 2, 3])
def test_linalg__powsum(ord):
    bench = LinalgPowsumBenchmark(
        op_name="linalg__powsum",
        torch_op=torch.ops.aten.linalg__powsum.default,
        dtypes=consts.FLOAT_DTYPES,
        ord=ord,
    )
    bench.run()

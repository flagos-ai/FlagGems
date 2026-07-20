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

# Representative 2D reduction shapes for logsumexp benchmarking
LOGSUMEXP_SHAPES = [
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]


class LogsumexpBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = LOGSUMEXP_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, 1


@pytest.mark.special_logsumexp
def test_special_logsumexp():
    bench = LogsumexpBenchmark(
        op_name="special_logsumexp",
        torch_op=torch.special.logsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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


class RenormBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # 2D shapes: (num_slices, slice_size) to benchmark renorm across different dimensions
        self.shapes = [
            (16, 256),
            (32, 512),
            (64, 1024),
            (128, 2048),
            (256, 4096),
            (512, 8192),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield inp, 2.0, 1, 1.0


@pytest.mark.renorm_
def test_renorm_():
    bench = RenormBenchmark(
        op_name="renorm_",
        torch_op=torch.ops.aten.renorm_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

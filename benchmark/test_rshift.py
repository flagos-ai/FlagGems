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

from typing import Generator

import pytest
import torch

from flag_gems.experimental_ops.__rshift__ import rshift_tensor

from . import base, consts


class RshiftBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative vector, matrix, and three-dimensional pointwise inputs.
        self.shapes = [(1024,), (1024, 1024), (16, 512, 256)]
        self.shape_desc = "SHAPE"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            value = torch.randint(0, 100, shape, dtype=cur_dtype, device=self.device)
            shift = torch.randint(0, 8, shape, dtype=cur_dtype, device=self.device)
            yield value, shift


@pytest.mark.rshift
def test_rshift():
    bench = RshiftBenchmark(
        op_name="rshift",
        torch_op=torch.bitwise_right_shift,
        dtypes=consts.INT_DTYPES + consts.EXTRA_INT_DTYPES,
    )
    bench.set_gems(rshift_tensor)
    bench.run()

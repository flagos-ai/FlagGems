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

from . import base, consts, utils

FLIPLR_SHAPES = [
    (4, 4),
    (64, 64),
    (512, 512),
    (1024, 4096),
    (4096, 4096),
    (64, 512, 512),
    (16, 128, 64, 64),
]


class FliplrBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = FLIPLR_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield utils.generate_tensor_input(shape, dtype, self.device), {}


@pytest.mark.fliplr
def test_fliplr():
    bench = FliplrBenchmark(
        op_name="fliplr",
        torch_op=torch.fliplr,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

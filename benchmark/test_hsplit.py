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

from . import base, consts


class HsplitBenchmark(base.Benchmark):
    """Benchmark for hsplit operator."""

    def set_shapes(self, shape_file_path=None):
        # Various 2D and 3D shapes for hsplit benchmarking
        self.shapes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (8, 16, 32),
            (16, 32, 64),
            (32, 64, 128),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            # Split horizontally into sections
            sections = 4 if shape[1] >= 4 else 2
            yield inp, sections


@pytest.mark.hsplit
def test_hsplit():
    bench = HsplitBenchmark(
        op_name="hsplit",
        torch_op=torch.hsplit,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

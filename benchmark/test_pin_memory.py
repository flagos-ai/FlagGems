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


class PinMemoryBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative host buffers used for asynchronous host-to-device copies.
        self.shapes = [(1024,), (1024, 1024), (4096, 4096)]
        self.shape_desc = "SHAPE"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield (torch.randn(shape, dtype=cur_dtype),)


@pytest.mark.pin_memory
def test_pin_memory():
    bench = PinMemoryBenchmark(
        op_name="pin_memory",
        torch_op=torch.ops.aten.pin_memory,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

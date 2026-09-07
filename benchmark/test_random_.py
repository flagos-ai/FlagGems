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


def random_input_fn(shape, dtype, device):
    # random_() overwrites the whole tensor, so the initial content is irrelevant.
    yield torch.empty(shape, dtype=dtype, device=device),


class RandomInplaceBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64,),
            (1024,),
            (16384,),
            (1024, 1024),
            (64, 128, 128),
        ]
        self.shape_desc = "input shape"


@pytest.mark.random_
def test_random_():
    bench = RandomInplaceBenchmark(
        input_fn=random_input_fn,
        op_name="random_",
        torch_op=torch.Tensor.random_,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()

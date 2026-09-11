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


class RreluWithNoiseFunctionalBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype: torch.dtype) -> Generator:
        for shape in self.shapes:
            inp = utils.generate_tensor_input(shape, dtype, self.device)
            noise = torch.rand_like(inp)
            lower = 0.125
            upper = 1.0 / 3.0
            training = True
            generator = None
            yield inp, noise, lower, upper, training, generator


@pytest.mark.rrelu_with_noise_functional
def test_rrelu_with_noise_functional():
    bench = RreluWithNoiseFunctionalBenchmark(
        op_name="rrelu_with_noise_functional",
        torch_op=torch.ops.aten.rrelu_with_noise_functional,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

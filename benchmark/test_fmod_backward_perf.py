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


class FmodBackwardBenchmark(base.Benchmark):
    def __init__(self, scalar):
        self.scalar = scalar
        super().__init__(
            op_name="fmod_backward_scalar" if scalar else "fmod_backward_tensor",
            torch_op=torch.fmod,
            gems_op=flag_gems.fmod_scalar if scalar else flag_gems.fmod_tensor,
            dtypes=consts.FLOAT_DTYPES,
            is_backward=True,
        )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = (
                torch.rand(shape, device=flag_gems.device, dtype=dtype) * 8 - 4
            ).requires_grad_(True)
            if self.scalar:
                yield x, 1.5
            else:
                y = (
                    torch.rand(shape, device=flag_gems.device, dtype=dtype) + 0.3
                ).requires_grad_(True)
                yield x, y


@pytest.mark.fmod_backward
def test_fmod_backward_tensor():
    FmodBackwardBenchmark(False).run()


@pytest.mark.fmod_backward
def test_fmod_backward_scalar():
    FmodBackwardBenchmark(True).run()

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


class HypotBackwardBenchmark(base.Benchmark):
    def __init__(self):
        super().__init__(
            op_name="hypot_backward",
            torch_op=torch.hypot,
            gems_op=flag_gems.hypot,
            dtypes=consts.FLOAT_DTYPES,
            is_backward=True,
        )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield (
                torch.randn(
                    shape, dtype=dtype, device=flag_gems.device, requires_grad=True
                ),
                torch.randn(
                    shape, dtype=dtype, device=flag_gems.device, requires_grad=True
                ),
            )


@pytest.mark.hypot_backward
def test_hypot_backward():
    HypotBackwardBenchmark().run()

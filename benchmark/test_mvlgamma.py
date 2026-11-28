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


class MvlgammaBackwardBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # p=5 requires x > 2; the margin also exercises non-integer inputs.
            yield torch.rand(shape, dtype=cur_dtype, device=self.device) + 3.0,


@pytest.mark.mvlgamma
def test_mvlgamma():
    bench = base.UnaryPointwiseBenchmark(
        op_name="mvlgamma",
        torch_op=lambda a: torch.mvlgamma(a, 5),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.mvlgamma_backward
def test_mvlgamma_backward():
    bench = MvlgammaBackwardBenchmark(
        op_name="mvlgamma_backward",
        torch_op=lambda a: torch.mvlgamma(a, 5),
        gems_op=lambda a: flag_gems.mvlgamma(a, 5),
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()

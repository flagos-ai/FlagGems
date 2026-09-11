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


@pytest.mark.amin
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="amin",
        torch_op=torch.amin,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.amin_
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_amin_(dtype):
    bench = base.UnaryReductionBenchmark(
        op_name="amin_",
        torch_op=lambda *a: a[0].copy_(torch.amin(*a, keepdim=True)),
        dtypes=[dtype],
        is_inplace=True,
        gems_op=flag_gems.amin_,
    )
    bench.run()

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


@pytest.mark.log10
def test_log10():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log10", torch_op=torch.log10, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.log10_
def test_log10_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="log10_",
        torch_op=torch.log10_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.log10_out
def test_log10_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="log10_out",
        torch_op=torch.log10,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

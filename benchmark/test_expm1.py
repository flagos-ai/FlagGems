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


@pytest.mark.expm1
def test_expm1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="expm1", torch_op=torch.expm1, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.expm1_
def test_expm1_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="expm1_",
        torch_op=torch.expm1_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.expm1_out
def test_expm1_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="expm1_out",
        torch_op=torch.expm1,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

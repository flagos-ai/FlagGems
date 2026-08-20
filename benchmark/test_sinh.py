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


@pytest.mark.sinh
def test_sinh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh",
        torch_op=torch.sinh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.sinh_
def test_sinh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="sinh_",
        torch_op=lambda a: a.sinh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()

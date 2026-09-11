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


@pytest.mark.gelu
def test_gelu():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu", torch_op=torch.nn.functional.gelu, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.gelu_
def test_gelu_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu_",
        torch_op=torch.ops.aten.gelu_.default,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.gelu_backward
def test_gelu_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="gelu_backward",
        torch_op=torch.nn.functional.gelu,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()

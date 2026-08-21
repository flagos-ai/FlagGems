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

from . import base, consts, utils


def input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    grad = utils.generate_tensor_input(shape, dtype, device)
    output = torch.cumprod(inp, dim=1)
    yield grad, inp, 1, output


@pytest.mark.cumprod_backward
def test_cumprod_backward():
    bench = base.GenericBenchmark2DOnly(
        input_fn=input_fn,
        op_name="cumprod_backward",
        torch_op=torch.ops.aten.cumprod_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()

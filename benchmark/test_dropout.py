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

from . import base, consts, utils


def _dropout_backward_input_fn(shape, dtype, device):
    grad_output = utils.generate_tensor_input(shape, dtype, device)
    mask = torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    scale = 1.0 / 0.5  # 1.0 / (1.0 - p), where p=0.5
    yield grad_output, mask, {"scale": scale}


@pytest.mark.dropout
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_dropout():
    bench = base.UnaryPointwiseBenchmark(
        op_name="dropout", torch_op=torch.nn.Dropout(p=0.5), dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.dropout_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_dropout_backward():
    bench = base.GenericBenchmark(
        op_name="dropout_backward",
        input_fn=_dropout_backward_input_fn,
        torch_op=torch.ops.aten.native_dropout_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

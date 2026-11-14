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
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(16, 4, 32, 32), (32, 8, 64, 64), (64, 16, 128, 128)]


class RnnReluBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "seq,batch,input,hidden"


def _case_fn(shape, dtype):
    del dtype
    seq, batch, input_size, hidden_size = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={
            "input": [seq, batch, input_size],
            "hx": [1, batch, hidden_size],
        },
        params={"layers": 1, "bidirectional": False, "batch_first": False},
        builder_args=((seq, batch, input_size, hidden_size),),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    seq, batch, input_size, hidden_size = plan.builder_args[0]
    value = torch.randn(seq, batch, input_size, dtype=dtype, device=device)
    hx = torch.randn(1, batch, hidden_size, dtype=dtype, device=device)
    module = torch.nn.RNN(input_size, hidden_size, 1, nonlinearity="relu").to(
        dtype=dtype, device=device
    )
    params = tuple(module._flat_weights)
    return value, hx, params, True, 1, 0.0, False, False, False


@pytest.mark.rnn_relu
def test_rnn_relu():
    bench = RnnReluBenchmark(
        op_name="rnn_relu",
        torch_op=torch.ops.aten.rnn_relu.input,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

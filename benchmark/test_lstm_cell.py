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
from .generated_operator_utils import OperatorBenchmark

LSTM_CELL_SHAPES = [
    (1, 128, 128),
    (8, 256, 256),
    (16, 512, 512),
    (32, 1024, 1024),
    (64, 2048, 512),
    (128, 4096, 1024),
]


def _case_fn(shape, dtype):
    del dtype
    batch, input_size, hidden = shape
    yield base.BenchmarkCasePlan(
        shape={"input": (batch, input_size), "hidden": (batch, hidden)},
        params={"bias": True},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, input_size, hidden = plan.builder_args[0]
    gates = 4 * hidden
    inp = utils.generate_tensor_input((batch, input_size), dtype, device)
    w_ih = utils.generate_tensor_input((gates, input_size), dtype, device)
    w_hh = utils.generate_tensor_input((gates, hidden), dtype, device)
    b_ih = utils.generate_tensor_input((gates,), dtype, device)
    b_hh = utils.generate_tensor_input((gates,), dtype, device)
    h0 = utils.generate_tensor_input((batch, hidden), dtype, device)
    c0 = utils.generate_tensor_input((batch, hidden), dtype, device)
    return inp, [h0, c0], w_ih, w_hh, b_ih, b_hh, {}


class LstmCellBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=LSTM_CELL_SHAPES)


@pytest.mark.lstm_cell
def test_lstm_cell():
    bench = LstmCellBenchmark(
        op_name="lstm_cell",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.lstm_cell,
        gems_op=getattr(flag_gems, "lstm_cell", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

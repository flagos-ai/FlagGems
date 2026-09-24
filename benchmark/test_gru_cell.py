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

GRU_CELL_SHAPES = [
    (32, 128, 64),
    (64, 512, 256),
    (128, 1024, 512),
    (256, 256, 256),
    (512, 128, 512),
    (1024, 512, 256),
]


def _case_fn(shape, dtype):
    del dtype
    batch, input_size, hidden_size = shape
    yield base.BenchmarkCasePlan(
        shape={
            "input": (batch, input_size),
            "hx": (batch, hidden_size),
            "w_ih": (3 * hidden_size, input_size),
            "w_hh": (3 * hidden_size, hidden_size),
        },
        params={"bias": True},
        builder_args=(batch, input_size, hidden_size),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, input_size, hidden_size = plan.builder_args
    inp = utils.generate_tensor_input((batch, input_size), dtype, device)
    hx = utils.generate_tensor_input((batch, hidden_size), dtype, device)
    w_ih = utils.generate_tensor_input((3 * hidden_size, input_size), dtype, device)
    w_hh = utils.generate_tensor_input((3 * hidden_size, hidden_size), dtype, device)
    b_ih = utils.generate_tensor_input((3 * hidden_size,), dtype, device)
    b_hh = utils.generate_tensor_input((3 * hidden_size,), dtype, device)
    return inp, hx, w_ih, w_hh, b_ih, b_hh, {}


class GruCellBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=GRU_CELL_SHAPES)


@pytest.mark.gru_cell
def test_gru_cell():
    bench = GruCellBenchmark(
        op_name="gru_cell",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.gru_cell,
        gems_op=getattr(flag_gems, "gru_cell", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

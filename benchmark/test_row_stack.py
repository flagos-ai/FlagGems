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

import math
from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils

# row_stack creates 3 inputs + 1 output. Cap total elements to avoid OOM /
# invalid-argument errors on the largest generated shapes.
MAX_ELEMENTS = 2**29


class RowStackBenchmark(base.Benchmark):
    # row_stack creates 3 inputs + 1 output. Cap total elements to avoid OOM /
    # invalid-argument errors on the largest generated shapes.
    MAX_ELEMENTS = 2**29

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def init_user_config(self):
        super().init_user_config()
        # Filter out shapes whose total element count exceeds MAX_ELEMENTS;
        # row_stack materializes 3 inputs plus the stacked output in memory.
        self.shapes = [s for s in self.shapes if math.prod(s) <= self.MAX_ELEMENTS]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def set_more_shapes(self):
        # Extra 2D / 3D shapes spanning a range of trailing-axis widths so the
        # benchmark covers both shallow-wide and deeper stacks.
        more_shapes_2d = [(1024, 2**i) for i in range(1, 11, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]
        return more_shapes_2d + more_shapes_3d


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)

    yield [inp1, inp2, inp3],


@pytest.mark.row_stack
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_row_stack():
    bench = RowStackBenchmark(
        op_name="row_stack",
        input_fn=_input_fn,
        torch_op=torch.row_stack,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.row_stack_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_row_stack_out():
    def _out_input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        inp3 = utils.generate_tensor_input(shape, dtype, device)
        # Pre-allocate the out tensor with the vertically-stacked shape.
        out = torch.empty(
            (inp1.shape[0] + inp2.shape[0] + inp3.shape[0],) + inp1.shape[1:],
            dtype=dtype,
            device=device,
        )
        yield [inp1, inp2, inp3], {"out": out}

    bench = RowStackBenchmark(
        op_name="row_stack_out",
        input_fn=_out_input_fn,
        torch_op=torch.row_stack,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

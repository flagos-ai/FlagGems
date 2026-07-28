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


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


class CatBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        self.input_fn = kwargs.pop("input_fn", _input_fn)
        super().__init__(*args, **kwargs)

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def set_more_shapes(self):
        more_shapes_2d = [(1024, 2**i) for i in range(1, 11, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]

        return more_shapes_2d + more_shapes_3d


@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.cat
def test_cat():
    bench = CatBenchmark(
        op_name="cat",
        input_fn=_input_fn,
        torch_op=torch.cat,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


class CatOutBenchmark(CatBenchmark):
    """Limit shapes to avoid OOM: cat_out needs 3 inputs + 1 output (3x size)."""

    MAX_ELEMENTS = 2**24

    def set_more_shapes(self):
        shapes = super().set_more_shapes()
        return [s for s in shapes if math.prod(s) <= self.MAX_ELEMENTS]

    def init_user_config(self):
        super().init_user_config()
        self.shapes = [s for s in self.shapes if math.prod(s) <= self.MAX_ELEMENTS]


def _cat_out_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    out_shape = list(shape)
    out_shape[0] = out_shape[0] * 3
    out = torch.empty(out_shape, dtype=dtype, device=device)
    yield [inp1, inp2, inp3], {"dim": 0, "out": out},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        out_shape_last = list(shape)
        out_shape_last[-1] = out_shape_last[-1] * 3
        out_last = torch.empty(out_shape_last, dtype=dtype, device=device)
        yield [inp1, inp2, inp3], {"dim": -1, "out": out_last},


@pytest.mark.cat_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_cat_out():
    bench = CatOutBenchmark(
        op_name="cat_out",
        input_fn=_cat_out_input_fn,
        torch_op=torch.cat,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()

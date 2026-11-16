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


def aminmax_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    # Test dim=None (whole tensor reduction)
    yield inp,
    # Test dim=-1 (last dimension)
    yield inp, {"dim": -1}
    # Test dim=0 (first dimension)
    if len(shape) > 1:
        yield inp, {"dim": 0}


class AminmaxBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from aminmax_input_fn(shape, dtype, self.device)


@pytest.mark.aminmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_aminmax():
    bench = AminmaxBenchmark(
        op_name="aminmax",
        torch_op=torch.aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


# ---------------------------------------------------------------------------
# aten::_aminmax / aten::_aminmax.out
#
# Operator ids stay `_aminmax` / `_aminmax_out`; the pytest markers are spelled
# `underscore_aminmax` / `underscore_aminmax_out` because pytest will not build a
# marker from an attribute beginning with an underscore.
# ---------------------------------------------------------------------------


def _aminmax_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    # ``_aminmax`` reduces the whole tensor; there is no dim argument.
    yield inp,


def _aminmax_out_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    min_out = torch.empty((), dtype=cur_dtype, device=device)
    max_out = torch.empty((), dtype=cur_dtype, device=device)
    # The aten ``.out`` overload takes keyword-only ``out0``/``out1``.
    yield inp, {"out0": min_out, "out1": max_out}


class AminmaxAllReduceBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _aminmax_input_fn(shape, cur_dtype, self.device)


class AminmaxOutBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _aminmax_out_input_fn(shape, cur_dtype, self.device)


@pytest.mark.underscore_aminmax
def test__aminmax():
    bench = AminmaxAllReduceBenchmark(
        op_name="_aminmax",
        torch_op=torch._aminmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.underscore_aminmax_out
def test__aminmax_out():
    bench = AminmaxOutBenchmark(
        op_name="_aminmax_out",
        torch_op=torch.ops.aten._aminmax.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

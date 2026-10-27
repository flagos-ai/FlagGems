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


def _take_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    index = torch.randint(0, inp.numel(), (inp.numel(),), device=device)
    yield inp, index


def _take_out_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    index = torch.randint(0, inp.numel(), (inp.numel(),), device=device)
    out = torch.empty(index.shape, dtype=cur_dtype, device=device)
    yield inp, index, {"out": out}


@pytest.mark.take
def test_take():
    bench = base.GenericBenchmark(
        op_name="take",
        input_fn=_take_input_fn,
        torch_op=torch.take,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.take_out
def test_take_out():
    bench = base.GenericBenchmark(
        op_name="take_out",
        input_fn=_take_out_input_fn,
        torch_op=torch.take,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0

    yield condition, inp1, inp2


@pytest.mark.where_self
def test_where_self():
    bench = base.GenericBenchmark(
        op_name="where_self",
        input_fn=_input_fn,
        torch_op=torch.where,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _input_fn_out(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    condition = inp1 > 0
    out = torch.empty(shape, dtype=cur_dtype, device=device)

    yield condition, inp1, inp2, {"out": out}


@pytest.mark.where_self_out
def test_where_self_out():
    bench = base.GenericBenchmark(
        op_name="where_self_out",
        input_fn=_input_fn_out,
        torch_op=torch.where,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

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


def _div_mode_input_fn_factory(rounding_mode):
    def inner(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        # Keep the divisor away from zero without changing the workload shape.
        inp2 = torch.where(inp2 >= 0, inp2 + 0.1, inp2 - 0.1)
        yield inp1, inp2, {"rounding_mode": rounding_mode}

    return inner


def _div_mode_dtypes(rounding_mode):
    # DIV_MODE_FLOAT_CASES in tests/test_div.py only covers trunc on float32.
    return [torch.float32] if rounding_mode == "trunc" else consts.FLOAT_DTYPES


@pytest.mark.div_mode
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_div_mode(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="div_mode",
        torch_op=torch.div,
        input_fn=_div_mode_input_fn_factory(rounding_mode),
        dtypes=_div_mode_dtypes(rounding_mode),
    )
    bench.set_gems(flag_gems.div_mode)
    bench.run()


@pytest.mark.div_mode_
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_div_mode_inplace(rounding_mode):
    bench = base.GenericBenchmark(
        op_name="div_mode_",
        torch_op=lambda a, b, rounding_mode: a.div_(b, rounding_mode=rounding_mode),
        input_fn=_div_mode_input_fn_factory(rounding_mode),
        dtypes=_div_mode_dtypes(rounding_mode),
        is_inplace=True,
    )
    bench.set_gems(flag_gems.div_mode_)
    bench.run()

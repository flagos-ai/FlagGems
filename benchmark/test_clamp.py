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


def _input_fn(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp3 = utils.generate_tensor_input(shape, cur_dtype, device)

    yield inp1, inp2, inp3

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # scalar or None situation
        yield inp1, inp2, None
        yield inp1, None, 3.14


@pytest.mark.clamp
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_clamp():
    bench = base.GenericBenchmark(
        op_name="clamp",
        input_fn=_input_fn,
        torch_op=torch.clamp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_clamp_inplace():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clamp_",
        torch_op=torch.clamp_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _clamp_tensor_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    mini = utils.generate_tensor_input(shape, cur_dtype, device)
    maxi = utils.generate_tensor_input(shape, cur_dtype, device)
    yield inp, {"min": mini, "max": maxi}

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, {"min": mini, "max": None}
        yield inp, {"min": None, "max": maxi}


@pytest.mark.clamp_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_clamp_tensor():
    bench = base.GenericBenchmark(
        op_name="clamp_tensor",
        input_fn=_clamp_tensor_input_fn,
        torch_op=torch.clamp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clamp_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_clamp_tensor_inplace():
    bench = base.GenericBenchmark(
        op_name="clamp_tensor_",
        input_fn=_clamp_tensor_input_fn,
        torch_op=torch.clamp_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()

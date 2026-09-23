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


def _tensor_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    yield inp1, inp2


def _scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.fmod_tensor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_tensor():
    bench = base.GenericBenchmark(
        input_fn=_tensor_input_fn,
        op_name="fmod_tensor",
        torch_op=torch.fmod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod_scalar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="fmod_scalar",
        torch_op=torch.fmod,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.fmod_tensor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_tensor_():
    bench = base.GenericBenchmark(
        input_fn=_tensor_input_fn,
        op_name="fmod_tensor_",
        torch_op=torch.Tensor.fmod_,
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()


@pytest.mark.fmod_scalar_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_scalar_():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="fmod_scalar_",
        torch_op=torch.Tensor.fmod_,
        dtypes=consts.FLOAT_DTYPES,
        inplace=True,
    )
    bench.run()

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


@pytest.mark.copysign
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copysign():
    bench = base.BinaryPointwiseBenchmark(
        op_name="copysign",
        torch_op=torch.copysign,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.copysign_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copysign_out():
    def input_fn(shape, dtype, device):
        inp1 = utils.generate_tensor_input(shape, dtype, device)
        inp2 = utils.generate_tensor_input(shape, dtype, device)
        out = torch.empty(shape, dtype=dtype, device=device)
        yield inp1, inp2, {"out": out}

    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="copysign_out",
        torch_op=torch.copysign,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

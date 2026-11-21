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


@pytest.mark.amax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_amax():
    bench = base.UnaryReductionBenchmark(
        op_name="amax", torch_op=torch.amax, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def amax_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = 1 if len(shape) > 1 else 0
    yield inp, dim


@pytest.mark.amax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_amax_dim():
    bench = base.GenericBenchmark(
        op_name="amax_dim",
        torch_op=torch.amax,
        input_fn=amax_dim_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

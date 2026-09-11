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


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype=dtype, device=device)
    yield inp,


@pytest.mark.trace
def test_trace():
    if flag_gems.vendor_name == "mthreads":
        dtypes = consts.FLOAT_DTYPES
    else:
        dtypes = consts.FLOAT_DTYPES + consts.INT_DTYPES

    bench = base.GenericBenchmark2DOnly(
        op_name="trace",
        input_fn=_input_fn,
        torch_op=torch.trace,
        dtypes=dtypes,
    )

    bench.run()

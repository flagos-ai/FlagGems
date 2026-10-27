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


def _input_fn_scalar(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    inp2 = 0
    yield inp1, inp2


@pytest.mark.lt
def test_lt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="lt",
        torch_op=torch.lt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.lt_scalar
def test_lt_scalar():
    bench = base.GenericBenchmark(
        op_name="lt_scalar",
        input_fn=_input_fn_scalar,
        torch_op=torch.lt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

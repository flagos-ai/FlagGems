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


def _tensor_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, inp2


def _scalar_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    yield inp1, 0.5


@pytest.mark.rsub_tensor
def test_rsub_tensor():
    bench = base.GenericBenchmark(
        input_fn=_tensor_input_fn,
        op_name="rsub_tensor",
        torch_op=torch.rsub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.rsub_scalar
def test_rsub_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="rsub_scalar",
        torch_op=torch.rsub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

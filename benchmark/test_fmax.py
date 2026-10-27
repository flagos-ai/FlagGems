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


@pytest.mark.fmax
def test_fmax():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmax",
        torch_op=torch.fmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def fmax_out_input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.fmax_out
def test_fmax_out():
    bench = base.GenericBenchmark(
        op_name="fmax_out",
        torch_op=torch.fmax,
        input_fn=fmax_out_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

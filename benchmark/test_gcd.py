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

from . import base, consts


@pytest.mark.gcd
def test_gcd():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gcd",
        torch_op=torch.gcd,
        dtypes=consts.INT_DTYPES,
    )
    bench.run()


def gcd_out_input_fn(shape, dtype, device):
    inp1 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device=device,
    )
    inp2 = torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device=device,
    )
    out = torch.empty(shape, dtype=dtype, device=device)
    yield inp1, inp2, {"out": out}


@pytest.mark.gcd_out
def test_gcd_out():
    bench = base.GenericBenchmark(
        op_name="gcd_out",
        torch_op=torch.gcd,
        dtypes=consts.INT_DTYPES,
        input_fn=gcd_out_input_fn,
    )
    bench.run()

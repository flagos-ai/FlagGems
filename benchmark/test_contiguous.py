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


def _input_fn(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=device)
    else:
        inp = torch.randint(
            low=-10000, high=10000, size=shape, dtype=dtype, device="cpu"
        ).to(device)
    inp = inp[::2]

    yield inp,


@pytest.mark.contiguous
def test_contiguous():
    bench = base.GenericBenchmark(
        op_name="contiguous",
        input_fn=_input_fn,
        torch_op=torch.Tensor.contiguous,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )

    bench.run()

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

from . import base, consts


def torch_normed_cumsum(inp, dim=-1):
    return torch.cumsum(inp, dim=dim) / inp.sum(dim=dim, keepdim=True)


def input_fn(shape, dtype, device):
    inp = torch.rand(shape, dtype=dtype, device=device) + 0.1
    dim = -1
    yield inp, dim


@pytest.mark.normed_cumsum
def test_normed_cumsum():
    bench = base.GenericBenchmark(
        input_fn=input_fn,
        op_name="normed_cumsum",
        torch_op=torch_normed_cumsum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.normed_cumsum)
    bench.run()

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

from . import base


def _input_fn(shape, dtype, device):
    x = torch.randn(size=shape, dtype=dtype, device=device)
    if flag_gems.vendor_name == "mthreads":
        yield x.conj(),
    else:
        yield x.conj().imag,


@pytest.mark.resolve_neg
def test_resolve_neg():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="resolve_neg",
        input_fn=_input_fn,
        dtypes=[torch.cfloat],
        torch_op=torch.resolve_neg,
    )
    bench.run()

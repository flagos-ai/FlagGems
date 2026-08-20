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

import math

import pytest
import torch

import flag_gems

from . import base

# _list_to_tensor builds a tensor from a *Python* list, so extremely large
# lengths are dominated by Python-side list construction rather than the kernel.
# Cap the list length to keep the benchmark meaningful and fast.
_MAX_LEN = 1 << 20


def _input_fn(shape, dtype, device):
    # aten::_list_to_tensor takes a Python list of ints; the shape drives the
    # length of the list. dtype/device are ignored by the op semantics.
    n = min(math.prod(shape), _MAX_LEN)
    yield ([i % 128 for i in range(n)],)


@pytest.mark.list_to_tensor
def test__list_to_tensor():
    # NOTE: aten::_list_to_tensor is a JIT-only prim op and cannot be
    # intercepted by flag_gems.use_gems(); we bench the GEMS implementation
    # directly via the gems_op hook.
    bench = base.GenericBenchmark(
        op_name="_list_to_tensor",
        input_fn=_input_fn,
        torch_op=torch.ops.aten._list_to_tensor,
        gems_op=flag_gems._list_to_tensor,
        dtypes=[torch.int32],
    )
    bench.run()

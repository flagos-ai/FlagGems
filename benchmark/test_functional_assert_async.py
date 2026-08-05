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


def functional_assert_async_input_fn(shape, dtype, device):
    # Always use single-element tensor (requirement of the op)
    inp = torch.ones(1, dtype=dtype, device=device)
    dep_token = torch.empty(0, dtype=dtype, device=device)
    yield inp, "assertion", dep_token


def functional_assert_async_torch_wrapper(inp, msg, dep_token):
    return torch.ops.aten._functional_assert_async.msg(inp, msg, dep_token)


@pytest.mark.functional_assert_async
def test_functional_assert_async():
    bench = base.GenericBenchmark(
        op_name="functional_assert_async",
        input_fn=functional_assert_async_input_fn,
        torch_op=functional_assert_async_torch_wrapper,
        dtypes=consts.INT_DTYPES + consts.FLOAT_DTYPES,
    )
    bench.run()

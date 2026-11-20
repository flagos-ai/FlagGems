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


def _reference():
    return torch.ops.aten.index.Tensor


def _inputs(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    count = 1 if shape[-1] == 6144 else 1024
    idx = torch.randint(-shape[0], shape[0], (count,), device=device, dtype=torch.int64)
    return (x, [idx]), {}


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.index
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="Atlas A5 specialization",
)
def test_index_perf():
    bench = A5Benchmark(
        op_name="a5_index",
        input_fn=_input_fn,
        torch_op=_reference(),
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.index)
    bench.run()

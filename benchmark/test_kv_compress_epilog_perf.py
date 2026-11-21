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
    import cann_ops_transformer  # noqa: F401

    return torch.ops.cann_ops_transformer.kv_compress_epilog.default


def _inputs(shape, dtype, device):
    t, width = shape
    cache = torch.full((16, 32, 1, 608), 17, device=device, dtype=torch.uint8)
    x = torch.randn(shape, device=device, dtype=dtype)
    slots = torch.arange(t, device=device, dtype=torch.int32) * 3 + 1
    return (cache, x, slots), dict(
        quant_group_size=64, quant_mode="fp8_bf16", round_scale=True, x_scale=1.0
    )


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.kv_compress_epilog
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="Atlas A5 specialization",
)
def test_kv_compress_epilog_perf():
    bench = A5Benchmark(
        op_name="kv_compress_epilog",
        input_fn=_input_fn,
        torch_op=_reference(),
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.kv_compress_epilog)
    bench.run()

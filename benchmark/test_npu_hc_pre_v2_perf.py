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
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    pytest.importorskip("vllm_ascend.vllm_ascend_C")
    return torch.ops._C_ascend.npu_hc_pre_v2.default


def _inputs(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    return (
        x,
        torch.randn((24, 16384), device=device) * 0.01,
        torch.randn(3, device=device) * 0.1,
        torch.randn(24, device=device) * 0.1,
        4,
        20,
        1e-6,
        1e-6,
    ), {}


def _input_fn(shape, dtype, device):
    args, kwargs = _inputs(shape, dtype, device)
    yield (*args, kwargs)


class A5Benchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.npu_hc_pre_v2
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="Atlas A5 specialization",
)
def test_npu_hc_pre_v2_perf():
    bench = A5Benchmark(
        op_name="npu_hc_pre_v2",
        input_fn=_input_fn,
        torch_op=_reference(),
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.npu_hc_pre_v2)
    bench.run()

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


def _inputs(shape, dtype, device):
    x1 = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.randn(shape[-1], dtype=dtype, device=device)
    yield x1, torch.randn_like(x1), gamma, torch.randn_like(gamma)


class AddRmsNormBiasBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.add_rms_norm_bias
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="Atlas A5 specialization",
)
def test_add_rms_norm_bias_perf():
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    pytest.importorskip("vllm_ascend.vllm_ascend_C")
    native = torch.ops._C_ascend.npu_add_rms_norm_bias.default
    bench = AddRmsNormBiasBenchmark(
        op_name="add_rms_norm_bias",
        input_fn=_inputs,
        torch_op=native,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.add_rms_norm_bias)
    bench.run()

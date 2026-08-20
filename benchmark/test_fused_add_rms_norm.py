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


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    residual = torch.randn(shape, dtype=dtype, device=device)
    layer_shape = (shape[-1],)
    weight = torch.randn(layer_shape, dtype=dtype, device=device)
    yield inp, residual, layer_shape, weight, 1e-5


def torch_op(x, residual, layer_shape, weight, eps):
    x = x + residual
    variance = x.pow(2).mean(-1, keepdim=True)
    hidden_states = x * torch.rsqrt(variance + eps)
    return weight * hidden_states


@pytest.mark.fused_add_rms_norm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fused_add_rms_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_input_fn,
        op_name="fused_add_rms_norm",
        torch_op=torch_op,
        gems_op=flag_gems.fused_add_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()

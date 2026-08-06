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


@pytest.mark.fused_rms_norm
def test_fused_rms_norm():
    """Benchmark for _fused_rms_norm which returns (output, inv_rms)."""

    def _torch_fused_rms_norm(x, normalized_shape, weight=None, eps=1e-5):
        """Reference implementation for comparison."""
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + eps)
        hidden_states = upcast_x * inv_rms
        hidden_states = hidden_states.to(x.dtype)
        if weight is not None:
            return weight * hidden_states, inv_rms.squeeze(-1)
        return hidden_states, inv_rms.squeeze(-1)

    def fused_rms_norm_input_fn(shape, dtype, device):
        M, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = base.GenericBenchmark2DOnly(
        input_fn=fused_rms_norm_input_fn,
        op_name="fused_rms_norm",
        torch_op=_torch_fused_rms_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._fused_rms_norm)
    bench.run()

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
import torch.nn.functional as F

import flag_gems
from flag_gems.runtime.backend._iluvatar.fused.mm_silu_mul import (
    _TILE_PACKED,
    _tile_packed_weights,
    mm_silu_mul,
)


@pytest.mark.skipif(flag_gems.vendor_name != "iluvatar", reason="Iluvatar only")
@pytest.mark.parametrize("M", [64, 2111])
def test_mm_silu_mul_matches_unfused_reference(M):
    K, N = 2048, 6144
    torch.manual_seed(20261103 + M)
    a = torch.randn(M, K, dtype=torch.bfloat16, device=flag_gems.device) * (K ** -0.5)
    b = torch.randn(2 * N, K, dtype=torch.bfloat16, device=flag_gems.device).t() * (K ** -0.5)
    reference = F.silu(torch.mm(a, b[:, :N])) * torch.mm(a, b[:, N:])

    result = mm_silu_mul(a, b)

    assert result.shape == (M, N)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, rtol=0.05, atol=0.05)


def test_tile_packed_weights_are_cached_once():
    _TILE_PACKED.clear()
    b = torch.randn(32, 512, dtype=torch.bfloat16)

    gate_first, up_first = _tile_packed_weights(b)
    gate_second, up_second = _tile_packed_weights(b)

    assert gate_first is gate_second
    assert up_first is up_second
    assert gate_first.shape == (1, 2, 128, 32)
    assert torch.equal(gate_first[0, 0], b[:, :128].t())
    assert torch.equal(up_first[0, 0], b[:, 256:384].t())
    assert len(_TILE_PACKED) == 1

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

from flag_gems.fused.dsv41 import mhc as gems
from tests.dsv41_helpers import (
    assert_exact,
    changed_graph,
    mhc_inputs,
    native_mhc_mix_stats,
)

pytestmark = [
    pytest.mark.dsv41_mhc,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="SM90 operator",
    ),
]


@pytest.mark.parametrize(
    "m",
    [
        1,
        3,
        5,
        6,
        10,
        12,
        15,
        18,
        20,
        24,
        25,
        30,
        35,
        36,
        40,
        42,
        48,
        50,
        60,
        70,
        72,
        80,
        84,
        96,
        100,
        120,
    ],
)
def test_stats_and_predecessor_combine(m):
    x, fn, _, _ = mhc_inputs(m)
    expected = (x.float() @ fn.T) * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    actual = gems.hc_mix_stats(x, fn, 1e-6)
    assert_exact(actual, native_mhc_mix_stats(x, fn, 1e-6))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-5)
    pre = torch.rand((m, 4), device="cuda")
    expected = (x.view(m, 4, -1).float() * pre[..., None]).sum(1).bfloat16()
    torch.testing.assert_close(
        gems.hc_combine(x, pre, 4, torch.bfloat16),
        expected,
        rtol=0.008,
        atol=0.016,
    )


@pytest.mark.parametrize("m", [36, 40, 42, 48, 50, 60, 80, 84, 96])
def test_exact_fused_sinkhorn_and_changed_graph(m):
    args = mhc_inputs(m)
    fn = lambda x, w, s, b: gems.hc_mix_stats_sinkhorn_exact(
        x, w, s, b, 4, 20, 1e-6, 1e-6
    )
    baseline = lambda x, w, s, b: gems.hc_split_sinkhorn(
        gems.hc_mix_stats(x, w, 1e-6), s, b
    )
    assert_exact(fn(*args), baseline(*args))

    def refresh(xs, seed):
        for x, y in zip(xs, mhc_inputs(m, seed=101 + seed)):
            x.copy_(y)

    changed_graph(fn, args, refresh, baseline)
    for _ in range(10):
        assert_exact(fn(*args), baseline(*args))

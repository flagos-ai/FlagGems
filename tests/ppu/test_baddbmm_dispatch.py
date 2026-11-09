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

import importlib
from contextlib import nullcontext

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead",
    reason="requires the T-Head BADDBMM implementation",
)


@pytest.mark.baddbmm
@pytest.mark.parametrize("use_aten", [False, True])
@pytest.mark.parametrize(
    "grad_enabled,requires_grad",
    [
        (True, (False, False, False)),
        (True, (True, False, False)),
        (True, (False, True, False)),
        (True, (False, False, True)),
        (False, (True, True, True)),
    ],
)
def test_baddbmm_dispatch_preserves_autograd(
    monkeypatch, use_aten, grad_enabled, requires_grad
):
    baddbmm = importlib.import_module(flag_gems.baddbmm.__module__)
    monkeypatch.setattr(baddbmm, "_can_use_ppu_baddbmm", lambda *args: True)
    launches = []

    def dispatch(a, b, out, *, bias, alpha, beta):
        launches.append((alpha, beta))
        return out

    monkeypatch.setattr(baddbmm, "_dispatch_ppu_bmm", dispatch)
    bias, a, b = (
        torch.empty((2, 4, 4), device=flag_gems.device, requires_grad=grad)
        for grad in requires_grad
    )
    op = torch.baddbmm if use_aten else flag_gems.baddbmm
    context = flag_gems.use_gems(include=["baddbmm"]) if use_aten else nullcontext()
    with context, torch.set_grad_enabled(grad_enabled):
        result = op(bias, a, b, beta=0.5, alpha=0.25)
    assert launches == [(0.25, 0.5)]
    assert result.requires_grad == (grad_enabled and any(requires_grad))
    if result.requires_grad:
        expected = "BaddbmmBackward0" if use_aten else "_PPUBaddbmmFunctionBackward"
        assert type(result.grad_fn).__name__ == expected

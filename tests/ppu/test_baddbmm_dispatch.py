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

import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead",
    reason="requires the T-Head BADDBMM implementation",
)


@pytest.mark.baddbmm
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
def test_baddbmm_dispatch_preserves_autograd(monkeypatch, grad_enabled, requires_grad):
    baddbmm = importlib.import_module(flag_gems.baddbmm.__module__)
    monkeypatch.setattr(baddbmm, "_can_use_ppu_baddbmm", lambda *args: True)
    generic_result = object()
    routed_result = object()
    monkeypatch.setattr(
        baddbmm, "_generic_baddbmm", lambda *args, **kwargs: generic_result
    )
    monkeypatch.setattr(
        baddbmm, "_dispatch_ppu_bmm", lambda *args, **kwargs: routed_result
    )
    bias, a, b = (torch.empty((2, 4, 4), requires_grad=grad) for grad in requires_grad)
    with torch.set_grad_enabled(grad_enabled):
        result = baddbmm.baddbmm(bias, a, b)
    assert result is (
        generic_result if grad_enabled and any(requires_grad) else routed_result
    )

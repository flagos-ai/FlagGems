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

from types import SimpleNamespace

import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.utils import triton_lang_helper


@triton.jit
def _fallback_hypot_kernel(x, y, out, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    x_values = tl.load(x + offsets, mask=mask)
    y_values = tl.load(y + offsets, mask=mask)
    result = triton_lang_helper._fallback_hypot(x_values, y_values)
    tl.store(out + offsets, result, mask=mask)


def test_patch_missing_nearbyint_uses_rint():
    rint = object()
    module = SimpleNamespace(rint=rint)

    patched = triton_lang_helper._patch_missing_symbols(module, ("nearbyint",))

    assert patched.nearbyint is rint


def test_patch_missing_hypot_uses_fallback():
    module = SimpleNamespace()

    patched = triton_lang_helper._patch_missing_symbols(module, ("hypot",))

    assert patched.hypot is triton_lang_helper._fallback_hypot


def test_fallback_hypot_special_values():
    x_cpu = torch.tensor(
        [3.0, 0.0, 2.0e38, float("inf"), float("nan"), float("nan")],
        dtype=torch.float32,
    )
    y_cpu = torch.tensor(
        [4.0, 0.0, 2.0e38, float("nan"), float("inf"), 1.0],
        dtype=torch.float32,
    )
    x = x_cpu.to(flag_gems.device)
    y = y_cpu.to(flag_gems.device)
    actual = torch.empty_like(x)

    _fallback_hypot_kernel[(1,)](
        x, y, actual, x.numel(), block=triton.next_power_of_2(x.numel())
    )

    torch.testing.assert_close(
        actual.cpu(), torch.hypot(x_cpu, y_cpu), rtol=2e-6, atol=0.0, equal_nan=True
    )

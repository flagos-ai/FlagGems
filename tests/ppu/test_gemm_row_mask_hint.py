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


"""A 32-row family cannot make larger autotune row tiles unmasked."""

import importlib
from contextlib import nullcontext

import pytest
import torch

import flag_gems

if flag_gems.vendor_name != "thead":
    pytest.skip("requires the T-Head PPU backend", allow_module_level=True)

mm = importlib.import_module(flag_gems.mm.__module__)


@pytest.mark.parametrize("m", [32, 64, 512])
@pytest.mark.parametrize("nt", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_main_row_hint_is_safe_for_every_candidate(monkeypatch, m, nt, dtype):
    seen = []

    class Capture:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: seen.append(kwargs)

    monkeypatch.setattr(mm, "mm_kernel_ppu", Capture())
    monkeypatch.setattr(mm, "_ppu_device_context", lambda *args: nullcontext())
    monkeypatch.setattr(mm, "_aiu_load_mask", lambda *args: 3)
    a = torch.empty((m, 4096), device="meta", dtype=torch.bfloat16)
    b = (
        torch.empty((512, 4096), device="meta", dtype=torch.bfloat16).t()
        if nt
        else torch.empty((4096, 512), device="meta", dtype=torch.bfloat16)
    )
    out = torch.empty((m, 512), device="meta", dtype=dtype)
    mm._run_ppu_mm(a, b, out)
    assert len(seen) == 1
    assert seen[0]["EVEN_M"] == (m % 512 == 0)


@pytest.mark.parametrize(
    "block,full", [(16, True), (32, True), (64, False), (128, False)]
)
def test_row_tile_divisibility_uses_the_selected_block(block, full):
    assert mm._full_m_tiles({"M": 32, "BLOCK_M": block}) is full

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
from types import SimpleNamespace

import pytest
import torch

import flag_gems

mm = importlib.import_module(flag_gems.mm.__module__)
pytestmark = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name != "thead"
    or not getattr(mm, "HAS_PPU_TLE", False),
    reason="requires the T-Head MM implementation",
)


@pytest.mark.parametrize(
    "kind", ["dense", "transpose", "sliced", "singleton", "empty", "offset"]
)
def test_dense_transpose_detection_matches_torch(kind):
    import flag_gems

    mm = importlib.import_module(flag_gems.mm.__module__)
    candidates = {
        "dense": torch.empty(3, 4),
        "transpose": torch.empty(4, 3).t(),
        "sliced": torch.empty(8, 6)[::2, ::2],
        "singleton": torch.empty(1, 4).t(),
        "empty": torch.empty(0, 4).t(),
        "offset": torch.empty(20)[2:14].view(4, 3).t(),
    }
    value = candidates[kind]
    expected = (
        value.ndim == 2 and not value.is_contiguous() and value.t().is_contiguous()
    )
    assert mm._is_transposed_contiguous_2d(value) == expected


def test_split_reducer_uses_this_launch_metadata_without_rebuilding_configs(
    monkeypatch,
):
    from contextlib import nullcontext

    import flag_gems

    mm = importlib.import_module(flag_gems.mm.__module__)
    selected = [2, 4, 2]
    seen = []

    class Main:
        best_config = SimpleNamespace(kwargs={"SPLIT_K": 8})

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                return self, {"SPLIT_K": selected.pop(0)}

            return launch

    class Reduce:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append(kwargs["SPLIT_K"])

            return launch

    def rebuild():
        raise AssertionError("candidate catalog rebuilt on hot path")

    monkeypatch.setattr(mm, "mm_split_k_kernel_ppu", Main())
    monkeypatch.setattr(mm, "mm_split_k_reduce_kernel_ppu", Reduce())
    monkeypatch.setattr(mm, "_ppu_split_k_configs", rebuild)
    monkeypatch.setattr(mm, "_ppu_device_context", lambda tensor: nullcontext())
    a = torch.empty(4, 2048, dtype=torch.bfloat16)
    b = torch.empty(2048, 64, dtype=torch.bfloat16)
    out = torch.empty(4, 64, dtype=torch.bfloat16)
    for _ in range(3):
        assert mm._run_split_k_mm(a, b, out) is out
    assert seen == [2, 4, 2]

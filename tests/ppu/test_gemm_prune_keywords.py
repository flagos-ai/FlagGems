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


"""PPU load legality must agree for positional and keyword launch arguments."""

import importlib

import pytest
import triton

import flag_gems

if flag_gems.vendor_name != "thead":
    pytest.skip("requires the T-Head PPU backend", allow_module_level=True)

gemm = importlib.import_module(
    flag_gems.mm.__module__.rsplit(".", 1)[0] + ".gemm_utils"
)


@pytest.mark.parametrize(
    "name",
    ["_prune_gemm_configs", "_prune_narrow_n_configs", "_prune_split_k_configs"],
)
@pytest.mark.parametrize("mask", [0, 1, 2, 3])
def test_keyword_aiu_mask_matches_combined_arguments(name, mask):
    configs = [
        triton.Config(
            {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "LOAD_MODE": mode,
                "SPLIT_K": 2,
            },
            num_warps=4,
            num_stages=3,
        )
        for mode in range(4)
    ]
    args = {"M": 2048, "N": 64, "K": 2048, "batch": 1}
    prune = getattr(gemm, name)
    positional = prune(configs, {**args, "aiu_load_mask": mask})
    keyword = prune(configs, args, aiu_load_mask=mask)
    expected = {0: {3}, 1: {1, 3}, 2: {2, 3}, 3: {0, 1, 2, 3}}[mask]
    assert {c.kwargs["LOAD_MODE"] for c in keyword} == expected
    assert [c.all_kwargs() for c in keyword] == [c.all_kwargs() for c in positional]

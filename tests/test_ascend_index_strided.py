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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend stride-aware index implementation required",
)


@pytest.mark.parametrize(
    "dtype", [torch.int64, torch.int32, torch.float32, torch.bfloat16]
)
@pytest.mark.parametrize(
    "case",
    ["strided_input", "strided_indices", "negative_multi", "scalar", "separated"],
)
def test_ascend_index_strided(dtype, case):
    assert flag_gems.index.__module__.endswith("_ascend.ops.index")
    inp = torch.arange(60, device=flag_gems.device).reshape(3, 4, 5).to(dtype)
    if case == "strided_input":
        inp = inp.transpose(0, 1)
        indices = [torch.tensor([0, 2, -1], device=inp.device)]
    elif case == "strided_indices":
        indices = [torch.tensor([0, 2, 1, 0, 2, 1], device=inp.device)[::2]]
    elif case == "negative_multi":
        indices = [
            torch.tensor([0, -1], device=inp.device),
            torch.tensor([-1, 1], device=inp.device),
        ]
    elif case == "scalar":
        indices = [None, torch.tensor(-1, device=inp.device)]
    else:
        indices = [
            torch.tensor([[0], [2]], device=inp.device),
            None,
            torch.tensor([[0, 2, 4]], device=inp.device),
        ]
    reference = torch.ops.aten.index.Tensor(inp, indices)
    with flag_gems.use_gems(include=["index"]):
        actual = torch.ops.aten.index.Tensor(inp, indices)
    assert actual.shape == reference.shape
    assert torch.equal(actual, reference)


@pytest.mark.parametrize(
    "case",
    ["model_sampler", "mixed_broadcast", "strided_storage", "one_dim", "zero_dim"],
)
def test_ascend_index_singleton(case):
    assert flag_gems.index.__module__.endswith("_ascend.ops.index")
    if case == "model_sampler":
        inp = torch.randn((2048, 2048), dtype=torch.bfloat16, device=flag_gems.device)
        indices = [torch.tensor([-1], device=inp.device)]
    elif case == "mixed_broadcast":
        inp = torch.arange(360, device=flag_gems.device).reshape(3, 4, 5, 6)
        indices = [
            torch.tensor([[[-1]]], device=inp.device),
            torch.tensor([[[0, 1, 3]], [[2, 1, 0]]], device=inp.device),
        ]
    elif case == "strided_storage":
        inp = torch.arange(60, device=flag_gems.device).reshape(3, 4, 5).transpose(1, 2)
        indices = [torch.tensor([99, -1, 88], device=inp.device)[1:2]]
    else:
        inp = torch.arange(7, device=flag_gems.device, dtype=torch.bfloat16)
        indices = [
            (
                torch.tensor([-1], device=inp.device)
                if case == "one_dim"
                else torch.tensor(2, device=inp.device)
            )
        ]
    reference = torch.ops.aten.index.Tensor(inp, indices)
    with flag_gems.use_gems(include=["index"]):
        actual = torch.ops.aten.index.Tensor(inp, indices)
    assert actual.shape == reference.shape
    assert torch.equal(actual, reference)

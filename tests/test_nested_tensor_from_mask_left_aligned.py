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

from . import accuracy_utils as utils

# Shapes are (N, L) for the 2D boolean padding mask. The data tensor `t` has
# shape (N, L, D) with a fixed hidden dim D.
SHAPES = [
    (1, 1),
    (1, 32),
    (3, 5),
    (16, 64),
    (64, 128),
    (2, 3),
    (300, 1000),
]


def _make_mask(N, L, kind, device):
    if kind == "left_aligned":
        lengths = torch.randint(0, L + 1, (N,), device="cpu")
        return torch.arange(L, device=device)[None, :] < lengths[:, None].to(device)
    if kind == "all_true":
        return torch.ones(N, L, dtype=torch.bool, device=device)
    if kind == "all_false":
        return torch.zeros(N, L, dtype=torch.bool, device=device)
    if kind == "gap":
        # A False followed by a True: not left-aligned.
        if L < 2:
            return torch.zeros(N, L, dtype=torch.bool, device=device)
        mask = torch.zeros(N, L, dtype=torch.bool, device=device)
        mask[:, 0] = False
        mask[:, L - 1] = True
        return mask
    # random
    return torch.randint(0, 2, (N, L), dtype=torch.bool, device=device)


@pytest.mark.nested_tensor_from_mask_left_aligned
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "kind", ["left_aligned", "all_true", "all_false", "gap", "random"]
)
def test_nested_tensor_from_mask_left_aligned(shape, dtype, kind):
    N, L = shape
    D = 8
    t = torch.randn(N, L, D, dtype=dtype, device=flag_gems.device)
    mask = _make_mask(N, L, kind, flag_gems.device)

    ref_out = torch.ops.aten._nested_tensor_from_mask_left_aligned(
        utils.to_reference(t), utils.to_reference(mask)
    )
    res_out = flag_gems._nested_tensor_from_mask_left_aligned(t, mask)

    utils.gems_assert_equal(torch.tensor(res_out), torch.tensor(ref_out))


@pytest.mark.nested_tensor_from_mask_left_aligned
@pytest.mark.parametrize("shape", [(0, 3), (4, 1)])
def test_nested_tensor_from_mask_left_aligned_edge(shape):
    # Empty batch (N == 0) returns True (vacuous truth); L == 1 always aligned.
    # fp32 suffices for these structural edge cases.
    N, L = shape
    D = 8
    t = torch.randn(N, L, D, dtype=torch.float32, device=flag_gems.device)
    mask = torch.zeros(N, L, dtype=torch.bool, device=flag_gems.device)

    ref_out = torch.ops.aten._nested_tensor_from_mask_left_aligned(
        utils.to_reference(t), utils.to_reference(mask)
    )
    res_out = flag_gems._nested_tensor_from_mask_left_aligned(t, mask)

    utils.gems_assert_equal(torch.tensor(res_out), torch.tensor(ref_out))

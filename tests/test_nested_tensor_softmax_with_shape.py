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

# (B, H, T_max, seq_lens)
# seq_lens[b] <= T_max encodes the per-batch sequence length.
CONFIGS = [
    (2, 2, 8, [3, 8]),
    (3, 1, 13, [1, 7, 13]),
    (4, 2, 24, [4, 12, 20, 24]),
]


def _reference(attn, seq_lens):
    """Manual reference: masked softmax over the last dim using seq_lens."""
    B, T_max, H, _ = attn.shape
    out = torch.zeros_like(attn)
    for b in range(B):
        T = int(seq_lens[b])
        if T <= 0:
            continue
        out[b, :T, :, :T] = torch.softmax(attn[b, :T, :, :T], dim=-1)
    return out


@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B, H, T_max, seq_lens", CONFIGS)
@pytest.mark.nested_tensor_softmax_with_shape
def test_nested_tensor_softmax_with_shape(dtype, B, H, T_max, seq_lens):
    # self: nested tensor [B, T_i, H, T_max] holding per-head attention scores.
    self_components = [
        torch.randn(seq_lens[b], H, T_max, dtype=dtype, device=flag_gems.device)
        for b in range(B)
    ]
    self_nt = torch.nested.nested_tensor(
        self_components, layout=torch.jagged, device=flag_gems.device
    )

    # query: nested tensor [B, T_i, E] whose ragged dim encodes the seq lengths.
    E = 4
    query_components = [
        torch.randn(seq_lens[b], E, dtype=dtype, device=flag_gems.device)
        for b in range(B)
    ]
    query_nt = torch.nested.nested_tensor(
        query_components, layout=torch.jagged, device=flag_gems.device
    )

    ref_padded = utils.to_reference(self_nt.to_padded_tensor(0.0))
    ref_out = _reference(ref_padded, seq_lens)
    res_out = flag_gems._nested_tensor_softmax_with_shape(self_nt, query_nt)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Full-length case has no masking to exercise; it validates the unmasked path
# across the same dtype set as the masked case.
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.nested_tensor_softmax_with_shape
def test_nested_tensor_softmax_with_shape_full_length(dtype):
    # All sequences have length T_max: no masking should occur.
    B, H, T_max = 2, 2, 16
    seq_lens = [T_max, T_max]

    self_components = [
        torch.randn(seq_lens[b], H, T_max, dtype=dtype, device=flag_gems.device)
        for b in range(B)
    ]
    self_nt = torch.nested.nested_tensor(
        self_components, layout=torch.jagged, device=flag_gems.device
    )
    query_components = [
        torch.randn(seq_lens[b], 4, dtype=dtype, device=flag_gems.device)
        for b in range(B)
    ]
    query_nt = torch.nested.nested_tensor(
        query_components, layout=torch.jagged, device=flag_gems.device
    )

    ref_padded = utils.to_reference(self_nt.to_padded_tensor(0.0))
    ref_out = _reference(ref_padded, seq_lens)
    res_out = flag_gems._nested_tensor_softmax_with_shape(self_nt, query_nt)

    utils.gems_assert_close(res_out, ref_out, dtype)

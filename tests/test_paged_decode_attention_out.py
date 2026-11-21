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

pytestmark = [
    pytest.mark.paged_decode_attention_out,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return lambda q, k, v, mask, table, lengths, workspace, out: torch.ops.npu.npu_fused_infer_attention_score.out(
        query=q,
        key=k,
        value=v,
        atten_mask=mask,
        block_table=table,
        input_layout="TND",
        block_size=128,
        actual_seq_lengths=[1],
        actual_seq_lengths_kv=lengths,
        num_key_value_heads=1,
        num_heads=8,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        scale=0.0625,
        sparse_mode=3,
        softmax_lse_flag=False,
        workspace=workspace,
        out=out,
    )


def _inputs(shape, dtype, device):
    (length,) = shape
    pages = (length + 127) // 128
    query = torch.randn((1, 8, 256), device=device, dtype=dtype) * 0.2
    cache = torch.randn((pages, 2, 128, 256), device=device, dtype=dtype) * 0.2
    mask = torch.ones(2048, 2048, device=device, dtype=torch.int8).triu(1)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None, :]
    workspace = torch.empty(41996288, device=device, dtype=dtype)
    return (
        query,
        cache[:, 0],
        cache[:, 1],
        mask,
        table,
        [length],
        workspace,
        [torch.zeros_like(query), torch.zeros(1, dtype=dtype, device=device)],
    ), {}


@pytest.mark.parametrize("shape", [(1025,), (1280,), (2047,)])
def test_paged_decode_attention_out_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    native_out = [torch.zeros_like(t) for t in args[-1]]
    expected = reference(*args[:-1], native_out, **kwargs)
    actual = flag_gems.paged_decode_attention_out(*args, **kwargs)
    assert actual[0] is args[-1][0] and actual[1] is args[-1][1]
    torch.testing.assert_close(actual[0], expected[0], atol=1e-4, rtol=0.016)


def test_paged_decode_attention_workspace_bound():
    args, kwargs = _inputs((1025,), torch.bfloat16, flag_gems.device)
    with pytest.raises(ValueError, match="workspace"):
        flag_gems.paged_decode_attention_out(
            *args[:-2], args[-2][:2], args[-1], **kwargs
        )


def test_paged_decode_attention_rejects_alias():
    args, kwargs = _inputs((1025,), torch.bfloat16, flag_gems.device)
    with pytest.raises(ValueError, match="independent"):
        flag_gems.paged_decode_attention_out(
            *args[:-1], [args[0], args[-1][1]], **kwargs
        )


def test_paged_decode_attention_mask_row():
    args, kwargs = _inputs((128,), torch.bfloat16, flag_gems.device)
    q, k, v, mask, table, lengths, workspace, out = args
    # A single visible lane has unit probability for every head. This also
    # checks the mapping from logical position to a physical page.
    q.zero_()
    mask[lengths[0] - 1, : lengths[0]] = 1
    mask[lengths[0] - 1, 7] = 0
    got, _ = flag_gems.paged_decode_attention_out(*args, **kwargs)
    torch.testing.assert_close(
        got, v[0, 7].view(1, 1, -1).expand_as(got), rtol=0, atol=0
    )

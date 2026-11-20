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
    pytest.mark.npu_sparse_attention_score,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    pytest.importorskip("vllm_ascend")
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    pytest.importorskip("vllm_ascend.vllm_ascend_C")
    return torch.ops._C_ascend.npu_sparse_attention_score.default


def _inputs(shape, dtype, device):
    tokens, heads, dim = shape
    pages = 64
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn((pages, 128, 1, dim), dtype=dtype, device=device)
    v = torch.randn_like(k)
    selected = torch.full((1, tokens, 16), -1, dtype=torch.int32)
    for t in range(tokens):
        last = t // 128
        prior = torch.randperm(last)[:15]
        selected[0, t, 0] = last
        selected[0, t, 1 : 1 + len(prior)] = prior
    counts = (selected >= 0).sum(-1, dtype=torch.int32).to(device)
    lengths = torch.tensor([tokens], dtype=torch.int32, device=device)
    return (
        q,
        k,
        v,
        selected.to(device),
        torch.arange(pages, dtype=torch.int32, device=device)[None, :],
    ), dict(
        select_num_idx=counts,
        actual_seq_lengths=lengths,
        actual_seq_lengths_kv=lengths,
        scale_value=dim**-0.5,
        inner_precise=4,
    )


@pytest.mark.parametrize("shape", [(8, 16, 128), (1024, 16, 128)])
def test_npu_sparse_attention_score_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.npu_sparse_attention_score(*args, **kwargs)
    torch.testing.assert_close(actual, expected)

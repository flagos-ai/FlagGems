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

import random

import pytest
import torch

import flag_gems

from . import conftest as cfg

random.seed(42)


try:
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.utils.import_utils import has_deep_gemm

    DEEPGEMM_AVAILABLE = has_deep_gemm()
except Exception:
    DEEPGEMM_AVAILABLE = False


def _cpu_reference(context_lens_2d, block_kv, num_sms):
    """CPU mirror of DeepGEMM's paged-MQA schedule.

    Each sequence is cut into ``ceil(ctx_len / block_kv)`` consecutive KV
    blocks and the block ranges are handed to the SMs in order, so row ``i``
    holds (q index, block index inside that sequence).  The extra row
    ``num_sms`` is the end sentinel.
    """
    lens = context_lens_2d[:, -1].to(torch.int64)
    num_blocks = torch.div(lens + block_kv - 1, block_kv, rounding_mode="floor")
    prefix = torch.cumsum(num_blocks, dim=0)
    total = int(prefix[-1]) if prefix.numel() else 0
    out = torch.zeros((num_sms + 1, 2), dtype=torch.int32)
    for sm in range(num_sms + 1):
        start = sm * (total // num_sms) + min(sm, total % num_sms)
        q_idx = int((prefix <= start).sum())
        previous = int(prefix[q_idx - 1]) if q_idx > 0 else 0
        out[sm, 0] = q_idx
        out[sm, 1] = start - previous
    return out


def _reference(context_lens_2d, block_kv, num_sms):
    if DEEPGEMM_AVAILABLE:
        return get_paged_mqa_logits_metadata(context_lens_2d, block_kv, num_sms)
    # vLLM's DeepGEMM plugin is not available on this host; the schedule is
    # still fully specified, so compare against the CPU mirror on the same
    # block size the caller asked for.
    return _cpu_reference(context_lens_2d.cpu(), block_kv, num_sms)


# Without DeepGEMM the reference runs on CPU, which this kernel test reaches
# through CUDA tensors; devices other than CUDA keep skipping as before.
pytestmark = pytest.mark.skipif(
    not DEEPGEMM_AVAILABLE and not torch.cuda.is_available(),
    reason="vllm with deep_gemm (or a CUDA device) is required.",
)

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    BATCH_NEXTN_SHAPES = [(4, 1)]
else:
    BATCH_NEXTN_SHAPES = [(4, 1), (2, 2)]


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.parametrize("batch_size, next_n", BATCH_NEXTN_SHAPES)
@pytest.mark.parametrize("avg_ctx_len", [1024, 2048])
@pytest.mark.parametrize("block_kv", [64, 256])
def test_get_paged_mqa_logits_metadata(batch_size, next_n, avg_ctx_len, block_kv):
    context_lens_2d = (
        torch.randint(
            int(0.8 * avg_ctx_len), int(1.2 * avg_ctx_len), (batch_size, next_n)
        )
        .cuda()
        .to(torch.int32)
    )

    if DEEPGEMM_AVAILABLE:
        num_sms = get_num_sms()
    else:
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count

    ref = _reference(context_lens_2d, block_kv, num_sms)
    res = flag_gems.get_paged_mqa_logits_metadata(context_lens_2d, block_kv, num_sms)

    assert torch.equal(ref.cpu(), res.cpu())

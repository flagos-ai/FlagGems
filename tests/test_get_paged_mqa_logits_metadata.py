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

# Shape configs for QUICK_MODE
if cfg.QUICK_MODE:
    BATCH_NEXTN_SHAPES = [(4, 1)]
else:
    BATCH_NEXTN_SHAPES = [(4, 1), (2, 2)]


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="vllm with deep_gemm is required.")
@pytest.mark.parametrize("batch_size, next_n", BATCH_NEXTN_SHAPES)
@pytest.mark.parametrize("avg_ctx_len", [1024, 2048])
def test_get_paged_mqa_logits_metadata(batch_size, next_n, avg_ctx_len):
    context_lens_2d = (
        torch.randint(
            int(0.8 * avg_ctx_len), int(1.2 * avg_ctx_len), (batch_size, next_n)
        )
        .cuda()
        .to(torch.int32)
    )

    ref = get_paged_mqa_logits_metadata(context_lens_2d, 64, get_num_sms())
    res = flag_gems.get_paged_mqa_logits_metadata(context_lens_2d, 64, get_num_sms())

    assert torch.equal(ref, res)

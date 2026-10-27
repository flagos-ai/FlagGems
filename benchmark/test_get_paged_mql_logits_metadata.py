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

from . import base

try:
    from vllm.utils.deep_gemm import (
        get_num_sms,
        get_paged_mqa_logits_metadata,
        has_deep_gemm,
    )

    DEEPGEMM_AVAILABLE = has_deep_gemm()
except ImportError:
    DEEPGEMM_AVAILABLE = False


class GetPagedMqaLogitsMetadataBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self):
        # (batch_size, next_n)
        self.shapes = [
            (4, 1),
            (8, 1),
            (16, 1),
            (32, 1),
            (64, 1),
            (128, 1),
            (256, 1),
            (512, 1),
            (4, 2),
            (8, 2),
            (16, 2),
            (32, 2),
        ]

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from self._input_fn(config, cur_dtype)

    def _input_fn(self, config, dtype):
        _ = dtype
        batch_size, next_n = config
        device = flag_gems.device

        avg_ctx_len = 2048
        context_lens = torch.randint(
            int(0.8 * avg_ctx_len),
            int(1.2 * avg_ctx_len),
            (batch_size,),
            device=device,
            dtype=torch.int32,
        )

        if next_n > 1:
            context_lens = (
                context_lens.unsqueeze(1).expand(batch_size, next_n).contiguous()
            )

        num_sms = get_num_sms()

        yield (context_lens, 64, num_sms)


@pytest.mark.get_paged_mqa_logits_metadata
@pytest.mark.skipif(
    not DEEPGEMM_AVAILABLE,
    reason="requires vLLM with DeepGEMM and NVIDIA Hopper architecture or newer",
)
def test_get_paged_mqa_logits_metadata():
    bench = GetPagedMqaLogitsMetadataBenchmark(
        op_name="get_paged_mqa_logits_metadata",
        torch_op=get_paged_mqa_logits_metadata,
        dtypes=[torch.int32],
    )

    bench.set_gems(flag_gems.get_paged_mqa_logits_metadata)
    bench.run()

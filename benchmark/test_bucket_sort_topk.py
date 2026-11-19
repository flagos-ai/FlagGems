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

from flag_gems.fused.DSA.bin_topk import bucket_sort_topk

from . import base


def _torch_bucket_sort_topk_wrapper(inputs, starts, ends, topk):
    """Reference: per-row torch.topk over the [starts[i], ends[i]) slice."""
    batch_size = inputs.shape[0]
    ref_indices = torch.full(
        (batch_size, topk), -1, dtype=torch.int32, device=inputs.device
    )
    for i in range(batch_size):
        start = int(starts[i].item())
        end = int(ends[i].item())
        if end > start:
            _, topk_indices = torch.topk(inputs[i, start:end], min(topk, end - start))
            ref_indices[i, : topk_indices.numel()] = (
                topk_indices.to(torch.int32) + start
            )
    return ref_indices


class BucketSortTopkBenchmark(base.Benchmark):
    """
    Benchmark for the FlagGems DSA bucket_sort_topk kernel vs per-row torch.topk.
    """

    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        # (batch_size, seq_len, topk)
        self.shapes = [
            (1, 1024, 16),
            (2, 4096, 32),
            (4, 8192, 64),
            (8, 16384, 128),
            (16, 32768, 128),
            (32, 65536, 256),
        ]
        self.shape_desc = "batch_size, seq_len, topk"

    def get_input_iter(self, cur_dtype):
        for batch_size, seq_len, topk in self.shapes:
            torch.manual_seed(0)
            inputs = torch.randn((batch_size, seq_len), device="cuda", dtype=cur_dtype)
            starts = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
            ends = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
            yield inputs, starts, ends, topk


@pytest.mark.bucket_sort_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_bucket_sort_topk_benchmark():
    """Benchmark FlagGems bucket_sort_topk vs per-row torch.topk (fp32)."""
    bench = BucketSortTopkBenchmark(
        op_name="bucket_sort_topk",
        torch_op=_torch_bucket_sort_topk_wrapper,
        dtypes=[torch.float32],
    )
    bench.set_gems(bucket_sort_topk)
    bench.run()

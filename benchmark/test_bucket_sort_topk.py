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
from tests.test_DSA.test_bin_topk import make_topk_input, reference_topk_implementation

from . import base

# (batch_size, seq_len, topk)
BUCKET_SORT_TOPK_SHAPES = [
    (1, 1024, 32),
    (8, 4096, 128),
    (32, 8192, 256),
    (96, 32768, 2048),
]


class BucketSortTopKBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = BUCKET_SORT_TOPK_SHAPES
        self.shape_desc = "batch, seq_len, topk"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for batch_size, seq_len, topk in self.shapes:
            inputs, starts, ends = make_topk_input(
                batch_size, seq_len, dtype, self.device
            )
            yield inputs, starts, ends, topk


@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.bucket_sort_topk
def test_bucket_sort_topk():
    bench = BucketSortTopKBenchmark(
        op_name="bucket_sort_topk",
        # Baseline reproduces the torch reference from
        # tests/test_DSA/test_bin_topk.py (per-row torch.topk over [start:end]):
        # the same semantic reference the accuracy gate uses, so benchmark and
        # acceptance share one definition.
        torch_op=reference_topk_implementation,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.bucket_sort_topk)
    bench.run()

# Copyright 2026, The FlagOS Contributors.
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

from . import base, consts

# embedding_bag benchmark
# (num_bags, embedding_dim, num_weights, num_samples_per_bag_avg)
# Shapes span the realistic range: the first few are tiny and launch-overhead
# bound (torch's fused CUDA kernel wins there), while the larger bag/sample
# counts are where the gather workload dominates and this kernel is faster.
EMBEDDING_BAG_SHAPES = [
    (128, 256, 500, 4),
    (512, 256, 10000, 8),
    (1024, 128, 10000, 16),
    (2048, 256, 20000, 16),
    (4096, 256, 50000, 32),
    (8192, 512, 100000, 32),
    (16384, 256, 100000, 16),
]


class EmbeddingBagBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_BAG_SHAPES

    def get_input_iter(self, cur_dtype):
        for num_bags, embedding_dim, num_weights, samples_per_bag in self.shapes:
            num_samples = num_bags * samples_per_bag
            weight = torch.randn(
                num_weights, embedding_dim, dtype=cur_dtype, device=self.device
            )
            indices = torch.randint(
                0, num_weights, (num_samples,), dtype=torch.long, device=self.device
            )
            offsets = torch.arange(
                0,
                num_samples,
                samples_per_bag,
                dtype=torch.long,
                device=self.device,
            )[:num_bags]
            yield (
                weight,
                indices,
                offsets,
                False,  # scale_grad_by_freq
                0,  # mode (sum)
                False,  # sparse
                None,  # per_sample_weights
                False,  # include_last_offset
            )


@pytest.mark.embedding_bag
def test_embedding_bag():
    bench = EmbeddingBagBenchmark(
        op_name="embedding_bag",
        torch_op=torch.ops.aten.embedding_bag,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

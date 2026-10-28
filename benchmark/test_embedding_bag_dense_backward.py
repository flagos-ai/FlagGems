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

from . import base, consts


SHAPES = [
    (8, 16, 50, 4),
    (16, 32, 100, 4),
    (32, 64, 100, 4),
    (64, 128, 200, 4),
    (128, 256, 500, 4),
    (256, 128, 500, 8),
    (512, 256, 1000, 8),
]


class EmbeddingBagDenseBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES


def _case_fn(shape, dtype):
    del dtype
    num_bags, embedding_dim, num_weights, samples_per_bag = shape
    yield base.BenchmarkCasePlan(
        shape={"grad": [num_bags, embedding_dim], "indices": [num_bags * samples_per_bag]},
        params={"num_weights": num_weights, "mode": 0, "padding_idx": -1},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(42)
    (shape,) = plan.builder_args
    num_bags, embedding_dim, num_weights, samples_per_bag = shape
    num_samples = num_bags * samples_per_bag
    weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)
    indices = torch.randint(0, num_weights, (num_samples,), device=device)
    offsets = torch.arange(0, num_samples, samples_per_bag, device=device)
    output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
        weight, indices, offsets, False, 0, False, None, False, -1
    )
    # Materialize the standalone backward metadata omitted by optimized
    # CPU sum forward without changing the bags or sample indices.
    if offset2bag.numel() == 0:
        offset2bag = torch.arange(
            num_bags, dtype=torch.long, device=indices.device
        ).repeat_interleave(samples_per_bag)
    grad = torch.randn_like(output)
    return grad, indices, offset2bag, bag_size, maximum_indices, num_weights, False, 0, None, -1


@pytest.mark.embedding_bag_dense_backward
@pytest.mark.underscore_embedding_bag_dense_backward
def test_embedding_bag_dense_backward():
    bench = EmbeddingBagDenseBackwardBenchmark(
        op_name="_embedding_bag_dense_backward",
        torch_op=torch.ops.aten._embedding_bag_dense_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

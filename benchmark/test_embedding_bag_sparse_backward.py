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
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(8, 64, 1024), (32, 128, 8192), (64, 256, 16384)]
CASES = ((0, False, -1), (1, False, -1), (0, True, -1), (0, False, 0))


class EmbeddingBagSparseBackwardBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "num_bags, embedding_dim, num_weights"


def _case_fn(shape, dtype):
    del dtype
    num_bags, embedding_dim, num_weights = tuple(shape)
    for mode, with_psw, padding_idx in CASES:
        yield base.BenchmarkCasePlan(
            shape={"grad": [num_bags, embedding_dim], "indices": [num_bags * 3]},
            params={
                "mode": mode,
                "per_sample_weights": with_psw,
                "padding_idx": padding_idx,
            },
            builder_args=(
                num_bags,
                embedding_dim,
                num_weights,
                mode,
                with_psw,
                padding_idx,
            ),
        )


def _build_inputs_fn(plan, dtype, device):
    num_bags, embedding_dim, num_weights, mode, with_psw, padding_idx = (
        plan.builder_args
    )
    torch.manual_seed(0)
    num_samples = num_bags * 3
    weight = torch.randn((num_weights, embedding_dim), dtype=dtype, device=device)
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=device
    )
    offsets = torch.arange(0, num_samples, 3, dtype=torch.long, device=device)
    per_sample_weights = (
        torch.rand(num_samples, dtype=dtype, device=device) if with_psw else None
    )
    output, offset2bag, bag_size, _ = torch.ops.aten._embedding_bag(
        weight,
        indices,
        offsets,
        False,
        mode,
        True,
        per_sample_weights,
        False,
        padding_idx,
    )
    if per_sample_weights is not None and offset2bag.numel() == 0:
        # Sum-mode forward may omit the map required by sparse backward.
        cpu_offsets = offsets.cpu()
        ends = torch.cat((cpu_offsets[1:], cpu_offsets.new_tensor([indices.numel()])))
        offset2bag = torch.repeat_interleave(
            torch.arange(cpu_offsets.numel()), ends - cpu_offsets
        ).to(indices.device)
    grad = torch.randn_like(output)
    return (
        grad,
        indices,
        offsets,
        offset2bag,
        bag_size,
        num_weights,
        False,
        mode,
        per_sample_weights,
        padding_idx,
    )


@pytest.mark.embedding_bag_sparse_backward
@pytest.mark.underscore_embedding_bag_sparse_backward
def test_embedding_bag_sparse_backward():
    EmbeddingBagSparseBackwardBenchmark(
        op_name="_embedding_bag_sparse_backward",
        torch_op=torch.ops.aten._embedding_bag_sparse_backward,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()

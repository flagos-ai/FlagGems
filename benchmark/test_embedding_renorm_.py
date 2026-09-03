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


class EmbeddingRenormBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # 2D shapes: (num_embeddings, embedding_dim) across growing table sizes.
        self.shapes = [
            (1024, 128),
            (4096, 256),
            (16384, 256),
            (32768, 512),
            (50000, 256),
            (50000, 512),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for num_embeddings, embedding_dim in self.shapes:
            weight = (
                torch.randn(
                    num_embeddings, embedding_dim, dtype=cur_dtype, device=self.device
                )
                * 1.5
            )
            num_indices = num_embeddings // 2
            indices = torch.randint(
                0, num_embeddings, (num_indices,), dtype=torch.long, device=self.device
            )
            # max_norm=1.0, norm_type=2.0 matches nn.Embedding's defaults.
            yield weight, indices, 1.0, 2.0


@pytest.mark.embedding_renorm_
def test_embedding_renorm_():
    bench = EmbeddingRenormBenchmark(
        op_name="embedding_renorm_",
        torch_op=torch.ops.aten.embedding_renorm_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

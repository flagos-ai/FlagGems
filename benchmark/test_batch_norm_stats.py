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


class NormStatsBenchmark(base.GenericBenchmarkExcluse1D):
    """Benchmark for batch_norm_stats that excludes 1D shapes."""

    def set_more_shapes(self):
        return [
            # 3D shapes: [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes: [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
            (8, 32, 224, 224),
        ]


def input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    eps = 1e-5
    yield inp, eps


@pytest.mark.batch_norm_stats
def test_batch_norm_stats():
    bench = NormStatsBenchmark(
        op_name="batch_norm_stats",
        input_fn=input_fn,
        torch_op=torch.batch_norm_stats,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()

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

from . import base, consts


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 32, 1024),
            (32, 64, 2048),
            # 4D shapes represented as [batch_size, channels, H, W]
            (32, 64, 56, 56),
            (64, 128, 28, 28),
            (16, 256, 14, 14),
            (8, 512, 7, 7),
        ]


def batch_norm_gather_stats_input_fn(shape, dtype, device):
    C = shape[1]
    N = shape[0]
    spatial = 1
    for s in shape[2:]:
        spatial *= s
    count = N * spatial
    # Typical distributed training uses 4 GPU segments
    num_segments = 4

    input_t = torch.randn(shape, dtype=dtype, device=device)
    mean = torch.randn(num_segments, C, dtype=torch.float32, device=device)
    invstd = torch.rand(num_segments, C, dtype=torch.float32, device=device) + 0.1
    running_mean = torch.randn(C, dtype=torch.float32, device=device)
    running_var = torch.rand(C, dtype=torch.float32, device=device) + 0.1

    yield input_t, mean, invstd, running_mean, running_var, 0.1, 1e-5, count


@pytest.mark.batch_norm_gather_stats
def test_batch_norm_gather_stats():
    bench = NormBenchmark(
        input_fn=batch_norm_gather_stats_input_fn,
        op_name="batch_norm_gather_stats",
        torch_op=torch.batch_norm_gather_stats,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.batch_norm_gather_stats)

    bench.run()

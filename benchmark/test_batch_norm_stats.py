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


def _substituted_batch_norm_stats(inp, eps):
    """Substituted (non-contract) torch baseline for vendors without an aten kernel.

    enflame GCU has no aten kernel for batch_norm_stats (not on CPU either), so
    the baseline is the eager torch decomposition of the same definition: the
    per-channel mean and 1/sqrt(biased var + eps) in float32. The comparison is
    therefore GPU kernel vs multi-launch eager decomposition, not vs an aten
    kernel; the ratio is reported as-is.
    """
    reduce_dims = (0,) + tuple(range(2, inp.dim()))
    inp_f32 = inp.to(torch.float32)
    mean = inp_f32.mean(dim=reduce_dims)
    var = inp_f32.var(dim=reduce_dims, unbiased=False)
    return mean, 1.0 / torch.sqrt(var + eps)


def input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    eps = 1e-5
    yield inp, eps


@pytest.mark.batch_norm_stats
def test_batch_norm_stats():
    torch_op = (
        _substituted_batch_norm_stats
        if flag_gems.vendor_name == "enflame"
        else torch.batch_norm_stats
    )
    bench = NormStatsBenchmark(
        op_name="batch_norm_stats",
        input_fn=input_fn,
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.batch_norm_stats)

    bench.run()

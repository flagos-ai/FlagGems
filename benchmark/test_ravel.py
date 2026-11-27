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

"""Benchmark for ``aten::ravel``.

``ravel`` returns a contiguous 1-D view when the flattened input is already
contiguous and otherwise materialises a copy. The timed grid keeps the
contiguous (view-path) and the transposed (copy-path) layouts, so the two
regimes are measured separately.
"""

import pytest
import torch

import flag_gems

from . import base, consts

# Shapes span the launch-bound and bandwidth-bound regimes; the largest holds
# 2**24 elements (64 MiB fp32) on the copy path.
RAVEL_SHAPES = [
    (1024,),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
]


class RavelBenchmark(base.Benchmark):
    """Benchmark for ``aten::ravel`` (view path and copy path)."""

    DEFAULT_SHAPE_DESC = "input shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = RAVEL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            yield (inp,)


@pytest.mark.ravel
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_ravel(dtype):
    bench = RavelBenchmark(
        op_name="ravel",
        torch_op=torch.ravel,
        gems_op=flag_gems.ravel,
        dtypes=[dtype],
    )
    bench.run()

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


def unfold_copy_input_fn(config, dtype, device):
    shape, dim, size, step = config
    inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, dim, size, step


@pytest.mark.unfold_copy
def test_unfold_copy():
    class UnfoldCopyBenchmark(base.Benchmark):
        def set_shapes(self, shape_file_path=None):
            # Shapes cover 2D and 3D input tensors with varying dimension, size, and step
            self.shapes = [
                # 2D case
                ((4, 8), 1, 3, 1),
                ((16, 32), 1, 8, 2),
                ((8, 15), 1, 4, 3),
                # 3D case with dim=1
                ((2, 6, 8), 1, 3, 1),
                ((4, 8, 16), 1, 4, 2),
                # 3D case with dim=2
                ((2, 6, 8), 2, 3, 1),
                ((4, 8, 16), 2, 4, 2),
                ((2, 6, 8), 2, 3, 2),
            ]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for config in self.shapes:
                yield from unfold_copy_input_fn(config, cur_dtype, self.device)

    bench = UnfoldCopyBenchmark(
        op_name="unfold_copy",
        torch_op=torch.unfold_copy,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.unfold_copy)
    bench.run()

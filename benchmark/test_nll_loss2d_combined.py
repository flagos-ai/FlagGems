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

from . import base, consts, utils


class NllLoss2dCombinedBenchmark(base.GenericBenchmark4DOnly):
    def set_shapes(self, shape_file_path=None):
        # Segmentation-style 4D shapes (N, C, H, W): small class count with large
        # spatial extent. The default core_shapes are generic 4D tensors with a huge
        # channel dim and tiny H/W, which are not representative of nll_loss2d
        # workloads, so lock the shapes here to prevent CI core_shapes override.
        self.shapes = [
            (2, 4, 4, 8),
            (16, 21, 128, 128),
            (8, 19, 256, 256),
            (32, 33, 512, 512),
            (16, 19, 512, 512),
        ]

    def set_more_shapes(self):
        return None


def nll_loss2d_input_fn(shape, cur_dtype, device):
    N, C, H, W = shape
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    target = torch.randint(0, C, (N, H, W), device=device)
    # reduction=1 (mean), no weight, default ignore_index.
    yield inp, target, None, 1, -100

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        weight = torch.randn(C, dtype=cur_dtype, device=device)
        yield inp, target, weight, 0, 1


@pytest.mark.nll_loss2d
def test_nll_loss2d():
    # aten.nll_loss2d is the combined op (returns only the loss). F.nll_loss on a
    # 4D input decomposes to nll_loss2d_forward, so call the aten op directly.
    bench = NllLoss2dCombinedBenchmark(
        input_fn=nll_loss2d_input_fn,
        op_name="nll_loss2d",
        torch_op=torch.ops.aten.nll_loss2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

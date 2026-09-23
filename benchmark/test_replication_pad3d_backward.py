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

from typing import Generator

import pytest
import torch

from . import base, consts


class ReplicationPad3dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Volumetric feature maps with symmetric and asymmetric padding.
        self.shapes = [
            (2, 3, 8, 16, 16, 1, 1, 1, 1, 1, 1),
            (2, 8, 16, 32, 32, 2, 1, 1, 2, 3, 1),
            (1, 16, 32, 32, 32, 1, 1, 1, 1, 1, 1),
        ]
        self.shape_desc = "N, C, D, H, W, PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM, PAD_FRONT, PAD_BACK"

    def get_input_iter(self, cur_dtype) -> Generator:
        for (
            n,
            c,
            d,
            h,
            w,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            pad_front,
            pad_back,
        ) in self.shapes:
            inp = torch.randn((n, c, d, h, w), dtype=cur_dtype, device=self.device)
            padding = (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            grad_output = torch.randn(
                (
                    n,
                    c,
                    d + pad_front + pad_back,
                    h + pad_top + pad_bottom,
                    w + pad_left + pad_right,
                ),
                dtype=cur_dtype,
                device=self.device,
            )
            yield grad_output, inp, padding


@pytest.mark.replication_pad3d_backward
def test_replication_pad3d_backward():
    bench = ReplicationPad3dBackwardBenchmark(
        op_name="replication_pad3d_backward",
        torch_op=torch.ops.aten.replication_pad3d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()

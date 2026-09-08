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


class UpsampleBilinear2dAaBackwardBenchmark(base.Benchmark):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # (N, C, H_in, W_in, H_out, W_out, align_corners, label): a spread of
        # up/down-sampling shapes covering both the fused (small) and 2-pass /
        # GEMM (large NC) backward paths.
        self._cfgs = [
            # Small / medium — fused path targets
            (4, 16, 4, 4, 1, 1, False, "tiny 4x down"),
            (4, 16, 4, 4, 16, 16, False, "small 4x up"),
            (4, 16, 16, 16, 4, 4, False, "small 4x down"),
            (4, 16, 16, 32, 64, 128, False, "small->med 4x up"),
            (1, 1, 64, 64, 16, 16, False, "C=1 4x down"),
            (1, 1, 64, 64, 32, 32, False, "C=1 2x down"),
            (1, 1, 64, 64, 128, 128, False, "C=1 2x up"),
            (4, 3, 256, 256, 128, 128, False, "C=3 2x down"),
            (4, 3, 128, 128, 256, 256, False, "C=3 2x up"),
            (4, 64, 64, 64, 32, 32, False, "C=64 2x down"),
            # Large — 2-pass / GEMM path targets
            (1, 64, 512, 512, 128, 128, False, "C=64 4x down"),
            (1, 64, 512, 512, 1024, 1024, False, "C=64 2x up"),
            (512, 1024, 32, 32, 8, 8, False, "NC=524K 4x down"),
            (256, 512, 64, 64, 16, 16, False, "NC=131K 4x down"),
            (256, 512, 64, 64, 32, 32, False, "NC=131K 2x down"),
            (256, 512, 64, 64, 128, 128, False, "NC=131K 2x up"),
        ]

    def get_input_iter(self, dtype):
        for N, C, Hi, Wi, Ho, Wo, ac, _label in self._cfgs:
            grad = torch.randn([N, C, Ho, Wo], device=self.device, dtype=dtype)
            yield grad, [Ho, Wo], [N, C, Hi, Wi], ac, None, None

    def get_tflops(self, op, *args, **kwargs):
        grad = args[0]
        return grad.numel() * 2


@pytest.mark.upsample_bilinear2d_aa_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_upsample_bilinear2d_aa_backward():
    bench = UpsampleBilinear2dAaBackwardBenchmark(
        op_name="upsample_bilinear2d_aa_backward",
        torch_op=torch.ops.aten._upsample_bilinear2d_aa_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()

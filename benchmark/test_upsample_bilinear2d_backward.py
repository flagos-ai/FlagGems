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

from . import base

_MUSA_NATIVE_BUG = False
if flag_gems.vendor_name == "mthreads":
    import torch_musa

    _MUSA_NATIVE_BUG = torch_musa.__version__ == "2.9.1+a18d871"

pytestmark = pytest.mark.skipif(
    _MUSA_NATIVE_BUG,
    reason="TorchMUSA 2.9.1+a18d871 native backward executes nearest rather than bilinear; no valid native baseline",
)

_CANN_VERSION = None
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN_VERSION = get_cann_version()
_CANN_850 = _CANN_VERSION == "8.5.0"

DTYPES = [
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name in ("ascend", "mthreads", "iluvatar"),
            reason="FP64 arithmetic unavailable on Iluvatar; native backward rejects FP64 on Ascend/MUSA",
        ),
    ),
    torch.float16,
    torch.bfloat16,
]
CASES = [
    ("identity", (2, 16, 64, 64), (64, 64), None, None, "contiguous"),
    ("2x_small", (1, 1, 7, 9), (14, 18), None, None, "contiguous"),
    ("2x_medium", (2, 16, 64, 64), (128, 128), None, None, "contiguous"),
    ("2x_large", (4, 32, 128, 128), (256, 256), None, None, "contiguous"),
    ("fractional", (2, 8, 31, 47), (53, 79), None, None, "contiguous"),
    ("downsample", (2, 16, 64, 96), (32, 48), None, None, "contiguous"),
    ("strong_downsample", (2, 16, 96, 128), (7, 9), None, None, "contiguous"),
    ("out_one", (2, 8, 31, 47), (1, 1), None, None, "contiguous"),
    ("in_h_one", (2, 4, 1, 33), (7, 65), None, None, "contiguous"),
    ("explicit", (2, 8, 17, 23), (27, 37), 2.1, 1.8, "contiguous"),
    ("explicit_downsample", (2, 8, 33, 41), (9, 13), 0.4, 0.5, "contiguous"),
    ("strided", (2, 8, 31, 47), (61, 89), None, None, "strided"),
    ("channels_last", (2, 16, 31, 47), (61, 89), None, None, "channels_last"),
]


class UpsampleBilinear2dBackwardBenchmark(base.Benchmark):
    out_variant = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_input_iter(self, dtype):
        for _, input_size, output_size, sh, sw, layout in self.shapes:
            for align_corners in (False, True):
                n, c, ih, iw = input_size
                oh, ow = output_size
                grad = torch.randn(
                    (n, c, oh, ow * 2 if layout == "strided" else ow),
                    device=self.device,
                    dtype=dtype,
                )
                if layout == "strided":
                    grad = grad[..., ::2]
                if layout == "channels_last":
                    grad = grad.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
                args = (grad, output_size, input_size, align_corners, sh, sw)
                if self.out_variant:
                    output = torch.empty(
                        (n, c, ih, iw * 2 if layout == "strided" else iw),
                        device=self.device,
                        dtype=dtype,
                    )
                    if layout == "strided":
                        output = output[..., ::2]
                    if layout == "channels_last":
                        output = (
                            output.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
                        )
                    yield (*args, {"grad_input": output})
                else:
                    yield args


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.skipif(
    _CANN_850,
    reason="CANN 8.5.0 is accuracy-only: benchmark execution has not been validated",
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_bilinear2d_backward(dtype):
    bench = UpsampleBilinear2dBackwardBenchmark(
        op_name="upsample_bilinear2d_backward",
        torch_op=torch.ops.aten.upsample_bilinear2d_backward.default,
        gems_op=flag_gems.ops.upsample_bilinear2d_backward,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.skipif(
    _CANN_850,
    reason="CANN 8.5.0 is accuracy-only: benchmark execution has not been validated",
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_upsample_bilinear2d_backward_grad_input(dtype):
    bench = UpsampleBilinear2dBackwardBenchmark(
        op_name="upsample_bilinear2d_backward.grad_input",
        torch_op=torch.ops.aten.upsample_bilinear2d_backward.grad_input,
        gems_op=flag_gems.ops.upsample_bilinear2d_backward_grad_input,
        dtypes=[dtype],
        out_variant=True,
    )
    bench.run()

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

_MUSA_NATIVE_IS_NEAREST = False
if flag_gems.vendor_name == "mthreads":
    import torch_musa

    _MUSA_NATIVE_IS_NEAREST = torch_musa.__version__ == "2.9.1+a18d871"
_INVALID_MUSA_NATIVE = pytest.mark.skipif(
    _MUSA_NATIVE_IS_NEAREST,
    reason=(
        "TorchMUSA 2.9.1+a18d871 native bicubic backward executes nearest; "
        "no valid bicubic performance baseline"
    ),
)

_CANN_VERSION = None
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN_VERSION = get_cann_version()

pytestmark = pytest.mark.skipif(
    _CANN_VERSION == "8.5.0",
    reason="CANN 8.5.0 is accuracy-only: benchmark execution has not been validated",
)

DTYPES = [
    pytest.param(torch.float32, marks=_INVALID_MUSA_NATIVE),
    pytest.param(
        torch.float64,
        marks=[
            pytest.mark.skipif(
                flag_gems.vendor_name in ("mthreads", "ascend"),
                reason="MUSA muDNN and Ascend aclnn bicubic backward reject DOUBLE; no native FP64 latency baseline",
            ),
            pytest.mark.skipif(
                flag_gems.vendor_name == "iluvatar",
                reason=(
                    "Iluvatar native FP64 bicubic backward returns zeros for "
                    "valid nonzero inputs; no valid native baseline"
                ),
            ),
        ],
    ),
    pytest.param(torch.float16, marks=_INVALID_MUSA_NATIVE),
    pytest.param(torch.bfloat16, marks=_INVALID_MUSA_NATIVE),
]
# This fixed manifest is shared by both overloads and all measured backends.
CASES = [
    ("identity_small", (1, 3, 8, 9), (8, 9), None, None, "contiguous"),
    ("identity_large", (8, 16, 128, 128), (128, 128), None, None, "contiguous"),
    ("double_small", (1, 3, 8, 9), (16, 18), None, None, "contiguous"),
    ("double_large", (4, 16, 64, 64), (128, 128), None, None, "contiguous"),
    ("noninteger", (2, 8, 37, 53), (61, 89), None, None, "contiguous"),
    ("downsample", (2, 8, 64, 80), (23, 31), None, None, "contiguous"),
    ("strong_downsample", (2, 8, 128, 160), (3, 5), None, None, "contiguous"),
    ("out_one", (2, 3, 13, 17), (1, 1), None, None, "contiguous"),
    ("input_one", (1, 3, 1, 1), (31, 37), None, None, "contiguous"),
    ("explicit_scales", (2, 3, 13, 17), (23, 31), 1.7, 1.9, "contiguous"),
    ("small_scale", (1, 3, 13, 17), (23, 31), 0.6, 0.8, "contiguous"),
    ("noncontiguous", (2, 3, 13, 17), (23, 31), None, None, "strided"),
]


class UpsampleBicubic2dBackwardBenchmark(base.Benchmark):
    out_variant = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_input_iter(self, cur_dtype):
        for _, shape, output_size, sh, sw, layout in self.shapes:
            for align_corners in (False, True):
                n, c = shape[:2]
                oh, ow = output_size
                if layout == "strided":
                    grad_output = torch.randn(
                        (n, c, oh, ow * 2), device=self.device, dtype=cur_dtype
                    )[..., ::2]
                else:
                    grad_output = torch.randn(
                        (n, c, oh, ow), device=self.device, dtype=cur_dtype
                    )
                args = (grad_output, output_size, shape, align_corners, sh, sw)
                if self.out_variant:
                    if layout == "strided":
                        out = torch.empty(
                            (*shape[:-1], shape[-1] * 2),
                            device=self.device,
                            dtype=cur_dtype,
                        )[..., ::2]
                    else:
                        out = torch.empty(shape, device=self.device, dtype=cur_dtype)
                    yield (*args, {"grad_input": out})
                else:
                    yield args


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
def test_upsample_bicubic2d_backward(dtype):
    bench = UpsampleBicubic2dBackwardBenchmark(
        op_name="upsample_bicubic2d_backward",
        torch_op=torch.ops.aten.upsample_bicubic2d_backward.default,
        gems_op=flag_gems.ops.upsample_bicubic2d_backward,
        dtypes=[dtype],
    )
    bench.run()


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.upsample_bicubic2d_backward_grad_input
def test_upsample_bicubic2d_backward_grad_input(dtype):
    bench = UpsampleBicubic2dBackwardBenchmark(
        op_name="upsample_bicubic2d_backward.grad_input",
        torch_op=torch.ops.aten.upsample_bicubic2d_backward.grad_input,
        gems_op=flag_gems.ops.upsample_bicubic2d_backward_grad_input,
        dtypes=[dtype],
        out_variant=True,
    )
    bench.run()

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
from flag_gems.runtime import torch_device_fn

from . import base

GROUP_SIZE = 128


def _fp8_available():
    if not hasattr(torch, "float8_e4m3fn") or not torch_device_fn.is_available():
        return False
    if flag_gems.vendor_name == "mthreads":
        return torch_device_fn.get_device_capability()[0] >= 3
    if flag_gems.vendor_name == "thead":
        return True
    return (
        flag_gems.vendor_name == "nvidia" and torch.cuda.get_device_capability()[0] >= 9
    )


def _gems_rms_norm(x, shape, weight_fp8, weight_scale, weight_ref):
    return flag_gems.rms_norm(x, shape, weight_ref, eps=1e-5)


def _gems_rms_norm_w8a16_fp8(x, shape, weight_fp8, weight_scale, weight_ref):
    return flag_gems.rms_norm_w8a16_fp8(
        x, shape, weight_fp8, weight_scale, eps=1e-5, group_size=GROUP_SIZE
    )


class RmsNormW8A16FP8Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
            (1024, 4096),
            (1, 8192),
            (64, 8192),
            (256, 8192),
            (1, 16384),
            (64, 16384),
            (1, 32768),
            (64, 32768),
            (4096, 4096),
            (16, 33024),
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            _, n = shape
            x = torch.randn(shape, dtype=dtype, device=self.device)
            weight = torch.randn(n, dtype=dtype, device=self.device)
            grouped = weight.float().reshape(-1, GROUP_SIZE)
            scales = (grouped.abs().amax(-1) / 448.0).clamp(min=1e-8).to(dtype)
            weight_fp8 = (
                (grouped / scales.float()[:, None])
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
                .reshape(n)
            )
            # The BF16 baseline uses the same effective quantized weight.
            # Quantization and reference dequantization are outside timing.
            weight_ref = (
                (weight_fp8.float().reshape(-1, GROUP_SIZE) * scales.float()[:, None])
                .reshape(n)
                .to(dtype)
            )
            yield x, (n,), weight_fp8, scales, weight_ref


@pytest.mark.rms_norm_w8a16_fp8
@pytest.mark.skipif(not _fp8_available(), reason="Requires FP8 E4M3FN support")
def test_rms_norm_w8a16_fp8():
    print("Baseline: FlagGems BF16 (flag_gems.rms_norm)")
    bench = RmsNormW8A16FP8Benchmark(
        op_name="rms_norm_w8a16_fp8",
        torch_op=_gems_rms_norm,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_rms_norm_w8a16_fp8)
    bench.run()

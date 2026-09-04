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

GROUP_SIZE = 128
FP8_DTYPE = torch.float8_e5m2


def _fp8_e5_available():
    return torch.cuda.is_available() and hasattr(torch, "float8_e5m2")


def _quantize_fp8_e5_grouped(x, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    *leading, n = x.shape
    padded = (n + group_size - 1) // group_size * group_size
    x_pad = torch.nn.functional.pad(x.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, padded // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (grouped / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return (
        q.reshape(*leading, padded)[..., :n].contiguous(),
        scale.squeeze(-1).to(x.dtype).contiguous(),
    )


def _dequant_fp8_e5(x_fp8, x_scale, group_size=GROUP_SIZE):
    *leading, n = x_fp8.shape
    num_groups = x_scale.shape[-1]
    padded = num_groups * group_size
    x_pad = torch.nn.functional.pad(x_fp8.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, num_groups, group_size)
    dequant = grouped * x_scale.unsqueeze(-1).float()
    return dequant.reshape(*leading, padded)[..., :n].to(x_scale.dtype)


def _torch_topk_w8a16(x_fp8, x_scale, k, dequant):
    return torch.topk(dequant, k, dim=-1, largest=True, sorted=True)


def _gems_topk_w8a16(x_fp8, x_scale, k, dequant):
    return flag_gems.topk_w8a16_fp8(x_fp8, x_scale, k, dim=-1, largest=True, sorted=True)


class TopKFp8W8A16Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 128, 8),
            (256, 256, 8),
            (128, 1024, 16),
            (64, 4096, 32),
            (32, 8192, 64),
            (16, 16384, 128),
            (8, 32768, 256),
        ]

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            x = torch.randn((m, n), dtype=dtype, device=self.device)
            x_fp8, x_scale = _quantize_fp8_e5_grouped(x)
            dequant = _dequant_fp8_e5(x_fp8, x_scale)
            yield x_fp8, x_scale, k, dequant


@pytest.mark.topk_w8a16_fp8
@pytest.mark.skipif(
    getattr(flag_gems, "vendor_name", None) != "thead",
    reason="topk_w8a16_fp8 is a THead/PPU operator",
)
@pytest.mark.skipif(not _fp8_e5_available(), reason="float8_e5m2 is unavailable")
def test_topk_w8a16_fp8():
    bench = TopKFp8W8A16Benchmark(
        op_name="topk_w8a16_fp8",
        torch_op=_torch_topk_w8a16,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_topk_w8a16)
    bench.run()

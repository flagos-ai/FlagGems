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
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def _is_fp8_topk_supported():
    return (
        FP8_DTYPE is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )


def _quantize_fp8_grouped_lastdim(x, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    n = x.shape[-1]
    num_groups = (n + group_size - 1) // group_size
    padded_n = num_groups * group_size
    if padded_n != n:
        pad_shape = x.shape[:-1] + (padded_n - n,)
        x_for_scale = torch.cat(
            [x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=-1
        )
    else:
        x_for_scale = x
    grouped = x_for_scale.reshape(*x.shape[:-1], num_groups, group_size).float()
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (grouped / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    q = q.reshape(*x.shape[:-1], padded_n)[..., :n].contiguous()
    return q, scale.squeeze(-1).to(x.dtype).contiguous()


def _quantize_fp8_row(x):
    fp8_info = torch.finfo(FP8_DTYPE)
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (x.float() / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return q.contiguous(), scale.to(x.dtype).contiguous()


def _torch_topk_bf16_baseline(x, x_fp8, x_scale, k, dim=-1):
    return torch.topk(x, k, dim=dim)


def _gems_topk_w8a16_fp8(x, x_fp8, x_scale, k, dim=-1):
    return flag_gems.topk_w8a16_fp8(
        x_fp8, x_scale, k, dim=dim, group_size=x_fp8.shape[-1]
    )


class TopKFp8W8A16Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 128, 8),
            (8, 256, 16),
            (64, 1024, 32),
            (64, 4096, 64),
            (64, 8192, 128),
            (128, 32768, 256),
        ]

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            x = torch.randn((m, n), device=self.device, dtype=dtype)
            x_fp8, x_scale = _quantize_fp8_row(x)
            yield x, x_fp8, x_scale, k, -1


@pytest.mark.topk_w8a16_fp8
@pytest.mark.skipif(
    not _is_fp8_topk_supported(),
    reason="FP8 TopK requires CUDA FP8 support on compute capability >= 9.0",
)
def test_topk_w8a16_fp8():
    bench = TopKFp8W8A16Benchmark(
        op_name="topk_w8a16_fp8",
        torch_op=_torch_topk_bf16_baseline,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_topk_w8a16_fp8)
    bench.run()

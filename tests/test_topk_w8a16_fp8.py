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

from . import accuracy_utils as utils

FP8_DTYPE = torch.float8_e5m2
FP8_GROUP_SIZE = 128


def _fp8_e5_available():
    return torch.cuda.is_available() and hasattr(torch, "float8_e5m2")


def _quantize_fp8_e5_grouped(x, group_size=FP8_GROUP_SIZE):
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


def _dequant_fp8_e5(x_fp8, x_scale, group_size=FP8_GROUP_SIZE):
    *leading, n = x_fp8.shape
    num_groups = x_scale.shape[-1]
    padded = num_groups * group_size
    x_pad = torch.nn.functional.pad(x_fp8.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, num_groups, group_size)
    dequant = grouped * x_scale.unsqueeze(-1).float()
    return dequant.reshape(*leading, padded)[..., :n].to(x_scale.dtype)


@pytest.mark.topk_w8a16_fp8
@pytest.mark.skipif(
    getattr(flag_gems, "vendor_name", None) != "thead",
    reason="topk_w8a16_fp8 is a THead/PPU operator",
)
@pytest.mark.skipif(not _fp8_e5_available(), reason="float8_e5m2 is unavailable")
@pytest.mark.parametrize(
    "shape, k",
    [
        ((4, 128), 5),
        ((8, 256), 8),
        ((4, 1024), 16),
        ((2, 4096), 32),
        ((2, 8192), 64),
        ((8, 32768), 256),
        ((2, 33, 128), 8),
    ],
)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_w8a16_fp8(shape, k, largest):
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_fp8, x_scale = _quantize_fp8_e5_grouped(x)
    dequant = _dequant_fp8_e5(x_fp8, x_scale)

    ref_value, ref_index = torch.topk(
        utils.to_reference(dequant), k, dim=-1, largest=largest, sorted=True
    )
    res_value, res_index = flag_gems.topk_w8a16_fp8(
        x_fp8,
        x_scale,
        k,
        dim=-1,
        largest=largest,
        sorted=True,
        group_size=FP8_GROUP_SIZE,
        out_dtype=dtype,
    )

    utils.gems_assert_close(res_value, ref_value, dtype)
    # FP8 E5M2 quantization creates many ties; index order among equal
    # values may differ from torch.topk. Check the selected values instead.
    utils.gems_assert_close(torch.gather(dequant, -1, res_index), ref_value, dtype)

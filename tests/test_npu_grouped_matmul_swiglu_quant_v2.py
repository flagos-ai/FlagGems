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

pytestmark = [
    pytest.mark.npu_grouped_matmul_swiglu_quant_v2,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2.default


def _inputs(shape, dtype, device):
    m, k, n, e, bits = shape
    counts = torch.full((e,), m // e, device=device, dtype=torch.int64)
    counts[: m % e] += 1
    group_list = counts.cumsum(0)
    if bits == 4:
        x = torch.randint(0, 256, (m, k // 2), device=device, dtype=torch.uint8)
        w = torch.randint(
            0, 256, (e, n, k // 2), device=device, dtype=torch.uint8
        ).transpose(1, 2)
        ws = torch.randint(
            122, 126, (e, n, k // 64, 2), device=device, dtype=torch.uint8
        ).transpose(1, 2)
        xs = torch.randint(122, 126, (m, k // 64, 2), device=device, dtype=torch.uint8)
    else:
        x = (torch.randn((m, k), device=device, dtype=dtype) * 0.25).to(
            torch.float8_e4m3fn
        )
        w = (torch.randn((e, k, n), device=device, dtype=dtype) * 0.25).to(
            torch.float8_e4m3fn
        )
        ws = torch.randint(
            124, 128, (e, k // 64, n, 2), device=device, dtype=torch.uint8
        )
        xs = torch.randint(
            124, 128, (m, k // 64, 2), device=device, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    return (x, [w], [ws], xs, group_list), dict(
        dequant_mode=2,
        dequant_dtype=torch.float32,
        quant_mode=2,
        quant_dtype=296 if bits == 4 else torch.float8_e4m3fn,
        x_dtype=296 if bits == 4 else None,
        weight_dtype=296 if bits == 4 else None,
        weight_scale_dtype=293,
        x_scale_dtype=293,
    )


@pytest.mark.parametrize("shape", [(16, 2048, 1024, 128, 8), (64, 6144, 4096, 32, 4)])
def test_npu_grouped_matmul_swiglu_quant_v2_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.npu_grouped_matmul_swiglu_quant_v2(*args, **kwargs)
    for computed, native in zip(actual, expected, strict=True):
        assert computed.shape == native.shape and computed.dtype == native.dtype
        torch.testing.assert_close(
            computed.view(torch.uint8), native.view(torch.uint8), rtol=0, atol=0
        )

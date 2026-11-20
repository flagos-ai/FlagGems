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
    pytest.mark.npu_dynamic_mx_quant,
    pytest.mark.skipif(
        flag_gems.vendor_name != "ascend"
        or not torch.npu.get_device_name().startswith("Ascend950"),
        reason="Atlas A5 specialization",
    ),
]


def _reference():
    return torch.ops.npu.npu_dynamic_mx_quant.default


def _inputs(shape, dtype, device):
    rows, hidden, bits = shape
    x = torch.randn((rows, hidden), device=device, dtype=torch.bfloat16)
    return (x,), dict(
        axis=-1,
        round_mode="rint",
        dst_type=296 if bits == 4 else torch.float8_e4m3fn,
        block_size=32,
        scale_alg=0,
    )


@pytest.mark.parametrize(
    "shape", [(1, 4096, 8), (1024, 4096, 8), (1024, 6144, 4), (8, 768, 8)]
)
def test_npu_dynamic_mx_quant_native(shape):
    args, kwargs = _inputs(shape, torch.bfloat16, flag_gems.device)
    reference = _reference()
    expected = reference(*args, **kwargs)
    actual = flag_gems.npu_dynamic_mx_quant(*args, **kwargs)
    for result, reference_value in zip(actual, expected, strict=True):
        assert (
            result.shape == reference_value.shape
            and result.dtype == reference_value.dtype
        )
        torch.testing.assert_close(
            result.view(torch.uint8), reference_value.view(torch.uint8), atol=0, rtol=0
        )

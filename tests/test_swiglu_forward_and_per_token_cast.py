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


def make_inputs(shape, device):
    m, h = shape
    x = torch.randn((m, 2 * h), device=device, dtype=torch.bfloat16)
    return (x, "e4m3", 32), dict(
        round_sf=True, use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True
    )


def assert_outputs(actual, expected):
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected)
        for x, y in zip(actual, expected):
            assert_outputs(x, y)
    elif expected is None:
        assert actual is None
    elif expected.dtype == torch.float8_e4m3fn:
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        torch.testing.assert_close(
            actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
    else:
        torch.testing.assert_close(actual, expected)


@pytest.mark.swiglu_forward_and_per_token_cast
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend"
    or not torch.npu.get_device_name().startswith("Ascend950"),
    reason="requires Atlas A5",
)
@pytest.mark.parametrize("shape", [(512, 2048), (512, 3584)])
def test_swiglu_forward_and_per_token_cast(shape):
    reference = pytest.importorskip(
        "tile_kernels.quant.swiglu_forward_kernel"
    ).swiglu_forward_and_per_token_cast
    args, kwargs = make_inputs(shape, flag_gems.device)
    expected = reference(*args, **kwargs)
    actual = flag_gems.swiglu_forward_and_per_token_cast(*args, **kwargs)
    assert_outputs(actual, expected)

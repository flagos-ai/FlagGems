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
from . import conftest as cfg

if cfg.QUICK_MODE:
    UPSAMPLE_BICUBIC2D_PARAMS = [
        (1, 1, 8, 8, 16, 16, False, False),
    ]
    UPSAMPLE_BICUBIC2D_DTYPES = [torch.float32]
else:
    UPSAMPLE_BICUBIC2D_PARAMS = [
        (1, 1, 8, 8, 16, 16, False, False),
        (2, 3, 15, 20, 30, 35, True, False),
        (4, 3, 7, 5, 14, 10, False, True),
        (1, 16, 32, 24, 48, 36, True, True),
    ]
    UPSAMPLE_BICUBIC2D_DTYPES = [torch.float16, torch.float32, torch.bfloat16]


@pytest.mark.upsample_bicubic2d
@pytest.mark.parametrize(
    "N, C, H, W, outH, outW, align_corners, use_scale",
    UPSAMPLE_BICUBIC2D_PARAMS,
)
@pytest.mark.parametrize("dtype", UPSAMPLE_BICUBIC2D_DTYPES)
def test_upsample_bicubic2d(N, C, H, W, outH, outW, align_corners, use_scale, dtype):
    x = torch.randn((N, C, H, W), dtype=dtype, device=flag_gems.device)

    if use_scale:
        output_size = None
        scale_factors = (outH / float(H), outW / float(W))
    else:
        output_size = (outH, outW)
        scale_factors = None

    ref_x = utils.to_reference(x, True)
    ref_out = torch._C._nn.upsample_bicubic2d(
        ref_x, output_size, align_corners, scale_factors
    ).to(dtype=dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_bicubic2d(
            x, output_size, align_corners, scale_factors
        )

    utils.gems_assert_close(res_out.to(dtype=dtype), ref_out, dtype, reduce_dim=16)

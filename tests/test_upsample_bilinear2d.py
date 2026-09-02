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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

if cfg.QUICK_MODE:
    # Quick-mode scale set: one integer scale covering the smoke path.
    UPSAMPLE_BILINEAR2D_SCALES = [(2, 2)]
else:
    # Scales cover integer upscale, fractional upscale, mixed-axis, and downscale.
    UPSAMPLE_BILINEAR2D_SCALES = [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.7)]


@pytest.mark.upsample_bilinear2d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", UPSAMPLE_BILINEAR2D_SCALES)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_bilinear2d(dtype, shape, scale, align_corners):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch._C._nn.upsample_bilinear2d(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_bilinear2d(
            input, output_size=output_size, align_corners=align_corners
        )
    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)
    # Bilinear interpolation uses 4 neighbors with weighted average. The source
    # coordinate is computed in fp32, so for large inputs (~1024 px) the weight
    # rounding alone contributes up to ~8e-4 abs error vs the fp64 reference
    # (fp32 ATen shows the same magnitude). reduce_dim=16 keeps ~2x headroom
    # over the observed worst-case diff so the test is not flaky on randn data.
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=16)

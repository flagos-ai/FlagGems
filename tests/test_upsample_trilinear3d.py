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


@pytest.mark.upsample_trilinear3d
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "scale", [(2, 2, 2), (1.5, 2.1, 3.7), (0.5, 0.5, 0.5), (0.3, 1.3, 0.7)]
)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_trilinear3d(dtype, shape, scale, align_corners):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(3)]
    ref_out = torch.ops.aten.upsample_trilinear3d.default(
        ref_i, output_size, align_corners, None, None, None
    ).to(dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_trilinear3d.default(
            input, output_size, align_corners, None, None, None
        )
    utils.gems_assert_close(res_out, ref_out, dtype)

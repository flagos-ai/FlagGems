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


@pytest.mark.upsample_bilinear2d_aa
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1)])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 16, 128, 128),
        (15, 37, 256, 256),
        (3, 5, 127, 127),
        (128, 192, 42, 51),
        (3, 7, 1023, 1025),
    ],
)
# bfloat16 excluded: insufficient precision for bilinear AA on large input sizes
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_upsample_bilinear2d_aa(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch.ops.aten._upsample_bilinear2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    res_out = flag_gems._upsample_bilinear2d_aa(
        input, output_size=output_size, align_corners=align_corners
    )

    def span(scale):
        support = 2 if (scale >= 1.0) else 2.0 / scale
        interpolate_range = int(support + 0.5) * 2 + 1
        return interpolate_range

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    # Bilinear uses 2x2 support window
    reduce_dim = span(scale[0]) * span(scale[1])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.upsample_bilinear2d_aa
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("scale", [(0.5, 0.5), (0.37, 0.61), (0.5, 1.7), (1.8, 0.4)])
@pytest.mark.parametrize("shape", [(4, 8, 64, 64), (3, 5, 127, 129)])
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_upsample_bilinear2d_aa_downsampling(dtype, shape, scale, align_corners):
    input = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch.ops.aten._upsample_bilinear2d_aa(
        ref_i, output_size=output_size, align_corners=align_corners
    )
    res_out = flag_gems._upsample_bilinear2d_aa(
        input, output_size=output_size, align_corners=align_corners
    )

    def span(scale_val):
        support = 1.0 if (scale_val >= 1.0) else 1.0 / scale_val
        return int(support + 0.5) * 2 + 1

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    reduce_dim = span(scale[0]) * span(scale[1])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.upsample_bilinear2d_aa
@pytest.mark.parametrize("layout", ["transpose", "sliced"])
@pytest.mark.parametrize("scale", [(0.5, 0.75), (1.5, 2.0)])
@pytest.mark.parametrize("dtype", utils.PRIMARY_FLOAT_DTYPES)
def test_upsample_bilinear2d_aa_non_contiguous(dtype, scale, layout):
    base = torch.rand((2, 6, 48, 64), dtype=dtype, device=flag_gems.device)
    if layout == "transpose":
        input = base.transpose(2, 3)
    else:
        input = base[:, ::2, ::2, 1::2]
    assert not input.is_contiguous()
    ref_i = utils.to_reference(input, True)
    output_size = tuple([int(input.shape[i + 2] * scale[i]) for i in range(2)])
    ref_out = torch.ops.aten._upsample_bilinear2d_aa(
        ref_i, output_size=output_size, align_corners=False
    )
    res_out = flag_gems._upsample_bilinear2d_aa(
        input, output_size=output_size, align_corners=False
    )

    def span(scale_val):
        support = 1.0 if (scale_val >= 1.0) else 1.0 / scale_val
        return int(support + 0.5) * 2 + 1

    if ref_out.dtype != res_out.dtype:
        ref_out = ref_out.to(res_out.dtype)

    reduce_dim = span(scale[0]) * span(scale[1])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.upsample_bilinear2d_aa
def test_upsample_bilinear2d_aa_empty_and_singleton():
    empty_in = torch.empty((0, 3, 16, 16), dtype=torch.float32, device=flag_gems.device)
    empty_out = flag_gems._upsample_bilinear2d_aa(empty_in, (8, 8))
    assert empty_out.shape == (0, 3, 8, 8)

    singleton_in = torch.rand(
        (2, 3, 5, 5), dtype=torch.float32, device=flag_gems.device
    )
    ref_in = utils.to_reference(singleton_in, True)
    ref_out = torch.ops.aten._upsample_bilinear2d_aa(ref_in, (1, 1), align_corners=True)
    res_out = flag_gems._upsample_bilinear2d_aa(
        singleton_in, (1, 1), align_corners=True
    )
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=25)

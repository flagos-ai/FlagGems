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
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

ADAPTIVE_AVGPOOL2D_CONFIGS = [
    # Test various combinations of input and output sizes
    # Cases where output size is smaller than input
    ((4, 3, 32, 32), (1, 1)),  # Downsize to 1x1
    ((4, 3, 32, 32), (2, 2)),  # Downsize to 2x2
    ((4, 3, 32, 32), (8, 8)),  # Downsize to 8x8
    ((4, 3, 32, 32), (16, 16)),  # Downsize to 16x16
    ((2, 16, 56, 56), (7, 7)),  # ResNet-like case
    # Test non-square inputs and outputs
    ((8, 16, 28, 40), (14, 10)),  # Non-square input to non-square output
    ((4, 8, 60, 80), (15, 20)),  # Non-square input to smaller non-square output
    # Test 1D output size
    ((4, 3, 32, 32), 8),  # Same output size for both dimensions
    # Large case
    ((1, 64, 224, 224), (7, 7)),  # Typical image classification case
    # Edge cases
    ((2, 4, 10, 10), (1, 5)),  # Different scaling for different dimensions
    ((4, 2, 50, 100), (25, 25)),  # 2x down one dimension, 4x down other
    # Non-divisible input/output ratios (windows follow ATen's ceil end rule)
    ((2, 3, 10, 10), (3, 3)),  # in=10, out=3: per-window ceil end
    ((2, 3, 5, 5), (3, 3)),  # in=5, out=3: middle window longer than cdiv
    ((1, 4, 224, 224), (13, 13)),  # Image classification-like, non-divisible
    ((4, 3, 17, 17), (5, 5)),  # Non-divisible both dimensions
    ((2, 3, 22, 33), (7, 9)),  # Non-square, non-divisible both dimensions
]


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVGPOOL2D_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_accuracy_adaptive_avg_pool2d_forward(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    ref_out = torch.ops.aten._adaptive_avg_pool2d(ref_inp, output_size)
    res_out = flag_gems.adaptive_avg_pool2d(inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "strided", "channels_last"])
@pytest.mark.parametrize(
    "shape, output_size",
    [
        ((1, 2, 130, 259), (1, 1)),
        ((1, 2, 225, 257), (3, 2)),
        ((2, 3, 3, 5), (7, 9)),
        ((2, 3, 1, 1), (2, 3)),
        ((0, 3, 17, 19), (3, 5)),
        ((1, 2, 65, 67), (2, 3)),
    ],
)
def test_accuracy_adaptive_avg_pool2d_boundary_layout(
    shape, output_size, layout, dtype
):
    if layout == "strided":
        storage_shape = (*shape[:-1], shape[-1] * 2)
        inp = torch.randn(storage_shape, dtype=dtype, device=flag_gems.device)[..., ::2]
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        if layout == "channels_last":
            if flag_gems.vendor_name == "ascend":
                inp = inp.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
            else:
                inp = inp.contiguous(memory_format=torch.channels_last)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten._adaptive_avg_pool2d(ref_inp, output_size)
    actual = flag_gems.adaptive_avg_pool2d(inp, output_size)
    utils.gems_assert_close(actual, ref_out, dtype)


@pytest.mark.adaptive_avg_pool2d_out
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVGPOOL2D_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_accuracy_adaptive_avg_pool2d_out(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    ref_out = torch.empty(
        (*shape[:2], *output_size), dtype=ref_inp.dtype, device=ref_inp.device
    )
    torch.ops.aten._adaptive_avg_pool2d.out(ref_inp, output_size, out=ref_out)

    res_out = torch.empty(
        (*shape[:2], *output_size), dtype=dtype, device=flag_gems.device
    )
    returned = flag_gems.adaptive_avg_pool2d_out(inp, output_size, out=res_out)

    # The out variant must fill and return the caller's buffer, not a new one.
    assert returned is res_out
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool2d_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("out_layout", ["contiguous", "strided", "channels_last"])
@pytest.mark.parametrize(
    "shape, output_size",
    [
        ((1, 2, 130, 259), (1, 1)),
        ((2, 3, 3, 5), (7, 9)),
        ((1, 2, 65, 67), (2, 3)),
        ((4, 3, 17, 17), (5, 5)),
    ],
)
def test_accuracy_adaptive_avg_pool2d_out_layout(shape, output_size, out_layout, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (*shape[:2], *output_size)
    # ATen accepts a non-contiguous out tensor, so the kernel must not assume a
    # flat layout when scattering results back.
    if out_layout == "strided":
        storage_shape = (*out_shape[:-1], out_shape[-1] * 2)
        res_out = torch.empty(storage_shape, dtype=dtype, device=flag_gems.device)[
            ..., ::2
        ]
    elif out_layout == "channels_last":
        res_out = torch.empty(
            out_shape, dtype=dtype, device=flag_gems.device
        ).contiguous(memory_format=torch.channels_last)
    else:
        res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._adaptive_avg_pool2d.out(ref_inp, output_size, out=ref_out)

    returned = flag_gems.adaptive_avg_pool2d_out(inp, output_size, out=res_out)

    assert returned is res_out
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool2d_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_avg_pool2d_out_resizes(dtype):
    shape, output_size = (2, 3, 16, 16), (4, 4)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty(
        (*shape[:2], *output_size), dtype=ref_inp.dtype, device=ref_inp.device
    )
    torch.ops.aten._adaptive_avg_pool2d.out(ref_inp, output_size, out=ref_out)

    # ATen resizes a mismatched out tensor rather than raising.
    res_out = torch.empty((1,), dtype=dtype, device=flag_gems.device)
    returned = flag_gems.adaptive_avg_pool2d_out(inp, output_size, out=res_out)

    assert returned is res_out
    assert tuple(res_out.shape) == (*shape[:2], *output_size)
    utils.gems_assert_close(res_out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize(
    "shape", [(3, 33, 33), (2, 4, 32, 64), (8, 16, 64, 64), (32, 64, 128, 256)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
        (4, 0, 4, 0),
    ],
)
def test_reflection_pad2d(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("padding", [[1, 1, 1, 1], [2, 3, 4, 5]])
def test_reflection_pad2d_list_padding(padding):
    # Test with list format: [pad_left, pad_right, pad_top, pad_bottom]
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
def test_reflection_pad2d_empty_padding():
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    padding = (0, 0, 0, 0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("padding", [(1, 1, 1, 1), (2, 3, 4, 5)])
def test_reflection_pad2d_3d_input(padding):
    # Test with 3D input (C, H, W) - no batch dimension
    shape = (3, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)

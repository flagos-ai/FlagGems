import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 8, 16, 16),
        (1, 3, 16, 32, 32),
        (2, 4, 16, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 2, 3, 1, 1),
        (1, 1, 2, 2, 2, 2),
        (0, 4, 0, 4, 0, 4),
        (4, 0, 4, 0, 4, 0),
    ],
)
def test_reflection_pad3d(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 4, 5, 2, 1),
    ],
)
def test_reflection_pad3d_list_padding(padding):
    shape = (2, 4, 8, 16, 16)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d
def test_reflection_pad3d_empty_padding():
    shape = (2, 4, 8, 16, 16)
    dtype = torch.float32
    padding = (0, 0, 0, 0, 0, 0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 4, 5, 2, 1),
    ],
)
def test_reflection_pad3d_4d_input(padding):
    shape = (3, 8, 16, 16)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)

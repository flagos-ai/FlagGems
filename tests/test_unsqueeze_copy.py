import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unsqueeze_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unsqueeze_copy(shape, dtype, dim):
    if len(shape) == 0:
        pytest.skip("skip scalar")

    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.unsqueeze_copy(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = flag_gems.unsqueeze_copy(res_inp, dim)

    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
    )

    # unsqueeze_copy must allocate new storage
    assert res_out.data_ptr() != res_inp.data_ptr()


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), -1),
        ((16, 32, 64), 0),
        ((16, 32, 64), 2),
        ((1024, 1024), 1),
    ],
)
def test_unsqueeze_copy_npu_semantics(shape, dim):
    """
    Validate Ascend/NPU unsqueeze_copy behavior.

    Checks:
    1. Correct output shape
    2. Output remains on NPU
    3. Output does not share storage with input
    4. Modifying input does not modify output
    """

    if not torch.npu.is_available():
        pytest.skip("NPU is not available")

    inp = torch.randn(
        shape,
        device="npu",
    )

    with flag_gems.use_gems():
        out = flag_gems.unsqueeze_copy(inp, dim)

    # shape check
    expected_shape = list(shape)

    if dim < 0:
        dim = dim + len(shape) + 1

    expected_shape.insert(dim, 1)

    assert list(out.shape) == expected_shape

    # device check
    assert out.device.type == "npu"

    # storage check
    assert (
        inp.untyped_storage().data_ptr()
        != out.untyped_storage().data_ptr()
    )

    # content check
    expected = out.clone()

    inp.fill_(100)

    torch.npu.synchronize()

    assert torch.equal(
        out.cpu(),
        expected.cpu(),
    )


def test_unsqueeze_copy_storage():

    inp = torch.randn(
        (2, 3),
        device=flag_gems.device,
    )

    with flag_gems.use_gems():
        out = flag_gems.unsqueeze_copy(inp, 1)

    # unsqueeze_copy must allocate new storage
    assert out.data_ptr() != inp.data_ptr()


def test_unsqueeze_copy_no_alias():

    inp = torch.randn(
        (2, 3),
        device=flag_gems.device,
    )

    with flag_gems.use_gems():
        out = flag_gems.unsqueeze_copy(inp, 1)

    out_before = out.clone()

    inp[0, 0] = 100

    # output should not change after modifying input
    assert torch.equal(out, out_before)

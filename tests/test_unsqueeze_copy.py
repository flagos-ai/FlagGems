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

    utils.gems_assert_close(res_out, ref_out, dtype)

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
    Validate Ascend/NPU specific semantics.

    Checks:
    1. output shape
    2. output device
    3. output has independent storage
    4. input modification does not affect output
    """

    # Only run on Ascend environment
    if not hasattr(torch, "npu"):
        pytest.skip("NPU test only")

    if not torch.npu.is_available():
        pytest.skip("NPU unavailable")

    inp = torch.randn(
        shape,
        device="npu",
        dtype=torch.float32,
    )

    with flag_gems.use_gems():
        out = flag_gems.unsqueeze_copy(inp, dim)

    ref = torch.ops.aten.unsqueeze_copy(inp, dim)

    # shape correctness
    assert out.shape == ref.shape

    # device correctness
    assert out.device.type == "npu"

    # storage independence
    assert out.data_ptr() != inp.data_ptr()

    # verify copy semantics
    out_clone = out.clone()

    inp.fill_(0)

    assert torch.equal(out, out_clone)

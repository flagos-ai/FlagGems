import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.set_
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096), (64, 512, 512)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_set_source_tensor(shape, dtype):
    """Test set_.source_Tensor: share storage with source."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    source = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_source = utils.to_reference(source)

    ref_inp.set_(ref_source)
    with flag_gems.use_gems():
        inp.set_(source)

    utils.gems_assert_close(utils.to_reference(inp), utils.to_reference(source), dtype)


@pytest.mark.set_
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (4096, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_set_source_tensor_storage_offset(shape, dtype):
    """Test set_.source_Tensor_storage_offset: set from tensor with explicit metadata."""
    source = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.empty(1, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_source = utils.to_reference(source)

    size = list(source.size())
    stride = list(source.stride())
    offset = source.storage_offset()

    ref_inp.set_(ref_source, offset, size, stride)
    with flag_gems.use_gems():
        inp.set_(source, offset, size, stride)

    utils.gems_assert_close(utils.to_reference(inp), utils.to_reference(source), dtype)


@pytest.mark.set_
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_set_default(dtype):
    """Test set_ (default): reset tensor to empty."""
    inp = torch.randn(64, 64, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_inp.set_()
    with flag_gems.use_gems():
        inp.set_()

    assert inp.numel() == 0, f"Expected 0 elements, got {inp.numel()}"
    assert ref_inp.numel() == 0

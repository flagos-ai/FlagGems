import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

device = flag_gems.device


@pytest.mark.normal_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_normal_(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    if flag_gems.vendor_name in ["metax", "iluvatar"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    loc = 3.0
    scale = 10.0
    res_out = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out.normal_(loc, scale)

    ref_out = utils.to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.normal_float_tensor
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_normal_float_tensor(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    loc = 3.0
    scale = torch.full(size=shape, fill_value=10.0, dtype=dtype, device=device)

    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)

    ref_out = utils.to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.normal_tensor_float
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_normal_tensor_float(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    loc = torch.full(size=shape, fill_value=3.0, dtype=dtype, device=device)
    scale = 10.0
    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)

    ref_out = utils.to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1


@pytest.mark.normal_tensor_tensor
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_normal_tensor_tensor(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    if flag_gems.vendor_name in ["metax", "iluvatar", "kunlunxin"]:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    loc = torch.full(size=shape, fill_value=3.0, dtype=dtype, device=device)
    scale = torch.full(size=shape, fill_value=10.0, dtype=dtype, device=device)

    with flag_gems.use_gems():
        res_out = torch.normal(loc, scale)

    ref_out = utils.to_reference(res_out)
    mean = torch.mean(ref_out)
    std = torch.std(ref_out)

    assert torch.abs(mean - 3.0) < 0.1
    assert torch.abs(std - 10.0) < 0.1

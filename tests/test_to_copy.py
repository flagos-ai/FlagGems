import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.COMPLEX_DTYPES,
)
def test_to_dtype(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = ref_x.to(dtype)
    with flag_gems.use_gems():
        out = x.to(dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("target_dtype", utils.ALL_FLOAT_DTYPES + utils.COMPLEX_DTYPES)
def test_to_copy_dtype_cast(shape, target_dtype):
    if flag_gems.vendor_name == "tsingmicro" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    src_dtype = torch.float32 if target_dtype != torch.float32 else torch.float16
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=target_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=target_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format],
)
def test_to_copy_preserve_strides(memory_format):
    base = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    x = base.transpose(0, 1)[::2]
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(
        ref_x,
        dtype=ref_x.dtype,
        memory_format=memory_format,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(
            x,
            dtype=x.dtype,
            memory_format=memory_format,
        )
    utils.gems_assert_equal(res_out, ref_out)
    if memory_format is torch.preserve_format:
        assert res_out.stride() == ref_out.stride()
    else:
        assert res_out.is_contiguous()

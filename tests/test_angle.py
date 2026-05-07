import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.angle
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.COMPLEX_DTYPES + utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.BOOL_TYPES,
)
def test_angle(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if dtype in utils.BOOL_TYPES:
        inp = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.ALL_INT_DTYPES:
        inp = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    elif dtype in utils.COMPLEX_DTYPES + utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu").to(flag_gems.device)

    ref_inp = utils.to_reference(inp)

    try:
        ref_out = torch.angle(ref_inp)
    except RuntimeError as e:
        if "angle_cpu" in str(e) and "ComplexHalf" in str(e):
            pytest.skip("Skipping angle ComplexHalf for unsupported dtype on CPU")
        elif "angle_cuda" in str(e) and "Half" in str(e):
            pytest.skip("Skipping angle Half for unsupported dtype on GPU")
        elif "angle_cuda" in str(e) and "BFloat16" in str(e):
            pytest.skip("Skipping angle BFloat16 for unsupported dtype on GPU")
        else:
            raise

    ref_out = torch.angle(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.angle(inp)

    dtype_out = res_out.dtype
    utils.gems_assert_close(res_out, ref_out, dtype_out)

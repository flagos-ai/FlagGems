import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.chalf
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chalf_real(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = ref_inp.chalf()
    with flag_gems.use_gems():
        res_out = res_inp.chalf()

    assert res_out.dtype == torch.complex32
    # Compare via view_as_real (complex32 -> float16 pairs). Leave ref_out on the
    # reference device so gems_assert_close's TO_CPU handling stays consistent.
    utils.gems_assert_close(
        torch.view_as_real(res_out),
        torch.view_as_real(ref_out),
        torch.float16,
    )


@pytest.mark.chalf
# Small representative shapes covering 1D/2D/3D + a large 2D case for the complex path.
@pytest.mark.parametrize("shape", [(256,), (32, 64), (2, 3, 4), (1024, 1024)])
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_chalf_complex(shape, dtype):
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    res_inp = torch.complex(real, imag).to(dtype)
    ref_inp = utils.to_reference(res_inp)

    ref_out = ref_inp.chalf()
    with flag_gems.use_gems():
        res_out = res_inp.chalf()

    assert res_out.dtype == torch.complex32
    utils.gems_assert_close(
        torch.view_as_real(res_out),
        torch.view_as_real(ref_out),
        torch.float16,
    )

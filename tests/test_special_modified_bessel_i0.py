import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_i0
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_i0(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # PyTorch's special.modified_bessel_i0 only supports float32 on CUDA,
    # so compute reference on CPU in float32 to avoid dtype limitations
    ref_inp_cpu = ref_inp.to("cpu").float()
    ref_out = torch.special.modified_bessel_i0(ref_inp_cpu)
    # Move back to original device for comparison
    ref_out = ref_out.to(flag_gems.device)

    with flag_gems.use_gems():
        res_out = torch.special.modified_bessel_i0(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_modified_bessel_i0_out
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_modified_bessel_i0_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)

    # PyTorch's special.modified_bessel_i0 only supports float32 on CUDA,
    # so compute reference on CPU in float32 to avoid dtype limitations
    ref_inp_cpu = ref_inp.to("cpu").float()
    ref_out_cpu = torch.empty_like(ref_inp_cpu)
    torch.special.modified_bessel_i0(ref_inp_cpu, out=ref_out_cpu)
    ref_out = ref_out_cpu.to(flag_gems.device)

    with flag_gems.use_gems():
        torch.special.modified_bessel_i0(inp, out=out)

    utils.gems_assert_close(out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", [(256,), (32, 64), (2, 3, 4)])
@pytest.mark.parametrize("is_complex", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_conj_physical(shape, is_complex, dtype):
    device = flag_gems.device

    if is_complex:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        input = torch.complex(real, imag)
        out_dtype = input.dtype
    else:
        input = torch.randn(shape, dtype=dtype, device=device)
        out_dtype = dtype

    ref_input = utils.to_reference(input, True)
    ref_out = torch.conj_physical(ref_input)
    with flag_gems.use_gems():
        res_out = torch.conj_physical(input)

    utils.gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=1)

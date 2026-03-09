import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.parametrize("shape", [(1,), (1024,), (1024, 1024), (4, 1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_log10(shape, dtype):
    # log10 requires positive input
    x = torch.rand(shape, dtype=dtype, device="cuda") + 0.01
    ref_inp = to_reference(x)
    ref_out = torch.log10(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.log10(x)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(1024,), (1024, 1024), (4, 1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_log10_(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device="cuda") + 0.01
    ref = torch.log10(x.float()).to(dtype)
    with flag_gems.use_gems():
        torch.log10_(x)
    gems_assert_close(x, ref, dtype)

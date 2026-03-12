import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.parametrize(
    "shape", [(1,), (1024,), (1024, 1024), (4, 1024, 1024), (2, 4, 256, 256)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_cosh(shape, dtype):
    # keep values small to avoid fp16 overflow (cosh grows fast)
    x = torch.randn(shape, dtype=dtype, device="cuda") * 5
    ref_inp = to_reference(x)
    ref_out = torch.cosh(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.cosh(x)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(1024,), (1024, 1024), (4, 1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_cosh_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda") * 5
    ref = torch.cosh(x.float()).to(dtype)
    with flag_gems.use_gems():
        torch.cosh_(x)
    gems_assert_close(x, ref, dtype)

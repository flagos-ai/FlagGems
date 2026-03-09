import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.parametrize(
    "shape", [(1,), (1024,), (1024, 1024), (4, 1024, 1024), (2, 4, 256, 256)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.2])
def test_accuracy_leaky_relu(shape, dtype, negative_slope):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref_inp = to_reference(x)
    ref_out = torch.nn.functional.leaky_relu(ref_inp, negative_slope=negative_slope)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", [(1024,), (1024, 1024), (4, 1024, 1024)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_leaky_relu_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch.nn.functional.leaky_relu(x.float()).to(dtype)
    with flag_gems.use_gems():
        torch.nn.functional.leaky_relu_(x)
    gems_assert_close(x, ref, dtype)

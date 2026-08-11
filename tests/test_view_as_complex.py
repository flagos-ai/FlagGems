import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import gems_assert_close


@pytest.mark.view_as_complex
@pytest.mark.parametrize("shape", [(10, 2), (100, 2), (1000, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_view_as_complex_accuracy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.view_as_complex(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.view_as_complex(inp)
    # Pass output dtype for complex tensors, not input dtype
    output_dtype = res_out.dtype
    gems_assert_close(res_out, ref_out, output_dtype)


@pytest.mark.view_as_complex
@pytest.mark.parametrize("shape", [(5, 10, 2), (20, 30, 2), (100, 50, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_view_as_complex_2d_accuracy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.view_as_complex(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.view_as_complex(inp)
    output_dtype = res_out.dtype
    gems_assert_close(res_out, ref_out, output_dtype)


@pytest.mark.view_as_complex
@pytest.mark.parametrize("shape", [(4, 8, 16, 2), (10, 20, 30, 2)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_view_as_complex_3d_accuracy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.view_as_complex(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.view_as_complex(inp)
    output_dtype = res_out.dtype
    gems_assert_close(res_out, ref_out, output_dtype)

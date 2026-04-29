import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference

HISTC_SHAPES = [(64,), (1024,), (4096,), (100, 100), (32, 64, 16)]
HISTC_BINS = [10, 50, 100]
HISTC_DTYPES = [torch.float32]


@pytest.mark.histc
@pytest.mark.parametrize("shape", HISTC_SHAPES)
@pytest.mark.parametrize("bins", HISTC_BINS)
@pytest.mark.parametrize("dtype", HISTC_DTYPES)
def test_accuracy_histc(shape, bins, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10
    ref_inp = to_reference(inp)
    ref_out = torch.histc(ref_inp, bins=bins, min=0, max=0)
    with flag_gems.use_gems():
        res_out = torch.histc(inp, bins=bins, min=0, max=0)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.histc
@pytest.mark.parametrize("shape", HISTC_SHAPES)
@pytest.mark.parametrize("bins", HISTC_BINS)
@pytest.mark.parametrize("dtype", HISTC_DTYPES)
def test_accuracy_histc_with_range(shape, bins, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 20 - 5
    ref_inp = to_reference(inp)
    ref_out = torch.histc(ref_inp, bins=bins, min=0, max=10)
    with flag_gems.use_gems():
        res_out = torch.histc(inp, bins=bins, min=0, max=10)
    gems_assert_close(res_out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.swish
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swish(shape, dtype):
    # Swish is mathematically equivalent to SiLU: x * sigmoid(x)
    # PyTorch exposes this as torch.nn.functional.silu
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    # Reference uses torch.nn.functional.silu (which is Swish)
    ref_out = torch.nn.functional.silu(ref_inp)
    # Our implementation is registered for aten.silu
    with flag_gems.use_gems():
        res_out = torch.ops.aten.silu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

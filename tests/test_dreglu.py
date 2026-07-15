import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "dreglu", None)
except ImportError:
    TE_OP = None


@pytest.mark.dreglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(TE_OP is None, reason="'dreglu' not found in TransformerEngine")
def test_dreglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )

    ref_out = TE_OP(grad_output, input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.dreglu(grad_output, input_tensor, None)
    utils.gems_assert_close(res_out, ref_out, dtype)

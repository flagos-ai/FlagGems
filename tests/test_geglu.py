import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_OP = getattr(tex, "geglu", None)
except ImportError:
    TE_OP = None


@pytest.mark.geglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(TE_OP is None, reason="'geglu' not found in TransformerEngine")
def test_geglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = TE_OP(input_tensor, None)
    ref_out = utils.to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = flag_gems.geglu(input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


def te_reglu_ref(input_tensor):
    # TransformerEngine's reglu kernel on this backend only accepts 2-D inputs
    # (its cast_gated kernel asserts input.data.shape.size() == 2), so collapse
    # all leading dims before calling it and restore the output shape afterwards.
    shape = input_tensor.shape
    last_dim = shape[-1]
    input_2d = input_tensor.contiguous().view(-1, last_dim)
    ref_out = tex.reglu(input_2d, None)
    return ref_out.view(*shape[:-1], last_dim // 2)


@pytest.mark.reglu
@pytest.mark.parametrize("shape", utils.GLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(not TE_AVAILABLE, reason="transformer engine is not available")
def test_reglu(shape, dtype):
    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = te_reglu_ref(input_tensor)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.reglu(input_tensor)

    utils.gems_assert_close(res_out, ref_out, dtype)

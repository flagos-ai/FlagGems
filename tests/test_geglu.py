import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


@pytest.mark.geglu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is required")
def test_geglu(shape, dtype):
    if len(shape) == 0:
        # GEGLU does not support 0-dim scalar tensors.
        return

    if shape[-1] % 2 != 0:
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = tex.geglu(input_tensor, None)
    ref_out = utils.to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = flag_gems.geglu(input_tensor)
    utils.gems_assert_close(res_out, ref_out, dtype)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


REGLU_SHAPES = [
    (),
    (2,),
    (512, 512),
    (1, 2048),
    (2048, 2),
    (1024, 1024),
    (20, 320, 16),
    (4096, 1024),
    (2048, 2048),
    (1024, 4096),
    (512, 512, 512),
    (512, 256, 512),
]


@pytest.mark.reglu
@pytest.mark.parametrize("shape", REGLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(not TE_AVAILABLE, reason="transformer engine is not available")
def test_reglu(shape, dtype):
    if len(shape) == 0:
        pytest.skip("reglu does not support 0-dim scalar tensors.")

    if shape[-1] % 2 != 0:
        pytest.skip(
            f"reglu requires the last dimension to be even, but got shape {shape}."
        )

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = tex.reglu(input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.reglu(input_tensor)

    utils.gems_assert_close(res_out, ref_out, dtype)

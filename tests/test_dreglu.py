import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

try:
    from transformer_engine.pytorch import cpp_extensions as tex

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


DREGLU_SHAPES = [
    (),
    (1,),
    (512, 512),
    (1, 2048),
    (2048, 1),
    (1024, 1024),
    (20, 320, 15),
    (4096, 1024),
    (2048, 2048),
    (1024, 4096),
    (512, 512, 512),
    (512, 256, 512),
]


@pytest.mark.dreglu
@pytest.mark.parametrize("shape", DREGLU_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(not TE_AVAILABLE, reason="TransformerEngine is required")
def test_dreglu(shape, dtype):
    if len(shape) == 0:
        # dreglu does not support 0-dim scalar tensors.
        return

    if shape[-1] % 2 != 0:
        shape = list(shape)
        shape[-1] += 1
        shape = tuple(shape)

    input_tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    grad_output_shape = list(shape)
    grad_output_shape[-1] //= 2
    grad_output = torch.randn(
        tuple(grad_output_shape), dtype=dtype, device=flag_gems.device
    )

    ref_out = tex.dreglu(grad_output, input_tensor, None)
    ref_out = utils.to_reference(ref_out)
    with flag_gems.use_gems():
        res_out = flag_gems.dreglu(grad_output, input_tensor, None)
    utils.gems_assert_close(res_out, ref_out, dtype)

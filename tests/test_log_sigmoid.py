import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SPECIAL_VALUES = [float("-inf"), float("inf"), -300]


@pytest.mark.log_sigmoid
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_sigmoid(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if len(shape) == 1:
        special_inputs = torch.tensor(
            SPECIAL_VALUES, dtype=dtype, device=flag_gems.device
        )
        inp = torch.cat((inp, special_inputs))
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.logsigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.logsigmoid(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)

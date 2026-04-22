import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)

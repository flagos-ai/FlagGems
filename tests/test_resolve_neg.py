import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.resolve_neg
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_accuracy_resolve_neg(shape, dtype):
    if flag_gems.vendor_name == "ascend":
        x = torch.randn(size=shape, dtype=dtype).to(device=flag_gems.device)
    else:
        x = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    y = x.conj()
    z = y.imag
    assert z.is_neg()

    with flag_gems.use_gems():
        out = z.resolve_neg()
    assert not out.is_neg()

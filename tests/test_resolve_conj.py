import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.resolve_conj
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", [torch.cfloat])
def test_resolve_conj(shape, dtype):
    x = torch.randn(size=shape, dtype=dtype, device="cpu")
    y = x.conj()

    assert y.is_conj()

    with flag_gems.use_gems():
        res_y = y.to(device=flag_gems.device)
        z = res_y.resolve_conj()

    assert not z.is_conj()

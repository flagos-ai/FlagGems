import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Dedicated shapes that exercise squeeze_copy semantics (removing size-1
# dimensions). The generic POINTWISE_SHAPES only has one case that actually
# triggers a squeeze, so these explicitly cover interleaved/leading/trailing
# size-1 dims as well as the no-op copy and 0-d scalar cases.
SQUEEZE_COPY_SHAPES = [
    (),  # 0-d scalar, nothing to squeeze
    (1,),  # all size-1 dims -> 0-d
    (1, 3, 1, 4),  # interleaved size-1 dims -> (3, 4)
    (1, 1, 5),  # leading size-1 dims -> (5,)
    (8, 1, 1),  # trailing size-1 dims -> (8,)
    (1024, 1024),  # no size-1 dim, no-op copy
]


@pytest.mark.squeeze_copy
@pytest.mark.parametrize("shape", SQUEEZE_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_squeeze_copy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.squeeze_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.squeeze_copy(inp)
    utils.gems_assert_equal(res_out, ref_out)

import pytest
import torch

import flag_gems
from flag_gems.ops.cdist import _cdist_forward

from . import accuracy_utils as utils

# Shapes for cdist test: (P, M), (R, M) -> (P, R)
# Note: torch.cdist doesn't support float16 on CUDA
CDIST_FORWARD_SHAPES = [
    ((4, 8), (6, 8)),
    ((8, 16), (8, 16)),
    ((16, 32), (16, 32)),
    ((32, 64), (32, 64)),
    ((1, 8), (1, 8)),
    ((2, 3, 8), (2, 4, 8)),  # batch dimension
    ((5, 3, 4), (3, 5, 2, 4)),  # unequal batch dims broadcast (5,) vs (3,5) -> (3,5)
]


@pytest.mark.cdist_forward
@pytest.mark.parametrize("shapes", CDIST_FORWARD_SHAPES)
# torch.cdist doesn't support float16 on CUDA; only float32 is numerically stable
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cdist_forward(shapes, dtype):
    shape1, shape2 = shapes
    x1 = torch.randn(*shape1, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(*shape2, dtype=dtype, device=flag_gems.device)

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)

    ref_out = torch.cdist(ref_x1, ref_x2, p=2.0)

    with flag_gems.use_gems():
        res_out = _cdist_forward(x1, x2, p=2.0)

    # cdist L2 computation accumulates error over feature dimension (M up to 128);
    # atol=0.01 is sufficient for float32 comparison
    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.01)

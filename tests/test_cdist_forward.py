import pytest
import torch

import flag_gems

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
]


@pytest.mark.cdist_forward
@pytest.mark.parametrize("shapes", CDIST_FORWARD_SHAPES)
# torch.cdist doesn't support float16 on CUDA; only float32 is numerically stable
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cdist_forward(shapes, dtype):
    shape1, shape2 = shapes
    x1 = torch.randn(*shape1, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(*shape2, dtype=dtype, device=flag_gems.device)

    # Skip if feature dimensions don't match
    if shape1[-1] != shape2[-1]:
        pytest.skip("Feature dimensions must match")

    ref_x1 = utils.to_reference(x1)
    ref_x2 = utils.to_reference(x2)

    ref_out = torch.cdist(ref_x1, ref_x2, p=2.0)

    # Import metax specialized version and call it directly
    from flag_gems.runtime.backend._metax.ops import _cdist_forward

    with flag_gems.use_gems():
        res_out = _cdist_forward(x1, x2, p=2.0)

    utils.gems_assert_close(res_out, ref_out, dtype)

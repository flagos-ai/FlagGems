import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Each case: (shape_a, shape_b, dims_a, dims_b). Contracted sizes must match.
TENSORDOT_CASES = [
    # Classic dims=2 style contraction (last d of a, first d of b).
    ((3, 4, 5), (4, 5, 6), [1, 2], [0, 1]),
    # Single contracted dim -> plain matmul-like.
    ((16, 32), (32, 24), [1], [0]),
    # Reordered / non-adjacent contracted dims.
    ((3, 5, 4, 6), (6, 4, 5, 3), [2, 1, 3], [1, 2, 0]),
    # Negative dim indices.
    ((8, 7, 9), (9, 7, 5), [-1, 1], [0, 1]),
    # Larger contraction dimension.
    ((64, 128), (128, 96), [1], [0]),
    # Outer product (no contracted dims).
    ((2, 3), (4, 5), [], []),
]


def _reduce_dim(shape_a, dims_a):
    k = 1
    for d in dims_a:
        k *= shape_a[d % len(shape_a)]
    return max(k, 1)


@pytest.mark.tensordot
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", TENSORDOT_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensordot(shape_a, shape_b, dims_a, dims_b, dtype):
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b))
    res_out = flag_gems.ops.tensordot(a, b, dims_a, dims_b)

    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape_a, dims_a)
    )


@pytest.mark.tensordot_out
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", TENSORDOT_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensordot_out(shape_a, shape_b, dims_a, dims_b, dtype):
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b))
    out = torch.empty_like(ref_out, dtype=dtype, device=flag_gems.device)
    flag_gems.ops.tensordot_out(a, b, dims_a, dims_b, out=out)

    utils.gems_assert_close(
        out, ref_out, dtype, reduce_dim=_reduce_dim(shape_a, dims_a)
    )

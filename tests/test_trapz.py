import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(128,), (64, 128), (16, 32, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dx", [1.0, 0.5, 2.0])
@pytest.mark.parametrize("dim", [-1, 0])
def test_trapz(shape, dtype, dx, dim):
    # Skip invalid dim for shape
    if dim >= len(shape) or dim < -len(shape):
        pytest.skip("Invalid dim for shape")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=dx, dim=dim)

    res_out = flag_gems.trapz(inp, dx=dx, dim=dim)

    # Trapezoid is a reduction; scale atol by the reduced dimension length
    reduce_dim = shape[dim]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.trapz
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_edge_cases(dtype):
    """Test edge cases: single element, two elements."""
    # Single element - should return zero
    inp = torch.randn(1, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=1.0)

    res_out = flag_gems.trapz(inp, dx=1.0)
    utils.gems_assert_close(res_out, ref_out, dtype)

    # Two elements
    inp = torch.randn(2, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.trapezoid(ref_inp, dx=1.0)

    res_out = flag_gems.trapz(inp, dx=1.0)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=2)


@pytest.mark.trapz
@pytest.mark.parametrize("shape", [(1024, 1024), (512, 2048)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_trapz_large_reduction(shape, dtype):
    """Test trapezoid with large reduction dimension."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.trapezoid(ref_inp, dx=0.01, dim=-1)

    res_out = flag_gems.trapz(inp, dx=0.01, dim=-1)

    # Large reduction accumulates more error
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])

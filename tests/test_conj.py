import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

@pytest.mark.conj
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_complex(shape, dtype):
    """Test conj on complex tensors: accuracy and view semantics."""
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    # Accuracy: conj should match torch.conj
    ref_out = torch.conj(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.conj(inp)

    utils.gems_assert_equal(res_out, ref_out)

    # View semantics: complex conj must share storage (zero copy)
    assert res_out.is_conj(), "Complex conj output must have is_conj() == True"
    if inp.numel() > 0:
        assert inp.data_ptr() == res_out.data_ptr(), (
            "conj must return a view sharing the same underlying storage"
        )

@pytest.mark.conj
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conj_real(shape, dtype):
    """Test conj on real tensors: identity (no-op)."""
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.conj(inp)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.data_ptr() == inp.data_ptr(), (
        "Real conj must return the input tensor itself (identity)"
    )

@pytest.mark.conj
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj_resolve_consistency(shape, dtype):
    """
    Verify that resolving the conj view produced by our implementation
    yields the same physical tensor as the reference.
    """
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    with flag_gems.use_gems():
        res_conj = torch.conj(inp)
        res_physical = res_conj.resolve_conj()

    ref_conj = torch.conj(ref_inp)
    ref_physical = ref_conj.resolve_conj()

    utils.gems_assert_equal(res_physical, ref_physical)

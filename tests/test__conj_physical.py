import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Core shapes exercised by CONJ_PHYSICAL_SHAPES (mirrors worktree CI branch).
CONJ_PHYSICAL_SHAPES = [(256,), (32, 64), (2, 3, 4)]


def _make_complex(shape, dtype, device):
    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float16
    real = torch.randn(shape, dtype=float_dtype, device=device)
    imag = torch.randn(shape, dtype=float_dtype, device=device)
    return torch.complex(real, imag).to(dtype)


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize("shape", CONJ_PHYSICAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test__conj_physical(shape, dtype):
    device = flag_gems.device
    inp = _make_complex(shape, dtype, device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch._conj_physical(ref_inp)
    res_out = flag_gems._conj_physical(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.underscore_conj_physical
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__conj_physical_real(shape, dtype):
    # For non-complex dtypes, _conj_physical is the identity.
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch._conj_physical(ref_inp)
    res_out = flag_gems._conj_physical(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.underscore_conj_physical_out
@pytest.mark.parametrize("shape", CONJ_PHYSICAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test__conj_physical_out(shape, dtype):
    device = flag_gems.device
    inp = _make_complex(shape, dtype, device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten._conj_physical.out(ref_inp, out=ref_out)

    res_out = torch.empty_like(inp)
    flag_gems._conj_physical_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ORGQR_SHAPES = [(3, 0), (4, 3), (8, 5), (16, 8), (32, 16), (64, 32), (128, 64)]
ORGQR_BATCH_SHAPES = [(0, 4, 3), (2, 8, 5), (2, 3, 16, 8)]
ORGQR_DTYPES = [torch.float32, torch.float64]


def make_reflectors(shape, dtype):
    matrix = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    return torch.geqrf(matrix)


@pytest.mark.orgqr
@pytest.mark.parametrize("shape", ORGQR_SHAPES + ORGQR_BATCH_SHAPES)
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
def test_accuracy_orgqr(shape, dtype):
    input, tau = make_reflectors(shape, dtype)
    ref = torch.orgqr(utils.to_reference(input), utils.to_reference(tau))
    result = flag_gems.orgqr(input, tau)
    utils.gems_assert_close(result, ref, dtype)


@pytest.mark.orgqr
@pytest.mark.parametrize("k", [0, 2, 5])
def test_orgqr_k_cases(k):
    input, tau = make_reflectors((7, 5), torch.float32)
    tau = tau[:k]
    ref = torch.orgqr(utils.to_reference(input), utils.to_reference(tau))
    result = flag_gems.orgqr(input, tau)
    utils.gems_assert_close(result, ref, torch.float32)


@pytest.mark.orgqr
def test_orgqr_invalid_shapes():
    with pytest.raises(RuntimeError):
        flag_gems.orgqr(
            torch.randn((2, 3), device=flag_gems.device),
            torch.randn((2,), device=flag_gems.device),
        )
    with pytest.raises(RuntimeError):
        flag_gems.orgqr(
            torch.randn((4, 3), device=flag_gems.device),
            torch.randn((4,), device=flag_gems.device),
        )


@pytest.mark.orgqr
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_orgqr_complex_is_explicitly_unsupported(dtype):
    input, tau = make_reflectors((4, 3), dtype)
    assert torch.orgqr(input, tau).dtype == dtype
    with pytest.raises(RuntimeError, match="complex dtypes are not supported"):
        flag_gems.orgqr(input, tau)


@pytest.mark.orgqr_out
@pytest.mark.parametrize("shape", ORGQR_SHAPES)
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
def test_accuracy_orgqr_out(shape, dtype):
    input, tau = make_reflectors(shape, dtype)
    ref_input = utils.to_reference(input)
    ref_tau = utils.to_reference(tau)
    ref_out = torch.empty(0, dtype=dtype, device=ref_input.device)
    ref = torch.orgqr(ref_input, ref_tau, out=ref_out)
    out = torch.empty(0, dtype=dtype, device=flag_gems.device)
    result = flag_gems.orgqr_out(input, tau, out=out)
    assert result is out
    assert ref is ref_out
    assert out.shape == input.shape
    utils.gems_assert_close(out, ref, dtype)


@pytest.mark.orgqr_out
def test_orgqr_out_stride_alias_dtype_and_device_contract():
    input, tau = make_reflectors((7, 5), torch.float32)
    ref_input = utils.to_reference(input)
    ref_tau = utils.to_reference(tau)

    out = torch.empty((5, 7), device=flag_gems.device).T
    ref_out = torch.empty((5, 7), device=ref_input.device).T
    result = flag_gems.orgqr_out(input, tau, out=out)
    ref = torch.orgqr(ref_input, ref_tau, out=ref_out)
    assert result is out
    assert ref is ref_out
    assert not result.is_contiguous()
    utils.gems_assert_close(result, ref, torch.float32)

    alias_input, alias_tau = make_reflectors((7, 5), torch.float32)
    ref_alias_input = utils.to_reference(alias_input).clone()
    ref_alias_tau = utils.to_reference(alias_tau)
    ref_alias = torch.orgqr(ref_alias_input, ref_alias_tau, out=ref_alias_input)
    result = flag_gems.orgqr_out(alias_input, alias_tau, out=alias_input)
    assert result is alias_input
    utils.gems_assert_close(result, ref_alias, torch.float32)

    wide_out = torch.empty((7, 5), dtype=torch.float64, device=flag_gems.device)
    wide_ref = torch.empty((7, 5), dtype=torch.float64, device=ref_input.device)
    result = flag_gems.orgqr_out(input, tau, out=wide_out)
    ref = torch.orgqr(ref_input, ref_tau, out=wide_ref)
    utils.gems_assert_close(result, ref, torch.float64)

    bad_out = torch.empty((7, 5), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.orgqr(input, tau, out=bad_out)
    with pytest.raises(RuntimeError):
        flag_gems.orgqr_out(input, tau, out=bad_out)

    cpu_out = torch.empty((7, 5), device="cpu")
    with pytest.raises(RuntimeError):
        torch.orgqr(input, tau, out=cpu_out)
    with pytest.raises(RuntimeError):
        flag_gems.orgqr_out(input, tau, out=cpu_out)

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
@pytest.mark.parametrize("dtype", ORGQR_DTYPES)
@pytest.mark.parametrize("layout", ["transposed_batch", "sliced_batch"])
def test_orgqr_out_multiple_noncontiguous_batch_dimensions(dtype, layout):
    input, tau = make_reflectors((2, 3, 7, 5), dtype)
    ref_input = utils.to_reference(input)
    ref_tau = utils.to_reference(tau)

    def make_out(device):
        if layout == "transposed_batch":
            return torch.full(
                (3, 2, 7, 5), float("nan"), dtype=dtype, device=device
            ).transpose(0, 1)
        return torch.full((4, 3, 7, 5), float("nan"), dtype=dtype, device=device)[::2]

    out = make_out(flag_gems.device)
    ref_out = make_out(ref_input.device)
    original_stride = out.stride()
    original_pointer = out.data_ptr()
    assert not out.is_contiguous()
    # These batch layouts cannot be flattened without allocating a copy.
    with pytest.raises(RuntimeError):
        out.view(6, 7, 5)

    reference = torch.orgqr(ref_input, ref_tau, out=ref_out)
    result = flag_gems.orgqr_out(input, tau, out=out)

    assert result is out
    assert out.stride() == original_stride
    assert out.data_ptr() == original_pointer
    utils.gems_assert_close(out, reference, dtype)


@pytest.mark.orgqr_out
@pytest.mark.skipif(not utils.fp64_is_supported, reason="FP64 is not supported")
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("nonflattenable", [False, True])
def test_orgqr_out_casts_only_final_result(output_dtype, nonflattenable):
    matrix = torch.randn((2, 3, 31, 19), dtype=torch.float64, device=flag_gems.device)
    input, tau = torch.geqrf(matrix)
    ref_input, ref_tau = utils.to_reference(input), utils.to_reference(tau)
    if nonflattenable:
        out = torch.empty(
            (3, 2, 31, 19), dtype=output_dtype, device=input.device
        ).transpose(0, 1)
        ref_out = torch.empty(
            (3, 2, 31, 19), dtype=output_dtype, device=ref_input.device
        ).transpose(0, 1)
    else:
        out = torch.empty(matrix.shape, dtype=output_dtype, device=input.device)
        ref_out = torch.empty(matrix.shape, dtype=output_dtype, device=ref_input.device)
    original_pointer, original_stride = out.data_ptr(), out.stride()
    reference = torch.orgqr(ref_input, ref_tau, out=ref_out)
    result = flag_gems.orgqr_out(input, tau, out=out)
    assert result is out and result.data_ptr() == original_pointer
    assert result.stride() == original_stride
    actual, expected = utils.to_cpu(result, reference), reference
    precision = torch.finfo(output_dtype)
    # Allow one final-output ULP, including CPU half-cast double rounding;
    # repeated low-precision reflector updates exceed this bound.
    torch.testing.assert_close(
        actual, expected, rtol=precision.eps, atol=precision.tiny * precision.eps
    )


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

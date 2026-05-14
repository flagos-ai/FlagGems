import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for linalg_eigvals - only 2x2 is currently supported
LINALG_EIGVALS_SHAPES = [
    (2, 2),  # 2x2 single matrix
    (3, 2, 2),  # batch of 3 2x2 matrices
]


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("shape", LINALG_EIGVALS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_linalg_eigvals(shape, dtype):
    # linalg.eigvals supports float, double, cfloat, cdouble
    # For real input, output is complex
    n = shape[-1]
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A)

    ref_out = torch.linalg.eigvals(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.eigvals(A)

    # Eigenvalues are complex, need special comparison
    # Order may differ, so we compare sets
    def assert_eigenvalues_close(res, ref, dtype):
        # Sort by real part then by imaginary part for comparison
        res_sorted = sorted(
            [(x.real, x.imag) for x in res.flatten()], key=lambda x: (x[0], x[1])
        )
        ref_sorted = sorted(
            [(x.real, x.imag) for x in ref.flatten()], key=lambda x: (x[0], x[1])
        )
        assert len(res_sorted) == len(ref_sorted)
        for (res_r, res_i), (ref_r, ref_i) in zip(res_sorted, ref_sorted):
            if dtype == torch.float32:
                assert (
                    abs(res_r - ref_r) < 1e-4
                ), f"Real part mismatch: {res_r} vs {ref_r}"
                assert (
                    abs(res_i - ref_i) < 1e-4
                ), f"Imag part mismatch: {res_i} vs {ref_i}"
            else:
                assert (
                    abs(res_r - ref_r) < 1e-10
                ), f"Real part mismatch: {res_r} vs {ref_r}"
                assert (
                    abs(res_i - ref_i) < 1e-10
                ), f"Imag part mismatch: {res_i} vs {ref_i}"

    assert_eigenvalues_close(res_out, ref_out, dtype)


@pytest.mark.linalg_eigvals
@pytest.mark.skip(
    reason="Triton kernel does not support complex64/complex128 input directly"
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_accuracy_linalg_eigvals_complex(dtype):
    # Test with complex input - only 2x2 supported
    shape = (2, 2)
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_A = to_reference(A)

    ref_out = torch.linalg.eigvals(ref_A)
    with flag_gems.use_gems():
        res_out = torch.linalg.eigvals(A)

    def assert_eigenvalues_close(res, ref, dtype):
        res_sorted = sorted(
            [(x.real, x.imag) for x in res.flatten()], key=lambda x: (x[0], x[1])
        )
        ref_sorted = sorted(
            [(x.real, x.imag) for x in ref.flatten()], key=lambda x: (x[0], x[1])
        )
        assert len(res_sorted) == len(ref_sorted)
        if dtype == torch.complex64:
            tol = 1e-4
        else:
            tol = 1e-10
        for (res_r, res_i), (ref_r, ref_i) in zip(res_sorted, ref_sorted):
            assert abs(res_r - ref_r) < tol, f"Real part mismatch: {res_r} vs {ref_r}"
            assert abs(res_i - ref_i) < tol, f"Imag part mismatch: {res_i} vs {ref_i}"

    assert_eigenvalues_close(res_out, ref_out, dtype)

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

_HAS_ATEN_SQRTH = hasattr(torch.ops.aten, "linalg_matrix_sqrth")


@pytest.fixture(autouse=True)
def _ieee_fp32_matmul():
    # The reconstruction GEMM must run in IEEE fp32.  Runners that start torch
    # with TF32 enabled (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 is set on CI's
    # a100-worker-1) turn that GEMM into a 10-bit-mantissa one, which moves the
    # fp32 result ~5e-4 away from the fp64 reference -- past the
    # atol=1e-4*reduce_dim budget on (4, 4) float32.  That is an ambient
    # precision policy, not a property of the kernel under test, so pin both
    # precision APIs off (torch >= 2.9 new API plus the legacy flag, which older
    # versions only have) and restore the process state afterwards.  Same
    # approach as tests/test_chain_matmul.py.
    saved_precision = None
    try:
        saved_precision = torch.backends.cuda.matmul.fp32_precision
    except (AttributeError, RuntimeError):
        pass
    saved_tf32 = None
    try:
        saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    except (AttributeError, RuntimeError):
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except (AttributeError, RuntimeError):
        pass
    yield
    if saved_precision is not None:
        try:
            torch.backends.cuda.matmul.fp32_precision = saved_precision
        except (AttributeError, RuntimeError):
            pass
    if saved_tf32 is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        except (AttributeError, RuntimeError):
            pass


DTYPES = [torch.float32, torch.complex64]
if not QUICK_MODE:
    DTYPES += [torch.float64, torch.complex128]

SHAPES = [(1, 1), (4, 4), (2, 8, 8), (2, 3, 4, 4), (0, 0), (0, 4, 4)]
if QUICK_MODE:
    SHAPES = [(4, 4), (2, 8, 8), (0, 0)]


def _make_hpd(shape, dtype, device):
    if 0 in shape:
        return torch.empty(shape, dtype=dtype, device=device)
    matrix = torch.randn(shape, dtype=dtype, device=device)
    n = shape[-1]
    return matrix @ matrix.mH + 0.5 * torch.eye(n, dtype=dtype, device=device)


def _reference(inp):
    # Compute the reference in the widest precision the reference backend
    # supports.  The candidate runs in the input dtype, and the CPU reference
    # kernel under ``--ref cpu`` is a second fp32/complex64 implementation, so
    # comparing two independent fp32 square-root kernels can differ by ~5e-4
    # on complex64 (measured in CI's quick-cpu lane) without either side being
    # wrong.  With the widened reference the check measures the candidate's
    # own rounding error (~1e-6 on H20, 2200 random draws).
    ref_inp = utils.to_reference(inp, True)
    if _HAS_ATEN_SQRTH:
        return torch.ops.aten.linalg_matrix_sqrth(ref_inp)
    # PyTorch 2.11 has no aten::linalg_matrix_sqrth.  Reconstruct the same
    # principal square root from the lower-triangle eigendecomposition the ATen
    # kernel performs: V diag(sqrt(clamp(lambda, 0))) V^H, Hermitian-projected.
    if ref_inp.numel() == 0:
        return ref_inp.clone()
    eigenvalues, eigenvectors = torch.linalg.eigh(ref_inp, UPLO="L")
    sqrt_eigenvalues = eigenvalues.clamp_min(0).sqrt()
    result = (eigenvectors * sqrt_eigenvalues.unsqueeze(-2)) @ eigenvectors.mH
    return (result + result.mH) * 0.5


@pytest.mark.linalg_matrix_sqrth
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_linalg_matrix_sqrth(dtype, shape):
    inp = _make_hpd(shape, dtype, flag_gems.device)
    reference = _reference(inp)

    result = flag_gems.linalg_matrix_sqrth(inp)

    utils.gems_assert_close(
        result,
        reference,
        dtype,
        reduce_dim=max(shape[-1], 1),
    )


@pytest.mark.linalg_matrix_sqrth
@pytest.mark.parametrize("dtype", DTYPES)
def test_linalg_matrix_sqrth_noncontiguous_and_lower_triangle(dtype):
    inp = _make_hpd((2, 8, 8), dtype, flag_gems.device).mH
    assert not inp.is_contiguous()
    upper_noise = torch.randn_like(inp)
    inp = torch.tril(inp) + torch.triu(upper_noise, diagonal=1)
    reference = _reference(inp)

    result = flag_gems.linalg_matrix_sqrth(inp)

    utils.gems_assert_close(result, reference, dtype, reduce_dim=8)


@pytest.mark.linalg_matrix_sqrth
@pytest.mark.parametrize(
    "shape,dtype",
    [((4,), torch.float32), ((3, 4), torch.float32), ((4, 4), torch.float16)],
)
def test_linalg_matrix_sqrth_invalid_input(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.linalg_matrix_sqrth(inp)


@pytest.mark.linalg_matrix_sqrth_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 4), (2, 8, 8)])
def test_linalg_matrix_sqrth_out(dtype, shape):
    inp = _make_hpd(shape, dtype, flag_gems.device)
    reference = _reference(inp)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device).mH
    assert not out.is_contiguous()

    returned = flag_gems.linalg_matrix_sqrth_out(inp, out=out)

    assert returned is out
    utils.gems_assert_close(out, reference, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_matrix_sqrth_out
def test_linalg_matrix_sqrth_out_resize():
    inp = _make_hpd((4, 4), torch.float32, flag_gems.device)
    reference = _reference(inp)
    out = torch.empty(0, dtype=inp.dtype, device=inp.device)

    returned = flag_gems.linalg_matrix_sqrth_out(inp, out=out)

    assert returned is out
    assert out.shape == inp.shape
    utils.gems_assert_close(out, reference, inp.dtype, reduce_dim=4)

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

DTYPES = [torch.float32, torch.complex64]
if not QUICK_MODE:
    DTYPES += [torch.float64, torch.complex128]

SHAPES = [(3, 2), (2, 3), (4, 4), (2, 5, 3), (2, 2, 4, 3)]
if QUICK_MODE:
    SHAPES = [(3, 2), (4, 4), (2, 5, 3)]

PINVERSE_ATOL = {
    torch.float32: 2e-2,
    torch.float64: 1e-10,
    torch.complex64: 2e-4,
    torch.complex128: 1e-10,
}


def _make_well_conditioned(shape, dtype, device):
    """Build matrices with singular values in [1, 2] on the CPU."""
    m, n = shape[-2:]
    k = min(m, n)
    batch_shape = shape[:-2]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17 + m * 31 + n)
    real_dtype = torch.float64
    matrix = torch.randn(shape, generator=generator, dtype=real_dtype)
    if dtype.is_complex:
        matrix = matrix + 1j * torch.randn(shape, generator=generator, dtype=real_dtype)
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    spectrum = torch.linspace(2.0, 1.0, k, dtype=real_dtype)
    spectrum = spectrum.expand(*batch_shape, k)
    matrix = (u * spectrum.unsqueeze(-2)) @ vh
    return matrix.to(dtype=dtype, device=device)


@pytest.mark.pinverse
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_pinverse(shape, dtype):
    inp = _make_well_conditioned(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)
    reference = torch.pinverse(ref_inp)

    result = flag_gems.pinverse(inp)
    utils.gems_assert_close(result, reference, dtype, atol=PINVERSE_ATOL[dtype])
    assert result.is_contiguous()


@pytest.mark.pinverse
def test_pinverse_rcond():
    inp = _make_well_conditioned((2, 5, 3), torch.float32, flag_gems.device)
    reference = torch.pinverse(utils.to_reference(inp), rcond=1e-4)

    result = flag_gems.pinverse(inp, rcond=1e-4)

    utils.gems_assert_close(result, reference, inp.dtype, atol=PINVERSE_ATOL[inp.dtype])


@pytest.mark.pinverse
def test_pinverse_noncontiguous():
    base = _make_well_conditioned((2, 3, 5), torch.float32, flag_gems.device)
    inp = base.transpose(-2, -1)
    assert not inp.is_contiguous()
    reference = torch.pinverse(utils.to_reference(inp))

    result = flag_gems.pinverse(inp)
    utils.gems_assert_close(result, reference, inp.dtype, atol=PINVERSE_ATOL[inp.dtype])


@pytest.mark.pinverse
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 0, 3)])
def test_pinverse_empty(shape):
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    reference = torch.pinverse(utils.to_reference(inp))

    result = flag_gems.pinverse(inp)
    utils.gems_assert_equal(result, reference)
    assert result.stride() == reference.stride()


@pytest.mark.pinverse
@pytest.mark.parametrize("rcond", [-1.0, 0.0, 1e-3, float("inf"), float("nan")])
def test_pinverse_rcond_and_rank_deficiency(rcond):
    inp = torch.diag(
        torch.tensor([2.0, 1e-4, 0.0], dtype=torch.float32, device=flag_gems.device)
    )
    reference = torch.pinverse(utils.to_reference(inp), rcond=rcond)

    result = flag_gems.pinverse(inp, rcond=rcond)
    utils.gems_assert_close(result, reference, inp.dtype, atol=5e-3)


@pytest.mark.pinverse
@pytest.mark.parametrize(
    "value,rcond",
    [
        (float("inf"), 1e-15),
        (float("nan"), 1e-15),
        (float("inf"), -1.0),
    ],
)
def test_pinverse_nonfinite(value, rcond):
    inp = torch.eye(3, dtype=torch.float32, device=flag_gems.device)
    inp[0, 0] = value
    ref_inp = utils.to_reference(inp)
    if ref_inp.device.type == "cpu" and (torch.isnan(ref_inp).any() or rcond < 0):
        pytest.skip("CPU and CUDA SVD differ for this non-finite input")
    reference = torch.pinverse(ref_inp, rcond=rcond)

    result = flag_gems.pinverse(inp, rcond=rcond)
    utils.gems_assert_equal(result, reference, equal_nan=True)


@pytest.mark.pinverse
@pytest.mark.parametrize(
    "shape,dtype", [((4,), torch.float32), ((3, 3), torch.float16)]
)
def test_pinverse_invalid_input(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.pinverse(inp)


@pytest.mark.pinverse
def test_pinverse_autograd():
    inp = _make_well_conditioned((4, 3), torch.float32, flag_gems.device)
    inp.requires_grad_(True)
    ref_inp = utils.to_reference(inp).detach().requires_grad_(True)
    reference = torch.pinverse(ref_inp)
    reference.square().sum().backward()

    result = flag_gems.pinverse(inp)
    result.square().sum().backward()

    utils.gems_assert_close(result, reference, inp.dtype, atol=1e-4)
    utils.gems_assert_close(inp.grad, ref_inp.grad, inp.dtype, atol=1e-3)

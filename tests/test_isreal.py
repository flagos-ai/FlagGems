# Copyright 2026, The FlagOS Contributors.
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


def _make_complex(shape, dtype, device):
    """Build a complex tensor whose imaginary part is zero on ~half of it."""
    float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, dtype=float_dtype, device=device)
    imag = torch.randn(shape, dtype=float_dtype, device=device)
    if imag.dim() > 0:
        imag[::2] = 0
    else:
        imag = torch.zeros_like(imag)
    return torch.complex(real, imag).to(dtype)


@pytest.mark.isreal
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
    + [torch.complex64, torch.complex128],
)
def test_isreal(shape, dtype):
    if dtype.is_complex:
        inp = _make_complex(shape, dtype, flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isreal(ref_inp)
    res_out = flag_gems.isreal(inp)

    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_isreal_special_values(dtype):
    """Zero, negative zero, NaN and infinity, including a NaN imaginary part."""
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), 1.0, 1.0, -0.0]
        imag = [0.0, -0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("inf"), -0.0]
        inp = torch.complex(
            torch.tensor(real, dtype=float_dtype, device=flag_gems.device),
            torch.tensor(imag, dtype=float_dtype, device=flag_gems.device),
        ).to(dtype)
    else:
        inp = torch.tensor(
            [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), 1.0, 1.0, -0.0],
            dtype=dtype,
            device=flag_gems.device,
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isreal(ref_inp)
    res_out = flag_gems.isreal(inp)

    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.complex128])
def test_isreal_non_contiguous(dtype):
    if dtype.is_complex:
        base = _make_complex((32, 48), dtype, flag_gems.device)
    else:
        base = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)
    inp = base.t()
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.isreal(ref_inp)
    res_out = flag_gems.isreal(inp)

    assert res_out.dtype == torch.bool
    assert res_out.shape == inp.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_isreal_empty(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isreal(ref_inp)
    res_out = flag_gems.isreal(inp)

    assert res_out.dtype == torch.bool
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isreal
def test_isreal_conj_bit():
    """A lazy conjugate only flips the sign of the imaginary part."""
    base = _make_complex((1024,), torch.complex64, flag_gems.device)
    inp = base.conj()
    assert inp.is_conj()
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.isreal(ref_inp)
    res_out = flag_gems.isreal(inp)

    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)

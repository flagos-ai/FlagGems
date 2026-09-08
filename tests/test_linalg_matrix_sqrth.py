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

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops.aten, "linalg_matrix_sqrth"),
    reason=(
        "linalg_matrix_sqrth requires PyTorch nightly 2026-06-27 or newer, "
        "or PyTorch 2.15+"
    ),
)

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
    ref_inp = utils.to_reference(inp)
    return torch.ops.aten.linalg_matrix_sqrth(ref_inp)


@pytest.mark.linalg_matrix_sqrth
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_linalg_matrix_sqrth(dtype, shape):
    inp = _make_hpd(shape, dtype, flag_gems.device)
    reference = _reference(inp)

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg_matrix_sqrth(inp)

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

    with flag_gems.use_gems():
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

    with flag_gems.use_gems():
        returned = torch.ops.aten.linalg_matrix_sqrth.out(inp, out=out)

    assert returned is out
    utils.gems_assert_close(out, reference, dtype, reduce_dim=shape[-1])


@pytest.mark.linalg_matrix_sqrth_out
def test_linalg_matrix_sqrth_out_resize():
    inp = _make_hpd((4, 4), torch.float32, flag_gems.device)
    reference = _reference(inp)
    out = torch.empty(0, dtype=inp.dtype, device=inp.device)

    with flag_gems.use_gems():
        returned = torch.ops.aten.linalg_matrix_sqrth.out(inp, out=out)

    assert returned is out
    assert out.shape == inp.shape
    utils.gems_assert_close(out, reference, inp.dtype, reduce_dim=4)

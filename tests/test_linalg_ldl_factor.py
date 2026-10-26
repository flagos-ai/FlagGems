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

LDL_FACTOR_SHAPES = [(4, 4), (8, 8), (16, 16), (32, 32)]
LDL_FACTOR_DTYPES = [torch.float32, torch.float64]


def _make_symmetric_positive_definite(shape, dtype, device):
    torch.manual_seed(0)
    n = shape[-1]
    # These backends cannot construct a faithful FP64 SPD matrix on device.
    factory_device = (
        "cpu"
        if dtype == torch.float64 and flag_gems.vendor_name in ("ascend", "iluvatar")
        else device
    )
    inp = torch.randn(shape, dtype=dtype, device=factory_device)
    inp = inp @ inp.transpose(-2, -1) + n * torch.eye(
        n, dtype=dtype, device=factory_device
    )
    return inp.to(device)


def _skip_unsupported_native_reference(dtype):
    if (
        flag_gems.vendor_name == "iluvatar"
        and dtype == torch.float64
        and not utils.TO_CPU
    ):
        pytest.skip("Iluvatar native ldl_factor does not support float64")


def _assert_factorization_equal(result, reference, dtype, expected_device):
    result_ld, result_pivots = result
    reference_ld, reference_pivots = reference
    assert result_ld.device == expected_device
    assert result_pivots.device == expected_device
    assert result_ld.shape == reference_ld.shape
    assert result_ld.dtype == dtype
    assert result_pivots.shape == reference_pivots.shape
    assert result_pivots.dtype == torch.int32
    utils.gems_assert_close(result_ld, reference_ld, dtype)
    utils.gems_assert_equal(result_pivots, reference_pivots)


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
@pytest.mark.parametrize("shape", LDL_FACTOR_SHAPES)
@pytest.mark.parametrize("dtype", LDL_FACTOR_DTYPES)
@pytest.mark.parametrize("hermitian", [False, True])
def test_linalg_ldl_factor(shape, dtype, hermitian):
    _skip_unsupported_native_reference(dtype)
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Not supported float64")
    inp = _make_symmetric_positive_definite(shape, dtype, flag_gems.device)
    inp_before = inp.clone()
    ref_inp = utils.to_reference(inp, True)

    reference = torch.linalg.ldl_factor(ref_inp, hermitian=hermitian)
    result = flag_gems.ldl_factor(inp, hermitian=hermitian)

    _assert_factorization_equal(result, reference, dtype, inp.device)
    assert inp.device == inp_before.device
    utils.gems_assert_equal(inp, utils.to_reference(inp_before, False))


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8)])
@pytest.mark.parametrize("dtype", LDL_FACTOR_DTYPES)
def test_linalg_ldl_factor_batched(shape, dtype):
    _skip_unsupported_native_reference(dtype)
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Not supported float64")
    inp = _make_symmetric_positive_definite(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    reference = torch.linalg.ldl_factor(ref_inp)
    result = flag_gems.ldl_factor(inp)

    _assert_factorization_equal(result, reference, dtype, inp.device)


@pytest.mark.linalg_ldl_factor
@pytest.mark.ldl_factor
@pytest.mark.parametrize(
    "shape,dtype,error_type,message",
    [
        ((8,), torch.float32, ValueError, "at least 2D"),
        ((4, 5), torch.float32, ValueError, "square"),
        ((4, 4), torch.int32, TypeError, "float32 and float64"),
        ((65, 65), torch.float32, ValueError, "exceeds maximum"),
    ],
)
def test_linalg_ldl_factor_rejects_invalid_inputs(shape, dtype, error_type, message):
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)

    with pytest.raises(error_type, match=message):
        flag_gems.ldl_factor(inp)

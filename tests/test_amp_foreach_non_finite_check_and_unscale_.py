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

# worktree explicitly excludes bfloat16 via PRIMARY_FLOAT_DTYPES
PRIMARY_FLOAT_DTYPES = [torch.float16, torch.float32]


def _assert_gradient_values(actual, expected, dtype, device, *, equal_nan=False):
    assert len(actual) == len(expected)
    for got, reference in zip(actual, expected):
        assert got.device == device
        assert got.dtype == dtype
        assert got.shape == reference.shape
        utils.gems_assert_close(got, reference, dtype, equal_nan=equal_nan)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale_(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with normal tensors."""
    # PyTorch expects inv_scale and found_inf to be float32
    inv_scale = torch.tensor(2.0, device=flag_gems.device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)

    tensors = [
        torch.randn(16, 32, device=flag_gems.device, dtype=dtype),
        torch.randn(8, 16, device=flag_gems.device, dtype=dtype),
    ]

    # Reference
    ref_tensors = [utils.to_reference(t.clone()) for t in tensors]
    ref_found_inf = utils.to_reference(found_inf.clone())
    ref_inv_scale = utils.to_reference(inv_scale.clone())
    getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
        ref_tensors, ref_found_inf, ref_inv_scale
    )

    # GEMS
    res_tensors = [t.clone() for t in tensors]
    res_found_inf = found_inf.clone()
    expected_device = res_found_inf.device
    flag_gems._amp_foreach_non_finite_check_and_unscale_(
        res_tensors, res_found_inf, inv_scale
    )

    # Compare mutated inputs (in-place operation)
    _assert_gradient_values(res_tensors, ref_tensors, dtype, expected_device)

    # Compare found_inf (also mutated in-place)
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale__inf(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with inf values."""
    # PyTorch expects inv_scale and found_inf to be float32
    inv_scale = torch.tensor(2.0, device=flag_gems.device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)

    tensors = [
        torch.tensor(
            [1.0, 2.0, float("inf"), 4.0], device=flag_gems.device, dtype=dtype
        ),
        torch.tensor([5.0, 6.0, 7.0], device=flag_gems.device, dtype=dtype),
    ]

    # Reference
    ref_tensors = [utils.to_reference(t.clone()) for t in tensors]
    ref_found_inf = utils.to_reference(found_inf.clone())
    ref_inv_scale = utils.to_reference(inv_scale.clone())
    getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
        ref_tensors, ref_found_inf, ref_inv_scale
    )

    # GEMS
    res_tensors = [t.clone() for t in tensors]
    res_found_inf = found_inf.clone()
    expected_device = res_found_inf.device
    flag_gems._amp_foreach_non_finite_check_and_unscale_(
        res_tensors, res_found_inf, inv_scale
    )

    # Compare mutated inputs (in-place operation)
    # Note: inf values remain unchanged, only finite values are scaled
    _assert_gradient_values(res_tensors, ref_tensors, dtype, expected_device)

    # Compare found_inf (also mutated in-place) - should be 1.0 when inf is present
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale__nan(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with nan values."""
    # PyTorch expects inv_scale and found_inf to be float32
    inv_scale = torch.tensor(2.0, device=flag_gems.device, dtype=torch.float32)
    found_inf = torch.tensor(0.0, device=flag_gems.device, dtype=torch.float32)

    tensors = [
        torch.tensor(
            [1.0, 2.0, float("nan"), 4.0], device=flag_gems.device, dtype=dtype
        ),
        torch.tensor([5.0, 6.0, 7.0], device=flag_gems.device, dtype=dtype),
    ]

    # Reference
    ref_tensors = [utils.to_reference(t.clone()) for t in tensors]
    ref_found_inf = utils.to_reference(found_inf.clone())
    ref_inv_scale = utils.to_reference(inv_scale.clone())
    getattr(torch, "_amp_foreach_non_finite_check_and_unscale_")(
        ref_tensors, ref_found_inf, ref_inv_scale
    )

    # GEMS
    res_tensors = [t.clone() for t in tensors]
    res_found_inf = found_inf.clone()
    expected_device = res_found_inf.device
    flag_gems._amp_foreach_non_finite_check_and_unscale_(
        res_tensors, res_found_inf, inv_scale
    )

    # Compare mutated inputs with equal_nan=True since tensors contain NaN
    _assert_gradient_values(
        res_tensors, ref_tensors, dtype, expected_device, equal_nan=True
    )

    # Compare found_inf - should be 1.0 when nan is present
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


def _layout_snapshot(tensors):
    return [(tensor, tensor.data_ptr(), tensor.stride(), tensor.storage_offset()) for tensor in tensors]


def _assert_layout_preserved(tensors, snapshots):
    for tensor, (original, pointer, stride, offset) in zip(tensors, snapshots):
        assert tensor is original
        assert tensor.data_ptr() == pointer
        assert tensor.stride() == stride
        assert tensor.storage_offset() == offset


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("initial_found_inf", [0.0, 2.0])
def test_amp_foreach_non_finite_check_and_unscale__mutation_contract(
    initial_found_inf
):
    """Every tensor and scalar keeps identity/storage while all mutations match ATen."""
    device = flag_gems.device
    dtype = torch.float32
    tensors = [
        torch.tensor([1.0, float("inf"), -3.0], dtype=dtype, device=device),
        torch.tensor([float("nan"), 5.0], dtype=dtype, device=device),
        torch.empty(0, dtype=dtype, device=device),
    ]
    found_inf = torch.tensor(initial_found_inf, dtype=torch.float32, device=device)
    inv_scale = torch.tensor(0.25, dtype=torch.float32, device=device)
    expected_device = found_inf.device
    tensor_layouts = _layout_snapshot(tensors)
    found_layout = _layout_snapshot([found_inf])
    inv_layout = _layout_snapshot([inv_scale])
    inv_before = utils.to_reference(inv_scale.clone(), False)
    ref_tensors = [utils.to_reference(tensor.clone(), False) for tensor in tensors]
    ref_found = utils.to_reference(found_inf.clone(), False)
    ref_scale = utils.to_reference(inv_scale.clone(), False)
    torch._amp_foreach_non_finite_check_and_unscale_(
        ref_tensors, ref_found, ref_scale
    )

    result = flag_gems._amp_foreach_non_finite_check_and_unscale_(
        tensors, found_inf, inv_scale
    )

    assert result is None
    _assert_layout_preserved(tensors, tensor_layouts)
    _assert_layout_preserved([found_inf], found_layout)
    _assert_layout_preserved([inv_scale], inv_layout)
    _assert_gradient_values(
        tensors, ref_tensors, dtype, expected_device, equal_nan=True
    )
    utils.gems_assert_equal(found_inf, ref_found)
    utils.gems_assert_equal(inv_scale, inv_before)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
def test_amp_foreach_non_finite_check_and_unscale__empty_list():
    found_inf = torch.tensor(0.0, device=flag_gems.device)
    inv_scale = torch.tensor(0.5, device=flag_gems.device)
    found_ptr = found_inf.data_ptr()
    inv_ptr = inv_scale.data_ptr()
    result = flag_gems._amp_foreach_non_finite_check_and_unscale_(
        [], found_inf, inv_scale
    )
    assert result is None
    assert found_inf.data_ptr() == found_ptr
    assert inv_scale.data_ptr() == inv_ptr
    utils.gems_assert_equal(found_inf, utils.to_reference(torch.zeros_like(found_inf)))
    utils.gems_assert_equal(
        inv_scale, utils.to_reference(torch.full_like(inv_scale, 0.5))
    )


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
def test_amp_foreach_non_finite_check_and_unscale__rejects_non_sequence():
    found_inf = torch.tensor(0.0, device=flag_gems.device)
    inv_scale = torch.tensor(0.5, device=flag_gems.device)
    with pytest.raises(
        TypeError, match=r"list or tuple|must be tuple of Tensors"
    ):
        flag_gems._amp_foreach_non_finite_check_and_unscale_(
            torch.ones(4, device=flag_gems.device), found_inf, inv_scale
        )

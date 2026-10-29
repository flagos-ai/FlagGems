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


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)


def _skip_native_half_reference(dtype):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype == torch.float16
        and not utils.TO_CPU
    ):
        pytest.skip(
            "Kunlunxin native Half AMP unscale returns incorrect values; CPU reference remains covered"
        )


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale_(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with normal tensors."""
    _skip_native_half_reference(dtype)
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
    _unscale_and_check(res_tensors, res_found_inf, inv_scale)

    # Compare mutated inputs (in-place operation)
    for i, (inp, ref_inp) in enumerate(zip(res_tensors, ref_tensors)):
        utils.gems_assert_close(inp, ref_inp, dtype)

    # Compare found_inf (also mutated in-place)
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale__inf(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with inf values."""
    _skip_native_half_reference(dtype)
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
    _unscale_and_check(res_tensors, res_found_inf, inv_scale)

    # Compare mutated inputs (in-place operation)
    # Note: inf values remain unchanged, only finite values are scaled
    for i, (inp, ref_inp) in enumerate(zip(res_tensors, ref_tensors)):
        utils.gems_assert_close(inp, ref_inp, dtype)

    # Compare found_inf (also mutated in-place) - should be 1.0 when inf is present
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("dtype", PRIMARY_FLOAT_DTYPES)
def test_amp_foreach_non_finite_check_and_unscale__nan(dtype):
    """Test _amp_foreach_non_finite_check_and_unscale_ with nan values."""
    _skip_native_half_reference(dtype)
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
    _unscale_and_check(res_tensors, res_found_inf, inv_scale)

    # Compare mutated inputs with equal_nan=True since tensors contain NaN
    for i, (inp, ref_inp) in enumerate(zip(res_tensors, ref_tensors)):
        utils.gems_assert_close(
            inp,
            ref_inp,
            dtype,
            equal_nan=True,
        )

    # Compare found_inf - should be 1.0 when nan is present
    utils.gems_assert_equal(res_found_inf, ref_found_inf)


def _tensor_metadata(tensor):
    storage = tensor.untyped_storage()
    return (
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.storage_offset(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
    )


def _layout_snapshot(tensors):
    return [(tensor, _tensor_metadata(tensor)) for tensor in tensors]


def _assert_layout_preserved(tensors, snapshots):
    assert len(tensors) == len(snapshots)
    for tensor, (original, metadata) in zip(tensors, snapshots):
        assert tensor is original
        assert _tensor_metadata(tensor) == metadata


def _unscale_and_check(tensors, found_inf, inv_scale):
    snapshots = _layout_snapshot([*tensors, found_inf, inv_scale])
    inv_bytes = bytes(inv_scale.untyped_storage().cpu())
    result = flag_gems._amp_foreach_non_finite_check_and_unscale_(
        tensors, found_inf, inv_scale
    )
    assert result is None
    _assert_layout_preserved([*tensors, found_inf, inv_scale], snapshots)
    assert bytes(inv_scale.untyped_storage().cpu()) == inv_bytes
    return result


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
@pytest.mark.parametrize("initial_found_inf", [0.0, 2.0])
def test_amp_foreach_non_finite_check_and_unscale__mutation_contract(initial_found_inf):
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
    tensor_layouts = _layout_snapshot(tensors)
    found_layout = _layout_snapshot([found_inf])
    inv_layout = _layout_snapshot([inv_scale])
    inv_before = utils.to_reference(inv_scale.clone(), False)
    ref_tensors = [utils.to_reference(tensor.clone(), False) for tensor in tensors]
    ref_found = utils.to_reference(found_inf.clone(), False)
    ref_scale = utils.to_reference(inv_scale.clone(), False)
    torch._amp_foreach_non_finite_check_and_unscale_(ref_tensors, ref_found, ref_scale)

    result = _unscale_and_check(tensors, found_inf, inv_scale)

    assert result is None
    _assert_layout_preserved(tensors, tensor_layouts)
    _assert_layout_preserved([found_inf], found_layout)
    _assert_layout_preserved([inv_scale], inv_layout)
    for got, expected in zip(tensors, ref_tensors):
        utils.gems_assert_close(
            got,
            expected,
            dtype,
            equal_nan=True,
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
    result = _unscale_and_check([], found_inf, inv_scale)
    assert result is None
    assert found_inf.data_ptr() == found_ptr
    assert inv_scale.data_ptr() == inv_ptr
    utils.gems_assert_equal(
        found_inf, utils.to_reference(torch.zeros_like(found_inf), False)
    )
    utils.gems_assert_equal(
        inv_scale, utils.to_reference(torch.full_like(inv_scale, 0.5), False)
    )


@pytest.mark.amp_foreach_non_finite_check_and_unscale_
@pytest.mark.underscore_amp_foreach_non_finite_check_and_unscale_
def test_amp_foreach_non_finite_check_and_unscale__rejects_non_sequence():
    found_inf = torch.tensor(0.0, device=flag_gems.device)
    inv_scale = torch.tensor(0.5, device=flag_gems.device)
    with pytest.raises(TypeError, match="list or tuple"):
        flag_gems._amp_foreach_non_finite_check_and_unscale_(
            torch.ones(4, device=flag_gems.device), found_inf, inv_scale
        )

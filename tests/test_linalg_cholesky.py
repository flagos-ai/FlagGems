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


def _cholesky_raw_cpu(tensor):
    return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
    )


def _cholesky_comparison_values(tensor):
    # Other backends retain their original tensor comparison path.
    if flag_gems.vendor_name not in ("nvidia", "ascend", "iluvatar"):
        return tensor
    values = _cholesky_raw_cpu(tensor)
    if tensor.is_conj():
        values = values.conj_physical()
    if tensor.is_neg():
        values = -values
    return values


def _cholesky_input_snapshot(tensor):
    metadata = (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage()._cdata,
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
    )
    values = (
        _cholesky_raw_cpu(tensor).contiguous().reshape(-1).view(torch.uint8).clone()
    )
    return metadata, values


def _checked_cholesky(inp, dtype, *, upper=False):
    if flag_gems.vendor_name not in ("nvidia", "ascend", "iluvatar"):
        if upper:
            return flag_gems.linalg_cholesky(inp, upper=True)
        return flag_gems.linalg_cholesky(inp)
    assert inp.dtype == dtype
    assert inp.device.type == torch.device(flag_gems.device).type
    metadata, values = _cholesky_input_snapshot(inp)
    try:
        result = flag_gems.linalg_cholesky(inp, upper=upper)
    finally:
        after_metadata, after_values = _cholesky_input_snapshot(inp)
        assert after_metadata == metadata
        assert torch.equal(after_values, values)
    # Check the target ABI before moving values to CPU for comparison.
    assert isinstance(result, torch.Tensor)
    assert result.device == metadata[0] and result.dtype == metadata[1]
    assert tuple(result.shape) == metadata[2]
    assert result.untyped_storage()._cdata != metadata[5]
    return result


@pytest.mark.linalg_cholesky
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky(shape, dtype):
    # Create a positive-definite matrix: A = B @ B^T + I
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    if (
        flag_gems.vendor_name in ("ascend", "iluvatar")
        and dtype == torch.float64
        and not utils.TO_CPU
    ):
        pytest.skip("backend native linalg_cholesky reference does not support float64")
    factory_device = (
        "cpu"
        if flag_gems.vendor_name in ("ascend", "iluvatar") and dtype == torch.float64
        else flag_gems.device
    )
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=factory_device)
    A = (
        B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=factory_device) * 0.1
    ).to(flag_gems.device)

    # For reference, convert to CPU and use torch.linalg.cholesky
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.cholesky(ref_A)

    # For gems, use aten.linalg_cholesky with flag_gems
    res_out = _checked_cholesky(A, dtype)

    utils.gems_assert_close(
        _cholesky_comparison_values(res_out),
        _cholesky_comparison_values(ref_out),
        dtype,
    )


@pytest.mark.linalg_cholesky
@pytest.mark.parametrize("shape", [(2, 2), (4, 4), (8, 8)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_upper(shape, dtype):
    # Test with upper=True
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    if (
        flag_gems.vendor_name in ("ascend", "iluvatar")
        and dtype == torch.float64
        and not utils.TO_CPU
    ):
        pytest.skip("backend native linalg_cholesky reference does not support float64")
    factory_device = (
        "cpu"
        if flag_gems.vendor_name in ("ascend", "iluvatar") and dtype == torch.float64
        else flag_gems.device
    )
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=factory_device)
    A = (
        B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=factory_device) * 0.1
    ).to(flag_gems.device)

    # For reference, convert to CPU and use torch.linalg.cholesky
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.cholesky(ref_A, upper=True)

    # For gems, use aten.linalg_cholesky with flag_gems
    res_out = _checked_cholesky(A, dtype, upper=True)

    utils.gems_assert_close(
        _cholesky_comparison_values(res_out),
        _cholesky_comparison_values(ref_out),
        dtype,
    )


@pytest.mark.linalg_cholesky
@pytest.mark.parametrize("shape", [(2, 4, 4), (3, 8, 8)])
# Cholesky only supports float32/float64; fp16/bf16 not supported by PyTorch
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_cholesky_batch(shape, dtype):
    # Create positive-definite matrices for batched input: A = B @ B^T + I
    if flag_gems.vendor_name == "cambricon" and dtype == torch.float64:
        pytest.skip("Issue #5253: Not supported")
    if (
        flag_gems.vendor_name in ("ascend", "iluvatar")
        and dtype == torch.float64
        and not utils.TO_CPU
    ):
        pytest.skip("backend native linalg_cholesky reference does not support float64")
    factory_device = (
        "cpu"
        if flag_gems.vendor_name in ("ascend", "iluvatar") and dtype == torch.float64
        else flag_gems.device
    )
    n = shape[-1]
    B = torch.randn(shape, dtype=dtype, device=factory_device)
    A = (
        B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=factory_device) * 0.1
    ).to(flag_gems.device)

    # For reference, convert to CPU and use torch.linalg.cholesky
    ref_A = utils.to_reference(A)
    ref_out = torch.linalg.cholesky(ref_A)

    # For gems, use aten.linalg_cholesky with flag_gems
    res_out = _checked_cholesky(A, dtype)

    utils.gems_assert_close(
        _cholesky_comparison_values(res_out),
        _cholesky_comparison_values(ref_out),
        dtype,
    )

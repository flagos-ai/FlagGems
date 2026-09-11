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

# Each case: (shape_a, shape_b, dims_a, dims_b). Contracted sizes must match.
TENSORDOT_CASES = [
    # Classic dims=2 style contraction (last d of a, first d of b).
    ((3, 4, 5), (4, 5, 6), [1, 2], [0, 1]),
    # Single contracted dim -> plain matmul-like.
    ((16, 32), (32, 24), [1], [0]),
    # Reordered / non-adjacent contracted dims.
    ((3, 5, 4, 6), (6, 4, 5, 3), [2, 1, 3], [1, 2, 0]),
    # Negative dim indices.
    ((8, 7, 9), (9, 7, 5), [-1, 1], [0, 1]),
    # Larger contraction dimension.
    ((64, 128), (128, 96), [1], [0]),
    # Outer product (no contracted dims).
    ((2, 3), (4, 5), [], []),
    # Zero-sized free dimensions.
    ((0, 3, 4), (4, 5), [2], [0]),
    ((3, 4), (4, 0, 5), [1], [0]),
    # Zero-sized contracted dimensions.
    ((3, 0), (0, 5), [1], [0]),
]


def _reduce_dim(shape_a, dims_a):
    k = 1
    for d in dims_a:
        k *= shape_a[d % len(shape_a)]
    return max(k, 1)


@pytest.mark.tensordot
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", TENSORDOT_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensordot(shape_a, shape_b, dims_a, dims_b, dtype):
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b))
    res_out = flag_gems.tensordot(a, b, dims_a, dims_b)

    utils.gems_assert_close(
        res_out, ref_out, dtype, reduce_dim=_reduce_dim(shape_a, dims_a)
    )


@pytest.mark.tensordot_out
@pytest.mark.parametrize("shape_a, shape_b, dims_a, dims_b", TENSORDOT_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensordot_out(shape_a, shape_b, dims_a, dims_b, dtype):
    a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
    b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=(dims_a, dims_b))
    out = torch.empty_like(ref_out, dtype=dtype, device=flag_gems.device)

    # Use flag_gems.tensordot_out instead of torch.tensordot with out=
    res = flag_gems.tensordot_out(a, b, dims_a, dims_b, out=out)
    assert res is out, "tensordot_out should return the same out tensor"

    utils.gems_assert_close(
        out, ref_out, dtype, reduce_dim=_reduce_dim(shape_a, dims_a)
    )


# Test invalid dimensions
@pytest.mark.tensordot
def test_tensordot_invalid_dims():
    a = torch.randn(3, 4, 5, device=flag_gems.device)
    b = torch.randn(5, 6, device=flag_gems.device)

    # Out-of-range dimension
    with pytest.raises(IndexError):
        flag_gems.tensordot(a, b, [5], [0])

    with pytest.raises(IndexError):
        flag_gems.tensordot(a, b, [-10], [0])

    # Mismatched number of contracted dims
    with pytest.raises(AssertionError):
        flag_gems.tensordot(a, b, [0, 1], [0])


# Test dtype and device validation
@pytest.mark.tensordot
def test_tensordot_dtype_device_validation():
    a_f32 = torch.randn(3, 4, device=flag_gems.device, dtype=torch.float32)
    b_f16 = torch.randn(4, 5, device=flag_gems.device, dtype=torch.float16)

    # Mismatched dtype
    with pytest.raises(RuntimeError, match="same dtype"):
        flag_gems.tensordot(a_f32, b_f16, [1], [0])

    # Mismatched device (if CPU is available)
    if torch.cuda.is_available():
        a_gpu = torch.randn(3, 4, device=flag_gems.device)
        b_cpu = torch.randn(4, 5, device="cpu")
        with pytest.raises(RuntimeError, match="same device"):
            flag_gems.tensordot(a_gpu, b_cpu, [1], [0])


# Test non-contiguous inputs
@pytest.mark.tensordot
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensordot_non_contiguous(dtype):
    # Create non-contiguous tensors via transpose
    a = torch.randn(3, 4, 5, dtype=dtype, device=flag_gems.device).transpose(0, 2)
    b = torch.randn(5, 6, 7, dtype=dtype, device=flag_gems.device).transpose(1, 2)

    assert not a.is_contiguous()
    assert not b.is_contiguous()

    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=([0], [0]))
    res_out = flag_gems.tensordot(a, b, [0], [0])

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=a.shape[0])


# Test out= contract with various tensor properties
@pytest.mark.tensordot_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensordot_out_contract(dtype):
    a = torch.randn(3, 4, dtype=dtype, device=flag_gems.device)
    b = torch.randn(4, 5, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a, upcast=True)
    ref_b = utils.to_reference(b, upcast=True)

    ref_out = torch.tensordot(ref_a, ref_b, dims=([1], [0]))

    # Test with correct size
    out = torch.empty(3, 5, dtype=dtype, device=flag_gems.device)
    result = flag_gems.tensordot_out(a, b, [1], [0], out=out)
    assert result is out
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=4)

    # Test with non-contiguous out
    out_large = torch.empty(6, 5, dtype=dtype, device=flag_gems.device)
    out_view = out_large[::2]  # Non-contiguous view
    result = flag_gems.tensordot_out(a, b, [1], [0], out=out_view)
    assert result is out_view
    utils.gems_assert_close(out_view, ref_out, dtype, reduce_dim=4)

    # Test with wrong size - should raise error
    out_wrong = torch.empty(2, 5, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="incorrect size"):
        flag_gems.tensordot_out(a, b, [1], [0], out=out_wrong)

    # Test with wrong dtype - should raise error
    out_wrong_dtype = torch.empty(3, 5, dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="incorrect dtype"):
        flag_gems.tensordot_out(a, b, [1], [0], out=out_wrong_dtype)


# Test unsupported dtypes
@pytest.mark.tensordot
def test_tensordot_unsupported_dtypes():
    # Integer dtypes should raise error
    a_int = torch.randint(0, 10, (3, 4), device=flag_gems.device, dtype=torch.int32)
    b_int = torch.randint(0, 10, (4, 5), device=flag_gems.device, dtype=torch.int32)

    with pytest.raises((RuntimeError, NotImplementedError)):
        flag_gems.tensordot(a_int, b_int, [1], [0])

    # Complex dtypes should raise error (if not supported)
    if hasattr(torch, "complex64"):
        a_complex = torch.randn(3, 4, device=flag_gems.device, dtype=torch.complex64)
        b_complex = torch.randn(4, 5, device=flag_gems.device, dtype=torch.complex64)

        with pytest.raises((RuntimeError, NotImplementedError)):
            flag_gems.tensordot(a_complex, b_complex, [1], [0])

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
from flag_gems.ops.copy import _can_use_triton
from flag_gems.ops.copy import copy_ as common_copy_

from . import accuracy_utils as utils

_COMMON_COPY_IS_ACTIVE = getattr(flag_gems, "copy_", None) is common_copy_


@pytest.mark.copy_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    (
        utils.FLOAT_DTYPES + [torch.int32, torch.int64]
        if flag_gems.vendor_name == "cambricon"
        else utils.FLOAT_DTYPES
    ),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_same_dtype(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        if dtype in utils.FLOAT_DTYPES:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        else:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
    else:
        src = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_dst = torch.zeros_like(ref_src)
    res_dst = torch.zeros_like(src)

    ref_dst.copy_(ref_src)
    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_broadcast():
    dst_shape = (2, 3)
    src = torch.arange(0, 3, dtype=torch.float32, device=flag_gems.device)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(
        torch.zeros(dst_shape, dtype=torch.float32, device=flag_gems.device)
    )
    res_dst = torch.zeros(dst_shape, dtype=torch.float32, device=flag_gems.device)

    ref_dst.copy_(ref_src)
    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not _COMMON_COPY_IS_ACTIVE,
    reason="active vendor copy_ does not use the common Triton gate",
)
@pytest.mark.parametrize(
    "src_negative,dst_negative",
    [(True, False), (False, True), (True, True)],
)
def test_copy_inplace_preserves_lazy_negative_semantics(src_negative, dst_negative):
    dtype = torch.float32
    src_base = torch.tensor([1, -2, 3], dtype=dtype, device=flag_gems.device)
    dst_base = torch.tensor([10, 20, 30], dtype=dtype, device=flag_gems.device)
    src = torch._neg_view(src_base) if src_negative else src_base
    dst = torch._neg_view(dst_base) if dst_negative else dst_base
    expected = torch.tensor(
        [-1, 2, -3] if src_negative else [1, -2, 3],
        dtype=dtype,
        device=flag_gems.device,
    )

    assert _can_use_triton(dst, src)
    with flag_gems.use_gems(include=["copy_"]):
        result = dst.copy_(src)

    assert result is dst
    assert dst.is_neg() == dst_negative
    # Some backends cannot safely materialize a lazy-negative destination in
    # their native comparison path.  Checking the raw view against the
    # sign-adjusted expectation proves the same logical value without invoking
    # that backend fallback.
    if not dst_negative:
        utils.gems_assert_equal(dst, expected)
    physical_dst = torch._neg_view(dst) if dst_negative else dst
    physical_expected = -expected if dst_negative else expected
    utils.gems_assert_equal(physical_dst, physical_expected)


@pytest.mark.copy_
@pytest.mark.skipif(
    not _COMMON_COPY_IS_ACTIVE,
    reason="active vendor copy_ does not use the common Triton gate",
)
def test_clone_materializes_lazy_negative_values():
    dtype = torch.float32
    raw = torch.tensor([-1, 0, 3], dtype=dtype, device=flag_gems.device)
    lazy_negative = torch._neg_view(raw)
    expected = torch.tensor([1, 0, -3], dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems(include=["copy_"]):
        result = lazy_negative.clone()

    assert not result.is_neg()
    assert result.data_ptr() != lazy_negative.data_ptr()
    utils.gems_assert_equal(result, expected)


@pytest.mark.copy_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_dtype_fallback():
    src = torch.arange(0, 8, dtype=torch.int32, device=flag_gems.device)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(
        torch.zeros(src.shape, dtype=torch.float32, device=flag_gems.device)
    )
    res_dst = torch.zeros(src.shape, dtype=torch.float32, device=flag_gems.device)

    ref_dst.copy_(ref_src)
    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="PyTorch does not support float8_e8m0fnu",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="mthreads does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="MetaX does not support float8_e8m0fnu dtype",
)
@pytest.mark.parametrize("shape", [(8,), (4, 4), (2, 3, 4)])
def test_copy_inplace_float8_e8m0fnu(shape):
    """Test that copy_ works correctly with float8_e8m0fnu (e8m0) dtype tensors.

    Triton does not recognize float8_e8m0fnu, so FlagGems should fallback to
    PyTorch's native copy_ implementation for this dtype.
    """
    device = flag_gems.device

    # e8m0 is an exponent-only format, create via view from uint8
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support uint8 generation.
        src_uint8 = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            device
        )
    else:
        src_uint8 = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
    src = src_uint8.view(torch.float8_e8m0fnu)
    ref_src = utils.to_reference(src)

    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support float8_e8m0fnu generation.
        ref_dst = utils.to_reference(
            torch.zeros(shape, dtype=torch.float8_e8m0fnu, device="cpu").to(device)
        )
        res_dst = torch.zeros(shape, dtype=torch.float8_e8m0fnu, device="cpu").to(
            device
        )
    else:
        ref_dst = utils.to_reference(
            torch.zeros(shape, dtype=torch.float8_e8m0fnu, device=device)
        )
        res_dst = torch.zeros(shape, dtype=torch.float8_e8m0fnu, device=device)
    ref_dst.copy_(ref_src)

    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="PyTorch does not support float8_e8m0fnu",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason="mthreads does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "metax",
    reason="MetaX does not support float8_e8m0fnu dtype",
)
def test_copy_inplace_float8_e8m0fnu_to_float32():
    """Test copy_ from float8_e8m0fnu to float32."""
    device = flag_gems.device
    shape = (8,)

    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support uint8 generation.
        src_uint8 = torch.randint(1, 200, shape, dtype=torch.uint8, device="cpu").to(
            device
        )
    else:
        src_uint8 = torch.randint(1, 200, shape, dtype=torch.uint8, device=device)
    src = src_uint8.view(torch.float8_e8m0fnu)
    ref_src = utils.to_reference(src)

    ref_dst = utils.to_reference(torch.zeros(shape, dtype=torch.float32, device=device))
    res_dst = torch.zeros(shape, dtype=torch.float32, device=device)
    ref_dst.copy_(ref_src)

    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.int32),
        (torch.int16, torch.float32),
        (torch.bool, torch.float32),
    ],
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_mixed_dtype_triton(src_dtype, dst_dtype):
    device = flag_gems.device
    numel = 8

    if src_dtype is torch.bool:
        base = torch.tensor([True, False, True, True, False, True, False, True])
        src = base.to(device=device)
    else:
        if flag_gems.vendor_name == "mthreads":
            src = torch.arange(numel, device="cpu", dtype=src_dtype).to(device)
        else:
            src = torch.arange(numel, device=device, dtype=src_dtype)

    dst = torch.zeros(numel, dtype=dst_dtype, device=device)

    assert _can_use_triton(dst, src)

    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())
    ref_dst.copy_(ref_src)

    with flag_gems.use_gems():
        res_dst = dst.clone()
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    (
        utils.FLOAT_DTYPES + [torch.int32, torch.int64]
        if flag_gems.vendor_name == "cambricon"
        else utils.FLOAT_DTYPES
    ),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_functional_same_dtype(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        if dtype in utils.FLOAT_DTYPES:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        else:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
    else:
        src = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    template = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_template = utils.to_reference(template)

    ref_out = torch.ops.aten.copy(ref_template, ref_src)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.copy(template, src)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.copy
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_functional_broadcast():
    src = torch.arange(0, 3, dtype=torch.float32, device=flag_gems.device)
    template = torch.empty((2, 3), dtype=torch.float32, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_template = utils.to_reference(template)

    ref_out = torch.ops.aten.copy(ref_template, ref_src)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.copy(template, src)

    utils.gems_assert_equal(res_out, ref_out)

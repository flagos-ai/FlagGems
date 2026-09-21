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

from . import accuracy_utils as utils


def _is_float8_dtype_supported(dtype: torch.dtype) -> bool:
    if dtype is None:
        return False
    if str(flag_gems.device).startswith("cuda"):
        device_index = 0
        if isinstance(flag_gems.device, str) and ":" in flag_gems.device:
            device_index = int(flag_gems.device.split(":")[1])

        cap = torch.cuda.get_device_capability(device_index)
        if cap[0] < 8 or (cap[0] == 8 and cap[1] < 9):
            return False
    try:
        t = torch.zeros(1, device=flag_gems.device, dtype=dtype)
        return t.dtype == dtype
    except (RuntimeError, TypeError):
        return False


_FLOAT8_DTYPES = []
for _dtype_name in ["float8_e4m3fn", "float8_e5m2"]:
    _dtype = getattr(torch, _dtype_name, None)
    if _dtype is not None and _is_float8_dtype_supported(_dtype):
        _FLOAT8_DTYPES.append(_dtype)


@pytest.mark.copy_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    (
        utils.FLOAT_DTYPES + [torch.int32, torch.int64, torch.int8, torch.uint8]
        if flag_gems.vendor_name == "cambricon"
        else utils.FLOAT_DTYPES + [torch.int8, torch.uint8]
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
        if dtype in [torch.int8, torch.uint8]:
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend does not support float8_e8m0 dtypes",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="KUNLUNXIN does not support float8_e8m0fnu dtype",
)
@pytest.mark.parametrize(
    "shape",
    [
        (8,),
        (4, 4),
        (2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
        (100, 200),
        (90, 60, 30),
        (1000, 2001),
        (7, 13, 17),
        (3, 31, 61, 127),
    ],
)
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
@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="KUNLUNXIN does not support float8_e8m0fnu dtype",
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend NPU does not support float8_e8m0fnu dtype for copy_d2d",
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
        (torch.int8, torch.float32),
        (torch.uint8, torch.float16),
        (torch.bool, torch.int8),
        (torch.bool, torch.uint8),
        (torch.float32, torch.bool),
        (torch.float16, torch.uint8),
        (torch.int8, torch.bool),
        (torch.uint8, torch.bool),
    ],
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_copy_inplace_mixed_dtype_triton(src_dtype, dst_dtype):
    device = flag_gems.device
    numel = 8

    if src_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        if src_dtype == torch.uint8:
            src = torch.randint(0, numel, (numel,), dtype=src_dtype, device=device)
        else:
            src = torch.randint(-numel, numel, (numel,), dtype=src_dtype, device=device)
    elif src_dtype is torch.bool:
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
        utils.FLOAT_DTYPES + [torch.int32, torch.int64, torch.int8, torch.uint8]
        if flag_gems.vendor_name == "cambricon"
        else utils.FLOAT_DTYPES + [torch.int8, torch.uint8]
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
        if dtype in [torch.int8, torch.uint8]:
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


@pytest.mark.copy
@pytest.mark.skipif(
    len(_FLOAT8_DTYPES) == 0, reason="No float8 is supported in current environment"
)
@pytest.mark.parametrize("dtype", _FLOAT8_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [
        (8,),
        (4, 4),
        (2, 3, 4),
        (1, 2, 3, 4),
        (1, 2, 3, 4, 5),
        (),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ],
)
def test_copy_functional_float8(dtype, shape):
    device = flag_gems.device

    src_uint8 = torch.randint(0, 256, shape, dtype=torch.uint8, device=device)

    src = src_uint8.view(dtype)
    ref_src = utils.to_reference(src)

    template = torch.zeros(shape, dtype=dtype, device=device)

    ref_dst = torch.empty_like(ref_src)

    ref_dst.copy_(ref_src)
    res_dst = flag_gems.copy(template, src)

    utils.gems_assert_equal(res_dst.view(torch.uint8), ref_dst.view(torch.uint8))

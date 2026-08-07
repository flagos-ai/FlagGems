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

import itertools

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def _to_copy_dtypes(dtypes):
    return [
        pytest.param(
            dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "kunlunxin"
                and not cfg.TO_CPU
                and dtype.is_complex,
                reason=(
                    "Kunlunxin PyTorch baseline does not implement real-to-complex "
                    "casts"
                ),
            ),
            id=str(dtype),
        )
        for dtype in dtypes
    ]


def _to_copy_pairs(src_dtypes, dst_dtypes):
    unsupported_xpu_baselines = {
        (torch.bfloat16, torch.int16),
        (torch.int16, torch.bfloat16),
    }
    return [
        pytest.param(
            src_dtype,
            dst_dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "kunlunxin"
                and not cfg.TO_CPU
                and (src_dtype, dst_dtype) in unsupported_xpu_baselines,
                reason=("Kunlunxin XDNN baseline does not implement this dtype cast"),
            ),
            id=f"{src_dtype}-{dst_dtype}",
        )
        for src_dtype, dst_dtype in itertools.product(src_dtypes, dst_dtypes)
    ]


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    _to_copy_dtypes(
        utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + utils.COMPLEX_DTYPES
    ),
)
def test_to_dtype(shape, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    x = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = ref_x.to(dtype)
    with flag_gems.use_gems():
        out = x.to(dtype)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "target_dtype", _to_copy_dtypes(utils.ALL_FLOAT_DTYPES + utils.COMPLEX_DTYPES)
)
def test_to_copy_dtype_cast(shape, target_dtype):
    if flag_gems.vendor_name == "tsingmicro" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("#2855: Skiping complex to_copy test on tsingmicro platform")
    if flag_gems.vendor_name == "ascend" and target_dtype in utils.COMPLEX_DTYPES:
        pytest.skip("Issues #3267: Ascend NPU does not support complex32 dtype")
    src_dtype = torch.float32 if target_dtype != torch.float32 else torch.float16
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=target_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=target_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize(
    "memory_format",
    [torch.preserve_format, torch.contiguous_format],
)
def test_to_copy_preserve_strides(memory_format):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and cfg.TO_CPU
        and memory_format is torch.preserve_format
    ):
        pytest.skip(
            "Kunlunxin and CPU baselines choose different suggested layouts "
            "for non-dense preserve_format inputs"
        )
    base = torch.randn((8, 16), dtype=torch.float32, device=flag_gems.device)
    x = base.transpose(0, 1)[::2]
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(
        ref_x,
        dtype=ref_x.dtype,
        memory_format=memory_format,
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(
            x,
            dtype=x.dtype,
            memory_format=memory_format,
        )
    utils.gems_assert_equal(res_out, ref_out)
    if memory_format is torch.preserve_format:
        assert res_out.stride() == ref_out.stride()
    else:
        assert res_out.is_contiguous()


# Generate (src, dst) pairs excluding same-dtype conversions
_FLOAT_TO_FLOAT_PAIRS = [
    (s, d)
    for s, d in itertools.product(utils.FLOAT_DTYPES, utils.ALL_FLOAT_DTYPES)
    if s != d
]


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _FLOAT_TO_FLOAT_PAIRS)
def test_to_copy_float_to_float(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and (
        src_dtype == torch.bfloat16 or dst_dtype == torch.bfloat16
    ):
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    _to_copy_pairs(
        utils.ALL_FLOAT_DTYPES,
        [torch.int8, torch.int16, torch.int32],
    ),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_float_to_int(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and src_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randn(shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    _to_copy_pairs(
        [torch.int8, torch.int16, torch.int32],
        utils.ALL_FLOAT_DTYPES,
    ),
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_int_to_float(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "ascend" and dst_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


# Generate (src, dst) int pairs excluding same-dtype conversions
_INT_DTYPES = [torch.int8, torch.int16, torch.int32]
_INT_TO_INT_PAIRS = list(itertools.permutations(_INT_DTYPES, 2))


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype,dst_dtype", _INT_TO_INT_PAIRS)
def test_to_copy_int_to_int(shape, src_dtype, dst_dtype):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(-100, 100, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("src_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_float_to_uint8(shape, src_dtype):
    if flag_gems.vendor_name == "ascend" and src_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    x = torch.randint(0, 255, shape, dtype=src_dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=torch.uint8)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=torch.uint8)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dst_dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_uint8_to_float(shape, dst_dtype):
    if flag_gems.vendor_name == "ascend" and dst_dtype == torch.bfloat16:
        pytest.skip("Ascend NPU may have issues with bfloat16")
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.to_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dst_dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy_uint8_to_int(shape, dst_dtype):
    if flag_gems.vendor_name == "cambricon":
        # Cambricon torch.randint currently does not support int8/int16 generation.
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device="cpu").to(
            flag_gems.device
        )
    else:
        x = torch.randint(0, 255, shape, dtype=torch.uint8, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten._to_copy(ref_x, dtype=dst_dtype)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._to_copy(x, dtype=dst_dtype)
    utils.gems_assert_equal(res_out, ref_out)

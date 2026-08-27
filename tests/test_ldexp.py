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


@pytest.mark.ldexp
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_ldexp(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randint(-8, 9, shape, device=flag_gems.device, dtype=torch.int32)

    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ldexp
@pytest.mark.parametrize(
    "self_dtype,other_dtype,expected_dtype",
    [
        (torch.bool, torch.int32, torch.float32),
        (torch.int32, torch.int64, torch.float32),
        (torch.float16, torch.int64, torch.float16),
        (torch.bfloat16, torch.int32, torch.bfloat16),
        (torch.float16, torch.float32, torch.float32),
        (torch.float32, torch.float64, torch.float64),
    ],
)
def test_ldexp_dtype_promotion(self_dtype, other_dtype, expected_dtype):
    shape = (37, 53)
    if self_dtype == torch.bool:
        self = torch.randint(0, 2, shape, device=flag_gems.device).bool()
    elif self_dtype.is_floating_point:
        self = torch.randn(shape, dtype=self_dtype, device=flag_gems.device)
    else:
        self = torch.randint(-8, 9, shape, dtype=self_dtype, device=flag_gems.device)
    if other_dtype.is_floating_point:
        other = torch.randn(shape, dtype=other_dtype, device=flag_gems.device) * 3
    else:
        other = torch.randint(-8, 9, shape, dtype=other_dtype, device=flag_gems.device)

    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    assert res_out.dtype == expected_dtype
    utils.gems_assert_close(res_out, ref_out, expected_dtype)


@pytest.mark.ldexp
def test_ldexp_broadcast_noncontiguous_and_empty():
    self = torch.randn((19, 7), device=flag_gems.device).T
    other = torch.randint(-8, 9, (19,), device=flag_gems.device, dtype=torch.int64)

    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)
    res_out = flag_gems.ldexp(self, other)
    utils.gems_assert_close(res_out, ref_out, torch.float32)

    empty = torch.empty((0, 7), device=flag_gems.device)
    empty_other = torch.empty((1, 7), dtype=torch.int32, device=flag_gems.device)
    empty_out = flag_gems.ldexp(empty, empty_other)
    assert empty_out.shape == (0, 7)
    assert empty_out.numel() == 0


@pytest.mark.ldexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_special_values(dtype):
    self = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    other = torch.tensor(
        [float("inf"), -float("inf"), 128.0, -128.0, 0.5, 0.0, 2.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
    res_cpu = res_out.cpu()
    ref_cpu = ref_out.cpu()
    valid = ~(torch.isnan(res_cpu) | torch.isnan(ref_cpu))
    utils.gems_assert_equal(
        torch.signbit(res_cpu)[valid],
        torch.signbit(ref_cpu)[valid],
    )


@pytest.mark.ldexp
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "tsingmicro"),
    reason="The backend does not support complex tensors",
)
def test_ldexp_complex():
    self = torch.randn((19, 7), dtype=torch.complex64, device=flag_gems.device)
    other = torch.randn((7,), dtype=torch.complex64, device=flag_gems.device)
    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other, True)
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    res_out = flag_gems.ldexp(self, other)

    utils.gems_assert_close(res_out, ref_out, torch.complex64)


@pytest.mark.ldexp_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_out_alias_resize_and_stride(dtype):
    self = torch.randn((11, 37), dtype=dtype, device=flag_gems.device)
    other = torch.randint(-8, 9, (37,), device=flag_gems.device, dtype=torch.int32)

    ref_self = utils.to_reference(self, True)
    ref_other = utils.to_reference(other)
    # Some supported PyTorch builds hit an internal CPU TensorIterator assertion
    # for BF16 ldexp.out. The functional overload has identical values, while
    # the GEMS call below still exercises the actual out overload semantics.
    ref_out = torch.ops.aten.ldexp.Tensor(ref_self, ref_other)

    storage = torch.empty((37, 11), dtype=dtype, device=flag_gems.device)
    res_out_buf = storage.T
    res_out = flag_gems.ldexp_out(self, other, out=res_out_buf)

    assert res_out is res_out_buf
    assert res_out.stride() == (1, 11)
    utils.gems_assert_close(res_out, ref_out, dtype)

    empty_out = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    resized = flag_gems.ldexp_out(self, other, out=empty_out)
    assert resized is empty_out
    assert resized.shape == self.shape


@pytest.mark.ldexp_out
def test_ldexp_out_rejects_invalid_dtype():
    self = torch.randn((17,), device=flag_gems.device)
    other = torch.randint(-4, 5, (17,), device=flag_gems.device)
    out = torch.empty((17,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ldexp_out(self, other, out=out)

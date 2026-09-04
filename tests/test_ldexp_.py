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


def _functional_reference(self, other):
    # Some supported PyTorch builds assert internally for CPU BF16 ldexp_.
    # The functional overload has the same values before the in-place cast.
    return torch.ops.aten.ldexp.Tensor(self, other).to(self.dtype)


@pytest.mark.ldexp_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_ldexp_(shape, dtype):
    self = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randint(-8, 9, shape, dtype=torch.int32, device=flag_gems.device)
    ref_self = utils.to_reference(self.clone(), True)
    ref_other = utils.to_reference(other)
    ref_out = _functional_reference(ref_self, ref_other)

    ptr = self.data_ptr()
    result = flag_gems.ldexp_(self, other)

    assert result is self
    assert result.data_ptr() == ptr
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.ldexp_
@pytest.mark.parametrize("other_dtype", [torch.int64, torch.float32, torch.float64])
@pytest.mark.parametrize("self_dtype", utils.FLOAT_DTYPES)
def test_ldexp_mixed_dtype_(self_dtype, other_dtype):
    shape = (37, 53)
    self = torch.randn(shape, dtype=self_dtype, device=flag_gems.device)
    if other_dtype.is_floating_point:
        other = torch.randn(shape, dtype=other_dtype, device=flag_gems.device) * 3
    else:
        other = torch.randint(-8, 9, shape, dtype=other_dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = _functional_reference(ref_self, ref_other)

    result = flag_gems.ldexp_(self, other)

    assert result.dtype == self_dtype
    utils.gems_assert_close(result, ref_out, self_dtype)


@pytest.mark.ldexp_
def test_ldexp_broadcast_noncontiguous_and_empty_():
    self = torch.randn((19, 7), device=flag_gems.device).T
    other = torch.randint(-8, 9, (19,), dtype=torch.int64, device=flag_gems.device)
    original_stride = self.stride()
    ref_self = utils.to_reference(self.clone(), True)
    ref_other = utils.to_reference(other)
    ref_out = _functional_reference(ref_self, ref_other)

    result = flag_gems.ldexp_(self, other)

    assert result is self
    assert result.stride() == original_stride
    utils.gems_assert_close(result, ref_out, torch.float32)

    empty = torch.empty((0, 7), device=flag_gems.device)
    empty_other = torch.empty((1, 7), dtype=torch.int32, device=flag_gems.device)
    result = flag_gems.ldexp_(empty, empty_other)
    assert result is empty
    assert result.shape == (0, 7)


@pytest.mark.ldexp_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ldexp_special_values_(dtype):
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
    ref_self = utils.to_reference(self.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = _functional_reference(ref_self, ref_other)

    result = flag_gems.ldexp_(self, other)

    utils.gems_assert_close(result, ref_out, dtype, equal_nan=True)
    result_cpu = result.cpu()
    ref_cpu = ref_out.cpu()
    valid = ~(torch.isnan(result_cpu) | torch.isnan(ref_cpu))
    utils.gems_assert_equal(
        torch.signbit(result_cpu)[valid], torch.signbit(ref_cpu)[valid]
    )


@pytest.mark.ldexp_
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "tsingmicro"),
    reason="The backend does not support complex tensors",
)
@pytest.mark.parametrize(
    "dtype",
    utils.COMPLEX_DTYPES + ([torch.complex128] if utils.fp64_is_supported else []),
)
def test_ldexp_complex_(dtype):
    self = torch.randn((19, 7), dtype=dtype, device=flag_gems.device)
    other = torch.randn((7,), dtype=dtype, device=flag_gems.device)
    ref_self = utils.to_reference(self.clone(), True)
    ref_other = utils.to_reference(other, True)
    ref_out = _functional_reference(ref_self, ref_other)

    result = flag_gems.ldexp_(self, other)

    assert result is self
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.ldexp_
@pytest.mark.parametrize("self_dtype", [torch.bool, torch.int32])
def test_ldexp_rejects_integer_self_(self_dtype):
    self = torch.ones((17,), dtype=self_dtype, device=flag_gems.device)
    other = torch.ones((17,), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ldexp_(self, other)


@pytest.mark.ldexp_
def test_ldexp_rejects_complex_to_real_():
    self = torch.ones((17,), dtype=torch.float32, device=flag_gems.device)
    other = torch.ones((17,), dtype=torch.complex64, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ldexp_(self, other)


@pytest.mark.ldexp_
def test_ldexp_rejects_expanding_self_():
    self = torch.ones((1, 7), device=flag_gems.device)
    other = torch.ones((3, 7), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.ldexp_(self, other)

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


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    if not dtype.is_floating_point:
        return torch.randint(-100, 101, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.msort
@pytest.mark.parametrize("shape", [(1,), (7,), (8, 17), (63, 129), (257, 7), (3, 5, 7)])
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.bool]
)
def test_msort(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.msort(ref_inp)

    result = flag_gems.msort(inp)

    assert result.dtype == inp.dtype
    assert result.stride() == inp.stride()
    utils.gems_assert_equal(result, ref_out)


@pytest.mark.msort
def test_msort_noncontiguous_empty_scalar_and_special_values():
    inp = torch.randn((19, 7), device=flag_gems.device).T
    ref_out = torch.msort(utils.to_reference(inp))
    result = flag_gems.msort(inp)
    assert result.stride() == inp.stride()
    utils.gems_assert_equal(result, ref_out)

    for shape in [(), (0,), (0, 7), (3, 0)]:
        inp = torch.empty(shape, device=flag_gems.device)
        result = flag_gems.msort(inp)
        assert result.shape == inp.shape
        assert result.stride() == inp.stride()

    inp = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 2.0, -3.0],
        device=flag_gems.device,
    )
    ref_out = torch.msort(utils.to_reference(inp))
    result = flag_gems.msort(inp)
    utils.gems_assert_equal(result, ref_out, equal_nan=True)

    # Exercise the large-first-dimension radix fallback.
    inp = torch.randn((1025, 3), device=flag_gems.device)
    ref_out = torch.msort(utils.to_reference(inp))
    utils.gems_assert_equal(flag_gems.msort(inp), ref_out)


@pytest.mark.msort_out
@pytest.mark.parametrize("shape", [(8, 17), (63, 129), (257, 7), (3, 5, 7)])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_msort_out(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_out = torch.msort(utils.to_reference(inp))
    out = torch.empty_like(inp)
    ptr = out.data_ptr()

    result = flag_gems.msort_out(inp, out=out)

    assert result is out
    assert result.data_ptr() == ptr
    utils.gems_assert_equal(result, ref_out)


@pytest.mark.msort_out
def test_msort_out_resize_noncontiguous_alias_and_dtype_error():
    inp = torch.randn((19, 7), device=flag_gems.device).T
    ref_out = torch.msort(utils.to_reference(inp))
    out = torch.empty((1,), device=flag_gems.device)
    result = flag_gems.msort_out(inp, out=out)
    assert result is out
    assert result.shape == inp.shape
    utils.gems_assert_equal(result, ref_out)

    alias = torch.randn((31, 13), device=flag_gems.device)
    ref_alias = torch.msort(utils.to_reference(alias))
    result = flag_gems.msort_out(alias, out=alias)
    assert result is alias
    utils.gems_assert_equal(result, ref_alias)

    with pytest.raises(RuntimeError):
        flag_gems.msort_out(
            inp,
            out=torch.empty(inp.shape, dtype=torch.float16, device=inp.device),
        )

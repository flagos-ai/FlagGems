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
    torch.manual_seed(0)
    if not dtype.is_floating_point:
        return torch.randint(-100, 101, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.msort_out
@pytest.mark.parametrize("shape", [(8, 17), (63, 129), (257, 7), (3, 5, 7)])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_msort_out_content_identity_and_storage(shape, dtype):
    inp = _make_input(shape, dtype)
    reference = torch.msort(utils.to_reference(inp))
    out = torch.empty_like(inp)
    pointer = out.data_ptr()

    result = flag_gems.msort_out(inp, out=out)

    assert result is out
    assert result.data_ptr() == pointer
    utils.gems_assert_equal(result, reference)


@pytest.mark.msort_out
@pytest.mark.parametrize("rows", [17, 512, 513, 1025])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("view_alias", [False, True])
def test_msort_out_alias_across_sort_paths(rows, dtype, view_alias):
    if flag_gems.vendor_name == "mthreads" and dtype == torch.float64 and not utils.TO_CPU:
        pytest.skip("MUSA native msort does not support float64")
    inp = _make_input((rows, 129), dtype)
    reference_input = utils.to_reference(inp).clone()
    reference = torch.msort(reference_input, out=reference_input)
    out = inp.view_as(inp) if view_alias else inp
    pointer, strides = out.data_ptr(), out.stride()

    result = flag_gems.msort_out(inp, out=out)

    assert result is out
    assert result.data_ptr() == pointer
    assert result.stride() == strides
    utils.gems_assert_equal(result, reference)


@pytest.mark.msort_out
def test_msort_out_resize_noncontiguous_alias_and_dtype_error():
    torch.manual_seed(0)
    inp = torch.randn((19, 7), device=flag_gems.device).T
    reference = torch.msort(utils.to_reference(inp))
    out = torch.empty((1,), device=flag_gems.device)

    result = flag_gems.msort_out(inp, out=out)

    assert result is out
    assert result.shape == inp.shape
    utils.gems_assert_equal(result, reference)

    alias = torch.randn((31, 13), device=flag_gems.device)
    reference_alias = torch.msort(utils.to_reference(alias))
    result = flag_gems.msort_out(alias, out=alias)
    assert result is alias
    utils.gems_assert_equal(result, reference_alias)

    with pytest.raises(RuntimeError):
        flag_gems.msort_out(
            inp,
            out=torch.empty(inp.shape, dtype=torch.float16, device=inp.device),
        )

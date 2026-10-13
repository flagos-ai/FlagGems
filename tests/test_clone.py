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


@pytest.mark.clone
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone(shape, dtype):
    """clone copies the tensor contents into an independent storage."""
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_out = ref_x.clone()
    res_out = x.clone()

    utils.gems_assert_equal(res_out, ref_out)
    # clone must not alias the source storage
    assert res_out.data_ptr() != x.data_ptr()


@pytest.mark.clone
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_memory_format(dtype):
    """clone(preserve_format) keeps the source memory layout."""
    base = torch.randn((16, 8), dtype=dtype, device=flag_gems.device)
    x = base.t().contiguous().t()[:12]  # non-contiguous view
    ref_x = utils.to_reference(x)
    ref_out = ref_x.clone(memory_format=torch.preserve_format)
    res_out = x.clone(memory_format=torch.preserve_format)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.is_contiguous() == x.is_contiguous()
    assert res_out.data_ptr() != x.data_ptr()

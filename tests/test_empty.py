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
from . import conftest as cfg

device = flag_gems.device


EMPTY_PERMUTED_LAYOUTS = [
    ((8,), [0]),
    ((4, 8), [0, 1]),
    ((4, 8), [1, 0]),
    ((2, 3, 4), [0, 1, 2]),
    ((2, 3, 4), [2, 0, 1]),
    ((2, 3, 4, 5), [0, 2, 3, 1]),
]


@pytest.mark.empty_permuted
@pytest.mark.parametrize("shape,physical_layout", EMPTY_PERMUTED_LAYOUTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_empty_permuted(shape, physical_layout, dtype, caplog):
    # empty_permuted returns uninitialized memory, so the layout contract is
    # verified against the reference instead of the element values. The
    # reference is built on the same device to keep the comparison about the
    # layout only.
    ref_out = torch.ops.aten.empty_permuted(
        shape, physical_layout, dtype=dtype, device=flag_gems.device
    )
    with caplog.at_level("DEBUG", logger="flag_gems.ops.empty_permuted"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.empty_permuted(
                shape, physical_layout, dtype=dtype, device=flag_gems.device
            )

    assert "GEMS EMPTY_PERMUTED" in caplog.text
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.dtype == ref_out.dtype
    assert res_out.numel() == ref_out.numel()
    assert res_out.device == ref_out.device
    # A permuted layout stays non-overlapping and dense.
    assert torch.ops.aten.is_non_overlapping_and_dense(res_out)


@pytest.mark.empty
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_empty(shape, dtype):
    expected_dev = "cpu" if cfg.TO_CPU else device
    with flag_gems.use_gems():
        res_out = torch.empty(*shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.zeros(*shape, dtype=dtype, device=expected_dev)
    ref_out = utils.to_reference(ref_out, True)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.empty
def test_empty_default_dtype():
    # Tests empty() with default dtype (not explicitly specified) to verify
    # proper dtype inference when only shape and device are given.
    expected_dev = "cpu" if cfg.TO_CPU else device
    with flag_gems.use_gems():
        res_out = torch.empty(10, 20, device=flag_gems.device)

    ref_out = torch.zeros(10, 20, device=expected_dev)
    ref_out = utils.to_reference(ref_out, True)
    utils.gems_assert_close(res_out, ref_out, torch.get_default_dtype())

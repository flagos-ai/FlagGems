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
from flag_gems.ops.sym_constrain_range import sym_constrain_range as gems_impl

# Scalar bounds for sym_constrain_range - covering min-only, max-only, both,
# unconstrained, and negative ranges. sym_constrain_range operates on a scalar
# symbolic size (not a tensor), so shapes/dtypes do not apply.
SYM_CONSTRAIN_RANGE_CASES = [
    # (size, min, max)
    (10, 0, 100),
    (50, 0, None),
    (50, None, 100),
    (42, None, None),
    (-5, -10, 0),
]

# Bounds violations that must raise, mirroring the ATen reference behaviour.
SYM_CONSTRAIN_RANGE_VIOLATIONS = [
    # (size, min, max)
    (5, 10, 100),
    (200, 0, 100),
    (-1, 0, None),
]


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_val, max_val", SYM_CONSTRAIN_RANGE_CASES)
def test_sym_constrain_range(size, min_val, max_val):
    """Test sym_constrain_range against the ATen reference.

    sym_constrain_range validates that a scalar symbolic integer lies in
    ``[min, max]`` and returns nothing (void). We assert the FlagGems
    implementation matches ATen: both accept in-range values and return None.
    """
    ref_out = torch.ops.aten.sym_constrain_range(size, min=min_val, max=max_val)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.sym_constrain_range(size, min=min_val, max=max_val)

    assert res_out is None
    assert ref_out is None


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_val, max_val", SYM_CONSTRAIN_RANGE_CASES)
def test_sym_constrain_range_gems_impl(size, min_val, max_val, caplog):
    """Directly exercise the FlagGems Triton implementation.

    ``torch.ops.aten.sym_constrain_range`` may resolve to the built-in ATen
    kernel, so we call the FlagGems wrapper directly to guarantee the Triton
    range-check kernel runs (verified via the debug log) and returns None.
    """
    with caplog.at_level("DEBUG", logger="flag_gems.ops.sym_constrain_range"):
        res_out = gems_impl(size, min=min_val, max=max_val)

    assert "GEMS SYM_CONSTRAIN_RANGE" in caplog.text
    assert res_out is None


@pytest.mark.sym_constrain_range
@pytest.mark.parametrize("size, min_val, max_val", SYM_CONSTRAIN_RANGE_VIOLATIONS)
def test_sym_constrain_range_out_of_bounds(size, min_val, max_val):
    """sym_constrain_range must raise when the value violates the bounds.

    The FlagGems Triton implementation and the ATen reference must agree.
    """
    with pytest.raises(RuntimeError):
        torch.ops.aten.sym_constrain_range(size, min=min_val, max=max_val)
    with pytest.raises(RuntimeError):
        gems_impl(size, min=min_val, max=max_val)

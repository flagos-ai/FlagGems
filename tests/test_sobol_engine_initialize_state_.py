# Copyright 2026, The FlagOS Contributors.
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

DIMENSIONS = [1, 2, 3, 5, 10, 20, 50, 100]


@pytest.mark.sobol_engine_initialize_state_
@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_sobol_engine_initialize_state_(dimension):
    """Test _sobol_engine_initialize_state_ operator.

    Note: The reference is computed on CPU because PyTorch's native CUDA
    implementation of this operator is unreliable (can segfault). The Sobol
    direction numbers are deterministic integer constants, so CPU and the
    gems implementation must produce identical results.
    """
    # gems input on the target device
    state = torch.zeros((dimension, 30), dtype=torch.int64, device=flag_gems.device)

    # Reference: compute Sobol direction numbers on CPU (deterministic constants),
    # then place them on the target device so to_reference handles the device the
    # same way it does for any other op. We compute on CPU because PyTorch's native
    # CUDA path for this op is unreliable (can segfault); the values are exact
    # integer constants either way.
    ref_state = torch.zeros((dimension, 30), dtype=torch.int64, device="cpu")
    torch._sobol_engine_initialize_state_(ref_state, dimension)
    ref_out = utils.to_reference(ref_state.to(flag_gems.device))

    # gems implementation (modifies state in place and returns it)
    res_out = flag_gems.ops._sobol_engine_initialize_state_(state, dimension)

    # Operator is in-place: the returned tensor must be the same object
    assert res_out is state, "Result should be the same object (in-place)"

    utils.gems_assert_equal(res_out, ref_out)

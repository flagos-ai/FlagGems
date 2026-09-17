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

DIMENSIONS = [1, 2, 3, 5, 8, 10]
N_VALUES = [1, 5, 10, 50, 100, 256, 1000]
NUM_GENERATED_VALUES = [0, 10, 50, 100, 512]


@pytest.mark.sobol_engine_ff_
@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("num_generated", NUM_GENERATED_VALUES)
def test_sobol_engine_ff_(dimension, n, num_generated):
    """Test _sobol_engine_ff_ operator for Sobol sequence fast-forward."""
    MAXBIT = 30

    # Create inputs on device
    quasi_gems = torch.zeros(dimension, dtype=torch.long, device=flag_gems.device)
    sobolstate_gems = torch.randint(
        0, 2**30, (dimension, MAXBIT), dtype=torch.long, device=flag_gems.device
    )

    # Create reference inputs
    quasi_ref = quasi_gems.clone().cpu()
    sobolstate_ref = sobolstate_gems.cpu()

    # Run reference
    torch._sobol_engine_ff_(quasi_ref, n, sobolstate_ref, dimension, num_generated)

    # Run gems kernel
    res_out = flag_gems.ops._sobol_engine_ff_(
        quasi_gems, n, sobolstate_gems, dimension, num_generated
    )

    # The native aten op only supports CPU (it segfaults on CUDA), so the
    # reference is computed on CPU. Move the gems result to CPU so both
    # tensors share the same device in every CI mode.
    utils.gems_assert_equal(res_out.cpu(), quasi_ref)

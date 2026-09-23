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
from torch.quasirandom import SobolEngine

import flag_gems


@pytest.mark.underscore_sobol_engine_draw
@pytest.mark.parametrize("n", [10, 100, 1000])
@pytest.mark.parametrize("dimension", [2, 5, 10, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_accuracy_underscore_sobol_engine_draw(n, dimension, dtype):
    # Initialize SobolEngine to get valid state
    eng = SobolEngine(dimension=dimension, scramble=False)

    # Copy to GPU
    quasi_cpu = eng.quasi.clone()
    sobolstate_cpu = eng.sobolstate.clone()

    quasi_cuda = quasi_cpu.cuda()
    sobolstate_cuda = sobolstate_cpu.cuda()

    # Generate reference on CPU
    result_ref, quasi_ref = torch._sobol_engine_draw(
        quasi_cpu, n, sobolstate_cpu, dimension, 0, dtype=dtype
    )

    # Generate with FlagGems
    result_gems, quasi_gems = flag_gems.underscore_sobol_engine_draw(
        quasi_cuda, n, sobolstate_cuda, dimension, 0, dtype=dtype
    )

    # Check results
    assert result_gems.shape == (n, dimension)
    assert result_gems.dtype == dtype
    assert quasi_gems.shape == (dimension,)
    assert quasi_gems.dtype == torch.int64

    # Check values are in valid range [0, 1)
    assert (result_gems >= 0).all()
    assert (result_gems < 1).all()

    # Check match with CPU reference
    torch.testing.assert_close(result_gems.cpu(), result_ref, rtol=1e-5, atol=1e-8)
    assert torch.equal(quasi_gems.cpu(), quasi_ref)


@pytest.mark.underscore_sobol_engine_draw
@pytest.mark.parametrize("num_generated", [0, 10, 100])
def test_accuracy_underscore_sobol_engine_draw_with_offset(num_generated):
    # Test with non-zero num_generated
    n = 50
    dimension = 3
    dtype = torch.float32

    eng = SobolEngine(dimension=dimension, scramble=False)

    # Advance state by num_generated
    if num_generated > 0:
        eng.draw(num_generated)

    quasi_cpu = eng.quasi.clone()
    sobolstate_cpu = eng.sobolstate.clone()

    quasi_cuda = quasi_cpu.cuda()
    sobolstate_cuda = sobolstate_cpu.cuda()

    # Generate reference on CPU (note: num_generated - 1 in the call, as per SobolEngine.draw)
    result_ref, quasi_ref = torch._sobol_engine_draw(
        quasi_cpu,
        n,
        sobolstate_cpu,
        dimension,
        num_generated - 1 if num_generated > 0 else 0,
        dtype=dtype,
    )

    # Generate with FlagGems
    result_gems, quasi_gems = flag_gems.underscore_sobol_engine_draw(
        quasi_cuda,
        n,
        sobolstate_cuda,
        dimension,
        num_generated - 1 if num_generated > 0 else 0,
        dtype=dtype,
    )

    # Check match
    torch.testing.assert_close(result_gems.cpu(), result_ref, rtol=1e-5, atol=1e-8)
    assert torch.equal(quasi_gems.cpu(), quasi_ref)


@pytest.mark.underscore_sobol_engine_draw
def test_accuracy_underscore_sobol_engine_draw_edge_cases():
    # Test edge cases
    dimension = 1
    n = 1
    dtype = torch.float32

    eng = SobolEngine(dimension=dimension, scramble=False)
    quasi_cuda = eng.quasi.cuda()
    sobolstate_cuda = eng.sobolstate.cuda()

    result_ref, quasi_ref = torch._sobol_engine_draw(
        eng.quasi, n, eng.sobolstate, dimension, 0, dtype=dtype
    )

    result_gems, quasi_gems = flag_gems.underscore_sobol_engine_draw(
        quasi_cuda, n, sobolstate_cuda, dimension, 0, dtype=dtype
    )

    torch.testing.assert_close(result_gems.cpu(), result_ref, rtol=1e-5, atol=1e-8)
    assert torch.equal(quasi_gems.cpu(), quasi_ref)


@pytest.mark.underscore_sobol_engine_draw
def test_accuracy_underscore_sobol_engine_draw_large():
    # Test larger dimensions
    n = 500
    dimension = 50
    dtype = torch.float32

    eng = SobolEngine(dimension=dimension, scramble=False)
    quasi_cuda = eng.quasi.cuda()
    sobolstate_cuda = eng.sobolstate.cuda()

    result_ref, quasi_ref = torch._sobol_engine_draw(
        eng.quasi, n, eng.sobolstate, dimension, 0, dtype=dtype
    )

    result_gems, quasi_gems = flag_gems.underscore_sobol_engine_draw(
        quasi_cuda, n, sobolstate_cuda, dimension, 0, dtype=dtype
    )

    torch.testing.assert_close(result_gems.cpu(), result_ref, rtol=1e-5, atol=1e-8)
    assert torch.equal(quasi_gems.cpu(), quasi_ref)

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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    flag_gems.exponential_(x)

    assert x.min() > 0


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_fast(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    flag_gems.exponential_(x)

    x_res = utils.to_reference(x)
    mean_res = torch.mean(x_res.to(torch.float32)).to(dtype)
    var_res = torch.var(x_res.to(torch.float32)).to(dtype)
    mean_ref = 1.0 / lambd
    var_ref = 1.0 / (lambd**2)

    assert torch.abs(mean_res - mean_ref) < mean_tol
    assert torch.abs(var_res - var_ref) < var_tol


def _exponential_samples(backend, seed, size):
    x = torch.empty(size, device=flag_gems.device, dtype=torch.float32)
    generator = torch.Generator(device=flag_gems.device).manual_seed(seed)
    if backend == "flaggems":
        flag_gems.exponential_(x, generator=generator)
    else:
        x.exponential_(generator=generator)
    return x


@pytest.mark.exponential_
@pytest.mark.parametrize("backend", ["torch", "flaggems"])
@pytest.mark.parametrize("seed", [260828, 2026, 7])
def test_exponential_lower_tail(backend, seed):
    x = _exponential_samples(backend, seed, 1048576)
    expected = 1 - math.exp(-0.1)
    observed = (x <= 0.1).float().mean().item()
    assert abs(observed - expected) < 0.005, (
        f"{backend} seed={seed}: P(X<=0.1)={observed}, "
        f"Exp(1) expected={expected}; tolerance=0.005"
    )


@pytest.mark.exponential_
@pytest.mark.parametrize("backend", ["torch", "flaggems"])
@pytest.mark.parametrize("seed", [260828, 2026, 7])
def test_exponential_no_repeated_blocks(backend, seed):
    x = _exponential_samples(backend, seed, 131072)
    fractions = {}
    for block in [64, 128, 256, 512]:
        tiles = x.view(-1, 8 * block)
        fractions[block] = (
            (tiles[:-1, 4 * block :] == tiles[1:, : 4 * block]).float().mean().item()
        )
    assert max(fractions.values()) < 0.01, (
        f"{backend} seed={seed}: repeated-block fractions={fractions}; "
        "different output coordinates must not reuse entire random blocks"
    )


@pytest.mark.exponential_
@pytest.mark.parametrize("seed", [260828, 2026, 7])
def test_exponential_far_left_tail(seed):
    """The far left tail must not carry an inflated atom (Issue #7040).

    The fast transform pinned every sample with ``u >= 1 - 1e-4`` to
    ``y == 0.5 * eps(float32) = 5.96e-8``, so ~1e-4 of the mass sat just above
    zero: ``P(X < 1e-5)`` measured ~1e-4 instead of ~1e-5. That atom is what
    breaks exponential-race / Gumbel-max sampling in vLLM.
    """
    x = _exponential_samples("flaggems", seed, 1 << 22)

    assert x.min().item() > 0, "exponential_ must not produce non-positive values"
    # Exp(1): P(X < t) = 1 - exp(-t) ~= t for small t. Bound each point by a few
    # times the true mass; the buggy atom overshoots by ~90x.
    for t, bound in ((1e-6, 5e-6), (1e-5, 3e-5)):
        observed = (x < t).float().mean().item()
        assert (
            observed < bound
        ), f"seed={seed}: P(X < {t:.0e}) = {observed:.3e}, expected ~{t:.0e}"

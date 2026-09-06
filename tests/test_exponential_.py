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
from contextlib import nullcontext

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        x.exponential_()

    assert x.min() > 0


@pytest.mark.exponential_
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_exponential_fast(shape, dtype):
    x = torch.empty(size=shape, dtype=dtype, device=flag_gems.device)
    lambd = 1.0
    mean_tol = 0.05
    var_tol = 0.05
    with flag_gems.use_gems():
        x.exponential_()

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
    context = (
        flag_gems.use_gems(include=["exponential_"])
        if backend == "flaggems"
        else nullcontext()
    )
    with context:
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

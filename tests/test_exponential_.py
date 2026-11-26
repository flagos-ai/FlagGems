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
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.random_utils import (
    philox_backend_seed_offset,
    set_philox_state,
)

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


def _assert_no_repeated_unroll8_boundaries(x):
    blocks = (64, 128, 256, 512)
    n = x.numel()
    for block in blocks:
        full_ctas = n // (8 * block)
        if full_ctas < 2:
            continue
        tiles = x[: full_ctas * 8 * block].view(full_ctas, 8, block)
        assert not torch.equal(tiles[:-1, 4:, :], tiles[1:, :4, :])


@pytest.mark.exponential_
def test_exponential_f32_unroll8_philox_ranges():
    n = 32768
    seed = 123
    device = torch_device_fn.current_device()
    generator = torch_device_fn.default_generators[device]
    original_state = generator.get_state()

    def sample(size=n):
        x = torch.empty((size,), dtype=torch.float32, device=flag_gems.device)
        flag_gems.exponential_(x)
        torch_device_fn.synchronize()
        return x

    def current_offset():
        return philox_backend_seed_offset(0)[1]

    try:
        set_philox_state(seed, 0, device)
        start_offset = current_offset()
        first = sample()
        first_end_offset = current_offset()
        second = sample()
        second_end_offset = current_offset()
        assert first_end_offset - start_offset == n
        assert second_end_offset - first_end_offset == n

        set_philox_state(seed, 0, device)
        replay = sample()
        assert torch.equal(first, replay)

        _assert_no_repeated_unroll8_boundaries(first)

        # Before the fix, the next call advanced by rounded ceil(N / 8),
        # which reused the first group of one earlier CTA.
        old_increment = ((n // 8 + 3) // 4) * 4
        for block in (64, 128, 256, 512):
            cta = old_increment // (4 * block)
            assert old_increment % (4 * block) == 0
            tiles = first.view(n // (8 * block), 8, block)
            next_tiles = second.view(n // (8 * block), 8, block)
            assert not torch.equal(tiles[cta, :4, :], next_tiles[0, :4, :])

        set_philox_state(seed, (1 << 32) - 4, device)
        near_wrap = sample()
        set_philox_state(seed, 1 << 32, device)
        at_wrap = sample()
        carry_blocks = []
        for block in (64, 128, 256, 512):
            if all(
                torch.equal(
                    near_wrap[part * block + 4 : (part + 1) * block],
                    at_wrap[part * block : (part + 1) * block - 4],
                )
                for part in range(4)
            ):
                carry_blocks.append(block)
        assert carry_blocks

        set_philox_state(seed, 0, device)
        tail = sample(n + 1)
        assert torch.isfinite(tail).all()
        assert tail.min() > 0
        _assert_no_repeated_unroll8_boundaries(tail)
    finally:
        generator.set_state(original_state)

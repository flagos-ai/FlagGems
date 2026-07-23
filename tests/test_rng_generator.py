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

"""Verify RNG ops actually use the user-supplied `generator=` argument.

Each test asserts two properties:
  - Two calls with same-seeded generators produce identical output.
  - Two calls with differently-seeded generators produce different output.

The first property catches the case where `generator=` is accepted in the
signature but silently dropped: both calls would then fall back to the
default generator, which advances between calls, so outputs differ and
the equality assert fires.
"""

import pytest
import torch

import flag_gems

device = flag_gems.device


def _gen(seed):
    return torch.Generator(device=device).manual_seed(seed)


@pytest.mark.rand
def test_rand_generator_propagation():
    shape = (128, 128)
    with flag_gems.use_gems():
        out_a = torch.rand(shape, device=device, generator=_gen(42))
        out_b = torch.rand(shape, device=device, generator=_gen(42))
        out_c = torch.rand(shape, device=device, generator=_gen(999))
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


@pytest.mark.randn
def test_randn_generator_propagation():
    shape = (128, 128)
    with flag_gems.use_gems():
        out_a = torch.randn(shape, device=device, generator=_gen(42))
        out_b = torch.randn(shape, device=device, generator=_gen(42))
        out_c = torch.randn(shape, device=device, generator=_gen(999))
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


@pytest.mark.normal_
def test_normal_inplace_generator_propagation():
    shape = (128, 128)
    with flag_gems.use_gems():
        buf = torch.empty(shape, device=device)
        buf.normal_(0.0, 1.0, generator=_gen(42))
        out_a = buf.clone()
        buf.normal_(0.0, 1.0, generator=_gen(42))
        out_b = buf.clone()
        buf.normal_(0.0, 1.0, generator=_gen(999))
        out_c = buf.clone()
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


@pytest.mark.normal_tensor_tensor
def test_normal_tensor_tensor_generator_propagation():
    shape = (128, 128)
    mean = torch.zeros(shape, device=device)
    std = torch.ones(shape, device=device)
    with flag_gems.use_gems():
        out_a = torch.normal(mean, std, generator=_gen(42))
        out_b = torch.normal(mean, std, generator=_gen(42))
        out_c = torch.normal(mean, std, generator=_gen(999))
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


@pytest.mark.normal_tensor_float
def test_normal_tensor_float_generator_propagation():
    shape = (128, 128)
    mean = torch.zeros(shape, device=device)
    with flag_gems.use_gems():
        out_a = torch.normal(mean, 1.0, generator=_gen(42))
        out_b = torch.normal(mean, 1.0, generator=_gen(42))
        out_c = torch.normal(mean, 1.0, generator=_gen(999))
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


@pytest.mark.normal_float_tensor
def test_normal_float_tensor_generator_propagation():
    shape = (128, 128)
    std = torch.ones(shape, device=device)
    with flag_gems.use_gems():
        out_a = torch.normal(0.0, std, generator=_gen(42))
        out_b = torch.normal(0.0, std, generator=_gen(42))
        out_c = torch.normal(0.0, std, generator=_gen(999))
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


@pytest.mark.multinomial
def test_multinomial_generator_propagation():
    # Cover both code paths: replacement=True and replacement=False (default),
    # since they use different RNG entry points internally.
    probs = torch.rand(64, device=device) + 0.1
    with flag_gems.use_gems():
        for replacement in (True, False):
            out_a = torch.multinomial(
                probs, 8, replacement=replacement, generator=_gen(42)
            )
            out_b = torch.multinomial(
                probs, 8, replacement=replacement, generator=_gen(42)
            )
            out_c = torch.multinomial(
                probs, 8, replacement=replacement, generator=_gen(999)
            )
            assert torch.equal(out_a, out_b), f"replacement={replacement}"
            assert not torch.equal(out_a, out_c), f"replacement={replacement}"

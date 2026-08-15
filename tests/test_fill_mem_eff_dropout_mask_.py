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

# `_fill_mem_eff_dropout_mask_` is an internal helper used by the memory
# efficient attention path. It only operates on 4D float32 tensors of
# shape ``(batch, heads, queries, keys)`` and fills the tensor in place
# with random uniform values in [0, 1) drawn from a Philox4x32-10 stream
# identified by ``seed`` and ``offset``. The values depend on neither
# ``dropout_p`` nor the tensor's prior content.
# Representative (batch, heads, queries, keys) attention mask shapes.
MASK_SHAPES = (
    [(1, 1, 4, 8)]
    if utils.QUICK_MODE
    else [
        (1, 1, 4, 8),
        (2, 3, 4, 8),
        (1, 1, 16, 8),
        (1, 1, 4, 16),
        (1, 1, 4, 32),
        (1, 2, 32, 64),
        (4, 8, 64, 64),
    ]
)
# The ATen op is float32-only (curand uniform); see kernel dtype assertion.
MASK_DTYPES = [torch.float32]


@pytest.mark.fill_mem_eff_dropout_mask_
@pytest.mark.parametrize("shape", MASK_SHAPES)
@pytest.mark.parametrize("seed_offset", [(42, 0), (42, 100), (7, 0), (7, 1024)])
def test_fill_mem_eff_dropout_mask_(shape, seed_offset):
    seed, offset = seed_offset

    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten._fill_mem_eff_dropout_mask_(ref_inp, 0.0, seed, offset)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._fill_mem_eff_dropout_mask_(inp, 0.0, seed, offset)

    # In-place semantics: the returned tensor aliases the input.
    utils.gems_assert_close(inp, ref_inp, torch.float32, atol=1e-6)
    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=1e-6)


@pytest.mark.fill_mem_eff_dropout_mask_
@pytest.mark.parametrize("shape", MASK_SHAPES)
def test_fill_mem_eff_dropout_mask__value_range(shape):
    # The kernel must produce values strictly in [0, 1).
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._fill_mem_eff_dropout_mask_(inp, 0.0, 42, 0)

    assert (res_out >= 0.0).all()
    assert (res_out < 1.0).all()


@pytest.mark.fill_mem_eff_dropout_mask_
@pytest.mark.parametrize("shape", MASK_SHAPES)
def test_fill_mem_eff_dropout_mask__inplace(shape):
    # The op writes into the supplied tensor in place.
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_ret = torch.ops.aten._fill_mem_eff_dropout_mask_(ref_inp, 0.0, 42, 0)
    with flag_gems.use_gems():
        res_ret = torch.ops.aten._fill_mem_eff_dropout_mask_(inp, 0.0, 42, 0)

    # The returned tensor is the same object as the input (in-place semantics).
    assert res_ret.data_ptr() == inp.data_ptr()
    utils.gems_assert_close(inp, ref_inp, torch.float32, atol=1e-6)
    utils.gems_assert_close(res_ret, ref_ret, torch.float32, atol=1e-6)

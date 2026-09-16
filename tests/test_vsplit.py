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

VSPLIT_CONFIGS = [
    # (shape, indices_or_sections)
    # Integer splits. torch.vsplit requires dim 0 to divide evenly.
    ((4, 6), 2),
    ((8, 4), 4),
    ((12, 8, 3), 4),
    ((16, 16), 2),
    ((20, 10, 5), 5),
    # Index splits (custom boundaries)
    ((4, 6), [2]),
    ((6, 8), [2, 4]),
    ((8, 4, 2), [3, 6]),
    ((10, 5), [5]),
    # Index splits that produce an empty chunk or clamp out of range
    ((4, 3), [2, 2]),
    ((4, 3), [10]),
    ((4, 3), []),
    # 4D
    ((4, 6, 8, 3), 2),
    ((8, 4, 2, 5), 4),
]


@pytest.mark.vsplit
@pytest.mark.parametrize("shape, indices_or_sections", VSPLIT_CONFIGS)
def test_accuracy_vsplit(shape, indices_or_sections):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(indices_or_sections, int):
        ref_out = torch.ops.aten.vsplit.int(ref_inp, indices_or_sections)
    else:
        ref_out = torch.ops.aten.vsplit.array(ref_inp, indices_or_sections)

    res_out = flag_gems.vsplit(inp, indices_or_sections)

    assert len(res_out) == len(
        ref_out
    ), f"Length mismatch: {len(res_out)} vs {len(ref_out)}"
    for res_chunk, ref_chunk in zip(res_out, ref_out):
        assert res_chunk.shape == ref_chunk.shape
        utils.gems_assert_close(res_chunk, ref_chunk, torch.float32)


@pytest.mark.vsplit
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8])
def test_accuracy_vsplit_dtypes(dtype):
    """vsplit only moves data, so it must work for non-float dtypes too."""
    if dtype == torch.int8:
        inp = torch.randint(-128, 127, (8, 4), dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn((8, 4), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.vsplit.int(ref_inp, 2)
    res_out = flag_gems.vsplit(inp, 2)

    assert len(res_out) == len(ref_out)
    for res_chunk, ref_chunk in zip(res_out, ref_out):
        assert torch.equal(res_chunk.cpu(), ref_chunk.cpu())


@pytest.mark.vsplit
def test_accuracy_vsplit_non_contiguous():
    inp = torch.randn((8, 6), dtype=torch.float32, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.vsplit.int(ref_inp, 2)
    res_out = flag_gems.vsplit(inp, 2)

    assert len(res_out) == len(ref_out)
    for res_chunk, ref_chunk in zip(res_out, ref_out):
        utils.gems_assert_close(res_chunk, ref_chunk, torch.float32)


@pytest.mark.vsplit
def test_vsplit_non_divisible_raises():
    """torch.vsplit rejects a dim 0 that does not divide evenly."""
    inp = torch.randn((7, 4), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.vsplit(inp, 2)


@pytest.mark.vsplit
def test_vsplit_requires_2d():
    """torch.vsplit requires a tensor with two or more dimensions."""
    inp = torch.randn((8,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.vsplit(inp, 2)

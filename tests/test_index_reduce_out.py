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


REDUCTIONS = ("prod", "mean", "amax", "amin")


def _inputs(dtype=torch.float32):
    torch.manual_seed(0)
    inp = torch.randn((4, 6), dtype=dtype, device=flag_gems.device)
    source = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    index = (torch.arange(8, device=flag_gems.device) * 3 + 1) % 6
    return inp, index, source


def _reference(inp, index, source, reduce, include_self, out):
    return torch.index_reduce(
        utils.to_reference(inp, True),
        1,
        utils.to_reference(index, False),
        utils.to_reference(source, True),
        reduce,
        include_self=include_self,
        out=out,
    )


@pytest.mark.index_reduce_out
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("include_self", (True, False))
def test_index_reduce_out_resizes(reduce, include_self):
    inp, index, source = _inputs()
    out = torch.empty((0,), dtype=inp.dtype, device=flag_gems.device)
    ref_out = torch.empty_like(utils.to_reference(inp, True))
    _reference(inp, index, source, reduce, include_self, ref_out)

    result = flag_gems.index_reduce_out(
        inp,
        1,
        index,
        source,
        reduce,
        include_self=include_self,
        out=out,
    )

    assert result is out
    assert out.device == inp.device
    assert out.shape == inp.shape
    assert out.dtype == inp.dtype
    utils.gems_assert_close(out, ref_out, dtype=inp.dtype, reduce_dim=source.size(1))


@pytest.mark.index_reduce_out
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_index_reduce_out_preserves_storage(reduce):
    inp, index, source = _inputs()
    out = torch.empty_like(inp)
    pointer = out.data_ptr()
    ref_out = torch.empty_like(utils.to_reference(inp, True))
    _reference(inp, index, source, reduce, True, ref_out)

    result = flag_gems.index_reduce_out(
        inp, 1, index, source, reduce, include_self=True, out=out
    )

    assert result is out
    assert out.device == inp.device
    assert out.data_ptr() == pointer
    utils.gems_assert_close(out, ref_out, dtype=inp.dtype, reduce_dim=source.size(1))


@pytest.mark.index_reduce_out
def test_index_reduce_out_aliases_input():
    inp, index, source = _inputs()
    ref_inp = utils.to_reference(inp.clone(), True)
    torch.index_reduce(
        ref_inp,
        1,
        utils.to_reference(index, False),
        utils.to_reference(source, True),
        "mean",
        out=ref_inp,
    )

    result = flag_gems.index_reduce_out(inp, 1, index, source, "mean", out=inp)

    assert result is inp
    assert result.device == inp.device
    utils.gems_assert_close(inp, ref_inp, dtype=inp.dtype, reduce_dim=source.size(1))


@pytest.mark.index_reduce_out
def test_index_reduce_out_rejects_unsafe_dtype():
    inp, index, source = _inputs()
    out = torch.full(
        inp.shape, 17, dtype=torch.int32, device=flag_gems.device
    )
    snapshot = out.clone()

    with pytest.raises(RuntimeError):
        flag_gems.index_reduce_out(inp, 1, index, source, "mean", out=out)

    utils.gems_assert_equal(out, utils.to_reference(snapshot, False))


@pytest.mark.index_reduce_out
def test_index_reduce_out_master_short_source_alias():
    # Master uses four source entries; preserve the original dev eight-entry
    # alias case independently, since they cover different duplicate patterns.
    dtype = torch.float32
    inp = torch.randn((4, 6), dtype=dtype, device=flag_gems.device)
    source = torch.randn((4, 4), dtype=dtype, device=flag_gems.device)
    index = (torch.arange(4, dtype=torch.int64, device=flag_gems.device) * 3 + 1) % 6
    ref_inp = utils.to_reference(inp.clone())
    torch.index_reduce(
        ref_inp,
        1,
        utils.to_reference(index),
        utils.to_reference(source),
        "mean",
        out=ref_inp,
    )
    result = flag_gems.index_reduce_out(inp, 1, index, source, "mean", out=inp)

    assert result is inp
    assert result.device == inp.device
    assert result.dtype == dtype
    assert result.shape == (4, 6)
    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=source.size(1))

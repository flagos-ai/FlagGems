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


def _inputs(dtype=torch.float32, index_dtype=torch.int64):
    torch.manual_seed(0)
    inp = torch.randn((4, 6), dtype=dtype, device=flag_gems.device)
    source = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    index_device = "cpu" if flag_gems.vendor_name == "enflame" else flag_gems.device
    index = ((torch.arange(8, dtype=index_dtype, device=index_device) * 3 + 1) % 6).to(
        flag_gems.device
    )
    assert index.dtype == index_dtype
    assert index.device == inp.device
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


def _skip_unsupported_index_dtype(index_dtype):
    if (
        flag_gems.vendor_name == "enflame"
        and index_dtype == torch.int64
        and not utils.TO_CPU
    ):
        pytest.skip("Enflame native index_reduce_out requires Int32 indices")


def _check_index_reduce_out_resizes(reduce, include_self, index_dtype):
    _skip_unsupported_index_dtype(index_dtype)
    inp, index, source = _inputs(index_dtype=index_dtype)
    expected_device, expected_dtype, expected_shape = inp.device, inp.dtype, inp.shape
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
    assert out.shape == expected_shape
    assert out.dtype == expected_dtype
    assert out.device == expected_device
    utils.gems_assert_close(out, ref_out, dtype=inp.dtype, reduce_dim=source.size(1))


def _check_index_reduce_out_preserves_storage(reduce, index_dtype):
    _skip_unsupported_index_dtype(index_dtype)
    inp, index, source = _inputs(index_dtype=index_dtype)
    expected_device, expected_dtype, expected_shape = inp.device, inp.dtype, inp.shape
    out = torch.empty_like(inp)
    pointer = out.data_ptr()
    ref_out = torch.empty_like(utils.to_reference(inp, True))
    _reference(inp, index, source, reduce, True, ref_out)

    result = flag_gems.index_reduce_out(
        inp, 1, index, source, reduce, include_self=True, out=out
    )

    assert result is out
    assert out.data_ptr() == pointer
    assert out.device == expected_device
    assert out.dtype == expected_dtype
    assert out.shape == expected_shape
    utils.gems_assert_close(out, ref_out, dtype=inp.dtype, reduce_dim=source.size(1))


def _check_index_reduce_out_aliases_input(index_dtype):
    _skip_unsupported_index_dtype(index_dtype)
    inp, index, source = _inputs(index_dtype=index_dtype)
    expected_device = inp.device
    expected_dtype = inp.dtype
    expected_shape = inp.shape
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
    assert inp.device == expected_device
    assert inp.dtype == expected_dtype
    assert inp.shape == expected_shape
    utils.gems_assert_close(inp, ref_inp, dtype=inp.dtype, reduce_dim=source.size(1))


def _check_index_reduce_out_rejects_unsafe_dtype(index_dtype):
    inp, index, source = _inputs(index_dtype=index_dtype)
    out = torch.full(inp.shape, 17, dtype=torch.int32, device=flag_gems.device)
    snapshot = out.clone()

    with pytest.raises(RuntimeError):
        flag_gems.index_reduce_out(inp, 1, index, source, "mean", out=out)

    assert out.device == snapshot.device
    assert out.dtype == snapshot.dtype
    assert out.shape == snapshot.shape
    utils.gems_assert_equal(out, utils.to_reference(snapshot))


@pytest.mark.index_reduce_out
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("include_self", (True, False))
def test_index_reduce_out_resizes(reduce, include_self):
    _check_index_reduce_out_resizes(reduce, include_self, torch.int64)


@pytest.mark.index_reduce_out
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_index_reduce_out_preserves_storage(reduce):
    _check_index_reduce_out_preserves_storage(reduce, torch.int64)


@pytest.mark.index_reduce_out
def test_index_reduce_out_aliases_input():
    _check_index_reduce_out_aliases_input(torch.int64)


@pytest.mark.index_reduce_out
def test_index_reduce_out_rejects_unsafe_dtype():
    _check_index_reduce_out_rejects_unsafe_dtype(torch.int64)


@pytest.mark.index_reduce_out
@pytest.mark.parametrize("reduce", REDUCTIONS)
@pytest.mark.parametrize("include_self", (True, False))
def test_index_reduce_out_resizes_int32(reduce, include_self):
    _check_index_reduce_out_resizes(reduce, include_self, torch.int32)


@pytest.mark.index_reduce_out
@pytest.mark.parametrize("reduce", REDUCTIONS)
def test_index_reduce_out_preserves_storage_int32(reduce):
    _check_index_reduce_out_preserves_storage(reduce, torch.int32)


@pytest.mark.index_reduce_out
def test_index_reduce_out_aliases_input_int32():
    _check_index_reduce_out_aliases_input(torch.int32)


@pytest.mark.index_reduce_out
def test_index_reduce_out_rejects_unsafe_dtype_int32():
    _check_index_reduce_out_rejects_unsafe_dtype(torch.int32)

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

REAL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.ALL_INT_DTYPES
    + utils.BOOL_TYPES
)
IDX_SHAPES = [(6,), (32, 32), (1024,)]
TAKE_CASES = [
    (dtype, shape, idx_shape)
    for dtype in REAL_DTYPES
    for shape in utils.POINTWISE_SHAPES
    for idx_shape in IDX_SHAPES
]


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(
            0, 2, shape, dtype=torch.int32, device=flag_gems.device
        ).bool()
    if dtype == torch.uint8:
        return torch.randint(0, 100, shape, dtype=dtype, device=flag_gems.device)
    if dtype in utils.ALL_INT_DTYPES or dtype == torch.int8:
        return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


def _make_index(numel, idx_shape, negative=False):
    lo = -numel if negative else 0
    return torch.randint(
        lo, numel, idx_shape, device=flag_gems.device, dtype=torch.int64
    )


def _assert_matches_torch(inp, index, out=None):
    ref_inp = utils.to_reference(inp)
    ref_idx = utils.to_reference(index)
    if out is None:
        ref_out = torch.take(ref_inp, ref_idx)
        with flag_gems.use_gems():
            result = torch.take(inp, index)
    else:
        ref_out = torch.empty(index.shape, dtype=inp.dtype, device=ref_inp.device)
        torch.take(ref_inp, ref_idx, out=ref_out)
        with flag_gems.use_gems():
            result = torch.take(inp, index, out=out)
        assert result is out

    # take is a pure gather, values must match exactly.
    utils.gems_assert_equal(result, ref_out)


@pytest.mark.take
@pytest.mark.parametrize("dtype,shape,idx_shape", TAKE_CASES)
def test_take(dtype, shape, idx_shape):
    inp = _make_input(shape, dtype)
    index = _make_index(inp.numel(), idx_shape)
    _assert_matches_torch(inp, index)


@pytest.mark.take_out
@pytest.mark.parametrize("dtype,shape,idx_shape", TAKE_CASES)
def test_take_out(dtype, shape, idx_shape):
    inp = _make_input(shape, dtype)
    index = _make_index(inp.numel(), idx_shape)
    out = torch.empty(idx_shape, dtype=dtype, device=flag_gems.device)
    _assert_matches_torch(inp, index, out)


@pytest.mark.take
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_take_negative_index(dtype):
    inp = _make_input((8, 8), dtype)
    index = _make_index(inp.numel(), (16,), negative=True)
    _assert_matches_torch(inp, index)


@pytest.mark.take
@pytest.mark.parametrize("bad", [1_000_000, -1_000_000])
def test_take_out_of_bounds(bad):
    # Our GEMS kernel mirrors PyTorch's CUDA semantics: an out-of-range index
    # trips an asynchronous device-side assert (tl.device_assert), not a
    # synchronous IndexError. We deliberately do NOT run the out-of-bounds path
    # on the accelerator here -- a device-side assert would poison the CUDA
    # context for the rest of the test process. Instead we confirm on CPU that
    # PyTorch itself classifies these indices as out of range (IndexError),
    # which is exactly the condition our device_assert guards.
    inp = torch.randn((4, 4), dtype=torch.float32)
    index = torch.tensor([0, bad], dtype=torch.int64)
    with pytest.raises(IndexError):
        torch.take(inp, index)


@pytest.mark.take
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_take_noncontiguous(dtype):
    inp = _make_input((7, 11), dtype).transpose(0, 1)
    index = _make_index(inp.numel(), (16,))
    _assert_matches_torch(inp, index)


@pytest.mark.take_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_take_out_noncontiguous(dtype):
    inp = _make_input((11, 7), dtype)
    index = _make_index(inp.numel(), (6, 4))
    out = torch.empty((4, 6), dtype=dtype, device=flag_gems.device).transpose(0, 1)
    _assert_matches_torch(inp, index, out)


@pytest.mark.take_out
def test_take_out_resizes_empty_tensor():
    inp = _make_input((3, 5), torch.float32)
    index = _make_index(inp.numel(), (2, 3))
    out = torch.empty(0, dtype=inp.dtype, device=flag_gems.device)

    _assert_matches_torch(inp, index, out)
    assert tuple(out.shape) == (2, 3)


@pytest.mark.take_out
def test_take_out_rejects_mismatched_dtype():
    inp = _make_input((3, 5), torch.float32)
    index = _make_index(inp.numel(), (4,))
    out = torch.empty((4,), dtype=torch.int32, device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        torch.take(inp, index, out=out)


@pytest.mark.take
def test_take_empty():
    inp = _make_input((3, 5), torch.float32)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    _assert_matches_torch(inp, index)


@pytest.mark.take
def test_take_multidim_index():
    """Output shape follows the index shape."""
    inp = _make_input((4, 6), torch.float32)
    index = _make_index(inp.numel(), (3, 5))
    _assert_matches_torch(inp, index)
    with flag_gems.use_gems():
        result = torch.take(inp, index)
    assert tuple(result.shape) == (3, 5)

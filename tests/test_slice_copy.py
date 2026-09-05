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

# Shape configurations for slice_copy testing: 1D, 2D, 3D and higher-rank tensors.
SLICE_COPY_SHAPES = [
    (16, 32, 64),
    (32, 64),
    (64,),
    (4, 8, 12),
    (2, 19, 7),
]


@pytest.mark.slice_copy
@pytest.mark.parametrize("shape", SLICE_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_basic(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        for start in [0, dim_size // 4, dim_size // 2]:
            for end in [dim_size // 2 + 1, dim_size]:
                for step in [1, 2, 3]:
                    if start >= end:
                        continue
                    ref_out = torch.slice_copy(ref_inp, dim, start, end, step)
                    res_out = flag_gems.slice_copy(inp, dim, start, end, step)
                    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_default_args(dtype):
    # When start/end are None and step defaults to 1, slice_copy reproduces the input.
    shape = (4, 8, 12)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        ref_out = torch.slice_copy(ref_inp, dim)
        res_out = flag_gems.slice_copy(inp, dim)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_negative_start_end(dtype):
    # 3-D shape so negative dim indices (-1, -2) resolve to distinct axes.
    shape = (8, 16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    cases = [
        (1, -8, None, 1),
        (1, -8, -1, 1),
        (2, -10, -2, 2),
        (0, -4, None, 1),
    ]
    for dim, start, end, step in cases:
        ref_out = torch.slice_copy(ref_inp, dim, start, end, step)
        res_out = flag_gems.slice_copy(inp, dim, start, end, step)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_step(dtype):
    # 2-D shape: vary step along both axes to exercise the strided kernel path.
    shape = (32, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        for start in [0, 1, dim_size // 4]:
            for step in [2, 3, 5, 7]:
                ref_out = torch.slice_copy(ref_inp, dim, start, dim_size, step)
                res_out = flag_gems.slice_copy(inp, dim, start, dim_size, step)
                utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_edge_cases(dtype):
    device = flag_gems.device

    # Empty slice: start >= end after clamping.
    inp = torch.randn((4, 8), dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.slice_copy(ref_inp, 1, 5, 3, 1)
    res_out = flag_gems.slice_copy(inp, 1, 5, 3, 1)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)

    # Out-of-bounds start and end get clamped.
    ref_out = torch.slice_copy(ref_inp, 0, 100, 200, 1)
    res_out = flag_gems.slice_copy(inp, 0, 100, 200, 1)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)

    # End clamped to the dimension size.
    ref_out = torch.slice_copy(ref_inp, 1, 2, 1000, 1)
    res_out = flag_gems.slice_copy(inp, 1, 2, 1000, 1)
    utils.gems_assert_equal(res_out, ref_out)

    # start == end produces an empty slice.
    ref_out = torch.slice_copy(ref_inp, 0, 4, 4, 1)
    res_out = flag_gems.slice_copy(inp, 0, 4, 4, 1)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)

    # 1D tensor.
    inp1d = torch.randn(64, dtype=dtype, device=device)
    ref_inp1d = utils.to_reference(inp1d)
    ref_out = torch.slice_copy(ref_inp1d, 0, 10, 50, 2)
    res_out = flag_gems.slice_copy(inp1d, 0, 10, 50, 2)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_negative_dim(dtype):
    # 3-D shape so negative dims -1/-2/-3 map to distinct axes.
    shape = (4, 8, 12)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in [-1, -2, -3]:
        ref_out = torch.slice_copy(ref_inp, dim, 1, 5, 2)
        res_out = flag_gems.slice_copy(inp, dim, 1, 5, 2)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_slice_copy_int_dtype(dtype):
    inp = torch.randint(-100, 100, (4, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in [0, 1]:
        for start in [0, 2, -8]:
            for end in [8, None]:
                for step in [1, 2]:
                    ref_out = torch.slice_copy(ref_inp, dim, start, end, step)
                    res_out = flag_gems.slice_copy(inp, dim, start, end, step)
                    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy_out
@pytest.mark.parametrize("shape", SLICE_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        start, end, step = dim_size // 4, dim_size - dim_size // 4, 2
        ref_out = torch.slice_copy(ref_inp, dim, start, end, step)

        out_shape = list(inp.shape)
        out_shape[dim] = (end - start + step - 1) // step
        res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        res_r = flag_gems.slice_copy_out(inp, dim, start, end, step, out=res_out)

        assert res_r is res_out
        utils.gems_assert_equal(res_r, ref_out)

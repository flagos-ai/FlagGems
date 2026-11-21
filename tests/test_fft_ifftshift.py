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

# 1-D, 2-D and 3-D shapes; every entry has at least one dimension so that the
# per-dimension shifts requested below are all valid.
FFT_IFFTSHIFT_SHAPES = [(2, 3), (128, 256), (512, 512), (1024,), (16, 17, 18)]
FFT_IFFTSHIFT_DIMS = [None, [0], [-1]]
FFT_IFFTSHIFT_EMPTY_SHAPES = [(0,), (0, 3), (2, 0, 4)]


@pytest.mark.fft_ifftshift
@pytest.mark.parametrize("shape", FFT_IFFTSHIFT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", FFT_IFFTSHIFT_DIMS)
def test_fft_ifftshift(shape, dtype, dim):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.fft_ifftshift(ref_inp, dim)
    res_out = flag_gems.fft_ifftshift(res_inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fft_ifftshift
@pytest.mark.parametrize("shape", FFT_IFFTSHIFT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fft_ifftshift_int_dim(shape, dtype):
    # dim is an int[1]? in the schema, so a bare int must be accepted too.
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.fft_ifftshift(ref_inp, 0)
    res_out = flag_gems.fft_ifftshift(res_inp, 0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fft_ifftshift
@pytest.mark.parametrize("shape", FFT_IFFTSHIFT_EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", FFT_IFFTSHIFT_DIMS)
def test_fft_ifftshift_empty(shape, dtype, dim):
    # Empty tensors take the clone fast path in the implementation.
    res_inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.fft_ifftshift(ref_inp, dim)
    res_out = flag_gems.fft_ifftshift(res_inp, dim)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fft_ifftshift
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fft_ifftshift_scalar(dtype):
    # Issue #8390: aten raises for a 0-dim input with no shifts ("`shifts`
    # required"), so the override raises too — the old clone fast path changed
    # the observable behaviour of torch.fft.ifftshift for scalar input.
    res_inp = torch.randn((), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="`shifts` required"):
        flag_gems.fft_ifftshift(res_inp)


@pytest.mark.fft_ifftshift
def test_fft_ifftshift_zero_dim_raises():
    # Issue #8390: a 0-dim input with no shifts raises in ATen ("`shifts` required");
    # the override used to return a clone instead
    x = torch.tensor(3.14, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="`shifts` required"):
        flag_gems.fft_ifftshift(x)


@pytest.mark.fft_ifftshift
def test_fft_ifftshift_empty_returns_clone():
    # the empty half of the old short-circuit was already right and stays
    x = torch.empty(0, device=flag_gems.device)
    out = flag_gems.fft_ifftshift(x)
    assert out.shape == (0,)

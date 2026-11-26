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
    # A 0-dim tensor takes the clone fast path and is returned unchanged.
    # aten raises here ("`shifts` required" / "tensor has no dimensions"), so
    # there is no reference to compare against: assert the clone semantics.
    res_inp = torch.randn((), dtype=dtype, device=flag_gems.device)
    # The comparison value must go through to_reference: under --ref=cpu the
    # reference side has to live on the CPU, and res_inp is still on device.
    ref_inp = utils.to_reference(res_inp)

    res_out = flag_gems.fft_ifftshift(res_inp)

    assert res_out.shape == ()
    assert res_out.dtype == dtype
    assert res_out is not res_inp
    utils.gems_assert_close(res_out, ref_inp, dtype)

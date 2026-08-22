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

from .accuracy_utils import (
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    gems_assert_equal,
    to_reference,
)

_MTHREADS = flag_gems.vendor_name == "mthreads"


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 1],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    rank = len(shape)
    num_pad = rank * 2
    pad = [torch.randint(0, 10, (1,)).item() for _ in range(num_pad)]
    value = 1.5

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 2],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_non_contiguous(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = inp[::2, ::2]
    rank = inp.ndim
    num_pad = rank * 2
    pad = [torch.randint(0, 5, (1,)).item() for _ in range(num_pad)]
    value = -2.0

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 1],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_zero_value(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    rank = len(shape)
    num_pad = rank * 2
    pad = [torch.randint(0, 10, (1,)).item() for _ in range(num_pad)]
    value = 0.0

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.parametrize(
    "shape",
    [s for s in POINTWISE_SHAPES if len(s) >= 2],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_constant_pad_nd_partial_dims(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad = [2, 3]
    value = 7.0

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)


@pytest.mark.constant_pad_nd
@pytest.mark.skipif(not _MTHREADS, reason="MThreads-specific dispatch coverage")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("last_dim", [511, 512, 513])
def test_constant_pad_nd_mthreads_rank3_gate_boundaries(last_dim, dtype):
    inp = torch.randn((10, 9, last_dim), dtype=dtype, device=flag_gems.device)
    pad = [0, 0, 0, 1, 0, 0]
    value = 1.25

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.constant_pad_nd
@pytest.mark.skipif(not _MTHREADS, reason="MThreads-specific dispatch coverage")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_constant_pad_nd_mthreads_rank6_fallback(dtype):
    inp = torch.randn(
        (2, 2, 2, 2, 5, 33),
        dtype=dtype,
        device=flag_gems.device,
    )
    pad = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    value = -0.75

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, value)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, value)

    gems_assert_equal(res_out, ref_out)
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.constant_pad_nd
@pytest.mark.skipif(not _MTHREADS, reason="MThreads-specific dispatch coverage")
def test_constant_pad_nd_mthreads_rank3_negative_crop():
    inp = torch.randn((10, 9, 511), dtype=torch.float16, device=flag_gems.device)
    pad = [-1, 0, 0, 1, 0, 0]

    ref_inp = to_reference(inp)
    ref_out = torch.constant_pad_nd(ref_inp, pad, 1.25)
    with flag_gems.use_gems():
        res_out = torch.constant_pad_nd(inp, pad, 1.25)

    gems_assert_equal(res_out, ref_out)
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.constant_pad_nd
@pytest.mark.skipif(not _MTHREADS, reason="MThreads-specific dispatch coverage")
def test_constant_pad_nd_mthreads_rank3_int32_gate():
    from flag_gems.runtime.backend._mthreads.ops.pad import _rank3_offsets_fit_int32

    assert _rank3_offsets_fit_int32(2**31 - 1)
    assert not _rank3_offsets_fit_int32(2**31)

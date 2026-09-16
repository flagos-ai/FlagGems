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
from . import conftest as cfg

if cfg.QUICK_MODE:
    ARGSORT_BATCH_SIZES = [4]
    ARGSORT_HIDDEN_SIZES = [256, 2048]
else:
    ARGSORT_BATCH_SIZES = [4, 8]
    ARGSORT_HIDDEN_SIZES = [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]


@pytest.mark.argsort
@pytest.mark.parametrize("batch_size", ARGSORT_BATCH_SIZES)
@pytest.mark.parametrize("hiddensize", ARGSORT_HIDDEN_SIZES)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + [torch.int8, torch.uint8]
)
@pytest.mark.parametrize("dim", [0, -1])
def test_accuracy_argsort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif not dtype.is_floating_point:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    ref_index = torch.argsort(ref_y, dim=dim, stable=True, descending=descending)

    res_index = flag_gems.argsort(y, dim=dim, descending=descending)

    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.argsort
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("length", [17, 2049, 8193])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_argsort_byte_boundaries(dtype, descending, dim, length, noncontiguous):
    limits = torch.iinfo(dtype)
    data = [limits.max, 0, limits.min, 127, limits.max, 1]
    data += [-1, -127] if dtype == torch.int8 else [128, 254]
    row = torch.tensor(data, dtype=dtype).repeat((length + len(data) - 1) // len(data))
    inp = row[:length].repeat(3, 1)
    if noncontiguous:
        inp = inp.repeat_interleave(2, dim=-1).to(flag_gems.device)[:, ::2]
    else:
        inp = inp.to(flag_gems.device)
    if dim == 0:
        inp = inp.t()
    ref = torch.argsort(
        utils.to_reference(inp), dim=dim, descending=descending, stable=True
    )
    res = flag_gems.argsort(inp, dim=dim, descending=descending)
    assert res.dtype == torch.int64
    assert res.shape == inp.shape
    utils.gems_assert_equal(res, ref)


@pytest.mark.argsort
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads",
    reason="Regression coverage for MThreads stable argsort kernels",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize("length", [17, 2048, 2049, 9333])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("dim", [0, -1])
def test_argsort_mthreads_stable_extrema(dtype, length, descending, dim):
    if dtype.is_floating_point:
        data = [
            float("nan"),
            -float("nan"),
            -float("inf"),
            -0.0,
            0.0,
            float("inf"),
            1.0,
            1.0,
            -1.0,
        ]
    else:
        limits = torch.iinfo(dtype)
        data = [limits.min, limits.min + 1, -1, 0, 0, 1, limits.max - 1, limits.max]
        if dtype == torch.int64:
            data += [2**60, 2**60 + 1, -(2**60), -(2**60) - 1]
    row = torch.tensor(data, dtype=dtype).repeat((length + len(data) - 1) // len(data))
    inp = row[:length].repeat(3, 1).repeat_interleave(2, dim=-1)
    inp = inp.to(flag_gems.device)[:, ::2]
    if dim == 0:
        inp = inp.t()
    ref_inp = utils.to_reference(inp)
    # Native MUSA stable sort separates -0.0 and +0.0. Use CPU semantics for
    # floating extrema, then restore the configured reference device.
    ref = torch.argsort(
        ref_inp.cpu() if dtype.is_floating_point else ref_inp,
        dim=dim,
        descending=descending,
        stable=True,
    ).to(ref_inp.device)
    result = flag_gems.argsort(inp, dim=dim, descending=descending)
    utils.gems_assert_equal(result, ref)

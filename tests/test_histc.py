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
    HISTC_SHAPES = [(64,), (100, 100)]
    HISTC_BINS = [10]
else:
    HISTC_SHAPES = [(64,), (1024,), (4096,), (100, 100), (32, 64, 16)]
    HISTC_BINS = [10, 50, 100]
HISTC_DTYPES = [torch.float32]


def _make_histc_input(
    shape,
    dtype,
    device,
    min_value,
    max_value,
    *,
    include_endpoints=False,
    include_outliers=False,
):
    numel = 1
    for dim in shape:
        numel *= dim

    bucket_ids = torch.arange(numel, device=device) % 100
    inp = min_value + (bucket_ids.to(dtype) + 0.5) * (
        (max_value - min_value) / 100
    )
    inp = inp.reshape(shape)

    if include_endpoints and numel >= 2:
        flat = inp.reshape(-1)
        flat[0] = min_value
        flat[1] = max_value
    elif include_outliers and numel >= 2:
        flat = inp.reshape(-1)
        flat[0] = min_value - 0.5
        flat[1] = max_value + 0.5

    return inp


@pytest.mark.histc
@pytest.mark.parametrize("shape", HISTC_SHAPES)
@pytest.mark.parametrize("bins", HISTC_BINS)
@pytest.mark.parametrize("dtype", HISTC_DTYPES)
def test_histc_automatic_range(shape, bins, dtype):
    inp = _make_histc_input(
        shape, dtype, flag_gems.device, 0.0, 10.0, include_endpoints=True
    )
    inp_before = utils.to_reference(inp.clone())
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.histc(ref_inp, bins=bins, min=0, max=0)
    res_out = flag_gems.histc(inp, bins=bins, min=0, max=0)

    assert res_out.shape == (bins,)
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(inp, inp_before)


@pytest.mark.histc
@pytest.mark.parametrize("shape", HISTC_SHAPES)
@pytest.mark.parametrize("bins", HISTC_BINS)
@pytest.mark.parametrize("dtype", HISTC_DTYPES)
def test_histc_explicit_range(shape, bins, dtype):
    inp = _make_histc_input(
        shape, dtype, flag_gems.device, 0.0, 10.0, include_outliers=True
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.histc(ref_inp, bins=bins, min=0, max=10)
    res_out = flag_gems.histc(inp, bins=bins, min=0, max=10)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.histc
def test_histc_empty_input():
    inp = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.histc(ref_inp, bins=10, min=0, max=10)
    res_out = flag_gems.histc(inp, bins=10, min=0, max=10)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.histc
def test_histc_noncontiguous_input():
    inp = _make_histc_input((12, 10), torch.float32, flag_gems.device, 0.0, 10.0)
    inp = inp.transpose(0, 1)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.histc(ref_inp, bins=20, min=0, max=10)
    res_out = flag_gems.histc(inp, bins=20, min=0, max=10)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.histc
@pytest.mark.parametrize("bins", [0, -1])
def test_histc_rejects_nonpositive_bins(bins):
    inp = torch.ones((16,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.histc(inp, bins=bins, min=0, max=10)


@pytest.mark.histc
def test_histc_rejects_reversed_range():
    inp = torch.ones((16,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.histc(inp, bins=10, min=2, max=1)

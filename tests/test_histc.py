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

from . import conftest as cfg
from .accuracy_utils import gems_assert_close, to_reference

if cfg.QUICK_MODE:
    HISTC_SHAPES = [(64,), (100, 100)]
    HISTC_BINS = [10]
else:
    HISTC_SHAPES = [(64,), (1024,), (4096,), (100, 100), (32, 64, 16)]
    HISTC_BINS = [10, 50, 100]
HISTC_DTYPES = [torch.float32]


def make_histc_input(
    shape,
    dtype,
    device,
    min_val,
    max_val,
    include_endpoints=False,
    include_outliers=False,
):
    numel = 1
    for dim in shape:
        numel *= dim

    # Keep values deterministic and away from histc bin boundaries, where tiny
    # rounding differences can move one element into an adjacent bin.
    bucket_ids = torch.arange(numel, device=device) % 100
    inp = min_val + (bucket_ids.to(dtype) + 0.5) * ((max_val - min_val) / 100)
    inp = inp.reshape(shape)

    if include_endpoints and numel >= 2:
        flat = inp.reshape(-1)
        flat[0] = min_val
        flat[1] = max_val
    elif include_outliers and numel >= 2:
        flat = inp.reshape(-1)
        flat[0] = min_val - 0.5
        flat[1] = max_val + 0.5

    return inp


@pytest.mark.histc
@pytest.mark.parametrize("shape", HISTC_SHAPES)
@pytest.mark.parametrize("bins", HISTC_BINS)
@pytest.mark.parametrize("dtype", HISTC_DTYPES)
def test_accuracy_histc(shape, bins, dtype):
    inp = make_histc_input(
        shape, dtype, flag_gems.device, 0.0, 10.0, include_endpoints=True
    )
    ref_inp = to_reference(inp)
    ref_out = torch.histc(ref_inp, bins=bins, min=0, max=0)
    with flag_gems.use_gems():
        res_out = torch.histc(inp, bins=bins, min=0, max=0)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.histc
@pytest.mark.parametrize("shape", HISTC_SHAPES)
@pytest.mark.parametrize("bins", HISTC_BINS)
@pytest.mark.parametrize("dtype", HISTC_DTYPES)
def test_accuracy_histc_with_range(shape, bins, dtype):
    inp = make_histc_input(
        shape, dtype, flag_gems.device, 0.0, 10.0, include_outliers=True
    )
    ref_inp = to_reference(inp)
    ref_out = torch.histc(ref_inp, bins=bins, min=0, max=10)
    with flag_gems.use_gems():
        res_out = torch.histc(inp, bins=bins, min=0, max=10)
    gems_assert_close(res_out, ref_out, dtype)

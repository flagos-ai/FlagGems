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

# torch.rand / torch.randn have no FP8 implementation, so FP8 inputs are built
# in FP32 and cast; FP8 results are also compared bitwise because torch rejects
# non-zero tolerances for 1-byte floats.
CONJ_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def _supported_fp8_dtypes(device):
    """Keep only the FP8 dtypes the current device can actually materialize."""
    supported = []
    for dtype in CONJ_FP8_DTYPES:
        try:
            torch.randn(1, device=device).to(dtype)
        except Exception:
            continue
        supported.append(dtype)
    return supported


# Dtype sweep: the low-precision dtypes first, then the float and int sweep.
CONJ_DTYPES = [
    torch.int8,
    torch.uint8,
    *_supported_fp8_dtypes(flag_gems.device),
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int32,
]

# Integer results must come back bit-identical, so they are compared exactly,
# together with FP8, which torch only compares bitwise.
CONJ_EXACT_DTYPES = (
    torch.int8,
    torch.uint8,
    *CONJ_FP8_DTYPES,
    torch.int32,
)

if cfg.QUICK_MODE:
    CONJ_SHAPES = [(2, 19, 7)]
    CONJ_VALUE_RANGES = ["[-1,1]"]
else:
    CONJ_SHAPES = [
        (),
        (1,),
        (256,),
        (1024, 1024),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
    CONJ_VALUE_RANGES = ["[-1,1]", "[0,1]", "[-1,0]", "[0,max]", "[min,0]"]


def _bounds(dtype, value_range):
    """Inclusive (low, high) bounds of the values to generate."""
    fixed_ranges = {
        "[-1,1]": (-1.0, 1.0),
        "[0,1]": (0.0, 1.0),
        "[-1,0]": (-1.0, 0.0),
    }
    if value_range in fixed_ranges:
        return fixed_ranges[value_range]

    info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
    if value_range == "[0,max]":
        return 0.0, info.max
    return info.min, 0.0


def _make_input(shape, dtype, device, value_range):
    """Sample values inside ``value_range`` and pin both of its ends."""
    low, high = _bounds(dtype, value_range)
    if dtype.is_floating_point:
        # torch.rand has no FP8 implementation, so sample in FP32 and cast.
        values = torch.rand(shape, dtype=torch.float32, device=device)
        out = (values * (high - low) + low).to(dtype)
        edges = torch.tensor([low, high], dtype=torch.float32, device=device).to(dtype)
    else:
        # Unsigned dtypes cannot hold the negative end of [-1,1] / [-1,0], so
        # the bounds are clamped to what the dtype can represent.
        info = torch.iinfo(dtype)
        low, high = max(int(low), info.min), min(int(high), info.max)
        upper = high + 1
        out = torch.randint(low, upper, shape, dtype=dtype, device=device)
        edges = torch.tensor([low, high], dtype=dtype, device=device)

    if out.numel() > 1:
        out.view(-1)[:2] = edges
    return out


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", CONJ_SHAPES)
@pytest.mark.parametrize("value_range", CONJ_VALUE_RANGES)
@pytest.mark.parametrize("dtype", CONJ_DTYPES)
def test_conj_physical(shape, dtype, value_range):
    device = flag_gems.device
    input = _make_input(shape, dtype, device, value_range)
    out_dtype = dtype

    if dtype in CONJ_EXACT_DTYPES:
        # Compare against an unmodified copy of the input, so the exact
        # comparison does not depend on the upcast reference.
        ref_input = utils.to_reference(input.clone())
    else:
        ref_input = utils.to_reference(input, True)

    ref_out = torch.conj_physical(ref_input)
    res_out = flag_gems.conj_physical(input)

    if dtype in CONJ_EXACT_DTYPES:
        utils.gems_assert_equal(res_out, ref_out.to(out_dtype))
    else:
        utils.gems_assert_close(res_out, ref_out, out_dtype, reduce_dim=1)


@pytest.mark.conj_physical
@pytest.mark.parametrize("shape", CONJ_SHAPES)
@pytest.mark.parametrize("value_range", CONJ_VALUE_RANGES)
def test_conj_physical_complex(shape, value_range):
    device = flag_gems.device
    real = _make_input(shape, torch.float32, device, value_range)
    imag = _make_input(shape, torch.float32, device, value_range)
    input = torch.complex(real, imag)
    out_dtype = input.dtype

    ref_input = utils.to_reference(input, True)
    ref_out = torch.conj_physical(ref_input)
    res_out = flag_gems.conj_physical(input)

    assert res_out.dtype == out_dtype
    # Not every device implements a complex comparison kernel: the Ascend NPU
    # rejects both isclose and equal for complex64 ("aclnnIsClose failed, error
    # code is 161002"), so compare the real and imaginary lanes as plain
    # float32, the same way tests/test_chalf.py does.
    utils.gems_assert_close(
        torch.view_as_real(res_out),
        torch.view_as_real(ref_out),
        torch.float32,
        reduce_dim=1,
    )

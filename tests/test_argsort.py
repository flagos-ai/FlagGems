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

import inspect
import itertools
import os

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# The default mark is a representative matrix that fits one CI phase. Keep the
# original Cartesian coverage available in this same file for extended runs.
FULL_MATRIX = os.environ.get("FLAGGEMS_ARGSORT_FULL_TESTS") == "1"
ARGSORT_DTYPES = (
    utils.FLOAT_DTYPES + utils.INT_DTYPES + [torch.int8, torch.uint8, torch.int64]
)
if FULL_MATRIX or cfg.QUICK_MODE:
    batches = [4] if cfg.QUICK_MODE else [4, 8]
    lengths = (
        [256, 2048]
        if cfg.QUICK_MODE
        else [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
    )
    ACCURACY_CASES = list(
        itertools.product(batches, lengths, [True, False], ARGSORT_DTYPES, [0, -1])
    )
else:
    # Each dtype covers short, tile-boundary, odd and long axes, both orders,
    # and both batch sizes. Dim 0 and noncontiguous paths are covered below.
    ACCURACY_CASES = [
        (batch, length, descending, dtype, -1)
        for dtype in ARGSORT_DTYPES
        for batch, length, descending in (
            (4, 256, False),
            (4, 2048, True),
            (8, 9333, False),
            (4, 65536, True),
        )
    ] + [
        (4, 262144, False, torch.float32, -1),
        (8, 262144, True, torch.int64, -1),
        (4, 131072, False, torch.float16, -1),
        (8, 131072, True, torch.uint8, -1),
        (4, 32768, False, torch.bfloat16, -1),
        (8, 32768, True, torch.int16, -1),
        (4, 65536, True, torch.float32, 0),
        (8, 262144, False, torch.uint8, 0),
    ]

SHORT_ROWS_CASES = (
    list(
        itertools.product(
            [32767, 32768, 32769, 65534, 65535, 65536, 98307],
            [torch.float16, torch.float32, torch.int8, torch.int64],
        )
    )
    if FULL_MATRIX
    else [
        (32767, torch.float16),
        (32768, torch.float32),
        (32769, torch.int8),
        (65534, torch.float16),
        (65535, torch.int64),
        (65536, torch.float32),
        (98307, torch.int8),
    ]
)
BYTE_CASES = (
    list(itertools.product([17, 2049, 8193], [False, True], [0, -1], [False, True]))
    if FULL_MATRIX
    else [
        (17, False, 0, False),
        (17, True, -1, True),
        (2049, True, 0, False),
        (2049, False, -1, True),
        (8193, False, -1, False),
        (8193, True, 0, True),
    ]
)
EXTREMA_CASES = (
    list(itertools.product([17, 2048, 2049, 9333], [False, True], [0, -1]))
    if FULL_MATRIX
    else [(17, False, 0), (2048, True, -1), (2049, False, -1), (9333, True, 0)]
)


def _argsort_reference(inp, dim, descending):
    ref_inp = utils.to_reference(inp)
    reference_device = ref_inp.device
    if flag_gems.vendor_name == "ascend" and not inp.dtype.is_floating_point:
        if inp.dtype in (torch.int8, torch.uint8, torch.int16):
            # These integer ranges are exactly representable in FP32. Avoid
            # native integer argsort's AiCPU fallback without introducing ties.
            ref_inp = ref_inp.to(torch.float32)
        else:
            # Full-range int32/int64 and adjacent extrema cannot be represented
            # exactly by NPU floating types. Keep an exact CPU integer oracle.
            ref_inp = ref_inp.cpu()
    return torch.argsort(ref_inp, dim=dim, stable=True, descending=descending).to(
        reference_device
    )


@pytest.mark.argsort
@pytest.mark.parametrize("rows,dtype", SHORT_ROWS_CASES)
def test_argsort_many_short_rows(rows, dtype):
    # Sorting dim 0 produces `rows` independent rows; offset and striding must
    # survive host batching, including ties and exact large integer ordering.
    data = [3, 1, 3, 0]
    if dtype == torch.int64:
        data = [2**60 + 3, 2**60 + 1, 2**60 + 3, 2**60]
    storage = torch.tensor(data, dtype=dtype)[:, None].repeat(1, rows * 2 + 1)
    inp = storage.to(flag_gems.device)[:, 1::2]
    for descending in (False, True):
        expected = torch.argsort(inp.cpu(), dim=0, stable=True, descending=descending)
        actual = flag_gems.argsort(inp, dim=0, descending=descending)
        utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.argsort
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim", [0, -1])
def test_argsort_many_tiny_rows_extrema(dtype, dim):
    tiny = torch.finfo(dtype).tiny
    values = [
        float("nan"),
        -float("inf"),
        float("inf"),
        -0.0,
        0.0,
        tiny / 2,
        -tiny / 2,
        float("nan"),
    ]
    inp = torch.tensor(values, dtype=dtype).repeat(4097, 1)
    if dim == 0:
        inp = inp.t()
    expected_input = inp.clone()
    inp = inp.to(flag_gems.device)
    for descending in (False, True):
        expected = torch.argsort(
            expected_input, dim=dim, stable=True, descending=descending
        )
        actual = flag_gems.argsort(inp, dim=dim, descending=descending)
        utils.gems_assert_equal(actual.cpu(), expected)


@pytest.mark.argsort
@pytest.mark.parametrize("batch_size,hiddensize,descending,dtype,dim", ACCURACY_CASES)
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

    ref_index = _argsort_reference(y, dim, descending)

    res_index = flag_gems.argsort(y, dim=dim, descending=descending)

    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.argsort
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8])
@pytest.mark.parametrize("length,descending,dim,noncontiguous", BYTE_CASES)
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
    ref = _argsort_reference(inp, dim, descending)
    res = flag_gems.argsort(inp, dim=dim, descending=descending)
    assert res.dtype == torch.int64
    assert res.shape == inp.shape
    utils.gems_assert_equal(res, ref)


@pytest.mark.argsort
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
@pytest.mark.parametrize("length,descending,dim", EXTREMA_CASES)
def test_argsort_stable_extrema(dtype, length, descending, dim):
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
            torch.finfo(dtype).tiny / 2,
            -torch.finfo(dtype).tiny / 2,
            torch.finfo(dtype).tiny * torch.finfo(dtype).eps,
            -torch.finfo(dtype).tiny * torch.finfo(dtype).eps,
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
    if dtype.is_floating_point:
        ref = torch.argsort(
            ref_inp.cpu(), dim=dim, descending=descending, stable=True
        ).to(ref_inp.device)
    else:
        ref = _argsort_reference(inp, dim, descending)
    result = flag_gems.argsort(inp, dim=dim, descending=descending)
    assert result.dtype == torch.int64
    assert result.shape == inp.shape
    utils.gems_assert_equal(result, ref)


@pytest.mark.argsort
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize(
    "shape, dim",
    [((), -1), ((), 0), ((0,), 0), ((0, 3), -1), ((2, 0, 3), 1)],
)
def test_argsort_scalar_empty(dtype, descending, shape, dim):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.argsort(ref_inp.cpu(), dim=dim, descending=descending, stable=True).to(
        ref_inp.device
    )
    result = flag_gems.argsort(inp, dim=dim, descending=descending)
    assert result.dtype == torch.int64
    assert result.shape == inp.shape
    assert result.device == inp.device
    utils.gems_assert_equal(result, ref)


@pytest.mark.argsort
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((), -2),
        ((), 1),
        ((0,), -2),
        ((0,), 1),
        ((2, 0, 3), -4),
        ((2, 0, 3), 3),
        ((2, 3), -3),
        ((2, 3), 2),
    ],
)
def test_argsort_invalid_dim(shape, dim):
    inp = torch.empty(shape, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(IndexError):
        torch.argsort(inp.cpu(), dim=dim, stable=True)
    with pytest.raises(IndexError):
        flag_gems.argsort(inp, dim=dim)


@pytest.mark.argsort
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("length, dim", [(19, 0), (19, 1), (2049, -1)])
def test_argsort_strided_3d(dtype, descending, length, dim):
    if dtype.is_floating_point:
        data = [float("nan"), -float("inf"), -0.0, 0.0, float("inf"), 1.0, 1.0]
    else:
        limits = torch.iinfo(dtype)
        data = [
            limits.min,
            limits.max,
            2**60,
            2**60 + 1,
            -(2**60),
            -(2**60) - 1,
            0,
            0,
        ]
    pattern = torch.tensor(data, dtype=dtype)
    positions = torch.arange(6 * (2 * length + 1), dtype=torch.int64)
    base = pattern[positions % len(data)].reshape(2, 3, 2 * length + 1)
    # Slice after the device transfer so it cannot make the tested view contiguous.
    inp = base.to(flag_gems.device)[:, :, 1::2].transpose(0, 1)
    assert not inp.is_contiguous()
    assert inp.storage_offset() > 0
    ref_inp = utils.to_reference(inp)
    ref = torch.argsort(ref_inp.cpu(), dim=dim, descending=descending, stable=True).to(
        ref_inp.device
    )
    result = flag_gems.argsort(inp, dim=dim, descending=descending)
    assert result.dtype == torch.int64
    assert result.shape == inp.shape
    assert result.device == inp.device
    utils.gems_assert_equal(result, ref)


@pytest.mark.argsort
@pytest.mark.skipif(flag_gems.vendor_name != "ascend", reason="Ascend launcher ABI")
def test_argsort_ascend_launcher_abi():
    from triton.backends.ascend import driver

    namespace = inspect.unwrap(flag_gems.argsort).__globals__
    check = namespace["_asc_sort_check_launcher"]
    target = driver.NPUDriver().get_current_target().arch
    for name, signature in namespace["_ASC_SORT_SIGNATURES"].items():
        metadata = namespace["_asc_sort_metadata"](name, "abi-test", target, 196608)
        wrapper = driver.make_launcher({}, dict(enumerate(signature)), metadata)
        check(wrapper, signature)
        marker = "struct __attribute__((packed)) {"
        assert marker in wrapper
        for comment in ("// debugger annotation\n", "/* debugger annotation */"):
            check(wrapper.replace(marker, marker + comment), signature)
        with pytest.raises(RuntimeError, match="launcher argument ABI"):
            check(
                wrapper.replace(
                    marker, marker + "uint32_t extra __attribute__((aligned(4)));"
                ),
                signature,
            )


@pytest.mark.argsort
@pytest.mark.parametrize(
    "shape,dtype", [((1024, 65536), torch.int16), ((4096, 4096), torch.int64)]
)
def test_argsort_large_tiled_launch_boundary(shape, dtype):
    # The merge tiles and integer pack path respectively launch 65536 tasks.
    row = torch.arange(shape[1], dtype=torch.int64) % 19
    if dtype == torch.int64:
        row += 2**60
    inp = row.to(dtype).expand(shape).contiguous().to(flag_gems.device)
    expected = torch.argsort(inp.cpu(), dim=-1, stable=True)
    actual = flag_gems.argsort(inp, dim=-1)
    utils.gems_assert_equal(actual.cpu(), expected)

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

import importlib

import pytest
import torch
import triton
import triton.language as tl

import flag_gems
from flag_gems.ops.rand import _uint_to_uniform_float_for_dtype, rand_kernel
from flag_gems.runtime import torch_device_fn

from . import accuracy_utils as utils

pytestmark = [pytest.mark.rand, pytest.mark.rand_like]
SHAPES = [(), (1,), (257,), (128, 128), (3, 7, 65), (0,), (2, 0, 3)]


def _assert_range(out, shape, dtype):
    assert out.shape == torch.Size(shape)
    assert out.dtype == dtype
    actual = out.cpu().double()
    assert bool(torch.isfinite(actual).all())
    assert bool(((actual >= 0) & (actual < 1)).all())


def _draw(op, shape, dtype):
    if op == "rand":
        return flag_gems.rand(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    return flag_gems.rand_like(inp)


@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", [0, 20261026, 2**40 + 3])
@pytest.mark.parametrize("op", ["rand", "rand_like"])
def test_uniform_output_half_open_range(dtype, shape, seed, op):
    torch.manual_seed(seed)
    _assert_range(_draw(op, shape, dtype), shape, dtype)


@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset_stride"])
@pytest.mark.parametrize("override_dtype", [False, True])
def test_rand_like_preserves_layout_and_input(dtype, layout, override_dtype):
    input_dtype = torch.float32 if override_dtype else dtype
    data = torch.arange(771, dtype=torch.float32).reshape(3, 257).to(input_dtype)
    if layout == "transpose":
        storage = data.t().contiguous()
        view = lambda x: x.t()
    elif layout == "offset_stride":
        storage = torch.full((3, 517), 57.0, dtype=input_dtype)
        storage[:, 1:515:2].copy_(data)
        view = lambda x: x[:, 1:515:2]
    else:
        storage = data.clone()
        view = lambda x: x
    device_storage = storage.to(flag_gems.device)
    inp = view(device_storage)
    metadata = (inp.shape, inp.stride(), inp.storage_offset())
    out = flag_gems.rand_like(inp, dtype=dtype if override_dtype else None)
    _assert_range(out, inp.shape, dtype)
    assert out.stride() == torch.empty_like(view(storage), dtype=dtype).stride()
    assert out.data_ptr() != inp.data_ptr()
    assert (inp.shape, inp.stride(), inp.storage_offset()) == metadata
    torch.testing.assert_close(device_storage.cpu(), storage, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("op", ["rand", "rand_like"])
def test_uniform_reseed_and_state_progression(dtype, op):
    torch.manual_seed(20261026)
    first = _draw(op, (4096,), dtype).cpu()
    second = _draw(op, (4096,), dtype).cpu()
    torch.manual_seed(20261026)
    replay = _draw(op, (4096,), dtype).cpu()
    assert torch.equal(first, replay)
    assert not torch.equal(first, second)


@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("op", ["rand", "rand_like"])
def test_uniform_distribution_controls(dtype, op):
    torch.manual_seed(20261120)
    out = _draw(op, (262144,), dtype)
    _assert_range(out, out.shape, dtype)
    sample = out.cpu().float()
    bins = torch.histc(sample, bins=64, min=0.0, max=1.0)
    expected = sample.numel() / 64
    chi_square = float(((bins - expected).square() / expected).sum())
    assert chi_square < 160.0, f"uniform histogram chi-square {chi_square}"
    assert abs(float(sample.mean()) - 0.5) < 0.006
    assert abs(float(sample.var(unbiased=False)) - 1 / 12) < 0.003


@triton.jit
def _convert_extreme_words(words, out, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(words + offsets, offsets < N, other=0).to(tl.uint32)
    result = _uint_to_uniform_float_for_dtype(value, out.dtype.element_ty)
    tl.store(out + offsets, result, offsets < N)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_uniform_conversion_excluded_endpoint(dtype):
    words = torch.tensor(
        [0, 1, 2**32 - 2, 2**32 - 1], dtype=torch.int64, device=flag_gems.device
    )
    out = torch.empty((4,), dtype=dtype, device=flag_gems.device)
    with torch_device_fn.device(out.device):
        _convert_extreme_words[(1,)](words, out, N=4, BLOCK=32)
    _assert_range(out, (4,), dtype)
    actual = out.cpu()
    assert actual[0].item() == 0.0
    upper = torch.nextafter(
        torch.tensor(1.0, dtype=dtype), torch.tensor(0.0, dtype=dtype)
    )
    assert torch.equal(actual[-2:], upper.expand(2))


@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("seed, offset", [(1, 0), (2**40 + 3, 2**40)])
def test_rand_kernel_explicit_philox_state(dtype, seed, offset):
    outputs = [
        torch.empty((1025,), dtype=dtype, device=flag_gems.device) for _ in range(3)
    ]
    grid = lambda meta: (triton.cdiv(1025, meta["BLOCK"] * 4),)
    with torch_device_fn.device(outputs[0].device):
        rand_kernel[grid](outputs[0], 1025, seed, offset)
        rand_kernel[grid](outputs[1], 1025, seed, offset)
        rand_kernel[grid](outputs[2], 1025, seed, offset + 4096)
    for out in outputs:
        _assert_range(out, (1025,), dtype)
    assert torch.equal(outputs[0].cpu(), outputs[1].cpu())
    assert not torch.equal(outputs[0].cpu(), outputs[2].cpu())


@pytest.mark.parametrize("op", ["rand", "rand_like"])
@pytest.mark.parametrize("size", [0, 1, 1025, 16384])
def test_uniform_reserves_once_with_existing_increment(monkeypatch, op, size):
    module = importlib.import_module("flag_gems.ops." + op)
    reservations = []

    def reserve(increment):
        reservations.append(increment)
        return 20261026, 0

    monkeypatch.setattr(module, "philox_backend_seed_offset", reserve)
    _assert_range(_draw(op, (size,), torch.float16), (size,), torch.float16)
    assert reservations == [(size + 3) // 4]

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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
else:
    DIM_LIST = [0, 1]

random.seed(time.time() // 100)


CONTIGUOUS_SUFFIX_CASES = [
    ((1024, 4), 0),
    ((1, 2048, 8), 1),
    ((2, 8, 2048, 16), 2),
    ((2, 8, 2048, 32), 2),
    ((1024, 64), 0),
]


def _make_repeated_index(index_len):
    index_range = max(index_len // 2, 1)
    return torch.arange(index_len, device=flag_gems.device) % index_range


@pytest.mark.index_add
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    res_out = flag_gems.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    res_out = flag_gems.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    flag_gems.index_add_(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_inplace_contiguous_suffix(shape, dim, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    index = _make_repeated_index(inp.size(dim))
    src = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    flag_gems.index_add_(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_invalid_index(inplace):
    shape = (2, 4, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.ones((2, 2, 8), device=flag_gems.device)
    index = torch.tensor([0, shape[dim]], device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"):
        if inplace:
            flag_gems.index_add_(inp, dim, index, src)
        else:
            flag_gems.index_add(inp, dim, index, src)

    utils.gems_assert_equal(inp, ref_inp)


# These regressions exercise the common implementation and the MetaX override
# that reuses its validation. Other vendor implementations have their own
# argument/layout handling and are covered by the existing tests above.
_COMMON_CONTRACT = flag_gems.index_add.__module__ == "flag_gems.ops.index_add" or (
    flag_gems.vendor_name == "metax"
)
_contract_test = pytest.mark.skipif(
    not _COMMON_CONTRACT, reason="common/MetaX index_add semantic regressions"
)


def _call_index_add(inp, dim, index, src, inplace, alpha=1):
    fn = flag_gems.index_add_ if inplace else flag_gems.index_add
    return fn(inp, dim, index, src, alpha=alpha)


@_contract_test
@pytest.mark.skipif(not utils.bf16_is_supported, reason="BF16 is not supported")
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("suffix", [1, 3, 33])
@pytest.mark.parametrize(
    "initial, alpha, receiver_value",
    [(0, 1, 1024), (256, 0.5, 768), (-256, -0.5, -768)],
)
def test_index_add_bf16_repeated_exact_sum(
    inplace, suffix, initial, alpha, receiver_value
):
    # Repeated BF16 writeback loses unit increments at 256. Preserve the
    # existing FP32 accumulation contract, including flat/tile and fallback.
    inp = torch.full(
        (2, 3, suffix), initial, dtype=torch.bfloat16, device=flag_gems.device
    )
    src = torch.ones((2, 1024, suffix), dtype=torch.bfloat16, device=flag_gems.device)
    index = torch.zeros(1024, dtype=torch.int64, device=flag_gems.device)
    original = inp.cpu().clone()
    reference = torch.index_add(
        original.double(), 1, index.cpu(), src.cpu().double(), alpha=alpha
    )
    expected = torch.full((2, 3, suffix), initial, dtype=torch.float64)
    expected[:, 0, :] = receiver_value
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)

    result = _call_index_add(inp, 1, index, src, inplace, alpha=alpha)

    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(result.cpu().double(), expected, rtol=0, atol=0)
    if inplace:
        assert result is inp
    else:
        assert result is not inp
        torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)


@_contract_test
@pytest.mark.skipif(not utils.bf16_is_supported, reason="BF16 is not supported")
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("suffix", [1, 3, 33])
def test_index_add_bf16_high_precision_reference(inplace, suffix):
    generator = torch.Generator().manual_seed(31)
    original = (torch.randn((2, 3, suffix), generator=generator) * 0.125).bfloat16()
    source = (torch.randn((2, 1024, suffix), generator=generator) * 0.25).bfloat16()
    indices = torch.zeros(1024, dtype=torch.int64)
    # Reference the actual quantized inputs, accumulate on CPU in FP64, and
    # retain the upstream BF16 tolerance; do not compare to a BF16 running sum.
    reference = torch.index_add(
        original.double(), 1, indices, source.double(), alpha=-0.5
    )
    inp = original.to(flag_gems.device)
    result = _call_index_add(
        inp,
        1,
        indices.to(flag_gems.device),
        source.to(flag_gems.device),
        inplace,
        alpha=-0.5,
    )

    utils.gems_assert_close(result.cpu(), reference, dtype=torch.bfloat16)
    if inplace:
        assert result is inp
    else:
        torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add_
@pytest.mark.parametrize("view_kind", ["mixed-dtype-disjoint", "identical-strided"])
def test_index_add_alias_byte_ranges(view_kind):
    if view_kind == "mixed-dtype-disjoint":
        storage = torch.zeros(8, dtype=torch.int64, device=flag_gems.device)
        inp = storage.view(torch.float32)[2:4]
        index = storage[2:4]
        index.copy_(torch.tensor([0, 1], device=flag_gems.device))
        src = torch.ones_like(inp)
    else:
        storage = torch.arange(8, dtype=torch.float32, device=flag_gems.device)
        inp, src = storage[::2], storage[::2]
        index = torch.arange(4, device=flag_gems.device)
    expected = torch.index_add(inp.cpu().clone(), 0, index.cpu(), src.cpu())
    assert flag_gems.index_add_(inp, 0, index, src) is inp
    torch.testing.assert_close(inp.cpu(), expected)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "dtype, alpha",
    [
        (torch.float32, 1j),
        (torch.float32, 1e100),
        (torch.int32, 1j),
        (torch.float32, None),
    ],
)
def test_index_add_empty_validates_alpha(inplace, dtype, alpha):
    inp = torch.zeros(3, dtype=dtype, device=flag_gems.device)
    index = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    src = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with pytest.raises((RuntimeError, TypeError)) as reference:
        torch.index_add(inp.cpu(), 0, index.cpu(), src.cpu(), alpha=alpha)
    with pytest.raises(type(reference.value)):
        _call_index_add(inp, 0, index, src, inplace, alpha=alpha)
    torch.testing.assert_close(inp.cpu(), torch.zeros(3, dtype=dtype))


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dim", [1, -2])
@pytest.mark.parametrize("layout", ["contiguous", "input", "source", "index"])
def test_index_add_layout_contract(inplace, dim, layout):
    inp = torch.arange(2 * 7 * 17, device=flag_gems.device).float().reshape(2, 7, 17)
    src = (
        (torch.arange(2 * 4 * 17, device=flag_gems.device) % 5)
        .float()
        .reshape(2, 4, 17)
    )
    index = torch.tensor([0, 2, 2, 6], device=flag_gems.device)
    if layout == "input":
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)
    elif layout == "source":
        src = src.transpose(0, 1).contiguous().transpose(0, 1)
    elif layout == "index":
        storage = torch.zeros(8, dtype=torch.int64, device=flag_gems.device)
        storage[::2] = index
        index = storage[::2]
    original = inp.cpu().clone()
    expected = torch.index_add(original, dim, index.cpu(), src.cpu(), alpha=2)
    stride = inp.stride()

    result = _call_index_add(inp, dim, index, src, inplace, alpha=2)

    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    if inplace:
        assert result is inp
        assert inp.stride() == stride
    else:
        torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("zero_prefix", [False, True])
def test_index_add_empty_contract(inplace, zero_prefix):
    shape = (0, 7, 17) if zero_prefix else (2, 7, 17)
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.empty((shape[0], 1 if zero_prefix else 0, 17), device=flag_gems.device)
    index = torch.tensor(
        [99] if zero_prefix else [], dtype=torch.int64, device=flag_gems.device
    )
    expected = torch.index_add(inp.cpu(), 1, index.cpu(), src.cpu())

    result = _call_index_add(inp, 1, index, src, inplace)

    torch.testing.assert_close(result.cpu(), expected)
    if inplace:
        assert result is inp


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "index_dtype", "index_rank", "dim"]
)
def test_index_add_rejects_invalid_metadata(inplace, invalid):
    inp = torch.zeros((2, 7, 17), device=flag_gems.device)
    src = torch.empty((2, 0, 17), device=flag_gems.device)
    index = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    dim = 1
    if invalid == "shape":
        src = torch.empty((3, 0, 17), device=flag_gems.device)
    elif invalid == "dtype":
        src = src.half()
    elif invalid == "index_dtype":
        index = index.float()
    elif invalid == "index_rank":
        index = index.reshape(0, 1)
    else:
        dim = -4
    original = inp.cpu().clone()
    with pytest.raises((IndexError, RuntimeError)) as reference:
        torch.index_add(inp.cpu(), dim, index.cpu(), src.cpu())
    with pytest.raises(type(reference.value)):
        _call_index_add(inp, dim, index, src, inplace)
    torch.testing.assert_close(inp.cpu(), original)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("scalar_input", [False, True])
def test_index_add_scalar_contract(inplace, scalar_input):
    inp = (
        torch.tensor(2.0, device=flag_gems.device)
        if scalar_input
        else torch.zeros((2, 3), device=flag_gems.device)
    )
    src = (
        torch.tensor(3.0, device=flag_gems.device)
        if scalar_input
        else torch.ones((2, 1), device=flag_gems.device)
    )
    index = torch.tensor(0 if scalar_input else 1, device=flag_gems.device)
    expected = torch.index_add(
        inp.cpu(), 0 if scalar_input else 1, index.cpu(), src.cpu()
    )
    result = _call_index_add(inp, 0 if scalar_input else 1, index, src, inplace)
    torch.testing.assert_close(result.cpu(), expected)


@_contract_test
@pytest.mark.index_add_
@pytest.mark.parametrize("overlap", [False, True])
def test_index_add_alias_contract(overlap):
    storage = torch.arange(12, device=flag_gems.device).float().reshape(4, 3)
    inp = storage[:2]
    src = inp if overlap else storage[2:]
    index = torch.tensor([1, 0], device=flag_gems.device)
    original = inp.cpu().clone()
    if overlap:
        with pytest.raises(RuntimeError, match="overlap|memory location"):
            flag_gems.index_add_(inp, 0, index, src)
        torch.testing.assert_close(inp.cpu(), original)
    else:
        expected = torch.index_add(original, 0, index.cpu(), src.cpu())
        assert flag_gems.index_add_(inp, 0, index, src) is inp
        torch.testing.assert_close(inp.cpu(), expected)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_tensor", ["input", "source"])
def test_index_add_lazy_negative_values(inplace, negative_tensor):
    inp = torch.ones((2, 7, 17), device=flag_gems.device)
    src = torch.ones((2, 4, 17), device=flag_gems.device)
    if negative_tensor == "input":
        inp = torch._neg_view(inp)
    else:
        src = torch._neg_view(src)
    index = torch.tensor([0, 2, 2, 6], device=flag_gems.device)
    expected = torch.index_add(inp.cpu().clone(), 1, index.cpu(), src.cpu())
    result = _call_index_add(inp, 1, index, src, inplace)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_index_add_lazy_negative_index(inplace, index_dtype):
    inp = torch.zeros((2, 7, 17), device=flag_gems.device)
    src = torch.ones((2, 4, 17), device=flag_gems.device)
    physical = torch.tensor([0, -2, -2, -6], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(physical)
    expected = torch.index_add(inp.cpu(), 1, index.cpu(), src.cpu(), alpha=-0.5)
    result = _call_index_add(inp, 1, index, src, inplace, alpha=-0.5)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add_
@pytest.mark.parametrize(
    "alias", ["partial", "empty-same", "empty-view", "strided-disjoint"]
)
def test_index_add_inplace_alias_views(alias):
    storage = torch.arange(12, device=flag_gems.device).float()
    if alias == "partial":
        inp, src = storage[:4], storage[2:6]
    elif alias == "empty-same":
        inp = src = storage[:0]
    elif alias == "empty-view":
        inp, src = storage[:4], storage[:0]
    else:
        inp, src = storage[::2], storage[1::2]
    index = torch.arange(src.numel(), device=flag_gems.device)
    original = inp.cpu().clone()
    if alias in ("partial", "empty-same"):
        with pytest.raises(RuntimeError):
            flag_gems.index_add_(inp, 0, index, src)
        torch.testing.assert_close(inp.cpu(), original)
    else:
        expected = torch.index_add(original, 0, index.cpu(), src.cpu())
        assert flag_gems.index_add_(inp, 0, index, src) is inp
        torch.testing.assert_close(inp.cpu(), expected)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_revalidates_mutated_index(inplace):
    inp = torch.zeros((2, 7, 17), device=flag_gems.device)
    src = torch.ones((2, 2, 17), device=flag_gems.device)
    index = torch.tensor([0, 1], device=flag_gems.device)
    _call_index_add(inp, 1, index, src, inplace)
    original = inp.cpu().clone()
    # .data writes do not update the original tensor's version counter.
    index.data[1] = 7
    with pytest.raises(AssertionError, match="0 <= index"):
        _call_index_add(inp, 1, index, src, inplace)
    torch.testing.assert_close(inp.cpu(), original)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("wrong_device", ["source", "index"])
def test_index_add_rejects_device_mismatch(inplace, wrong_device):
    inp = torch.zeros((2, 3), device=flag_gems.device)
    src = torch.ones(
        (2, 1), device="cpu" if wrong_device == "source" else flag_gems.device
    )
    index = torch.tensor(
        [0], device="cpu" if wrong_device == "index" else flag_gems.device
    )
    with pytest.raises(RuntimeError, match="same device"):
        _call_index_add(inp, 1, index, src, inplace)
    torch.testing.assert_close(inp.cpu(), torch.zeros((2, 3)))


@_contract_test
@pytest.mark.index_add_
def test_index_add_large_source_address():
    if torch.cuda.mem_get_info()[0] < 6 * 1024**3:
        pytest.skip("large-address regression requires 6 GiB free memory")
    length, suffix, receivers = 1_100_001, 513, 17
    inp = torch.zeros((1, receivers, suffix), device=flag_gems.device)
    src = torch.ones((1, length, suffix), device=flag_gems.device)
    index = torch.arange(length, device=flag_gems.device) % receivers
    quotient, remainder = divmod(length, receivers)
    counts = torch.full((receivers,), float(quotient))
    counts[:remainder] += 1
    assert flag_gems.index_add_(inp, 1, index, src) is inp
    expected = counts[None, :, None].expand_as(inp.cpu())
    torch.testing.assert_close(inp.cpu(), expected, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("suffix", [1, 3, 33])
@pytest.mark.skipif(not utils.bf16_is_supported, reason="BF16 is not supported")
def test_index_add_bf16_preserves_fp32_alpha(inplace, suffix):
    # Preserve the existing FP32 scratch multiply. Rounding 0.1 to BF16
    # before multiplying gives 0.25 instead of the exact cancellation 0.
    inp = torch.full(
        (1, 1, suffix), -256, dtype=torch.bfloat16, device=flag_gems.device
    )
    src = torch.full_like(inp, 2560)
    index = torch.zeros(1, dtype=torch.int64, device=flag_gems.device)
    original = inp.cpu().clone()
    result = _call_index_add(inp, 1, index, src, inplace, alpha=0.1)
    torch.testing.assert_close(result.cpu(), torch.zeros_like(original), rtol=0, atol=0)
    if not inplace:
        torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("alpha", [65505, 100000])
def test_index_add_fp16_rejects_finite_alpha_overflow(inplace, empty, alpha):
    inp = torch.ones((1, 3, 3), dtype=torch.float16, device=flag_gems.device)
    src = torch.ones((1, 0 if empty else 1, 3), dtype=inp.dtype, device=inp.device)
    index = torch.zeros(src.size(1), dtype=torch.int64, device=inp.device)
    original = inp.cpu().clone()
    with pytest.raises(RuntimeError, match="overflow"):
        # The GPU checked conversion differs from CPU's multi-dimensional
        # empty fast return. Compare the actual target backend's Native API.
        torch.index_add(inp, 1, index, src, alpha=alpha)
    with pytest.raises(RuntimeError, match="overflow"):
        _call_index_add(inp, 1, index, src, inplace, alpha=alpha)
    torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("alpha", [float("inf"), float("nan")])
def test_index_add_fp16_accepts_explicit_nonfinite_alpha_empty(inplace, alpha):
    inp = torch.ones(3, dtype=torch.float16, device=flag_gems.device)
    src = torch.empty(0, dtype=inp.dtype, device=inp.device)
    index = torch.empty(0, dtype=torch.int64, device=inp.device)
    expected = torch.index_add(inp.cpu(), 0, index.cpu(), src.cpu(), alpha=alpha)
    result = _call_index_add(inp, 0, index, src, inplace, alpha=alpha)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@_contract_test
@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_singleton_noncanonical_stride(inplace):
    inp = torch.arange(3, device=flag_gems.device).float().reshape(3, 1).t()
    assert inp.is_contiguous() and inp.stride() == (1, 1)
    src = torch.arange(6, device=inp.device).float().reshape(2, 3)
    index = torch.zeros(2, dtype=torch.int64, device=inp.device)
    original = inp.cpu().clone()
    expected = torch.tensor([[3.0, 6.0, 9.0]])
    torch.testing.assert_close(
        torch.index_add(original, 0, index.cpu(), src.cpu()), expected
    )
    result = _call_index_add(inp, 0, index, src, inplace)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    if inplace:
        assert result is inp and inp.stride() == (1, 1)
    else:
        torch.testing.assert_close(inp.cpu(), original, rtol=0, atol=0)

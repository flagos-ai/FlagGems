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


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.int32, torch.int64]
)
def test_index_add_empty_index(inplace, dtype):
    """An empty index is a no-op, not a division by zero (issue #6657)."""
    inp = torch.ones((2, 4, 8), dtype=dtype, device=flag_gems.device)
    index = torch.empty(0, dtype=torch.int64, device=flag_gems.device)
    src = torch.empty((2, 0, 8), dtype=dtype, device=flag_gems.device)
    # Snapshot through to_reference so the baseline follows the reference
    # device (CPU in quick-cpu mode) that gems_assert_equal expects.
    expected = utils.to_reference(inp.clone())

    if inplace:
        flag_gems.index_add_(inp, 1, index, src)
        utils.gems_assert_equal(inp, expected)
    else:
        res = flag_gems.index_add(inp, 1, index, src)
        utils.gems_assert_equal(res, expected)
        # The out-of-place form must leave the input untouched.
        utils.gems_assert_equal(inp, expected)


def _hygon_fp16_word_cas_device_view(values, shape, offset):
    # Construct the view on the real device. Moving a CPU view with .to(device)
    # could lose its storage offset or padding and weaken this regression.
    backing = torch.full(
        (offset + len(values) + 4,),
        -13.0,
        dtype=torch.float16,
        device=flag_gems.device,
    )
    view = backing[offset : offset + len(values)].view(shape)
    view.copy_(
        torch.tensor(values, dtype=torch.float16, device=flag_gems.device).reshape(
            shape
        )
    )
    assert view.device.type != "cpu"
    assert view.storage_offset() == offset
    assert view.is_contiguous()
    return backing, view


def _hygon_fp16_word_cas_cpu_reference(inp, dim, index, src):
    # Exactly one update per CPU call forces FP16 rounding after each update.
    # Each destination receives identical dyadic increments in these fixtures,
    # so every legal atomic serialization has the same result. Do not use one
    # wide accumulated sum for the 2050 +/- 1 tie cases.
    reference = inp.detach().cpu().clone()
    cpu_index = index.detach().cpu()
    cpu_src = src.detach().cpu()
    for j in range(cpu_index.numel()):
        reference.index_add_(
            dim, cpu_index[j : j + 1], cpu_src.narrow(dim, j, 1), alpha=1
        )
    return reference


def _hygon_fp16_word_cas_check_case(shape, dim, index_values, inp_values, src_values):
    src_shape = list(shape)
    src_shape[dim] = len(index_values)
    inp_backing, inp = _hygon_fp16_word_cas_device_view(inp_values, shape, 3)
    src_backing, src = _hygon_fp16_word_cas_device_view(src_values, src_shape, 1)
    index_backing = torch.full(
        (len(index_values) + 4,),
        -7,
        dtype=torch.int64,
        device=flag_gems.device,
    )
    index = index_backing[2 : 2 + len(index_values)]
    index.copy_(torch.tensor(index_values, dtype=torch.int64, device=flag_gems.device))
    assert inp.device == src.device == index.device
    assert inp.numel() % 2 == 0
    assert inp.is_contiguous() and src.is_contiguous() and index.is_contiguous()
    backings = (inp_backing, src_backing, index_backing)
    for backing in backings:
        assert backing.storage_offset() == 0
        assert backing.untyped_storage().nbytes() == (
            backing.numel() * backing.element_size()
        )
    before = [backing.detach().cpu().clone() for backing in backings]
    reference = _hygon_fp16_word_cas_cpu_reference(inp, dim, index, src)

    with torch.inference_mode():
        result = flag_gems.index_add(inp, dim, index, src, alpha=1)

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype
    assert result.untyped_storage().data_ptr() not in {
        backing.untyped_storage().data_ptr() for backing in backings
    }
    result_cpu = result.detach().cpu().contiguous()
    # Exact finite FP16 bits, including untouched same-word halves. No relaxed
    # relative tolerance at magnitude 2050: that could accept a wrong RTZ result.
    assert torch.equal(
        result_cpu.view(torch.int16), reference.contiguous().view(torch.int16)
    )
    for backing, expected in zip(backings, before):
        assert torch.equal(
            backing.detach().cpu().view(torch.uint8), expected.view(torch.uint8)
        ), "index_add mutated input/source/index backing storage or padding"
    return result_cpu, reference


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon FP16 word-CAS repair"
)
def test_index_add_hygon_fp16_word_cas_odd_suffix_tail_neighbor():
    # 51 source elements leave masked tail lanes. Updating element 2 must not
    # overwrite element 3, the untouched adjacent half of the same output word.
    result, reference = _hygon_fp16_word_cas_check_case(
        (2, 3),
        0,
        [0] * 17,
        [1.0, 2.0, 3.0, -4.0, 5.0, -6.0],
        [1 / 128, 0.0, -1 / 128] * 17,
    )
    assert result[1, 0].item() == reference[1, 0].item() == -4.0
    assert result[0, 1].item() == reference[0, 1].item() == 2.0


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon FP16 word-CAS repair"
)
def test_index_add_hygon_fp16_word_cas_multicta_positive65():
    # 1040 elements span five CTAs at the repair's BLOCK_SIZE=256, including
    # a tail CTA. Positive increments expose lost updates hidden by cancellation.
    result, reference = _hygon_fp16_word_cas_check_case(
        (2, 16), 0, [0] * 65, [0.0] * 32, [1 / 128] * (65 * 16)
    )
    assert torch.equal(
        reference[0], torch.full((16,), 65 / 128, dtype=torch.float16, device="cpu")
    )
    assert torch.equal(result[0], reference[0])


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon FP16 word-CAS repair"
)
@pytest.mark.parametrize(
    "sign",
    [pytest.param(1, id="positive-2050"), pytest.param(-1, id="negative-2050")],
)
def test_index_add_hygon_fp16_word_cas_odd_tie(sign):
    result, reference = _hygon_fp16_word_cas_check_case(
        (2, 3),
        0,
        [0] * 5,
        [sign * 2050.0, 11.0, 7.0, -9.0, 5.0, 3.0],
        [sign * 1.0, 0.0, 0.0] * 5,
    )
    # The first odd tie rounds away from zero to the even mantissa. Subsequent
    # identical updates stay at +/-2052. RTZ stays at +/-2050; a wide sum of all
    # five updates rounds to +/-2056. Neither is this FP16 per-update result.
    assert reference[0, 0].item() == sign * 2052
    assert result[0, 0].item() == sign * 2052
    assert result[0, 1].item() == 11.0


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="Hygon FP16 word-CAS repair"
)
@pytest.mark.parametrize(
    "dim, suffix",
    [
        pytest.param(1, 3, id="dim1-suffix3"),
        pytest.param(1, 4, id="dim1-suffix4"),
        pytest.param(-2, 3, id="dim-neg2-suffix3"),
        pytest.param(-2, 4, id="dim-neg2-suffix4"),
    ],
)
def test_index_add_hygon_fp16_word_cas_rank3_source_offset(dim, suffix):
    # Prefix=2, odd/even suffixes and repeated indices exercise flattened
    # address decomposition. The helper creates source-offset=1 on device and
    # verifies complete initialized backing storage, not just the logical view.
    _hygon_fp16_word_cas_check_case(
        (2, 4, suffix),
        dim,
        [0, 1, 1, 3, 0],
        [(i % 9 - 4) / 8 for i in range(2 * 4 * suffix)],
        [1 / 128] * (2 * 5 * suffix),
    )

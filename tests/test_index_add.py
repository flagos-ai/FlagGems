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


def _run_index_add(inp, dim, index, src, inplace, alpha=1):
    if inplace:
        result = inp.index_add_(dim, index, src, alpha=alpha)
        assert result is inp
        return result
    return torch.index_add(inp, dim, index, src, alpha=alpha)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_empty_index(inplace, dtype):
    inp = torch.randn((2, 7, 17), dtype=dtype, device=flag_gems.device)
    src = torch.empty((2, 0, 17), dtype=dtype, device=flag_gems.device)
    index = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with flag_gems.use_gems():
        if inplace:
            result = inp.index_add_(1, index, src)
            assert result is inp
        else:
            result = torch.index_add(inp, 1, index, src)

    utils.gems_assert_equal(result, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_negative_dim_contiguous_suffix(inplace):
    shape = (2, 7, 17)
    dim = -2
    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    src = torch.ones((2, 4, 17), dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([0, 2, 2, 6], device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)

    ref_result = _run_index_add(ref_inp, dim, ref_index, ref_src, inplace)
    with flag_gems.use_gems():
        result = _run_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "dim, src_shape, index_values",
    [
        (3, (4, 7, 17), [0, 1, 1, 0]),
        (-4, (2, 7, 4), [0, 2, 2, 6]),
    ],
)
def test_index_add_invalid_dim(inplace, dim, src_shape, index_values):
    inp = torch.zeros((2, 7, 17), dtype=torch.float32, device=flag_gems.device)
    src = torch.ones(src_shape, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor(index_values, device=flag_gems.device)
    original = utils.to_reference(inp.clone())
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)

    with pytest.raises(IndexError):
        _run_index_add(ref_inp, dim, ref_index, ref_src, inplace)
    with flag_gems.use_gems(), pytest.raises(IndexError):
        _run_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(inp, original)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize(
    "inp_noncontiguous, src_noncontiguous",
    [(True, False), (False, True)],
    ids=["noncontiguous-input", "noncontiguous-source"],
)
def test_index_add_noncontiguous_input_and_source(
    inplace, inp_noncontiguous, src_noncontiguous
):
    inp = torch.arange(
        2 * 7 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 7, 17)
    src = torch.arange(
        2 * 4 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 4, 17)
    if inp_noncontiguous:
        inp = inp.transpose(0, 1).contiguous().transpose(0, 1)
    if src_noncontiguous:
        src = src.transpose(0, 1).contiguous().transpose(0, 1)

    assert inp.is_contiguous() == (not inp_noncontiguous)
    assert src.is_contiguous() == (not src_noncontiguous)
    index = torch.tensor([0, 2, 2, 6], device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src.clone())
    ref_index = utils.to_reference(index)
    ref_result = _run_index_add(ref_inp, 1, ref_index, ref_src, inplace)

    with flag_gems.use_gems():
        result = _run_index_add(inp, 1, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add_
def test_index_add_inplace_input_source_alias():
    inp = torch.arange(
        2 * 7 * 17, dtype=torch.float32, device=flag_gems.device
    ).reshape(2, 7, 17)
    index = torch.tensor([0, 2, 2, 4, 4, 6, 0], device=flag_gems.device)
    original = utils.to_reference(inp.clone())
    ref_inp = utils.to_reference(inp.clone())
    ref_index = utils.to_reference(index)

    with pytest.raises(RuntimeError):
        ref_inp.index_add_(1, ref_index, ref_inp)
    utils.gems_assert_equal(ref_inp, original)

    with flag_gems.use_gems(), pytest.raises(RuntimeError):
        inp.index_add_(1, index, inp)
    utils.gems_assert_equal(inp, original)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_partial_contiguous_suffix(inplace, dtype):
    shape = (2, 257, 513)
    dim = 1
    index_len = 129
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    src = torch.ones((2, index_len, 513), dtype=dtype, device=flag_gems.device)
    index = torch.arange(index_len, device=flag_gems.device) % 65
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)

    ref_result = _run_index_add(ref_inp, dim, ref_index, ref_src, inplace, alpha=2)
    with flag_gems.use_gems():
        result = _run_index_add(inp, dim, index, src, inplace, alpha=2)

    utils.gems_assert_close(result, ref_result, dtype=dtype, reduce_dim=dim)


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
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

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
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

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
    with flag_gems.use_gems():
        inp.index_add_(dim, index, src, alpha=alpha)

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
    with flag_gems.use_gems():
        inp.index_add_(dim, index, src, alpha=alpha)

    utils.gems_assert_equal(inp, ref_inp)


# Randomized, non-exactly-representable values: zeros/ones/alpha=2 cannot
# reveal rounding differences between fp32-accumulate-then-cast and native
# bf16 accumulation. A deterministic seed keeps the loose-tolerance native
# parity check stable across runs.
CONTIGUOUS_SUFFIX_STRESS_CASES = [
    ((2, 8, 2048, 32), 2),  # flat path, narrow suffix
    ((2, 8, 2048, 72), 2),  # flat path, partially filled second tile
    ((2, 8, 2048, 256), 2),  # tile path, wide suffix
    ((1024, 64), 0),  # dim == 0 always routed to flat
]


def _make_dup_index(index_len, dup_factor):
    receiver_range = max(index_len // dup_factor, 1)
    return torch.arange(index_len, device=flag_gems.device) % receiver_range


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("shape, dim", CONTIGUOUS_SUFFIX_STRESS_CASES)
@pytest.mark.parametrize("dup_factor", [2, 32])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_contiguous_suffix_randomized(shape, dim, dup_factor, inplace):
    torch.manual_seed(2024 + dup_factor * 8 + (1 if inplace else 0))
    dtype = torch.bfloat16
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    index_len = shape[dim] // 2
    index = _make_dup_index(index_len, dup_factor)
    src_shape = list(shape)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 0.7

    # Native bf16-accumulate reference (what the vendor torch does).
    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    if inplace:
        ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    else:
        ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)

    # High-precision reference models accumulate then cast once, which is the
    # contract of the optimized bf16 fallback. Per-add bf16 rounding would
    # fail this check at dup_factor=32.
    exact_inp = utils.to_reference(inp, upcast=True)
    exact_src = utils.to_reference(src, upcast=True)
    exact_out = torch.index_add(exact_inp, dim, ref_index, exact_src, alpha=alpha)

    with flag_gems.use_gems():
        result = _run_index_add(inp, dim, index, src, inplace, alpha=alpha)

    target = ref_inp if inplace else ref_out
    utils.gems_assert_close(result, exact_out, dtype=dtype, reduce_dim=1, atol=0.05)
    utils.gems_assert_close(result, target, dtype=dtype, reduce_dim=1, atol=0.5)


@pytest.mark.index_add
@pytest.mark.skipif(
    flag_gems.vendor_name != "metax", reason="MetaX-specific routing policy"
)
@pytest.mark.parametrize(
    "suffix_size, expected_route",
    [(64, "flat"), (65, "flat"), (79, "flat"), (80, "tile"), (512, "tile")],
)
def test_index_add_metax_contiguous_suffix_route(
    monkeypatch, suffix_size, expected_route
):
    metax_index_add = importlib.import_module(
        "flag_gems.runtime.backend._metax.ops.index_add"
    )
    selected = []

    def record_flat(out, dim, index, src, alpha):
        selected.append("flat")
        return out

    def record_tile(out, dim, index, src, alpha):
        selected.append("tile")
        return out

    monkeypatch.setattr(
        metax_index_add, "_run_contiguous_suffix_flat_path", record_flat
    )
    monkeypatch.setattr(
        metax_index_add, "_run_contiguous_suffix_tile_path", record_tile
    )
    out = torch.empty((1, 2, suffix_size), dtype=torch.float32, device=flag_gems.device)
    src = torch.empty((1, 1, suffix_size), dtype=torch.float32, device=flag_gems.device)
    index = torch.zeros((1,), dtype=torch.int64, device=flag_gems.device)

    metax_index_add._run_contiguous_suffix_path(out, 1, index, src, 1.0)

    assert selected == [expected_route]


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

    with (
        flag_gems.use_gems(),
        pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"),
    ):
        if inplace:
            inp.index_add_(dim, index, src)
        else:
            torch.index_add(inp, dim, index, src)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_lazy_negative_index(inplace, index_dtype):
    shape = (1, 2, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.stack(
        (
            torch.ones((1, 8), device=flag_gems.device),
            torch.full((1, 8), 2.0, device=flag_gems.device),
        ),
        dim=1,
    )
    raw_index = torch.tensor([-1, 0], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(raw_index)
    assert index.is_neg()

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = torch.tensor([1, 0], dtype=index_dtype, device=ref_inp.device)
    ref_result = _run_index_add(ref_inp, dim, ref_index, ref_src, inplace)

    with flag_gems.use_gems():
        result = _run_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_lazy_negative_index_fallback(inplace, index_dtype):
    shape = (1, 2, 8)
    dim = 1
    inp = torch.zeros(
        (shape[0], shape[2], shape[1]), device=flag_gems.device
    ).transpose(1, 2)
    src = torch.stack(
        (
            torch.ones((1, 8), device=flag_gems.device),
            torch.full((1, 8), 2.0, device=flag_gems.device),
        ),
        dim=2,
    ).transpose(1, 2)
    assert not inp.is_contiguous()
    assert not src.is_contiguous()

    raw_index = torch.tensor([-1, 0], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(raw_index)
    assert index.is_neg()

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src)
    ref_index = torch.tensor([1, 0], dtype=index_dtype, device=ref_inp.device)
    ref_result = _run_index_add(ref_inp, dim, ref_index, ref_src, inplace)

    with flag_gems.use_gems():
        result = _run_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(result, ref_result)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_lazy_negative_oob_does_not_write_input(
    inplace, index_dtype, fallback
):
    shape = (1, 2, 8)
    dim = 1
    if fallback:
        storage_shape = (shape[0], shape[2], shape[1])
        inp = torch.zeros(storage_shape, device=flag_gems.device).transpose(1, 2)
        src = torch.ones(storage_shape, device=flag_gems.device).transpose(1, 2)
        assert not inp.is_contiguous()
        assert not src.is_contiguous()
    else:
        inp = torch.zeros(shape, device=flag_gems.device)
        src = torch.ones(shape, device=flag_gems.device)
    # The physical values are valid, while the lazy-negative logical value -1
    # is invalid. This distinguishes correct materialization from accidentally
    # validating the un-negated storage.
    raw_index = torch.tensor([0, 1], dtype=index_dtype, device=flag_gems.device)
    index = torch._neg_view(raw_index)
    original = utils.to_reference(inp.clone())

    with (
        flag_gems.use_gems(),
        pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"),
    ):
        _run_index_add(inp, dim, index, src, inplace)

    utils.gems_assert_equal(inp, original)


@pytest.mark.index_add_
def test_index_add_inplace_revalidates_cached_index_after_mutation():
    shape = (2, 4, 8)
    dim = 1
    inp = torch.zeros(shape, device=flag_gems.device)
    src = torch.ones((2, 2, 8), device=flag_gems.device)
    index = torch.tensor([0, 1], device=flag_gems.device)

    with flag_gems.use_gems():
        inp.index_add_(dim, index, src)
    before_invalid_call = utils.to_reference(inp.clone())
    index[-1] = shape[dim]

    with (
        flag_gems.use_gems(),
        pytest.raises(AssertionError, match=r"0 <= index < self\.size\(dim\)"),
    ):
        inp.index_add_(dim, index, src)

    utils.gems_assert_equal(inp, before_invalid_call)

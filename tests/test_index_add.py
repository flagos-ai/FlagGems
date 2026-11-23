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

import flag_gems
import pytest
import torch

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

    ref_inp = utils.to_reference(inp.clone())
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

    ref_inp = utils.to_reference(inp.clone())
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
    ref_inp = utils.to_reference(inp.clone())

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
    expected = utils.to_reference(inp.clone())

    if inplace:
        flag_gems.index_add_(inp, 1, index, src)
        utils.gems_assert_equal(inp, expected)
    else:
        res = flag_gems.index_add(inp, 1, index, src)
        utils.gems_assert_equal(res, expected)
        utils.gems_assert_equal(inp, expected)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("src_kind", ["transposed_4d", "sliced_3d", "expanded_3d"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_noncontiguous_source(inplace, src_kind, dtype):
    """Regression for issue #9904: non-contiguous source (e.g. index_select backward grad)."""
    gen = torch.Generator(device="cpu").manual_seed(42)
    if src_kind == "transposed_4d":
        dim = 0
        inp = torch.zeros((5, 16, 1, 64), dtype=dtype, device=flag_gems.device)
        src_contig = torch.randn((13, 16, 1, 64), generator=gen, dtype=dtype).to(
            flag_gems.device
        )
        src = src_contig.transpose(1, 3).contiguous().transpose(1, 3)
        index = torch.tensor(
            [0] + [1] * 2 + [2] * 3 + [3] * 3 + [4] * 4,
            dtype=torch.long,
            device=flag_gems.device,
        )
    elif src_kind == "sliced_3d":
        dim = -2
        inp = torch.randn((3, 7, 19), generator=gen, dtype=dtype).to(flag_gems.device)
        src_backing = torch.randn((3, 10, 38), generator=gen, dtype=dtype).to(
            flag_gems.device
        )
        src = src_backing[:, ::2, 1::2]  # shape (3, 5, 19), non-contiguous on two axes
        index = torch.tensor([0, 2, 2, 5, 6], dtype=torch.long, device=flag_gems.device)
    else:
        dim = 1
        inp = torch.zeros((4, 6, 17), dtype=dtype, device=flag_gems.device)
        src_base = torch.randn((1, 5, 17), generator=gen, dtype=dtype).to(
            flag_gems.device
        )
        src = src_base.expand(4, 5, 17)  # stride=(0, 17, 1)
        index = torch.tensor([1, 3, 1, 5, 0], dtype=torch.long, device=flag_gems.device)

    assert not src.is_contiguous()
    alpha = 0.5
    atol = (
        1e-2 if dtype == torch.bfloat16 else (1e-3 if dtype == torch.float16 else 1e-4)
    )
    orig_inp = utils.to_reference(inp.clone())
    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src.clone())
    ref_index = utils.to_reference(index.clone())
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)

    if inplace:
        res = flag_gems.index_add_(inp, dim, index, src, alpha=alpha)
        assert res is inp
        utils.gems_assert_close(
            inp, ref_out, dtype=dtype, reduce_dim=src.size(dim), atol=atol
        )
    else:
        res = flag_gems.index_add(inp, dim, index, src, alpha=alpha)
        utils.gems_assert_close(
            res, ref_out, dtype=dtype, reduce_dim=src.size(dim), atol=atol
        )
        utils.gems_assert_equal(inp, orig_inp)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("inp_kind", ["transposed", "sliced_holes"])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_add_noncontiguous_self(inplace, inp_kind, dtype):
    """Verify strided target `self` updates exact logical slots without corrupting storage holes."""
    gen = torch.Generator(device="cpu").manual_seed(7)
    if inp_kind == "transposed":
        dim = 1
        backing = torch.randn((2, 17, 7), generator=gen, dtype=dtype).to(
            flag_gems.device
        )
        ref_backing = utils.to_reference(backing.clone())
        inp = backing.transpose(1, 2)  # shape (2, 7, 17), stride (119, 1, 7)
        ref_inp = ref_backing.transpose(1, 2)
    else:
        dim = -1
        backing = torch.full((3, 12, 9), -7.0, dtype=dtype, device=flag_gems.device)
        backing[:, ::2, :] = torch.randn((3, 6, 9), generator=gen, dtype=dtype).to(
            flag_gems.device
        )
        ref_backing = utils.to_reference(backing.clone())
        inp = backing[:, ::2, :]  # shape (3, 6, 9), storage holes at [:, 1::2, :]
        ref_inp = ref_backing[:, ::2, :]

    assert not inp.is_contiguous()
    inp_stride = inp.stride()
    src_shape = list(inp.shape)
    src_shape[dim] = 4
    src = torch.randn(src_shape, generator=gen, dtype=dtype).to(flag_gems.device)
    index = torch.tensor([0, 2, 2, inp.size(dim) - 1], device=flag_gems.device)
    atol = (
        1e-2 if dtype == torch.bfloat16 else (1e-3 if dtype == torch.float16 else 1e-4)
    )

    orig_backing = utils.to_reference(backing.clone())
    ref_src = utils.to_reference(src.clone())
    ref_index = utils.to_reference(index.clone())

    if inplace:
        ref_inp.index_add_(dim, ref_index, ref_src, alpha=2)
        res = flag_gems.index_add_(inp, dim, index, src, alpha=2)
        assert res is inp
        assert inp.stride() == inp_stride
        utils.gems_assert_close(
            inp, ref_inp, dtype=dtype, reduce_dim=src.size(dim), atol=atol
        )
        if inp_kind == "sliced_holes":
            holes = utils.to_reference(backing[:, 1::2, :].clone())
            expected_holes = orig_backing[:, 1::2, :]
            utils.gems_assert_equal(holes, expected_holes)
    else:
        ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=2)
        res = flag_gems.index_add(inp, dim, index, src, alpha=2)
        utils.gems_assert_close(
            res, ref_out, dtype=dtype, reduce_dim=src.size(dim), atol=atol
        )
        utils.gems_assert_equal(backing, orig_backing)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("suffix_gt_one", [False, True])
def test_index_add_noncontiguous_index_with_storage_holes(inplace, suffix_gt_one):
    """Strided index views must read logical index elements and ignore storage holes."""
    shape = (2, 5, 8) if suffix_gt_one else (5,)
    dim = 1 if suffix_gt_one else 0
    src_shape = (2, 3, 8) if suffix_gt_one else (3,)

    inp = torch.zeros(shape, dtype=torch.float32, device=flag_gems.device)
    src = (
        torch.arange(1, 1 + torch.Size(src_shape).numel(), dtype=torch.float32)
        .reshape(src_shape)
        .to(flag_gems.device)
    )
    # Logical index is [0, 4, 2]; storage holes hold 99 (out-of-bounds) and 1 (wrong in-bounds).
    idx_storage = torch.tensor(
        [0, 99, 4, 1, 2], dtype=torch.int64, device=flag_gems.device
    )
    index = idx_storage[::2]
    assert not index.is_contiguous()

    ref_inp = utils.to_reference(inp.clone())
    ref_src = utils.to_reference(src.clone())
    ref_index = torch.tensor([0, 4, 2], dtype=torch.int64, device=ref_inp.device)
    expected = torch.index_add(ref_inp, dim, ref_index, ref_src)

    if inplace:
        res = flag_gems.index_add_(inp, dim, index, src)
        assert res is inp
        utils.gems_assert_equal(inp, expected)
    else:
        res = flag_gems.index_add(inp, dim, index, src)
        utils.gems_assert_equal(res, expected)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_mismatched_non_dim_shape_raises(inplace):
    inp = torch.zeros((2, 4, 8), dtype=torch.float32, device=flag_gems.device)
    src = torch.ones((3, 2, 8), dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(AssertionError, match="src.size"):
        if inplace:
            flag_gems.index_add_(inp, 1, index, src)
        else:
            flag_gems.index_add(inp, 1, index, src)


@pytest.mark.index_add
@pytest.mark.index_add_
@pytest.mark.parametrize("inplace", [False, True])
def test_index_add_zero_dim_scalar(inplace):
    inp = torch.tensor(2.5, dtype=torch.float32, device=flag_gems.device)
    src = torch.tensor(1.5, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor(0, dtype=torch.int64, device=flag_gems.device)
    expected = utils.to_reference(
        torch.tensor(5.5, dtype=torch.float32, device=inp.device)
    )

    if inplace:
        res = flag_gems.index_add_(inp, 0, index, src, alpha=2)
        assert res is inp and inp.ndim == 0
        utils.gems_assert_equal(inp, expected)
    else:
        res = flag_gems.index_add(inp, 0, index, src, alpha=2)
        assert res.ndim == 0
        utils.gems_assert_equal(res, expected)


@pytest.mark.index_add_
def test_index_add_inplace_storage_alias_and_overlap():
    base = torch.arange(1, 9, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())
    inp = base[:5]
    src = base[2:6]
    index = torch.tensor([0, 2, 1, 4], dtype=torch.int64, device=flag_gems.device)
    ref_index = utils.to_reference(index.clone())

    ref_base[:5].index_add_(0, ref_index, ref_base[2:6].clone())
    res = flag_gems.index_add_(inp, 0, index, src)
    assert res is inp
    utils.gems_assert_equal(base, ref_base)

    with pytest.raises(RuntimeError, match="unsupported operation"):
        flag_gems.index_add_(inp, 0, torch.arange(5, device=flag_gems.device), inp)

    expanded = torch.zeros((1, 4), dtype=torch.float32, device=flag_gems.device).expand(
        3, 4
    )
    src_2d = torch.ones((3, 2), dtype=torch.float32, device=flag_gems.device)
    idx_2d = torch.tensor([0, 2], dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="unsupported operation"):
        flag_gems.index_add_(expanded, 1, idx_2d, src_2d)

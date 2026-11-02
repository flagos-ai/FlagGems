# Copyright 2026, The FlagOS Contributors.
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

# Each entry is one tensor list. The single-tensor entries check the zero-copy
# view fast path, the two-to-four contiguous entries check the 2-D grid kernel,
# and the five-plus entries check the metadata fallback. Mixed ranks and
# unequal sizes make sure the running output offset is right.
if cfg.QUICK_MODE:
    FLATTEN_SHAPES = [
        [(4, 5)],
        [(4, 5), (4, 5)],
        [(64,), (32,), (16,)],
    ]
else:
    FLATTEN_SHAPES = [
        [()],
        [(3,)],
        [(), ()],
        [(256,), (256,)],
        [(4, 5)],
        [(4, 5), (4, 5)],
        [(4, 5), (3,), (7, 2)],
        [(8, 8), (8, 8), (8, 8)],
        [(8, 8), (8, 8), (8, 8), (8, 8)],
        [(9,), (9,), (9,), (9,), (9,)],
        [(2, 3, 4), (5, 6)],
        [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2)],
    ]

# Zero-element inputs still contribute their slot, so the output can be empty
# or merely have an empty tensor somewhere in the middle.
FLATTEN_ZERO_SHAPES = [
    [(0,), (0,)],
    [(0,)],
    [(2, 0), (3,)],
    [(4,), (0,), (5,)],
]


def _make_inputs(shapes, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return [torch.randn(s, dtype=dtype, device=flag_gems.device) for s in shapes]
    if dtype == torch.bool:
        return [
            torch.randint(0, 2, s, dtype=dtype, device="cpu").to(flag_gems.device)
            for s in shapes
        ]
    return [
        torch.randint(0, 0x7FFF, s, dtype=dtype, device="cpu").to(flag_gems.device)
        for s in shapes
    ]


def _reference(tensors):
    return torch.ops.aten.flatten_dense_tensors(
        [utils.to_reference(t) for t in tensors]
    )


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("shapes", FLATTEN_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_flatten_dense_tensors(shapes, dtype):
    inp = _make_inputs(shapes, dtype)

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    assert res_out.shape == ref_out.shape
    # The op only moves elements, so the result must match bit for bit.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("shapes", FLATTEN_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + utils.BOOL_TYPES)
def test_flatten_dense_tensors_int_bool(shapes, dtype):
    inp = _make_inputs(shapes, dtype)

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("shapes", FLATTEN_ZERO_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_flatten_dense_tensors_zero_sized(shapes, dtype):
    inp = _make_inputs(shapes, dtype)

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_flatten_dense_tensors_element_order(count):
    # Disjoint integer ramps per input: swapping two inputs, or copying them in
    # storage order instead of list order, changes the result.
    inp = [
        torch.arange(i * 100, i * 100 + 12, dtype=torch.int32, device=flag_gems.device)
        for i in range(count)
    ]

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    utils.gems_assert_equal(res_out, ref_out)


def _strided_inputs(dtype):
    base = torch.randn((8, 10, 12), dtype=dtype, device=flag_gems.device)
    return [base[::2, :, ::3], base[::2, :, 1::3]]


def _transposed_inputs(dtype):
    base = torch.randn((8, 12), dtype=dtype, device=flag_gems.device)
    return [base.T, base.T]


def _expanded_inputs(dtype):
    base = torch.randn((1, 9), dtype=dtype, device=flag_gems.device)
    return [base.expand(6, 9), base.expand(6, 9)]


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize(
    "make_inputs", [_strided_inputs, _transposed_inputs, _expanded_inputs]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_flatten_dense_tensors_non_contiguous(make_inputs, dtype):
    # The copy kernels address each input by flat linear index, so any view
    # whose storage order differs from its logical order has to be walked
    # through its strides; otherwise the elements come out permuted.
    inp = make_inputs(dtype)
    assert not inp[0].is_contiguous()

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_flatten_dense_tensors_single_non_contiguous(dtype):
    base = torch.randn((8, 12), dtype=dtype, device=flag_gems.device)
    inp = [base.T]
    assert not inp[0].is_contiguous()

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_flatten_dense_tensors_nan_inf(dtype):
    values = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5]
    inp = [
        torch.tensor(values, dtype=dtype, device=flag_gems.device),
        torch.tensor(values[::-1], dtype=dtype, device=flag_gems.device),
    ]

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_mixed_dtype_promotion():
    inp = [
        torch.randn((4, 5), dtype=torch.float16, device=flag_gems.device),
        torch.randn((3,), dtype=torch.float32, device=flag_gems.device),
        torch.randn((2,), dtype=torch.bfloat16, device=flag_gems.device),
    ]

    ref_out = _reference(inp)
    res_out = flag_gems.flatten_dense_tensors(inp)

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_rejects_empty_list():
    with pytest.raises(RuntimeError):
        flag_gems.flatten_dense_tensors([])


@pytest.mark.flatten_dense_tensors
def test_flatten_dense_tensors_rejects_non_tensor_element():
    inp = [torch.randn((4,), device=flag_gems.device), 1.0]

    with pytest.raises(TypeError):
        flag_gems.flatten_dense_tensors(inp)

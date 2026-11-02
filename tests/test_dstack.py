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

# Each entry is one tensor list. 0-D and 1-D entries check the atleast_3d
# promotion ((1,1,1) and (1,N,1)), where every input contributes a single depth
# slot and the kernel takes its interleaving path; the 3-D and higher entries
# have depth > 1 so they take the block-copy path instead. Lists with unequal
# depths check the running output offset.
if cfg.QUICK_MODE:
    # One list per code path: 0-D for the promotion, 2-D for interleaving,
    # unequal 3-D depths for the block copy and the output offset.
    DSTACK_SHAPES = [
        [(), ()],
        [(4, 5), (4, 5)],
        [(4, 5, 6), (4, 5, 7)],
    ]
else:
    # Ranks 0 through 5, input counts 1 through 4, and both equal and unequal
    # depths, so every branch of the shape promotion and both kernels are hit.
    DSTACK_SHAPES = [
        [()],
        [(), ()],
        [(), (), (), ()],
        [(3,), (3,)],
        [(1,), (1,)],
        [(256,), (256,)],
        [(4, 5), (4, 5)],
        [(3, 33), (3, 33)],
        [(4, 5, 6), (4, 5, 6)],
        [(4, 5, 6), (4, 5, 7)],
        [(16, 16, 33), (16, 16, 33), (16, 16, 33)],
        [(8, 8, 16, 16), (8, 8, 16, 16)],
        [(13, 3, 64, 5, 2), (13, 3, 96, 5, 2), (13, 3, 32, 5, 2)],
    ]

# dstack concatenates along dim 2, so a mismatch on any other promoted
# dimension must be rejected.
DSTACK_EXCEPTION_SHAPES = [
    [(4, 5, 6), (4, 7, 6)],
    [(4, 5, 6), (8, 5, 6)],
    [(4, 5, 6, 2), (4, 5, 6, 3)],
]

# Zero-element inputs still contribute depth slots, so the output can be empty
# or merely have an empty inner extent.
DSTACK_ZERO_SHAPES = [
    [(0,), (0,)],
    [(2, 0), (2, 0)],
    [(0, 3, 4), (0, 3, 4)],
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


@pytest.mark.dstack
@pytest.mark.parametrize("shapes", DSTACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dstack(shapes, dtype):
    inp = _make_inputs(shapes, dtype)

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    # dstack only moves elements, so the result must match bit for bit.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shapes", DSTACK_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + utils.BOOL_TYPES)
def test_dstack_int_bool(shapes, dtype):
    inp = _make_inputs(shapes, dtype)

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shapes", DSTACK_ZERO_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dstack_zero_sized(shapes, dtype):
    inp = _make_inputs(shapes, dtype)

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("count", [2, 3, 4])
def test_dstack_element_order(count):
    # Disjoint integer ramps per input: swapping two inputs, or interleaving
    # them in the wrong order, changes the result in a way randn could mask.
    inp = [
        torch.arange(i * 100, i * 100 + 12, dtype=torch.int32, device=flag_gems.device)
        for i in range(count)
    ]

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

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


@pytest.mark.dstack
@pytest.mark.parametrize(
    "make_inputs", [_strided_inputs, _transposed_inputs, _expanded_inputs]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dstack_non_contiguous(make_inputs, dtype):
    # Both kernels address their input by flat linear index, so any view whose
    # storage order differs from its logical order has to be materialised by
    # the wrapper; otherwise the elements come out permuted.
    inp = make_inputs(dtype)
    assert not inp[0].is_contiguous()

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dstack_nan_inf(dtype):
    values = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5]
    inp = [
        torch.tensor(values, dtype=dtype, device=flag_gems.device),
        torch.tensor(values[::-1], dtype=dtype, device=flag_gems.device),
    ]

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.dstack
def test_dstack_mixed_dtype_promotion():
    inp = [
        torch.randn((4, 5, 2), dtype=torch.float16, device=flag_gems.device),
        torch.randn((4, 5, 3), dtype=torch.float32, device=flag_gems.device),
    ]

    ref_inp = [utils.to_reference(x) for x in inp]
    ref_out = torch.dstack(ref_inp)
    res_out = flag_gems.dstack(inp)

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.dstack
@pytest.mark.parametrize("shapes", DSTACK_EXCEPTION_SHAPES)
def test_dstack_rejects_mismatched_shapes(shapes):
    inp = _make_inputs(shapes, torch.float32)

    with pytest.raises(RuntimeError):
        flag_gems.dstack(inp)


@pytest.mark.dstack
def test_dstack_rejects_empty_list():
    with pytest.raises(RuntimeError):
        flag_gems.dstack([])


@pytest.mark.dstack
def test_dstack_rejects_non_tensor():
    inp = [torch.randn((4,), device=flag_gems.device), 1.0]

    with pytest.raises(TypeError):
        flag_gems.dstack(inp)

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

# ``diagflat`` flattens its input first, so only the element count matters for
# the kernel; the ranks here just confirm the C-order flatten. Sizes straddle
# the two internal crossovers: the fused tile path (m * m <= 2 ** 21) and the
# separate fill + scatter path above it.
DIAGFLAT_SHAPES = [
    (),
    (0,),
    (1,),
    (64,),
    (1024,),
    (4096,),
    (32, 32),
    (16, 16, 16),
]

# 0 exercises the main diagonal, +/-3 the short off-diagonals, and +/-1024 an
# offset large enough to dominate the output extent (m = n + |offset|).
DIAGFLAT_OFFSETS = [0, 3, -3, 1024, -1024]


def _make_input(shape, dtype):
    if dtype in utils.FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    return torch.randint(0, 0x7FFF, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", DIAGFLAT_SHAPES)
@pytest.mark.parametrize("offset", DIAGFLAT_OFFSETS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagflat(shape, offset, dtype):
    inp = _make_input(shape, dtype)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    # diagflat only moves elements and writes exact zeros, so the result must
    # match bit for bit rather than within a tolerance.
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("shape", DIAGFLAT_SHAPES)
@pytest.mark.parametrize("offset", [0, 3, -3])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + utils.BOOL_TYPES)
def test_diagflat_int_bool(shape, offset, dtype):
    inp = _make_input(shape, dtype)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [0, 2, -2])
def test_diagflat_element_order(offset):
    # A strictly increasing int32 ramp: any transposition of the diagonal or a
    # wrong flatten order changes the result, which randn inputs could mask.
    inp = torch.arange(1, 25, dtype=torch.int32, device=flag_gems.device).reshape(
        2, 3, 4
    )

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [0, 3, -3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagflat_non_contiguous(offset, dtype):
    # The kernel reads the flattened input through its element stride, so a
    # zero-copy strided view must give the same answer as a contiguous copy.
    base = torch.randn((48,), dtype=dtype, device=flag_gems.device)
    inp = base[::2]
    assert not inp.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [0, 3, -3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagflat_transposed(offset, dtype):
    # A transposed 2-D view flattens in a different memory order than its
    # storage, which checks the reshape rather than the stride arithmetic.
    inp = torch.randn((8, 12), dtype=dtype, device=flag_gems.device).T
    assert not inp.is_contiguous()

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, offset)
    res_out = flag_gems.diagflat(inp, offset)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagflat_nan_inf(dtype):
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, 1)
    res_out = flag_gems.diagflat(inp, 1)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.diagflat
def test_diagflat_default_offset():
    inp = torch.randn((128,), device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp)
    res_out = flag_gems.diagflat(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [0, 3, -3])
def test_diagflat_backward(offset):
    inp = torch.randn((64,), device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach(), False).requires_grad_()

    m = 64 + abs(offset)
    weight = torch.arange(m * m, device=flag_gems.device, dtype=torch.float32).reshape(
        m, m
    )
    ref_weight = utils.to_reference(weight, False)

    (torch.diagflat(ref_inp, offset) * ref_weight).sum().backward()
    (flag_gems.diagflat(inp, offset) * weight).sum().backward()

    utils.gems_assert_equal(inp.grad, ref_inp.grad)


@pytest.mark.diagflat
@pytest.mark.parametrize("offset", [0, 3])
def test_diagflat_large_output_is_deterministic(offset):
    # Above the fused-tile crossover a single kernel both zeroes a span and
    # writes the diagonal cells inside it. The two stores come from different
    # lanes of the same block, so without a barrier a few diagonal cells are
    # nondeterministically left at zero. One pass hits only a handful of the
    # 16 M cells, hence the repeats.
    inp = torch.randn((4096,), device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diagflat(ref_inp, offset)

    for _ in range(8):
        utils.gems_assert_equal(flag_gems.diagflat(inp, offset), ref_out)


@pytest.mark.diagflat
def test_diagflat_rejects_non_int_offset():
    inp = torch.randn((8,), device=flag_gems.device)

    with pytest.raises(TypeError):
        flag_gems.diagflat(inp, 1.5)

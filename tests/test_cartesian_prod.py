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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Lengths of the 1-D input vectors. Covers the single-tensor pass-through, the
# zero-length cases (which make the output empty) and up to four tensors.
CARTESIAN_PROD_SIZES = [
    (8,),
    (0,),
    (1, 7),
    (3, 5),
    (16, 16),
    (5, 0),
    (0, 3),
    (3, 1, 3),
    (2, 4, 3),
    (8, 16, 32),
    (2, 5, 8, 3),
]

CARTESIAN_PROD_BACKWARD_SIZES = [(8,), (3, 5), (2, 4, 3), (16, 16)]


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", CARTESIAN_PROD_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cartesian_prod(sizes, dtype):
    inputs = [torch.randn((s,), dtype=dtype, device=flag_gems.device) for s in sizes]
    ref_inputs = [utils.to_reference(inp) for inp in inputs]

    ref_out = torch.cartesian_prod(*ref_inputs)
    res_out = flag_gems.cartesian_prod(*inputs)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", CARTESIAN_PROD_SIZES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_cartesian_prod_int_bool(sizes, dtype):
    inputs = [
        torch.randint(0, 2, (s,), dtype=dtype, device=flag_gems.device) for s in sizes
    ]
    ref_inputs = [utils.to_reference(inp) for inp in inputs]

    ref_out = torch.cartesian_prod(*ref_inputs)
    res_out = flag_gems.cartesian_prod(*inputs)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cartesian_prod_sequence_form(dtype):
    inputs = [torch.randn((s,), dtype=dtype, device=flag_gems.device) for s in (3, 5)]
    ref_inputs = [utils.to_reference(inp) for inp in inputs]

    ref_out = torch.cartesian_prod(*ref_inputs)
    # aten::cartesian_prod takes a TensorList, so the list form must work too.
    res_out = flag_gems.cartesian_prod(inputs)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cartesian_prod_non_contiguous(dtype):
    # Stride-2 slices of a larger buffer: the gather has to honour the input
    # strides instead of reading the underlying storage linearly.
    base = [torch.randn((2 * s,), dtype=dtype, device=flag_gems.device) for s in (3, 5)]
    inputs = [b[::2] for b in base]
    assert not any(inp.is_contiguous() for inp in inputs)
    ref_inputs = [utils.to_reference(inp) for inp in inputs]

    ref_out = torch.cartesian_prod(*ref_inputs)
    res_out = flag_gems.cartesian_prod(*inputs)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_cartesian_prod_row_order(dtype):
    # The last tensor varies fastest (row-major), so the exact row order is
    # part of the contract, not just the set of combinations.
    a = torch.tensor([0, 1], dtype=dtype, device=flag_gems.device)
    b = torch.tensor([10, 20, 30], dtype=dtype, device=flag_gems.device)

    ref_out = torch.cartesian_prod(utils.to_reference(a), utils.to_reference(b))
    res_out = flag_gems.cartesian_prod(a, b)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("sizes", CARTESIAN_PROD_BACKWARD_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cartesian_prod_backward(sizes, dtype):
    inputs = [
        torch.randn((s,), dtype=dtype, device=flag_gems.device, requires_grad=True)
        for s in sizes
    ]
    ref_inputs = [utils.to_reference(inp) for inp in inputs]

    ref_out = torch.cartesian_prod(*ref_inputs)
    res_out = flag_gems.cartesian_prod(*inputs)

    out_grad = torch.randn_like(res_out)
    ref_out_grad = utils.to_reference(out_grad)

    ref_in_grads = torch.autograd.grad(ref_out, ref_inputs, ref_out_grad)
    res_in_grads = torch.autograd.grad(res_out, inputs, out_grad)

    # Every element of input k collects rows / sizes[k] gradient terms, so the
    # tolerance has to scale with the length of that accumulation.
    rows = math.prod(sizes)
    for res_grad, ref_grad, size in zip(res_in_grads, ref_in_grads, sizes):
        utils.gems_assert_close(res_grad, ref_grad, dtype, reduce_dim=rows // size)


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_cartesian_prod_nan_inf(dtype):
    inputs = [torch.randn((s,), dtype=dtype, device=flag_gems.device) for s in (3, 5)]
    inputs[0][0] = float("nan")
    inputs[0][1] = float("inf")
    inputs[1][0] = float("-inf")
    ref_inputs = [utils.to_reference(inp) for inp in inputs]

    ref_out = torch.cartesian_prod(*ref_inputs)
    res_out = flag_gems.cartesian_prod(*inputs)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_empty_list():
    with pytest.raises(RuntimeError):
        flag_gems.cartesian_prod()


@pytest.mark.cartesian_prod
@pytest.mark.parametrize("shape", [(), (3, 4)])
def test_cartesian_prod_rejects_multidim_input(shape):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.cartesian_prod(inp, inp)


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_mixed_dtype():
    a = torch.randn((3,), dtype=torch.float32, device=flag_gems.device)
    b = torch.randn((5,), dtype=torch.float16, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.cartesian_prod(a, b)


@pytest.mark.cartesian_prod
def test_cartesian_prod_rejects_non_tensor():
    a = torch.randn((3,), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems.cartesian_prod(a, 1.0)

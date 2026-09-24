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


import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
INT_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.int64]
FLOAT_DTYPES = [torch.float32, torch.bfloat16, torch.float16] + (
    [torch.float64] if utils.fp64_is_supported else []
)

COMPLEX_DTYPES = [torch.complex64] + (
    [torch.complex128] if utils.fp64_is_supported else []
)
VIEW_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES + INT_DTYPES + [torch.bool] + FP8_DTYPES
ARITH_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES + INT_DTYPES + [torch.bool]
CONTRACT_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES

VIEW_ROWS = tu.selected_cases(
    [("...->...", (shape,)) for shape in tu.selected_shapes()]
    + [("...ij->...ji", (shape,)) for shape in tu.selected_shapes() if len(shape) >= 2]
    + [("ii->i", ((1024, 1024),)), ("...ii->...i", ((16, 128, 60, 60),))],
    quick=[("...->...", ((2, 19, 7),)), ("...ij->...ji", ((2, 19, 7),))],
)

MULTIPLY_ROWS = tu.selected_cases(
    [
        ("ij,ij->ij", ((1024, 1024), (1024, 1024))),
        ("i,j->ij", ((256,), (256,))),
    ],
    quick=[("ijk,ijk->ijk", ((2, 19, 7), (2, 19, 7)))],
)

REDUCE_ROWS = tu.selected_cases(
    [
        ("i->", ((256,),)),
        ("ii->", ((1024, 1024),)),
        ("ijk->", ((20, 320, 15),)),
        ("ijk->j", ((20, 320, 15),)),
        ("abcd->ac", ((16, 128, 64, 60),)),
    ],
    quick=[("ijk->j", ((2, 19, 7),))],
)

CONTRACT_ROWS = tu.selected_cases(
    [
        ("i,i->", ((256,), (256,))),
        ("ij,jk->ik", ((512, 512), (512, 512))),
        ("bij,bjk->bik", ((16, 128, 64), (16, 64, 60))),
        ("...ij,...jk->...ik", ((2, 4, 64, 64), (2, 4, 64, 128))),
    ],
    quick=[("bij,bjk->bik", ((2, 19, 7), (2, 7, 19)))],
)

# Preserve the original suites' shapes when migrating their value generation.
VIEW_ROWS += tu.selected_cases(
    [("ii->i", ((n, n),)) for n in (32, 64, 128)]
    + [("ij->ji", (shape,)) for shape in ((16, 32), (32, 64))]
    + [("ijk->kji", ((64, 128, 256),))],
    quick=[],
)
MULTIPLY_ROWS += tu.selected_cases(
    [("i,j->ij", ((m,), (n,))) for m, n in ((16, 32), (32, 64), (128, 256))]
    + [("...,...->...", (shape, shape)) for shape in tu.selected_shapes()],
    quick=[],
)
REDUCE_ROWS += tu.selected_cases(
    [("ii->", ((n, n),)) for n in (32, 64, 128)]
    + [
        (equation, (shape,))
        for equation in ("ijk->", "ijk->j")
        for shape in ((16, 32, 64), (32, 64, 128))
    ]
    + [("...->", (shape,)) for shape in tu.selected_shapes()],
    quick=[],
)
CONTRACT_ROWS += tu.selected_cases(
    [
        ("ij,jk->ik", ((m, k), (k, n)))
        for m, k, n in ((16, 32, 64), (32, 64, 128), (16, 256, 32))
    ]
    + [
        ("bij,bjk->bik", ((b, m, k), (b, k, n)))
        for b, m, k, n in ((2, 16, 32, 64), (4, 32, 64, 128), (8, 16, 256, 32))
    ]
    + [("i,i->", ((n,), (n,))) for n in (64, 1024)]
    + [
        ("...ij,...jk->...ik", ((2, 3, 32, 64), (2, 3, 64, 128))),
        ("ij,jk->ik", ((2, 0), (0, 3))),
        ("ij,jk,kl->il", ((32, 16), (16, 8), (8, 4))),
    ]
    + [("...,...->", (shape, shape)) for shape in tu.selected_shapes()],
    quick=[],
)

VIEW_CASES = [
    (row, dtype, value_range)
    for row in VIEW_ROWS
    for dtype in VIEW_DTYPES
    for value_range in tu.selected_ranges()
]
MULTIPLY_CASES = [
    (row, dtype, value_range)
    for row in MULTIPLY_ROWS
    for dtype in ARITH_DTYPES
    for value_range in tu.selected_ranges()
]
REDUCE_CASES = [
    (row, dtype, value_range)
    for row in REDUCE_ROWS
    for dtype in ARITH_DTYPES
    for value_range in tu.selected_ranges()
]
CONTRACT_CASES = [
    (row, dtype, value_range)
    for row in CONTRACT_ROWS
    for dtype in CONTRACT_DTYPES
    for value_range in tu.selected_ranges()
]

BROADCAST_CASES = tu.selected_cases(
    [
        ("...a,a->...a", ((1024, 1024), (1024,)), torch.float32),
        ("a,...a->...a", ((1024,), (1024, 1024)), torch.int32),
        ("...a,...a->...a", ((20, 320, 15), (20, 1, 15)), torch.bfloat16),
        ("...a,...a->...a", ((20, 320, 15), (1, 320, 15)), torch.int64),
    ],
    quick=[],
)

BACKWARD_CASES = tu.selected_cases(
    [
        (case, dtype)
        for case in [
            ("ij,jk->ik", ((256, 256), (256, 256))),
            ("...a,a->...a", ((20, 320, 15), (15,))),
            ("i,j->ij", ((256,), (256,))),
            ("...ii->...i", ((16, 128, 60, 60),)),
        ]
        for dtype in CONTRACT_DTYPES
    ]
    + [(("ij->ji", ((3, 5),)), dtype) for dtype in FP8_DTYPES],
    quick=[],
)

SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(FLOAT_DTYPES), quick=[])
FP8_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(FP8_DTYPES), quick=[])

VIEW_STATE_CASES = tu.selected_cases(
    [
        (torch.float32, "strided"),
        (torch.complex128, "strided"),
        (torch.complex128, "conj_offset"),
    ],
    quick=[],
)

PATH_CASES = tu.selected_cases([[0, 1], [1, 0], None], quick=[])

NEGATIVE_CASES = [
    ("i->", ((1,), (1,))),
    ("ij,jk->ik", ((1, 1),)),
    ("ij->ii", ((4, 4),)),
    ("ij->jk", ((4, 4),)),
    ("1i->i", ((4, 4),)),
    ("ii->i", ((3, 4),)),
    ("i->", ()),
    ("ij,jk->ik", ((2, 3), (4, 5))),
]

ARITHMETIC_CASES = MULTIPLY_CASES + REDUCE_CASES + CONTRACT_CASES
ARITHMETIC_CASES += [
    ((equation, shapes), dtype, ["-1", "1"])
    for equation, shapes, dtype in BROADCAST_CASES
]


@pytest.mark.einsum
@pytest.mark.parametrize("case,dtype,value_range", VIEW_CASES)
def test_einsum_view(case, dtype, value_range):
    equation, operand_shapes = case
    operands = [tu.make_input(dtype, shape, value_range) for shape in operand_shapes]
    ref_operands = [tu.to_reference(operand) for operand in operands]

    ref_out = torch.ops.aten.einsum(equation, ref_operands)
    res_out = flag_gems.einsum(equation, operands)

    tu.assert_result_equal(res_out, ref_out)

    assert (
        res_out.untyped_storage().data_ptr() == operands[0].untyped_storage().data_ptr()
    )
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()


@pytest.mark.einsum
@pytest.mark.parametrize("case,dtype,value_range", ARITHMETIC_CASES)
def test_einsum_arithmetic(case, dtype, value_range):
    equation, operand_shapes = case
    operands = [tu.make_input(dtype, shape, value_range) for shape in operand_shapes]
    ref_operands = [tu.to_reference(operand) for operand in operands]

    ref_out = torch.ops.aten.einsum(equation, ref_operands)
    res_out = flag_gems.einsum(equation, operands)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.einsum
@pytest.mark.parametrize("case,dtype", BACKWARD_CASES)
def test_einsum_backward(case, dtype):
    equation, operand_shapes = case
    operands = [
        tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
        for shape in operand_shapes
    ]
    ref_operands = [tu.to_reference(operand) for operand in operands]

    ref_out = torch.ops.aten.einsum(equation, ref_operands)
    res_out = flag_gems.einsum(equation, operands)

    ref_grads = torch.autograd.grad(ref_out, ref_operands, torch.ones_like(ref_out))
    res_grads = torch.autograd.grad(res_out, operands, torch.ones_like(res_out))

    tu.assert_result_close(res_out, ref_out)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.einsum
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_einsum_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    ref_payload = tu.to_reference(payload)

    ref_out = torch.ops.aten.einsum("i,i->i", [ref_payload, ref_payload])
    res_out = flag_gems.einsum("i,i->i", [payload, payload])

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.einsum
@pytest.mark.parametrize("dtype,scenario", FP8_SPECIAL_CASES)
def test_einsum_fp8_special_values(dtype, scenario):
    payload = tu.make_special_input(dtype, scenario)
    ref_payload = tu.to_reference(payload)

    ref_out = torch.ops.aten.einsum("...->...", [ref_payload])
    res_out = flag_gems.einsum("...->...", [payload])

    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.einsum
@pytest.mark.parametrize("dtype,state", VIEW_STATE_CASES)
def test_einsum_view_state(dtype, state):
    base = tu.make_input(dtype, (4, 64, 64), ["-1", "1"])
    if state == "conj_offset":
        operand = base[1:3, 1:33:2, 2:].conj()
    else:
        operand = base[:, 0:64:2, :]

    ref_operand = tu.to_reference(operand)
    ref_out = torch.ops.aten.einsum("...ij->...ji", [ref_operand])
    res_out = flag_gems.einsum("...ij->...ji", [operand])

    tu.assert_result_equal(res_out, ref_out)

    assert res_out.untyped_storage().data_ptr() == operand.untyped_storage().data_ptr()
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.is_conj() == ref_out.is_conj()


@pytest.mark.einsum
@pytest.mark.parametrize("path", PATH_CASES)
def test_einsum_contraction_path(path):
    operands = [tu.make_input(torch.float32, (256, 256), ["-1", "1"]) for _ in range(2)]
    ref_operands = [tu.to_reference(operand) for operand in operands]

    ref_out = torch.ops.aten.einsum("ij,jk->ik", ref_operands, path=path)
    res_out = flag_gems.einsum("ij,jk->ik", operands, path=path)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.einsum
@pytest.mark.parametrize("equation,operand_shapes", NEGATIVE_CASES)
def test_einsum_invalid_equation(equation, operand_shapes):
    operands = [
        tu.make_input(torch.float32, shape, ["-1", "1"]) for shape in operand_shapes
    ]

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.einsum(equation, operands)


@pytest.mark.einsum
@pytest.mark.parametrize("dtype", INT_DTYPES + [torch.bool] + FP8_DTYPES)
def test_einsum_unsupported_contraction_dtype(dtype):
    operands = [tu.make_input(dtype, shape, ["-1", "1"]) for shape in ((2, 3), (3, 4))]
    with pytest.raises(RuntimeError):
        flag_gems.einsum("ij,jk->ik", operands)


@pytest.mark.einsum
@pytest.mark.parametrize("path", [[], [0]])
def test_einsum_invalid_path(path):
    operands = [tu.make_input(torch.float32, (3, 3), ["-1", "1"]) for _ in range(2)]
    with pytest.raises(RuntimeError):
        flag_gems.einsum("ij,jk->ik", operands, path=path)

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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    SHAPE_DIM_CASES = [((2, 32), [1])]
    ORD_LIST = [2]
    KEEP_DIM = [False]
    DTYPES = [torch.float32]
else:
    SHAPE_DIM_CASES = [
        ((37,), None),
        ((17, 33), [1]),
        ((5, 7, 11), [-1]),
        ((4, 9, 13), [0, 2]),
    ]
    ORD_LIST = [0, 1, 2, 3, 0.5, -1]
    KEEP_DIM = [False, True]
    DTYPES = utils.ALL_FLOAT_DTYPES


def _reduction_size(shape, dim):
    if dim is None or dim == []:
        return math.prod(shape)
    dims = dim if isinstance(dim, (list, tuple)) else [dim]
    return math.prod(shape[d % len(shape)] for d in dims)


@pytest.mark.linalg__powsum
@pytest.mark.parametrize(("shape", "dim"), SHAPE_DIM_CASES)
@pytest.mark.parametrize("ord", ORD_LIST)
@pytest.mark.parametrize("keepdim", KEEP_DIM)
@pytest.mark.parametrize("dtype", DTYPES)
def test_linalg__powsum(shape, dim, ord, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)
    ref = torch.ops.aten.linalg__powsum.default(ref_inp, ord, dim, keepdim)

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(inp, ord, dim, keepdim)

    utils.gems_assert_close(
        result,
        ref,
        dtype=dtype,
        reduce_dim=_reduction_size(shape, dim),
    )


@pytest.mark.linalg__powsum
def test_linalg__powsum_dtype_and_public_api():
    inp = torch.randn((11, 19), dtype=torch.float16, device=flag_gems.device)
    ref_inp = utils.to_reference(inp).to(torch.float32)
    ref = torch.ops.aten.linalg__powsum.default(
        ref_inp, 2, [1], True, dtype=torch.float32
    )

    result = flag_gems.linalg__powsum(inp, 2, [1], True, dtype=torch.float32)

    assert result.dtype == torch.float32
    utils.gems_assert_close(result, ref, dtype=torch.float32, reduce_dim=19)


@pytest.mark.linalg__powsum
def test_linalg__powsum_integer_input_with_float_dtype():
    inp = torch.arange(-16, 16, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.ops.aten.linalg__powsum.default(
        ref_inp, 2, None, False, dtype=torch.float32
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(
            inp, 2, None, False, dtype=torch.float32
        )

    utils.gems_assert_equal(result, ref)


@pytest.mark.linalg__powsum
@pytest.mark.skipif(
    cfg.TO_CPU,
    reason="native CPU and CUDA differ for complex input with a real dtype",
)
@pytest.mark.parametrize(
    ("input_dtype", "requested_dtype"),
    [
        (torch.complex64, None),
        (torch.complex128, None),
        (torch.complex64, torch.complex128),
        (torch.complex128, torch.float32),
    ],
)
def test_linalg__powsum_complex(input_dtype, requested_dtype):
    inp = torch.randn((5, 7), dtype=input_dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.ops.aten.linalg__powsum.default(
        ref_inp, 3, [1], True, dtype=requested_dtype
    )

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(
            inp, 3, [1], True, dtype=requested_dtype
        )

    utils.gems_assert_close(result, ref, dtype=ref.dtype, reduce_dim=7)


@pytest.mark.linalg__powsum
@pytest.mark.parametrize("keepdim", [False, True])
def test_linalg__powsum_empty(keepdim):
    inp = torch.empty((3, 0, 5), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.ops.aten.linalg__powsum.default(ref_inp, -1, [1], keepdim)

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(inp, -1, [1], keepdim)

    utils.gems_assert_equal(result, ref)


@pytest.mark.linalg__powsum
def test_linalg__powsum_noncontiguous():
    inp = torch.randn((7, 5, 9), device=flag_gems.device).transpose(0, 2)
    ref_inp = utils.to_reference(inp)
    ref = torch.ops.aten.linalg__powsum.default(ref_inp, 3, [0, 2], False)

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(inp, 3, [0, 2], False)

    utils.gems_assert_close(result, ref, dtype=inp.dtype, reduce_dim=63)


@pytest.mark.linalg__powsum
@pytest.mark.parametrize("dim", [None, [], [0], [-1]])
def test_linalg__powsum_scalar(dim):
    inp = torch.tensor(-3.0, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref = torch.ops.aten.linalg__powsum.default(ref_inp, 2, dim, True)

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(inp, 2, dim, True)

    utils.gems_assert_equal(result, ref)


@pytest.mark.linalg__powsum
@pytest.mark.parametrize("ord", [0, 1, 2, -1, float("inf"), -float("inf")])
def test_linalg__powsum_special_values(ord):
    inp = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("nan"), float("inf"), -float("inf")],
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)
    ref = torch.ops.aten.linalg__powsum.default(ref_inp, ord)

    with flag_gems.use_gems():
        result = torch.ops.aten.linalg__powsum.default(inp, ord)

    utils.gems_assert_equal(result, ref, equal_nan=True)


@pytest.mark.linalg__powsum
@pytest.mark.parametrize("dim", [[3], [1, 1]])
def test_linalg__powsum_invalid_dim(dim):
    inp = torch.randn((2, 3, 4), device=flag_gems.device)

    with flag_gems.use_gems(), pytest.raises((IndexError, RuntimeError)):
        torch.ops.aten.linalg__powsum.default(inp, 2, dim)

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

"""Accuracy tests for ``aten::chain_matmul(Tensor[] matrices) -> Tensor``.

``chain_matmul`` multiplies a chain of 2-D matrices with an optimal
parenthesization, matching ``torch.linalg.multi_dot``. The implementation has
four paths, all covered below:

* generic Triton matmul kernels for every intermediate product,
* a fused single-program kernel for tiny 3-matrix chains (all dims <= 16),
* a fused last-two-gemm kernel for fp16/bf16 4-matrix ``A@(B@(C@D))`` chains,
* a single-matrix no-op.

Both contiguous and transposed (column-major) operands are exercised because
the kernels are stride-generic. The reference is the native operator in the
same dtype on the same device, as in ``test_linalg_multi_dot.py``; the
low-precision intermediate rounding is part of the operator's contract, so
comparing against an upcast reference would measure a different computation.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# (rows, cols) for each matrix in the chain. The 3-matrix tiny case triggers
# the fused _chain3 kernel; the 4-matrix 64x128... case has the
# A@(B@(C@D)) plan and triggers the fused last-two-gemm kernel.
CHAIN_SHAPES = [
    [(64, 32), (32, 128)],
    [(64, 32), (32, 128), (128, 16)],
    [(4, 5), (5, 6), (6, 7)],
    [(64, 128), (128, 256), (256, 128), (128, 16)],
    [(64, 32), (32, 128), (128, 64), (64, 32)],
    [(64, 32), (32, 128), (128, 64), (64, 32), (32, 16)],
]
if QUICK_MODE:
    CHAIN_SHAPES = CHAIN_SHAPES[:3]


def _make_chain(shapes, dtype):
    return [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for shape in shapes
    ]


def _reference(mats):
    """Native chain product in the input dtype; TF32 disabled for fp32."""
    ref_mats = [utils.to_reference(m) for m in mats]
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.ops.aten.chain_matmul(ref_mats)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _reduce_dim(shapes):
    return max([shapes[0][0]] + [shape[1] for shape in shapes])


def _atol(dtype):
    # The fp64 path is documented as an fp32 hi/lo-split emulation, so it is
    # graded at the wider atol the implementation targets (5e-4*K) rather than
    # fp64-exact precision.
    return 5e-4 if dtype == torch.float64 else 1e-4


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", CHAIN_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul(shapes, dtype):
    mats = _make_chain(shapes, dtype)
    ref_out = _reference(mats)

    res_out = flag_gems.chain_matmul(mats)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=_reduce_dim(shapes),
        atol=_atol(dtype),
    )


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul_single_matrix(dtype):
    mat = torch.randn((7, 11), dtype=dtype, device=flag_gems.device)

    res_out = flag_gems.chain_matmul([mat])
    ref_out = _reference([mat])

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=11, atol=_atol(dtype))


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul_transposed_inputs(dtype):
    # Column-major operands: matrix i is stored as (cols, rows) and transposed.
    shapes = [(64, 128), (128, 32), (32, 64)]
    base = [
        torch.randn((k, m), dtype=dtype, device=flag_gems.device) for m, k in shapes
    ]
    mats = [t.t() for t in base]
    assert not mats[0].is_contiguous()

    ref_out = _reference(mats)
    res_out = flag_gems.chain_matmul(mats)

    utils.gems_assert_close(
        res_out,
        ref_out,
        dtype,
        reduce_dim=_reduce_dim(shapes),
        atol=_atol(dtype),
    )


@pytest.mark.chain_matmul
def test_chain_matmul_empty_list():
    # The ATen reference raises for an empty matrix list; the implementation
    # rejects it before touching any kernel.
    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.chain_matmul([])

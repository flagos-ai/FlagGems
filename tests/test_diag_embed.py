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

import itertools
import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


def get_dim1_dim2(o_rank):
    dims = list(range(-o_rank, o_rank))
    return [
        p for p in itertools.permutations(dims, 2) if (p[0] % o_rank) != (p[1] % o_rank)
    ]


def get_diag_embed_shape_and_dims():
    if cfg.QUICK_MODE:
        shapes = [(1024,)]
    else:
        shapes = [
            (1024,),
            (1024, 1024),
        ]
    # [(shape, dim1, dim2)]
    result = []

    for s in shapes:
        dim_pairs = get_dim1_dim2(len(s) + 1)
        if dim_pairs:
            dim1, dim2 = random.choice(dim_pairs)
            result.append((s, dim1, dim2))

    return result


@pytest.mark.diag_embed
@pytest.mark.parametrize("shape, dim1, dim2", get_diag_embed_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_accuracy_diag_embed(shape, dtype, offset, dim1, dim2):
    if dtype in utils.FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    elif dtype in utils.INT_DTYPES:
        inp = torch.randint(
            low=0, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        inp = torch.randint(low=0, high=2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    ref_inp = utils.to_reference(inp)
    ref_out = torch.diag_embed(ref_inp, offset, dim1, dim2)
    res_out = flag_gems.diag_embed(inp, offset, dim1, dim2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.diag_embed_backward
@pytest.mark.parametrize("shape, dim1, dim2", get_diag_embed_shape_and_dims())
@pytest.mark.parametrize("offset", [-1, 0, 1])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diag_embed_backward(shape, dtype, offset, dim1, dim2):
    """Gradient of ``diag_embed``: the diagonal of the upstream gradient.

    ATen's ``DiagEmbedBackward0`` returns a ``torch.diagonal`` view of the
    upstream gradient at (``offset``, ``dim1``, ``dim2``); the FlagGems kernel
    gathers the same elements into a contiguous tensor and is bit-exact.
    """
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out_shape = torch.diag_embed(torch.empty_like(inp), offset, dim1, dim2).shape
    grad_out = torch.randn(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_out, True)
    ref_out = torch.diag_embed(ref_inp, offset, dim1, dim2)
    (ref_grad_in,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    ref_grad_in = ref_grad_in.to(dtype)

    res_grad = flag_gems.diag_embed_backward(grad_out, inp, offset, dim1, dim2)

    utils.gems_assert_equal(res_grad, ref_grad_in)


@pytest.mark.diag_embed_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diag_embed_backward_default_dims(dtype):
    """The default (offset, -2, -1) fast path through the Triton kernel."""
    inp = torch.randn((32, 48), dtype=dtype, device=flag_gems.device)
    grad_out = torch.randn((32, 48, 48), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_out, True)
    (ref_grad_in,) = torch.autograd.grad(torch.diag_embed(ref_inp), ref_inp, ref_grad)
    ref_grad_in = ref_grad_in.to(dtype)

    res_grad = flag_gems.diag_embed_backward(grad_out, inp)

    utils.gems_assert_equal(res_grad, ref_grad_in)


@pytest.mark.diag_embed_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diag_embed_backward_non_contiguous(dtype):
    """A transposed upstream gradient (non-contiguous grad_output)."""
    base = torch.randn((64, 64), dtype=dtype, device=flag_gems.device)
    grad_out = base.t()

    inp = torch.randn((64,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_out, True)
    (ref_grad_in,) = torch.autograd.grad(torch.diag_embed(ref_inp), ref_inp, ref_grad)
    ref_grad_in = ref_grad_in.to(dtype)

    res_grad = flag_gems.diag_embed_backward(grad_out, inp)

    utils.gems_assert_equal(res_grad, ref_grad_in)

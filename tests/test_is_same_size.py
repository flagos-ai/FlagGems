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

# 1D / 2D / 3D shape pairs. Each entry pairs a shape with itself (same size) and
# with a differently shaped tensor (not the same size), so both branches of the
# boolean result are covered at every rank.
SAME_SIZE_SHAPE_PAIRS = [
    ((8,), (8,)),
    ((8,), (16,)),
    ((4, 8), (4, 8)),
    ((4, 8), (8, 4)),
    ((2, 3, 4), (2, 3, 4)),
    ((2, 3, 4), (2, 4, 3)),
    # Differing ranks must also compare as not the same size.
    ((4, 8), (4, 8, 1)),
]


@pytest.mark.is_same_size
@pytest.mark.parametrize("shape_a,shape_b", SAME_SIZE_SHAPE_PAIRS)
@pytest.mark.parametrize(
    "dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES + utils.BOOL_TYPES
)
def test_is_same_size(shape_a, shape_b, dtype, caplog):
    if dtype in utils.FLOAT_DTYPES:
        inp_a = torch.randn(shape_a, dtype=dtype, device=flag_gems.device)
        inp_b = torch.randn(shape_b, dtype=dtype, device=flag_gems.device)
    else:
        inp_a = torch.ones(shape_a, dtype=dtype, device=flag_gems.device)
        inp_b = torch.ones(shape_b, dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(inp_a)
    ref_b = utils.to_reference(inp_b)

    ref_out = torch.ops.aten.is_same_size(ref_a, ref_b)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_same_size"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_same_size(inp_a, inp_b)

    assert "GEMS IS_SAME_SIZE" in caplog.text
    assert res_out == ref_out, f"Expected {ref_out}, got {res_out}"
    assert res_out == (shape_a == shape_b)


@pytest.mark.is_same_size
def test_is_same_size_non_contiguous(caplog):
    # is_same_size compares logical shapes, so a transposed view of a square
    # tensor is still the same size as the original despite different strides.
    inp = torch.randn(4, 4, device=flag_gems.device)
    inp_t = inp.t()
    ref_inp = utils.to_reference(inp)
    ref_inp_t = ref_inp.t()

    ref_out = torch.ops.aten.is_same_size(ref_inp, ref_inp_t)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_same_size"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_same_size(inp, inp_t)

    assert "GEMS IS_SAME_SIZE" in caplog.text
    assert res_out == ref_out
    assert res_out is True


@pytest.mark.is_same_size
def test_is_same_size_scalar_tensor(caplog):
    # Zero-dimensional tensors have an empty shape and must compare equal to
    # each other but not to a one-element 1D tensor.
    inp_a = torch.tensor(1.0, device=flag_gems.device)
    inp_b = torch.tensor(2.0, device=flag_gems.device)
    inp_c = torch.tensor([1.0], device=flag_gems.device)
    ref_a = utils.to_reference(inp_a)
    ref_b = utils.to_reference(inp_b)
    ref_c = utils.to_reference(inp_c)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_same_size"):
        with flag_gems.use_gems():
            res_same = torch.ops.aten.is_same_size(inp_a, inp_b)
            res_diff = torch.ops.aten.is_same_size(inp_a, inp_c)

    assert "GEMS IS_SAME_SIZE" in caplog.text
    assert res_same == torch.ops.aten.is_same_size(ref_a, ref_b)
    assert res_diff == torch.ops.aten.is_same_size(ref_a, ref_c)
    assert res_same is True
    assert res_diff is False

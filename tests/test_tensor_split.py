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
from . import conftest as cfg

# Define shapes for tensor_split tests
if cfg.QUICK_MODE:
    TENSOR_SPLIT_SHAPES = [
        (64,),
        (64, 128),
    ]
else:
    TENSOR_SPLIT_SHAPES = [
        (64,),
        (64, 128),
        (8, 16, 32),
        (4, 8, 16, 32),
    ]


@pytest.mark.tensor_split_sections
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_by_int(shape, dtype):
    """Test tensor_split with integer sections."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test splitting into 3 sections
    sections = 3
    ref_out = torch.tensor_split(ref_inp, sections, dim=0)

    res_out = flag_gems.tensor_split(inp, sections, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split_indices
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_by_list(shape, dtype):
    """Test tensor_split with list of indices."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test splitting at specific indices
    indices = [shape[0] // 3, shape[0] * 2 // 3]
    ref_out = torch.tensor_split(ref_inp, indices, dim=0)

    res_out = flag_gems.tensor_split(inp, indices, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split_sections
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_dim(shape, dtype):
    """Test tensor_split with different dim values."""
    if len(shape) < 2:
        pytest.skip("Need at least 2 dimensions for dim test")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test splitting along dim=1
    sections = 2
    ref_out = torch.tensor_split(ref_inp, sections, dim=1)

    res_out = flag_gems.tensor_split(inp, sections, dim=1)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split_sections
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_uneven(dtype):
    """Test tensor_split with uneven splits."""
    # Specific shape for uneven split test
    shape = (10, 20)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Split 10 into 3 sections: [4, 3, 3]
    sections = 3
    ref_out = torch.tensor_split(ref_inp, sections, dim=0)

    res_out = flag_gems.tensor_split(inp, sections, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split_tensor_indices_or_sections
@pytest.mark.parametrize("shape", TENSOR_SPLIT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_tensor_split_by_tensor(shape, dtype):
    """Test tensor_split with a 1-D tensor of split indices."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    indices = torch.tensor([shape[0] // 3, shape[0] * 2 // 3])
    ref_out = torch.tensor_split(ref_inp, indices, dim=0)

    res_out = flag_gems.tensor_split(inp, indices, dim=0)

    # Compare number of outputs
    assert len(res_out) == len(ref_out)

    # Compare each split
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        utils.gems_assert_equal(res, ref)


@pytest.mark.tensor_split_indices
@pytest.mark.parametrize("layout", ["contiguous", "strided", "empty"])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize(
    "form", ["sections", "indices", "tensor_sections", "tensor_indices"]
)
def test_ascend_tensor_split_preserves_views(layout, dim, form):
    if flag_gems.vendor_name != "ascend":
        pytest.skip("Ascend tensor_split view implementation required")
    assert flag_gems.tensor_split.__module__.endswith("_ascend.ops.tensor_split")

    inp = torch.arange(48, dtype=torch.float32, device=flag_gems.device).reshape(6, 8)
    if layout == "strided":
        inp = inp[1::2, 1::2].transpose(0, 1)
    elif layout == "empty":
        inp = inp[:, :0]
    ref_inp = utils.to_reference(inp).clone()
    boundaries = 12 if "sections" in form else [4, 1, 1, -2, -100, 100]
    if form.startswith("tensor_"):
        boundaries = torch.tensor(boundaries, dtype=torch.int64, device="cpu")
    ref_out = torch.tensor_split(ref_inp, boundaries, dim=dim)
    previous_registrar = flag_gems.current_work_registrar
    lib = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.only_enable(lib=lib, include=["tensor_split"])
        res_out = torch.tensor_split(inp, boundaries, dim=dim)

    finally:
        lib._destroy()
        flag_gems.current_work_registrar = previous_registrar
    assert len(res_out) == len(ref_out)
    for res, ref in zip(res_out, ref_out):
        utils.gems_assert_equal(res, ref)
        assert res.stride() == inp.stride()
        assert res.untyped_storage()._cdata == inp.untyped_storage()._cdata
    # Overlapping outputs must observe sequential writes into the same input.
    for i, (res, ref) in enumerate(zip(res_out, ref_out)):
        res.fill_(100 + i)
        ref.fill_(100 + i)
        utils.gems_assert_equal(inp, ref_inp)

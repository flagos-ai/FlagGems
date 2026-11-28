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
from .accuracy_utils import SCALARS


@pytest.mark.subtract
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.BYTE_DTYPES)
def test_subtract(shape, alpha, dtype):
    # randn has no int8 kernel: build in fp32 and cast down.
    inp1 = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    inp2 = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    if dtype in utils.BYTE_DTYPES:
        alpha = int(alpha)  # ATen rejects float alpha for integral inputs.
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.subtract(ref_inp1, ref_inp2, alpha=alpha)
    res_out = flag_gems.subtract(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.subtract
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_subtract_float_alpha_rejected(dtype):
    """ATen rejects a fractional alpha for integral inputs; FlagGems must too."""
    inp1 = torch.randint(-50, 50, (4,), dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    inp2 = torch.randint(-50, 50, (4,), dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    with pytest.raises(
        RuntimeError,
        match="argument alpha must not be a floating point number",
    ):
        flag_gems.subtract(inp1, inp2, alpha=0.5)


@pytest.mark.subtract_
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_subtract__float_alpha_rejected(dtype):
    """The in-place variant with a scalar other must reject fractional alpha too."""
    inp1 = torch.randint(-50, 50, (4,), dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    with pytest.raises(
        RuntimeError,
        match="argument alpha must not be a floating point number",
    ):
        flag_gems.subtract_(inp1, 2, alpha=0.5)


@pytest.mark.subtract_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.BYTE_DTYPES)
def test_subtract_(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    inp2 = torch.randn(shape, dtype=torch.float32, device=flag_gems.device).to(dtype)
    if dtype in utils.BYTE_DTYPES:
        alpha = int(alpha)  # ATen rejects float alpha for integral inputs.
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.subtract_(ref_inp2, alpha=alpha)
    res_out = flag_gems.subtract_(inp1, inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp1, dtype)

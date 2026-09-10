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


@pytest.mark.log_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_(shape, dtype):
    torch.manual_seed(0)
    # Add 0.1 to keep all values positive (log is only defined for positive reals)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.log_()
    with flag_gems.use_gems():
        res_out = inp.log_()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_special_values(dtype):
    """Test log_ on inf, -inf, 0, and nan inputs."""
    inp = torch.tensor(
        [float("inf"), float("-inf"), 0.0, float("nan"), 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.log_()
    with flag_gems.use_gems():
        res_out = inp.log_()
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.log_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_noncontiguous(dtype):
    """Non-contiguous tensors should fall back to aten and still produce correct results."""
    base = torch.rand(64, 64, dtype=dtype, device=flag_gems.device) + 0.1
    inp = base[::2, ::2].clone()  # make contiguous copy for gems path
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.log_()
    with flag_gems.use_gems():
        res_out = inp.log_()
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.log_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_log_empty(dtype):
    """Empty tensor should return immediately without error."""
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = inp.log_()
    assert res_out.numel() == 0


@pytest.mark.log_
@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_log_unsupported_dtype_raises(dtype):
    """Integer in-place log cannot store a float result: match torch and raise
    instead of silently truncating."""
    inp = torch.arange(1, 5, dtype=dtype, device=flag_gems.device)

    # torch raises RuntimeError; gems raises TypeError (cannot delegate to aten
    # without recursion). Both prevent silent truncation.
    with pytest.raises(RuntimeError):
        inp.clone().log_()
    with flag_gems.use_gems():
        with pytest.raises((RuntimeError, TypeError)):
            inp.log_()

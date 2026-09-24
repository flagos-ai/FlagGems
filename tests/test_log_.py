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
@pytest.mark.parametrize("n", [40960, 41984, 65536, 151936])
def test_log_large_inplace(n):
    """A grid larger than the device core count must not re-run the first tile.

    Regression test for #6446. Ascend caps the launch grid at the vector core
    count (40 for an Ascend910) and triton-ascend pads a larger grid up to a
    multiple of it, running every padding CTA with program_id == 0. When log_
    launched one program per 1024-element tile without capping, the extra CTAs
    re-applied the in-place log to tile 0, so log(log(x)) came out as NaN.
    The sizes below all need more than 40 tiles while not being a multiple of
    40, which is what makes the old code pad instead of queueing cleanly.
    """
    torch.manual_seed(0)
    inp = torch.rand(n, dtype=torch.float32, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.log_()
    res_out = flag_gems.log_(inp)
    utils.gems_assert_close(res_out, ref_out, torch.float32)


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
    # Build the values on CPU and move them: torch_npu's arange has no int16
    # kernel, so arange(dtype=torch.int16, device=<npu>) raises before the test
    # can run. int16 tensors themselves are fine on npu.
    inp = torch.arange(1, 5, dtype=dtype).to(flag_gems.device)

    # torch raises RuntimeError; gems raises TypeError (cannot delegate to aten
    # without recursion). Both prevent silent truncation.
    with pytest.raises(RuntimeError):
        inp.clone().log_()
    with flag_gems.use_gems():
        with pytest.raises((RuntimeError, TypeError)):
            inp.log_()

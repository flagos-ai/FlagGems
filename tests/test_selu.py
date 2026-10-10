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


@pytest.mark.selu
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.selu(ref_inp)
    res_out = flag_gems.selu(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.selu_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.selu_(ref_inp)
    res_out = flag_gems.selu_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="empty-input handling is specific to the Ascend kernel",
)
@pytest.mark.selu_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu__empty(dtype):
    # An empty tensor is a valid no-op; a zero-sized grid used to abort the
    # Ascend process with "coreDim is invalid".
    from flag_gems.runtime.backend._ascend.ops import selu_ as ascend_selu_

    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.selu_(ref_inp)
    res_out = ascend_selu_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="contiguity guard is specific to the Ascend kernel",
)
@pytest.mark.selu_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_selu__non_contiguous_rejected(dtype):
    # A strided view would be traversed linearly by the kernel and corrupt
    # neighbouring elements, so it must be rejected instead.
    from flag_gems.runtime.backend._ascend.ops import selu_ as ascend_selu_

    base = torch.randn(16, dtype=dtype, device=flag_gems.device)
    view = base[::2]
    assert not view.is_contiguous()

    with pytest.raises(ValueError):
        ascend_selu_(view)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="dtype validation is specific to the Ascend kernel",
)
@pytest.mark.selu_
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_selu__integral_rejected(dtype):
    # ATen's elu_ has no integral kernel; we must raise rather than compute in
    # fp32 and write the result back in the original dtype.
    from flag_gems.runtime.backend._ascend.ops import selu_ as ascend_selu_

    inp = torch.ones(4, dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        ascend_selu_(inp)

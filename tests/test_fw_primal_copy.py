# Copyright 2026, The FlagOS Contributors.
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

# CI derives the pytest marker from the implementation file name via the
# #6359 convention; the stripped name also needs a marker, so declare both.
pytestmark = [
    pytest.mark.fw_primal_copy,
]


@pytest.mark.fw_primal_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_fw_primal_copy(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # The native reference only runs under InferenceMode; the GEMS path is
    # dtype-exact for the copy, so compare against the InferenceMode reference.
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with torch.inference_mode():
        ref_inp = res_inp.clone()
        ref_out = flag_gems._fw_primal_copy(ref_inp, 0)

    res_out = flag_gems._fw_primal_copy(res_inp, 0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fw_primal_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
def test_fw_primal_copy_int(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    res_inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_out = res_inp.clone()

    res_out = flag_gems._fw_primal_copy(res_inp, 0)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.fw_primal_copy
def test_fw_primal_copy_noncontiguous():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # Non-contiguous inputs must materialize a fresh contiguous copy.
    res_inp = torch.randn((16, 32), device=flag_gems.device)[:, ::2]
    with torch.inference_mode():
        ref_inp = res_inp.clone()
        ref_out = flag_gems._fw_primal_copy(ref_inp, 0)

    res_out = flag_gems._fw_primal_copy(res_inp, 0)

    assert res_out.is_contiguous()
    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.fw_primal_copy_out
def test_fw_primal_copy_out():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    res_inp = torch.randn((16, 16), device=flag_gems.device)
    res_out = torch.empty_like(res_inp)
    # The native .out reference only runs under InferenceMode with
    # inference tensors, so build the reference inputs inside the block.
    with torch.inference_mode():
        ref_inp = res_inp.clone()
        ref_out = torch.empty_like(ref_inp)
        flag_gems._fw_primal_copy_out(ref_inp, 0, out=ref_out)
    res_ret = flag_gems._fw_primal_copy_out(res_inp, 0, out=res_out)

    assert res_ret is res_out
    utils.gems_assert_close(res_out, ref_out, torch.float32)

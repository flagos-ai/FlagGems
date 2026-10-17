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


@pytest.mark.make_dual_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_make_dual_copy(shape, dtype):
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # The tangent is only readable inside the forward-AD level that created
    # the dual; the native reference additionally requires InferenceMode
    # tensors, so build and unpack each side inside its own block.
    res_primal = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_tangent = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with torch.inference_mode():
        with torch.autograd.forward_ad.dual_level():
            ref_out = flag_gems._make_dual_copy(
                res_primal.clone(), res_tangent.clone(), 0
            )
            ref_primal_v, ref_tangent_v = torch.autograd.forward_ad.unpack_dual(ref_out)

    with torch.autograd.forward_ad.dual_level():
        res_out = flag_gems._make_dual_copy(res_primal, res_tangent, 0)
        res_primal_v, res_tangent_v = torch.autograd.forward_ad.unpack_dual(res_out)

    utils.gems_assert_close(res_primal_v, ref_primal_v, dtype)
    assert res_tangent_v is not None
    utils.gems_assert_close(res_tangent_v, res_tangent, dtype)
    # The primal is a copy: it must not alias the input.
    assert res_primal_v.data_ptr() != res_primal.data_ptr()


@pytest.mark.make_dual_copy
def test_make_dual_copy_noncontiguous():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    res_primal = torch.randn((16, 32), device=flag_gems.device)[:, ::2]
    res_tangent = torch.randn((16, 16), device=flag_gems.device)
    with torch.inference_mode():
        with torch.autograd.forward_ad.dual_level():
            ref_out = flag_gems._make_dual_copy(
                res_primal.clone(), res_tangent.clone(), 0
            )
            ref_primal_v, _ = torch.autograd.forward_ad.unpack_dual(ref_out)

    with torch.autograd.forward_ad.dual_level():
        res_out = flag_gems._make_dual_copy(res_primal, res_tangent, 0)
        res_primal_v, _ = torch.autograd.forward_ad.unpack_dual(res_out)

    assert res_primal_v.is_contiguous()
    utils.gems_assert_close(res_primal_v, ref_primal_v, torch.float32)


@pytest.mark.make_dual_copy_out
def test_make_dual_copy_out():
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    # The op output is device-independent, but --ref=cpu moves the native
    # reference to CPU; this suite is validated on the device path.
    if utils.TO_CPU:
        pytest.skip("reference-device suite; skipped under --ref=cpu")
    res_primal = torch.randn((16, 16), device=flag_gems.device)
    res_tangent = torch.randn((16, 16), device=flag_gems.device)
    res_out = torch.empty_like(res_primal)

    with torch.inference_mode():
        with torch.autograd.forward_ad.dual_level():
            ref_ret = flag_gems._make_dual_copy_out(
                res_primal.clone(),
                res_tangent.clone(),
                0,
                out=torch.empty_like(res_primal),
            )
            ref_primal_v, ref_tangent_v = torch.autograd.forward_ad.unpack_dual(ref_ret)

    with torch.autograd.forward_ad.dual_level():
        res_ret = flag_gems._make_dual_copy_out(res_primal, res_tangent, 0, out=res_out)
        res_primal_v, res_tangent_v = torch.autograd.forward_ad.unpack_dual(res_ret)

    utils.gems_assert_close(res_primal_v, ref_primal_v, torch.float32)
    assert res_tangent_v is not None
    utils.gems_assert_close(res_tangent_v, res_tangent, torch.float32)

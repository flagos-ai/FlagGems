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

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.testing import RESOLUTION

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.dropout_
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3] if cfg.QUICK_MODE else [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_dropout_(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    else:
        utils.init_seed(0)

    if cfg.TO_CPU or shape == (1,):
        # Statistical validation needs a large sample; scalar/CPU shapes are
        # replaced with a 32K-element tensor so the drop ratio is meaningful.
        shape = (32768,)

    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    # dropout_ overwrites its argument, so snapshot the original values before
    # the call to validate the scaling of the surviving elements afterwards.
    ref_orig = utils.to_reference(res_inp.clone())

    # Compute the expected scale factor for validation. On some vendors it's
    # computed as Python float division; on cambricon it's plain float.
    if flag_gems.vendor_name == "cambricon":
        one_minus_p = 1.0 - p
    else:
        one_minus_p = np.float32(1.0) - np.float32(p)

    res_out = flag_gems.dropout_(res_inp, p, True)

    # In-place contract: the returned tensor is the mutated input itself.
    assert res_out.data_ptr() == res_inp.data_ptr()

    # dropout is probabilistic, so validate statistically rather than against a
    # reference mask (Rule 30): the drop ratio must match p, and every surviving
    # element must equal its original value scaled by 1 / (1 - p).
    res_out = utils.to_reference(res_out)
    zero_equal = torch.eq(res_out, torch.zeros_like(res_out))
    num_zero = torch.sum(zero_equal).item()
    assert abs(num_zero / res_out.numel() - p) <= 0.05
    scale_equal = torch.isclose(
        res_out, ref_orig / one_minus_p, rtol=RESOLUTION[dtype], atol=1e-3
    )
    assert torch.all(torch.logical_or(zero_equal, scale_equal))


@pytest.mark.dropout_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dropout__noop(dtype):
    # train=False must be an exact no-op that returns the same tensor.
    # 8192 elements is a small, cheap buffer sufficient for an exact check.
    res_inp = torch.randn((8192,), dtype=dtype, device=flag_gems.device)
    # to_reference moves the snapshot to CPU under TO_CPU so gems_assert_equal's
    # device check holds; the no-op must leave the input bit-for-bit unchanged.
    expected = utils.to_reference(res_inp.clone())
    res_out = flag_gems.dropout_(res_inp, 0.5, False)
    assert res_out.data_ptr() == res_inp.data_ptr()
    utils.gems_assert_equal(res_out, expected)

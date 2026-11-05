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

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


def _skip_if_device_ref_int(dtype):
    # The XPU (xpu) device reference does not implement the min reduction for
    # integer dtypes (e.g. "kint16 is unsupported in xdnn_pytorch_wrapper"), so
    # the on-device torch reference raises for them. The integer semantics are
    # covered by the --ref=cpu run (see test-op.sh TEST_CASES_CPU); the device
    # run records them as SKIP (reference-side limitation, not a kernel
    # defect).
    if not cfg.TO_CPU and dtype in utils.ALL_INT_DTYPES:
        pytest.skip(
            f"XPU device reference does not implement min for {dtype}; "
            "run with --ref=cpu"
        )


def _skip_if_device_ref_all_inf():
    # The XPU (xpu) device reduction initializes with +FLT_MAX, so an all-inf
    # input reduces to +FLT_MAX instead of inf; skipped on the device run
    # (reference-side deviation, not a kernel defect) and covered by the
    # --ref=cpu run (test-op.sh TEST_CASES_CPU).
    if not cfg.TO_CPU:
        pytest.skip("XPU device reduction mishandles all +inf; run with --ref=cpu")


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min(shape, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    res_out = flag_gems.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.min
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_min_all_inf(shape, dtype):
    _skip_if_device_ref_all_inf()
    # ensure that padding value used in min is inf, not max value
    inp = torch.full(
        shape, fill_value=float("inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.min(ref_inp)
    res_out = flag_gems.min(inp)

    utils.gems_assert_equal(res_out, ref_out)


# Issue #2832: fails at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.min_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_min_dim(shape, dim, keepdim, dtype):
    _skip_if_device_ref_int(dtype)
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.min(ref_inp, dim=dim, keepdim=keepdim)
    res_out_value, res_out_index = flag_gems.min_dim(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)

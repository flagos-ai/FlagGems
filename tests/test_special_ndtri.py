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

# PyTorch implements ndtri for single and double precision only: ndtri_cuda and
# ndtri_cpu both raise NotImplementedError for Half and BFloat16, so there is no
# reference to compare against for those dtypes. Both supported precisions are
# exercised here; float64 is only included when the backend supports it.
NDTRI_DTYPES = [torch.float32] + ([torch.float64] if utils.fp64_is_supported else [])
NDTRI_UNSUPPORTED_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.special_ndtri
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", NDTRI_DTYPES)
def test_special_ndtri(shape, dtype, caplog):
    # ndtri is the inverse of the standard normal CDF, so the input is a
    # probability in [0, 1]. The interior of the range is sampled here; the
    # boundaries are covered separately below because they map to infinities.
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(0.01, 0.99)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.special_ndtri(ref_inp)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_ndtri"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_ndtri(inp)

    assert "GEMS SPECIAL_NDTRI" in caplog.text
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_ndtri
@pytest.mark.parametrize("dtype", NDTRI_DTYPES)
def test_special_ndtri_edge_values(dtype, caplog):
    # p == 0 and p == 1 map to -inf and +inf, and inputs outside [0, 1] as well
    # as nan map to nan. These are exact values rather than approximations, so
    # they are compared with equality semantics instead of a tolerance.
    inp = torch.tensor(
        [0.0, 1.0, 0.5, -0.1, 1.1, float("nan")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.special_ndtri(ref_inp)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_ndtri"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_ndtri(inp)

    assert "GEMS SPECIAL_NDTRI" in caplog.text
    torch.testing.assert_close(
        res_out.to(torch.float32).cpu(),
        ref_out.to(torch.float32).cpu(),
        equal_nan=True,
        rtol=0,
        atol=0,
    )


@pytest.mark.special_ndtri
@pytest.mark.parametrize("dtype", NDTRI_DTYPES)
def test_special_ndtri_non_contiguous(dtype, caplog):
    # A transposed view exercises the strided path of the pointwise kernel.
    inp = torch.empty((16, 32), dtype=dtype, device=flag_gems.device).uniform_(
        0.01, 0.99
    )
    inp = inp.t()
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.special_ndtri(ref_inp)
    with caplog.at_level("DEBUG", logger="flag_gems.ops.special_ndtri"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.special_ndtri(inp)

    assert "GEMS SPECIAL_NDTRI" in caplog.text
    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_ndtri
@pytest.mark.parametrize("dtype", NDTRI_UNSUPPORTED_DTYPES)
def test_special_ndtri_unsupported_dtype(dtype):
    # Half and BFloat16 are rejected by the operator itself, matching the
    # NotImplementedError that the PyTorch reference raises for these dtypes.
    inp = torch.empty((8,), dtype=dtype, device=flag_gems.device).uniform_(0.01, 0.99)
    with pytest.raises(NotImplementedError):
        with flag_gems.use_gems():
            torch.ops.aten.special_ndtri(inp)

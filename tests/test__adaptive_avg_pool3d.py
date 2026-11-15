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

# (input_shape, output_size) pairs, ported from the experimental unit test.
ADAPTIVE_AVG_POOL3D_CASES = [
    ((1, 1, 4, 4, 4), (1, 1, 1)),
    ((1, 1, 4, 4, 4), (2, 2, 2)),
    ((1, 1, 4, 4, 4), (4, 4, 4)),
    ((2, 3, 8, 7, 6), (1, 1, 1)),
    ((2, 3, 8, 7, 6), (2, 3, 3)),
    ((2, 3, 8, 7, 6), (4, 5, 6)),
    ((4, 16, 16, 16, 16), (1, 1, 1)),
    ((4, 16, 16, 16, 16), (4, 4, 4)),
    ((4, 16, 16, 16, 16), (8, 8, 8)),
]


# The yaml ids are ``adaptive_avg_pool3d`` / ``adaptive_avg_pool3d_out``, but the
# aten name starts with an underscore and the stripped name collides with the
# separate ``adaptive_avg_pool3d`` op, so the #6359 convention maps the
# implementation file to ``underscore_adaptive_avg_pool3d``. CI selects tests by
# that derived marker (tools/test-op.sh), while the rule-check marker gate uses
# the yaml id -- so declare both names (cf. tests/test__sobol_engine_scramble_.py).
@pytest.mark.adaptive_avg_pool3d
@pytest.mark.underscore_adaptive_avg_pool3d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d(shape, output_size, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    res_out = flag_gems.adaptive_avg_pool3d(res_inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool3d_out
@pytest.mark.underscore_adaptive_avg_pool3d_out
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVG_POOL3D_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_out(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], shape[1], *output_size)
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._adaptive_avg_pool3d.out(ref_inp, output_size, out=ref_out)

    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.adaptive_avg_pool3d_out(inp, output_size, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.underscore_adaptive_avg_pool3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_4d_input(dtype):
    # A 4D (C, D, H, W) input takes the unsqueeze-to-5D branch.
    shape = (3, 8, 7, 6)
    output_size = (4, 5, 6)
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten._adaptive_avg_pool3d(ref_inp, output_size)
    res_out = flag_gems.adaptive_avg_pool3d(res_inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.adaptive_avg_pool3d_out
@pytest.mark.underscore_adaptive_avg_pool3d_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_out_4d_input(dtype):
    # A 4D (C, D, H, W) input takes the unsqueeze-to-5D branch of the out variant;
    # the result is squeezed back to 4D (C, D_out, H_out, W_out).
    shape = (3, 8, 7, 6)
    output_size = (4, 5, 6)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    out_shape = (shape[0], *output_size)
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten._adaptive_avg_pool3d.out(ref_inp, output_size, out=ref_out)

    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.adaptive_avg_pool3d_out(inp, output_size, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)

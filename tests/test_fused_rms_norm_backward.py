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


@pytest.mark.fused_rms_norm_backward
# Covers one normalized dimension and a multi-dimensional normalized suffix.
@pytest.mark.parametrize(
    "shape,normalized_shape", [((3, 8), (8,)), ((2, 3, 4), (3, 4))]
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "output_mask", [(True, True), (True, False), (False, True), (False, False)]
)
def test_fused_rms_norm_backward(shape, normalized_shape, dtype, output_mask):
    # aten._fused_rms_norm_backward has no CPU backend registration, so the
    # CPU reference path (--ref=cpu) cannot compute the expected result.
    if utils.TO_CPU:
        pytest.skip("aten._fused_rms_norm_backward is not implemented on CPU")
    inp = torch.randn(shape, device=flag_gems.device, dtype=dtype)
    weight = torch.randn(normalized_shape, device=flag_gems.device, dtype=dtype)
    grad = torch.randn_like(inp)
    reduce_dims = tuple(range(inp.ndim - len(normalized_shape), inp.ndim))
    rstd = torch.rsqrt(inp.float().pow(2).mean(dim=reduce_dims) + 1e-5)
    ref_grad = utils.to_reference(grad)
    ref_inp = utils.to_reference(inp)
    ref_rstd = utils.to_reference(rstd)
    ref_weight = utils.to_reference(weight)
    expected = torch.ops.aten._fused_rms_norm_backward(
        ref_grad, ref_inp, normalized_shape, ref_rstd, ref_weight, output_mask
    )

    with flag_gems.use_gems():
        actual = torch.ops.aten._fused_rms_norm_backward(
            grad, inp, normalized_shape, rstd, weight, output_mask
        )

    for result, reference in zip(actual, expected):
        if reference is None:
            assert result is None
        else:
            utils.gems_assert_close(
                result, reference, dtype, reduce_dim=torch.tensor(shape).prod().item()
            )


@pytest.mark.fused_rms_norm_backward
def test_fused_rms_norm_backward_without_weight():
    # aten._fused_rms_norm_backward has no CPU backend registration, so the
    # CPU reference path (--ref=cpu) cannot compute the expected result.
    if utils.TO_CPU:
        pytest.skip("aten._fused_rms_norm_backward is not implemented on CPU")
    inp = torch.randn((3, 8), device=flag_gems.device)
    grad = torch.randn_like(inp)
    rstd = torch.rsqrt(inp.pow(2).mean(dim=-1) + 1e-5)
    expected = torch.ops.aten._fused_rms_norm_backward(
        utils.to_reference(grad),
        utils.to_reference(inp),
        [8],
        utils.to_reference(rstd),
        None,
        [True, False],
    )
    with flag_gems.use_gems():
        actual = torch.ops.aten._fused_rms_norm_backward(
            grad, inp, [8], rstd, None, [True, False]
        )
    utils.gems_assert_close(actual[0], expected[0], torch.float32, reduce_dim=8)
    assert actual[1] is None

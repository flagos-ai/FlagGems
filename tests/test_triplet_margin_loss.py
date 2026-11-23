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
    TRIPLET_SHAPES = [(4, 8)]
    TRIPLET_MARGINS = [1.0]
    TRIPLET_P_VALUES = [2.0]
    TRIPLET_REDUCTIONS = [1]
    TRIPLET_SWAP_VALUES = [False]
else:
    TRIPLET_SHAPES = [(4, 8), (32, 128), (128, 256), (1024, 512)]
    TRIPLET_MARGINS = [0.0, 1.0, 2.0]
    TRIPLET_P_VALUES = [1.0, 2.0, 3.0]
    TRIPLET_REDUCTIONS = [0, 1, 2]  # none, mean, sum
    TRIPLET_SWAP_VALUES = [False, True]


@pytest.mark.triplet_margin_loss
@pytest.mark.parametrize("shape", TRIPLET_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("margin", TRIPLET_MARGINS)
@pytest.mark.parametrize("p", TRIPLET_P_VALUES)
@pytest.mark.parametrize("reduction", TRIPLET_REDUCTIONS)
@pytest.mark.parametrize("swap", TRIPLET_SWAP_VALUES)
def test_triplet_margin_loss(shape, dtype, margin, p, reduction, swap):
    anchor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    positive = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_anchor = utils.to_reference(anchor, upcast=True)
    ref_positive = utils.to_reference(positive, upcast=True)
    ref_negative = utils.to_reference(negative, upcast=True)

    ref_out = torch.ops.aten.triplet_margin_loss(
        ref_anchor,
        ref_positive,
        ref_negative,
        margin,
        p,
        1e-6,
        swap,
        reduction,
    )
    res_out = flag_gems.triplet_margin_loss(
        anchor, positive, negative, margin, p, 1e-6, swap, reduction
    )

    # The reference is computed in float64 (upcast) because torch's native
    # triplet_margin_loss accumulates in the input dtype. For fp16/bf16 that is
    # too imprecise: the Lp-distance terms are ~O(D) and their difference
    # suffers catastrophic cancellation when margin=0 and dist_ap ~= dist_an,
    # so a same-dtype reference can be off by O(0.1). FlagGems accumulates in
    # fp32, so a high-precision reference matches it to within output-dtype
    # ulp. The residual fp32 accumulation error on large D (~2e-4) is covered
    # by the loosened atol below.
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.triplet_margin_loss
@pytest.mark.parametrize("eps", [1e-6, 0.5, 1.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_triplet_margin_loss_p1_eps(eps, dtype):
    # ATen folds eps *inside* the abs for p=1: sum(abs(x1 - x2 + eps)). A large
    # non-default eps makes the "eps before abs" vs "eps after norm" difference
    # observable, so this pins the correct placement.
    shape = (32, 64)
    anchor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    positive = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_anchor = utils.to_reference(anchor, upcast=True)
    ref_positive = utils.to_reference(positive, upcast=True)
    ref_negative = utils.to_reference(negative, upcast=True)

    ref_out = torch.ops.aten.triplet_margin_loss(
        ref_anchor, ref_positive, ref_negative, 1.0, 1.0, eps, False, 1
    )
    res_out = flag_gems.triplet_margin_loss(
        anchor, positive, negative, 1.0, 1.0, eps, False, 1
    )
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)


@pytest.mark.triplet_margin_loss
@pytest.mark.parametrize("p", [0.0, float("inf"), float("-inf")])
@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_triplet_margin_loss_extremal_p(p, swap, reduction):
    # p=0 is the count norm, p=+inf/-inf are the max/min coordinate distances.
    # The general 1/p formula cannot reproduce these, so they take dedicated
    # kernels; run in fp32 and check parity against ATen.
    shape = (64, 128)
    dtype = torch.float32
    anchor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    positive = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_anchor = utils.to_reference(anchor, upcast=True)
    ref_positive = utils.to_reference(positive, upcast=True)
    ref_negative = utils.to_reference(negative, upcast=True)

    ref_out = torch.ops.aten.triplet_margin_loss(
        ref_anchor, ref_positive, ref_negative, 1.0, p, 1e-6, swap, reduction
    )
    res_out = flag_gems.triplet_margin_loss(
        anchor, positive, negative, 1.0, p, 1e-6, swap, reduction
    )
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-4)


@pytest.mark.triplet_margin_loss
@pytest.mark.parametrize("margin", [0.0, 0.7, 2.0])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_triplet_margin_loss_empty_feature_dim(margin, reduction):
    # An empty feature dimension (N, 0) yields a zero Lp distance for every
    # sample, so ATen returns clamp_min(margin, 0) per sample. The kernel is
    # skipped for D=0, so this exercises the explicit host-side fill.
    shape = (8, 0)
    dtype = torch.float32
    anchor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    positive = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    negative = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # to_reference decides (per TO_CPU) whether the reference runs on cpu or the
    # device and upcasts to fp64, so gems_assert_close's internal cpu assertion
    # holds in both GPU and quick-cpu modes.
    ref_anchor = utils.to_reference(anchor, upcast=True)
    ref_positive = utils.to_reference(positive, upcast=True)
    ref_negative = utils.to_reference(negative, upcast=True)

    ref_out = torch.ops.aten.triplet_margin_loss(
        ref_anchor, ref_positive, ref_negative, margin, 2.0, 1e-6, False, reduction
    )
    res_out = flag_gems.triplet_margin_loss(
        anchor, positive, negative, margin, 2.0, 1e-6, False, reduction
    )
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-4)


@pytest.mark.triplet_margin_loss
def test_triplet_margin_loss_broadcast_and_promotion():
    # ATen accepts same-rank broadcastable shapes and promotes mixed dtypes.
    dtype_a = torch.float32
    dtype_p = torch.float64
    anchor = torch.randn(4, 16, dtype=dtype_a, device=flag_gems.device)
    positive = torch.randn(1, 16, dtype=dtype_p, device=flag_gems.device)
    negative = torch.randn(4, 16, dtype=dtype_a, device=flag_gems.device)

    # to_reference places the reference on cpu/device per TO_CPU and upcasts to
    # fp64, matching the promoted output dtype without a manual .to(device).
    ref_anchor = utils.to_reference(anchor, upcast=True)
    ref_positive = utils.to_reference(positive, upcast=True)
    ref_negative = utils.to_reference(negative, upcast=True)

    ref_out = torch.ops.aten.triplet_margin_loss(
        ref_anchor, ref_positive, ref_negative, 1.0, 2.0, 1e-6, False, 1
    )
    res_out = flag_gems.triplet_margin_loss(
        anchor, positive, negative, 1.0, 2.0, 1e-6, False, 1
    )
    # Output follows type promotion -> float64.
    assert res_out.dtype == torch.float64
    utils.gems_assert_close(res_out, ref_out, torch.float64, atol=1e-6)

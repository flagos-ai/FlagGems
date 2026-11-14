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

import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

device = flag_gems.device

if cfg.QUICK_MODE:
    N_TOKEN_LIST = [8]
    N_EXPERT_LIST = [16]
    N_GROUP_LIST = [4]
    TOPK_LIST = [2]
    RENORMALIZE_LIST = [True]
    SCORING_FUNC_LIST = [0]
    DTYPE_LIST = [torch.float32]
else:
    N_TOKEN_LIST = [1, 3, 8]
    N_EXPERT_LIST = [16]
    N_EXPERT_LIST = [8, 16]
    N_GROUP_LIST = [2, 4]
    TOPK_LIST = [1, 2]
    RENORMALIZE_LIST = [True, False]
    SCORING_FUNC_LIST = [0, 1]
    DTYPE_LIST = [torch.bfloat16, torch.float32]

try:
    from vllm._custom_ops import grouped_topk as vllm_grouped_topk

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    vllm_grouped_topk = None


def get_tolerance(dtype, scoring_func, renormalize):
    if dtype == torch.bfloat16:
        return 5e-3, 1e-3

    if dtype == torch.float16:
        if scoring_func == 1:
            return 1e-3, 1e-4
        else:
            return 5e-3, 1e-3

    if renormalize:
        return 5e-4, 1e-4
    return 1e-5, 1e-5


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("n_token", N_TOKEN_LIST)
@pytest.mark.parametrize("n_expert", N_EXPERT_LIST)
@pytest.mark.parametrize("n_group", N_GROUP_LIST)
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("scoring_func", SCORING_FUNC_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_grouped_topk(
    n_token,
    n_expert,
    n_group,
    topk,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk accuracy against vLLM CUDA implementation"""

    if n_expert % n_group != 0:
        return

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    topk_group = topk
    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_topk_weights, ref_topk_ids = vllm_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    ref_topk_weights = utils.to_reference(ref_topk_weights)
    ref_topk_ids = utils.to_reference(ref_topk_ids)

    res_topk_weights, res_topk_ids = flag_gems.grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    utils.gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_topk_weights = utils.to_reference(res_topk_weights)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("n_token", [32, 64])
@pytest.mark.parametrize("n_expert", [64])
@pytest.mark.parametrize("n_group", [8])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_topk_large_scale(
    n_token,
    n_expert,
    n_group,
    topk,
    topk_group,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk with larger scale configurations"""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_topk_weights, ref_topk_ids = vllm_grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )
    ref_topk_weights = utils.to_reference(ref_topk_weights)
    ref_topk_ids = utils.to_reference(ref_topk_ids)

    res_topk_weights, res_topk_ids = flag_gems.grouped_topk(
        scores.clone(),
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    utils.gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_topk_weights = utils.to_reference(res_topk_weights)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("topk", [3, 5, 10, 12, 14, 22])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_grouped_topk_non_power_of_2_topk(topk, renormalize, scoring_func):
    """topk need not be a power of two; only the Triton block dims must be.

    Nemotron-H sets ``num_experts_per_tok`` per layer to 4/8/10/12/14 (MTP: 22).
    Using topk directly as a block dimension made the Triton kernel fail with
    ``ValueError: Shape element 0 must be a power of 2``; the padded dimension
    must not change results for power-of-two topk nor the stored prefix for the
    others.
    """

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    n_expert, n_group, topk_group = 128, 8, 2
    scores = torch.randn((8, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), n_group, topk_group, topk, renormalize, 1.0, bias, scoring_func
    )
    res_weights, res_ids = flag_gems.grouped_topk(
        scores.clone(), n_group, topk_group, topk, renormalize, 1.0, bias, scoring_func
    )

    utils.gems_assert_equal(res_ids, utils.to_reference(ref_ids))

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(
        utils.to_reference(res_weights),
        utils.to_reference(ref_weights),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("topk", [10, 12, 14, 22])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_grouped_topk_non_power_of_2_topk_torch_reference(
    topk, renormalize, scoring_func
):
    """Dependency-free regression for the topk values Nemotron-H actually uses.

    The vLLM-reference case above also covers non-power-of-2 topk, but vLLM's
    CUDA extension is optional and absent on non-NVIDIA backends (such as the
    PPU target this fix was validated on). This case compares against plain
    torch so the padded block dimension is exercised on every backend.
    """

    dtype = torch.float32
    n_expert = 128
    n_group, topk_group = 1, 1
    routed_scaling_factor = 2.0
    # Strictly increasing values: no ties, so the expected ids are unambiguous.
    scores = torch.linspace(
        -2, 2, n_expert, dtype=dtype, device=flag_gems.device
    ).reshape(1, -1)
    bias = torch.zeros(n_expert, dtype=dtype, device=flag_gems.device)

    weights, ids = flag_gems.grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )

    processed = torch.sigmoid(scores.float()) if scoring_func else scores
    expected_ids = processed.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    expected_weights = processed.gather(1, expected_ids)
    if renormalize:
        expected_weights = expected_weights / (
            expected_weights.sum(-1, keepdim=True) + 1e-20
        )

    torch.testing.assert_close(ids, expected_ids.to(torch.int32))
    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(
        weights, expected_weights * routed_scaling_factor, atol=atol, rtol=rtol
    )
    assert weights.shape == (1, topk) and ids.shape == (1, topk)


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("renormalize", [True, False])
def test_grouped_topk_scaling_factor(routed_scaling_factor, renormalize):
    """Test grouped_topk with different scaling factors"""

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights, res_ids = flag_gems.grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
    )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 0, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_grouped_topk_single_token(renormalize, scoring_func):
    """Test grouped_topk with single token"""

    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((1, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights, res_ids = flag_gems.grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("renormalize", [True, False])
def test_grouped_topk_sigmoid(renormalize):
    """Test grouped_topk with sigmoid scoring function"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights, res_ids = flag_gems.grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 1, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)

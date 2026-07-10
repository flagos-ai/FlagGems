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


def apply_scoring_func_cpu(scores, scoring_func):
    if scoring_func == 1:
        half_scores = (0.5 * scores).to(scores.dtype)
        tanh_scores = torch.tanh(half_scores.to(torch.float32)).to(scores.dtype)
        routing_scores = (0.5 * tanh_scores).to(scores.dtype)
        return (routing_scores + 0.5).to(scores.dtype)
    return scores


def torch_grouped_topk_ref(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    """ATen CPU reference implementation for grouped_topk."""
    scores = scores.detach().cpu()
    bias = bias.detach().cpu().flatten().to(scores.dtype)

    routing_scores = apply_scoring_func_cpu(scores, scoring_func)

    n_token, n_expert = routing_scores.shape
    expert_per_group = n_expert // n_group

    selection_scores = routing_scores + bias
    group_scores = selection_scores.reshape(n_token, n_group, expert_per_group)
    group_scores = torch.topk(group_scores.to(torch.float32), k=2, dim=-1)
    group_scores = group_scores.values.sum(dim=-1).to(scores.dtype).to(torch.float32)

    group_ids = torch.argsort(group_scores, dim=-1, descending=True, stable=True)[
        :, :topk_group
    ]
    group_mask = torch.zeros(
        (n_token, n_group), dtype=torch.bool, device=selection_scores.device
    )
    group_mask.scatter_(1, group_ids, True)

    expert_group_ids = torch.arange(n_expert, device=selection_scores.device)
    expert_group_ids = expert_group_ids // expert_per_group
    expert_mask = group_mask.gather(1, expert_group_ids.expand(n_token, -1))

    masked_selection_scores = selection_scores.to(torch.float32)
    masked_selection_scores = masked_selection_scores.masked_fill(
        ~expert_mask, -float("inf")
    )
    topk_ids = torch.argsort(
        masked_selection_scores, dim=-1, descending=True, stable=True
    )[:, :topk].to(torch.int32)

    topk_weights = routing_scores.gather(1, topk_ids.to(torch.int64)).to(torch.float32)
    if renormalize:
        topk_weights = (
            topk_weights
            * routed_scaling_factor
            / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        )
    else:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def grouped_topk_reference(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    if HAS_VLLM and torch.cuda.is_available() and device == "cuda":
        return vllm_grouped_topk(
            scores,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            bias,
            scoring_func,
        )

    scores_ref = utils.to_reference(scores, True)
    bias_ref = utils.to_reference(bias, True)
    topk_weights, topk_ids = torch_grouped_topk_ref(
        scores_ref,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias_ref,
        scoring_func,
    )
    if not cfg.TO_CPU:
        topk_weights = topk_weights.to(scores.device)
        topk_ids = topk_ids.to(scores.device)
    return topk_weights, topk_ids


@pytest.mark.grouped_topk
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
    """Test grouped_topk accuracy against vLLM or ATen CPU reference."""

    if n_expert % n_group != 0:
        return

    torch.manual_seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(45)

    topk_group = topk
    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_topk_weights, ref_topk_ids = grouped_topk_reference(
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

    with flag_gems.use_gems():
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    routed_scaling_factor = 1.0

    scores = torch.randn((n_token, n_expert), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((n_expert,), dtype=dtype, device=flag_gems.device)

    ref_topk_weights, ref_topk_ids = grouped_topk_reference(
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

    with flag_gems.use_gems():
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
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("renormalize", [True, False])
def test_grouped_topk_scaling_factor(routed_scaling_factor, renormalize):
    """Test grouped_topk with different scaling factors"""

    torch.manual_seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = grouped_topk_reference(
        scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
        )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 0, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_grouped_topk_single_token(renormalize, scoring_func):
    """Test grouped_topk with single token"""

    torch.manual_seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((1, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = grouped_topk_reference(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
        )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
def test_grouped_topk_sigmoid(renormalize):
    """Test grouped_topk with sigmoid scoring function"""
    torch.manual_seed(45)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = grouped_topk_reference(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
        )

    utils.gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 1, renormalize)
    res_weights = utils.to_reference(res_weights)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)

import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_equal
from .conftest import QUICK_MODE

try:
    from vllm._custom_ops import grouped_topk as vllm_grouped_topk

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    vllm_grouped_topk = None


N_TOKEN_LIST = [1, 3, 8] if not QUICK_MODE else [8]
N_EXPERT_LIST = [8, 16] if not QUICK_MODE else [16]
N_GROUP_LIST = [2, 4] if not QUICK_MODE else [4]
TOPK_LIST = [1, 2] if not QUICK_MODE else [2]
RENORMALIZE_LIST = [True, False] if not QUICK_MODE else [True]
SCORING_FUNC_LIST = [0, 1] if not QUICK_MODE else [0]
DTYPE_LIST = [torch.bfloat16, torch.float32] if not QUICK_MODE else [torch.float32]

LARGE_SCALE_DTYPE_LIST = [torch.float32, torch.bfloat16]


def check_valid_config(n_expert, n_group, topk):
    if n_expert % n_group != 0:
        return False
    return True


def get_tolerance(dtype, scoring_func, renormalize):
    if dtype == torch.bfloat16:
        return 5e-3, 1e-3
    elif dtype == torch.float16:
        if scoring_func == 1:
            return 1e-3, 1e-4
        else:
            return 5e-3, 1e-3
    else:
        if renormalize:
            return 5e-4, 1e-4
        else:
            return 1e-5, 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", N_TOKEN_LIST)
@pytest.mark.parametrize("n_expert", N_EXPERT_LIST)
@pytest.mark.parametrize("n_group", N_GROUP_LIST)
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("scoring_func", SCORING_FUNC_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
def test_accuracy_grouped_topk(
    n_token,
    n_expert,
    n_group,
    topk,
    renormalize,
    scoring_func,
    dtype,
):
    """Test grouped_topk accuracy against vLLM CUDA implementation"""
    if not check_valid_config(n_expert, n_group, topk):
        pytest.skip("Invalid config")

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

    gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("n_token", [32, 64])
@pytest.mark.parametrize("n_expert", [64])
@pytest.mark.parametrize("n_group", [8])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("topk_group", [2])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
@pytest.mark.parametrize("dtype", LARGE_SCALE_DTYPE_LIST)
def test_accuracy_grouped_topk_large_scale(
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

    gems_assert_equal(res_topk_ids, ref_topk_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(res_topk_weights, ref_topk_weights, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("renormalize", [True, False])
def test_accuracy_grouped_topk_scaling_factor(routed_scaling_factor, renormalize):
    """Test grouped_topk with different scaling factors"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
    )

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, routed_scaling_factor, bias, 0
        )

    gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 0, renormalize)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", [0, 1])
def test_accuracy_grouped_topk_single_token(renormalize, scoring_func):
    """Test grouped_topk with single token"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((1, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
    )

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, 1.0, bias, scoring_func
        )

    gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, scoring_func, renormalize)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.grouped_topk
@pytest.mark.parametrize("renormalize", [True, False])
def test_accuracy_grouped_topk_sigmoid(renormalize):
    """Test grouped_topk with sigmoid scoring function"""
    torch.manual_seed(45)
    torch.cuda.manual_seed(45)

    dtype = torch.float32
    scores = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((16,), dtype=dtype, device=flag_gems.device)

    ref_weights, ref_ids = vllm_grouped_topk(
        scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
    )

    with flag_gems.use_gems():
        res_weights, res_ids = flag_gems.grouped_topk(
            scores.clone(), 4, 2, 2, renormalize, 1.0, bias, 1
        )

    gems_assert_equal(res_ids, ref_ids)

    atol, rtol = get_tolerance(dtype, 1, renormalize)
    torch.testing.assert_close(res_weights, ref_weights, atol=atol, rtol=rtol)

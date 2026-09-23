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
import torch.nn.functional as F

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device

if cfg.QUICK_MODE:
    NUM_TOKENS_LIST = [1, 33]
    NUM_EXPERTS_LIST = [128, 256]
    TOPK_LIST = [6, 8]
    DTYPE_LIST = [torch.bfloat16, torch.float32]
    RENORMALIZE_LIST = [True]
    RSF_LIST = [1.0]
else:
    NUM_TOKENS_LIST = [1, 33, 128]
    NUM_EXPERTS_LIST = [128, 256, 384, 512]
    TOPK_LIST = [6, 8, 16]
    DTYPE_LIST = [torch.bfloat16, torch.float16, torch.float32]
    RENORMALIZE_LIST = [True, False]
    RSF_LIST = [1.0, 1.5]

# Hash mode configurations
HASH_NUM_TOKENS_LIST = [1, 33, 128] if not cfg.QUICK_MODE else [1, 33]
HASH_NUM_EXPERTS_LIST = [256, 384, 512] if not cfg.QUICK_MODE else [256]
HASH_TOPK_LIST = [6, 8, 16] if not cfg.QUICK_MODE else [6, 8]
HASH_RSF_LIST = [1.0, 2.5] if not cfg.QUICK_MODE else [1.0]

try:
    from vllm._custom_ops import topk_hash_softplus_sqrt as vllm_topk_softplus_sqrt

    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    vllm_topk_softplus_sqrt = None


def _torch_topk_softplus_sqrt_reference(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    tid2eid: torch.Tensor | None = None,
):
    """Pure PyTorch reference implementation."""
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores
    if correction_bias is not None:
        scores_for_choice = scores + correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    if tid2eid is not None:
        assert input_ids is not None
        topk_ids = tid2eid[input_ids.long()]
    else:
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _check_topk_results(
    res_weights, res_ids, ref_weights, ref_ids, atol=2e-2, rtol=1e-2
):
    """Compare topk results by sorting indices first."""
    sorted_ids, idx_ops = res_ids.sort(dim=-1)
    sorted_ref_ids, idx_ref = ref_ids.sort(dim=-1)

    # Check indices match
    utils.gems_assert_equal(sorted_ids, sorted_ref_ids)

    # Check weights match (after sorting by indices)
    sorted_w = res_weights.gather(1, idx_ops)
    sorted_w_ref = ref_weights.gather(1, idx_ref)
    sorted_w = utils.to_reference(sorted_w)
    sorted_w_ref = utils.to_reference(sorted_w_ref)
    torch.testing.assert_close(sorted_w, sorted_w_ref, atol=atol, rtol=rtol)


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_LIST)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS_LIST)
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("routed_scaling_factor", RSF_LIST)
def test_topk_softplus_sqrt(
    num_tokens, num_experts, topk, dtype, renormalize, routed_scaling_factor
):
    """Test topk_softplus_sqrt in standard mode (with bias) against PyTorch reference."""
    torch.manual_seed(0)

    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device=device)
    correction_bias = torch.randn((num_experts,), dtype=torch.float32, device=device)

    ref_weights, ref_ids = _torch_topk_softplus_sqrt_reference(
        gating_output,
        topk,
        renormalize,
        routed_scaling_factor,
        correction_bias=correction_bias,
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    res_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    res_tei = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    with flag_gems.use_gems():
        flag_gems.topk_softplus_sqrt(
            res_weights,
            res_ids,
            res_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            correction_bias=correction_bias,
        )

    _check_topk_results(res_weights, res_ids, ref_weights, ref_ids)


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("num_tokens", HASH_NUM_TOKENS_LIST)
@pytest.mark.parametrize("num_experts", HASH_NUM_EXPERTS_LIST)
@pytest.mark.parametrize("topk", HASH_TOPK_LIST)
@pytest.mark.parametrize("dtype", DTYPE_LIST)
@pytest.mark.parametrize("renormalize", RENORMALIZE_LIST)
@pytest.mark.parametrize("routed_scaling_factor", HASH_RSF_LIST)
def test_topk_softplus_sqrt_hash(
    num_tokens, num_experts, topk, dtype, renormalize, routed_scaling_factor
):
    """Test topk_softplus_sqrt in hash mode against PyTorch reference."""
    torch.manual_seed(0)

    vocab_size = 1024
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device=device)
    tid2eid = torch.stack(
        [torch.randperm(num_experts)[:topk] for _ in range(vocab_size)]
    ).to(device=device, dtype=torch.int32)
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), dtype=torch.int32, device=device
    )

    ref_weights, ref_ids = _torch_topk_softplus_sqrt_reference(
        gating_output,
        topk,
        renormalize,
        routed_scaling_factor,
        input_ids=input_ids,
        tid2eid=tid2eid,
    )
    ref_weights = utils.to_reference(ref_weights)
    ref_ids = utils.to_reference(ref_ids)

    res_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    res_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    res_tei = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    with flag_gems.use_gems():
        flag_gems.topk_softplus_sqrt(
            res_weights,
            res_ids,
            res_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            input_ids=input_ids,
            tid2eid=tid2eid,
        )

    _check_topk_results(res_weights, res_ids, ref_weights, ref_ids)


@pytest.mark.topk_softplus_sqrt
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
@pytest.mark.parametrize("num_tokens", [1, 10, 128])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("topk", [6])
@pytest.mark.parametrize("renormalize", [True, False])
def test_topk_softplus_sqrt_vs_vllm(num_tokens, num_experts, topk, renormalize):
    """Test topk_softplus_sqrt against vLLM CUDA kernel."""
    torch.manual_seed(0)

    dtype = torch.bfloat16
    routed_scaling_factor = 1.0
    gating_output = torch.randn((num_tokens, num_experts), dtype=dtype, device=device)
    correction_bias = torch.randn((num_experts,), dtype=torch.float32, device=device)

    # vLLM CUDA kernel
    vllm_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    vllm_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    vllm_tei = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    vllm_topk_softplus_sqrt(
        vllm_weights,
        vllm_ids,
        vllm_tei,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias,
        None,
        None,
    )
    vllm_weights = utils.to_reference(vllm_weights)
    vllm_ids = utils.to_reference(vllm_ids)

    # FlagGems Triton kernel
    res_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    res_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    res_tei = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    with flag_gems.use_gems():
        flag_gems.topk_softplus_sqrt(
            res_weights,
            res_ids,
            res_tei,
            gating_output,
            renormalize,
            routed_scaling_factor,
            correction_bias=correction_bias,
        )

    _check_topk_results(res_weights, res_ids, vllm_weights, vllm_ids)

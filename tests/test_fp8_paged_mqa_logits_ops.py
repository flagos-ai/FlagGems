import random

import pytest
import torch

try:
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import (
        fp8_paged_mqa_logits as fp8_paged_mqa_logits_deepgemm,
    )
    from vllm.utils.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata
    from vllm.utils.import_utils import has_deep_gemm

    VLLM_AVAILABLE = True
    DEEPGEMM_AVAILABLE = has_deep_gemm()
    SM90_AVAILABLE = current_platform.has_device_capability(90)

except ImportError:
    VLLM_AVAILABLE = False
    DEEPGEMM_AVAILABLE = False
    SM90_AVAILABLE = False

import flag_gems
from flag_gems.ops.fp8_paged_mqa_logits import fp8_paged_mqa_logits

from .accuracy_utils import gems_assert_close, to_reference


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf_bytes

    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def kv_cache_cast_to_fp8_triton(x: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)

    out = torch.empty(
        (num_blocks, block_size, num_heads, head_dim + 4),
        device=x.device,
        dtype=torch.uint8,
    )
    out[..., :head_dim] = x_scaled.view(torch.uint8)

    sf_scaled = sf.squeeze(-1).squeeze(-1)  # [num_blocks, block_size]
    sf_bytes = sf_scaled.view(torch.int32).view(torch.uint8)
    out[..., head_dim:] = sf_bytes.view(num_blocks, block_size, num_heads, 4)

    return out


def _build_mask(context_lens, batch_size, next_n, max_model_len, device):
    positions = (
        torch.arange(max_model_len, device=device)
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    row_indices = torch.arange(batch_size * next_n, device=device) // next_n
    next_n_offset = torch.arange(batch_size * next_n, device=device) % next_n
    return positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(
        1
    )


@pytest.mark.fp8_paged_mqa_logits
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vllm is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="DeepGEMM not available")
@pytest.mark.skipif(not SM90_AVAILABLE, reason="SM90 and SM100 only")
@pytest.mark.parametrize("clean_logits", [True, False])
def test_accuracy_fp8_paged_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    batch_size, next_n = 4, 1
    heads, index_dim = 32, 128
    avg_kv = 2048
    num_blocks, blocksize = max_model_len * 2, 64

    q = torch.randn(
        (batch_size, next_n, heads, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device=flag_gems.device, dtype=torch.float32
    )

    context_lens = torch.randint(
        int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,), device=flag_gems.device
    ).to(torch.int32)
    max_num_blocks_per_seq = (context_lens.max().item() + blocksize - 1) // blocksize
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), device=flag_gems.device, dtype=torch.int32
    )

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        for j in range((ctx_len + blocksize - 1) // blocksize):
            block_tables[i][j] = block_idx_pool[counter]
            counter += 1

    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache_fp8_deepgemm = kv_cache_cast_to_fp8(kv_cache)
    kv_cache_fp8_triton = kv_cache_cast_to_fp8_triton(kv_cache)

    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, blocksize, get_num_sms()
    )
    ref_out = fp8_paged_mqa_logits_deepgemm(
        q_fp8,
        kv_cache_fp8_deepgemm,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )
    ref_out = to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_triton,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    mask = _build_mask(
        context_lens, batch_size, next_n, max_model_len, flag_gems.device
    )
    res_out_masked = torch.nan_to_num(res_out.masked_fill(~mask, 0), 0.0)
    ref_out_masked = torch.nan_to_num(ref_out.masked_fill(~mask, 0), 0.0)

    gems_assert_close(
        res_out_masked,
        ref_out_masked,
        res_out_masked.dtype,
        equal_nan=True,
        atol=5e-2,
        rtol=1e-3,
        reduce_dim=1,
    )


@pytest.mark.fp8_paged_mqa_logits
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vllm is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.skipif(not DEEPGEMM_AVAILABLE, reason="DeepGEMM not available")
@pytest.mark.skipif(not SM90_AVAILABLE, reason="SM90 and SM100 only")
@pytest.mark.parametrize("batch_size, next_n", [(4, 1), (2, 2)])
@pytest.mark.parametrize("heads, index_dim", [(32, 128)])
def test_accuracy_fp8_paged_mqa_logits_param(batch_size, next_n, heads, index_dim):
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    avg_kv = 2048
    num_blocks, blocksize = max_model_len * 2, 64

    q = torch.randn(
        (batch_size, next_n, heads, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim),
        device=flag_gems.device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, heads), device=flag_gems.device, dtype=torch.float32
    )

    context_lens = torch.randint(
        int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,), device=flag_gems.device
    ).to(torch.int32)
    max_num_blocks_per_seq = (context_lens.max().item() + blocksize - 1) // blocksize
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), device=flag_gems.device, dtype=torch.int32
    )

    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        for j in range((ctx_len + blocksize - 1) // blocksize):
            block_tables[i][j] = block_idx_pool[counter]
            counter += 1

    q_fp8 = q.to(torch.float8_e4m3fn)

    kv_cache_fp8_deepgemm = kv_cache_cast_to_fp8(kv_cache)
    kv_cache_fp8_triton = kv_cache_cast_to_fp8_triton(kv_cache)

    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, blocksize, get_num_sms()
    )
    ref_out = fp8_paged_mqa_logits_deepgemm(
        q_fp8,
        kv_cache_fp8_deepgemm,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=True,
    )
    ref_out = to_reference(ref_out)

    with flag_gems.use_gems():
        res_out = fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8_triton,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    mask = _build_mask(
        context_lens, batch_size, next_n, max_model_len, flag_gems.device
    )
    res_out_masked = torch.nan_to_num(res_out.masked_fill(~mask, 0), 0.0)
    ref_out_masked = torch.nan_to_num(ref_out.masked_fill(~mask, 0), 0.0)

    gems_assert_close(
        res_out_masked,
        ref_out_masked,
        res_out_masked.dtype,
        equal_nan=True,
        atol=5e-2,
        rtol=1e-3,
        reduce_dim=1,
    )

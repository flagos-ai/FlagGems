import pytest
import torch

import flag_gems
from flag_gems.fused import fp8_fp4_paged_mqa_logits

device = flag_gems.device

# DeepSeek-V4 model parameters
NUM_HEADS = 64
HEAD_DIM = 128
BLOCK_KV = 64  # KV cache page size
MAX_MODEL_LEN = 111 * 1024

# Test shapes: (batch_size, next_n, avg_context_len)
TEST_SHAPES = [
    (4, 1, 512),
    (4, 1, 1024),
    (4, 1, 2048),
    (8, 1, 1024),
    (8, 2, 1024),
    (16, 1, 2048),
    (32, 1, 4096),
    (64, 1, 4096),
    (128, 1, 2048),
]


def _ceil_div(a, b):
    return (a + b - 1) // b


def _kv_cache_cast_to_fp8(x):
    """Cast bf16 KV cache to FP8 format matching DeepGEMM layout.

    Layout: [num_blocks, block_size, 1, head_dim + 4] where trailing 4 bytes
    store per-token float32 scale factors.
    """
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
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def _make_inputs(batch_size, next_n, avg_kv):
    """Generate test inputs for FP8 paged MQA logits."""
    num_total_blocks = max(MAX_MODEL_LEN * 3 // BLOCK_KV, 1000)

    q_bf16 = torch.randn(
        (batch_size, next_n, NUM_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache_bf16 = torch.randn(
        (num_total_blocks, BLOCK_KV, 1, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, NUM_HEADS), device=device, dtype=torch.float
    )

    base_ctx = torch.randint(
        max(1, int(0.7 * avg_kv)),
        int(1.3 * avg_kv) + 1,
        (batch_size,),
        device=device,
        dtype=torch.int32,
    )
    base_ctx = base_ctx.clamp(max=MAX_MODEL_LEN)
    context_lens = base_ctx.unsqueeze(1).expand(-1, next_n).contiguous()

    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    kv_fp8 = _kv_cache_cast_to_fp8(kv_cache_bf16)

    max_ctx = int(base_ctx.max().item())
    num_blocks_per_query = _ceil_div(max_ctx, BLOCK_KV)
    block_table = torch.zeros(
        (batch_size, num_blocks_per_query), device=device, dtype=torch.int32
    )
    block_idx_pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i in range(batch_size):
        n_blocks = _ceil_div(base_ctx[i].item(), BLOCK_KV)
        if offset + n_blocks > num_total_blocks:
            block_idx_pool = torch.randperm(
                num_total_blocks, device=device, dtype=torch.int32
            )
            offset = 0
        block_table[i, :n_blocks] = block_idx_pool[offset : offset + n_blocks]
        offset += n_blocks

    return q_fp8, kv_fp8, weights, context_lens, block_table


try:
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits as _vllm_fp8_fp4_paged_mqa_logits,
    )
    from vllm.utils.deep_gemm import get_paged_mqa_logits_metadata as _vllm_get_metadata

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _reference_fn(q_fp8, kv_fp8, weights, context_lens, block_table):
    """Reference: call vLLM DeepGEMM CUDA kernel."""
    assert _HAS_VLLM, "vLLM is required for reference implementation"
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule_meta = _vllm_get_metadata(context_lens, BLOCK_KV, num_sms)
    q_input = (q_fp8, None)
    return _vllm_fp8_fp4_paged_mqa_logits(
        q=q_input,
        kv_cache=kv_fp8,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_table,
        schedule_metadata=schedule_meta,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
    )


@pytest.mark.fp8_fp4_paged_mqa_logits
@pytest.mark.parametrize(
    "batch_size, next_n, avg_kv",
    TEST_SHAPES,
    ids=[f"B{b}_N{n}_L{l}" for b, n, l in TEST_SHAPES],
)
@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not available")
def test_fp8_fp4_paged_mqa_logits(batch_size, next_n, avg_kv):
    torch.manual_seed(0)
    q_fp8, kv_fp8, weights, context_lens, block_table = _make_inputs(
        batch_size, next_n, avg_kv
    )

    # Run Triton kernel
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    schedule_meta = _vllm_get_metadata(context_lens, BLOCK_KV, num_sms)
    q_input = (q_fp8, None)
    triton_out = fp8_fp4_paged_mqa_logits(
        q=q_input,
        kv_cache=kv_fp8,
        weights=weights,
        context_lens=context_lens,
        block_tables=block_table,
        schedule_metadata=schedule_meta,
        max_model_len=MAX_MODEL_LEN,
        clean_logits=False,
    )

    # Run reference (vLLM DeepGEMM CUDA)
    ref_out = _reference_fn(q_fp8, kv_fp8, weights, context_lens, block_table)

    # Compare valid positions only (up to each row's context length)
    total_rows = batch_size * next_n
    ctx_flat = context_lens.reshape(-1)[:total_rows]
    diffs = []
    for row in range(total_rows):
        ctx = ctx_flat[row].item()
        if ctx == 0:
            continue
        t_row = triton_out[row, :ctx].float()
        r_row = ref_out[row, :ctx].float()
        # Mean relative difference
        denom = r_row.abs().clamp(min=1e-6)
        rel_diff = ((t_row - r_row).abs() / denom).mean().item()
        diffs.append(rel_diff)

    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    assert (
        mean_diff < 1e-3
    ), f"Mean relative diff {mean_diff:.6f} exceeds threshold 1e-3"
    assert not torch.isnan(triton_out).any(), "NaN detected in Triton output"

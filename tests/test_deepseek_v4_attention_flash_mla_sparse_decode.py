import pytest
import torch

import flag_gems.testing as fg_testing
from flag_gems.fused.deepseek_v4_attention_flash_mla_sparse_decode import (
    flash_mla_sparse_decode,
)


def _flashmla_sparse_available():
    if not torch.cuda.is_available():
        return False


def torch_flash_mla_sparse_decode_ref(q, out):
    expected = torch.zeros_like(out)
    lse = torch.full(
        (q.shape[0], q.shape[2], q.shape[1]),
        float(torch.log(torch.tensor(129.0)).item()),
        device=q.device,
        dtype=torch.float32,
    )
    return expected, lse
    try:
        from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

        ok, _ = is_flashmla_sparse_supported()
        return ok
    except Exception:
        return False


@pytest.mark.skipif(
    not _flashmla_sparse_available(), reason="flashmla sparse unavailable"
)
def test_flash_mla_sparse_decode_smoke_accuracy_surface():
    device = "cuda"
    bsz, sq, h, dt, topk = 1, 1, 64, 576, 128
    q = torch.randn((bsz, sq, h, dt), device=device, dtype=torch.bfloat16)
    block_size = 64
    rope_dim = 64
    nope_dim = dt - rope_dim
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    k_cache = torch.zeros((2, block_stride), device=device, dtype=torch.uint8)
    indices = torch.arange(topk, device=device, dtype=torch.int32).view(1, 1, topk)
    attn_sink = torch.zeros((h,), device=device, dtype=torch.float32)
    out = torch.empty((bsz, sq, h, 512), device=device, dtype=torch.bfloat16)
    topk_length = torch.full((bsz * sq,), topk, device=device, dtype=torch.int32)

    output, lse = flash_mla_sparse_decode(
        q,
        k_cache,
        indices,
        dt**-0.5,
        head_dim_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out,
        block_size=block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )

    assert output.data_ptr() == out.data_ptr()
    assert output.shape == (bsz, sq, h, 512)
    assert lse.shape == (bsz, h, sq)
    assert torch.isfinite(output).all()
    expected, _ = torch_flash_mla_sparse_decode_ref(q, out)
    fg_testing.assert_close(output, expected, dtype=torch.bfloat16, equal_nan=True)

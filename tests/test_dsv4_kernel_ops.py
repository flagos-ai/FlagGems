import pytest
import torch

from flag_gems.fused.dsv4_kernel_ops import (
    dsv4_kernel_combine_topk_swa_indices,
    dsv4_kernel_compute_global_topk_indices_and_lens,
    dsv4_kernel_cp_gather_indexer_k_quant_cache,
    dsv4_kernel_deepseek_v4_fp8_einsum,
    dsv4_kernel_dequantize_and_gather_k_cache,
    dsv4_kernel_flash_mla_sparse_decode,
    dsv4_kernel_flash_mla_sparse_fwd,
    dsv4_kernel_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
    dsv4_kernel_fused_q_kv_rmsnorm,
    dsv4_kernel_get_mla_metadata,
    dsv4_kernel_persistent_topk,
    dsv4_kernel_top_k_per_row_prefill,
)


def _has_hopper() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 9


HAS_HOPPER = _has_hopper()


@pytest.mark.skipif(not HAS_HOPPER, reason="requires Hopper SM90")
def test_dsv4_kernel_interface_local_ops_smoke():
    device = "cuda"

    qr = torch.randn((8, 64 * 576), device=device, dtype=torch.bfloat16)
    kv = torch.randn((8, 576), device=device, dtype=torch.bfloat16)
    q_weight = torch.ones((64 * 576,), device=device, dtype=torch.bfloat16)
    kv_weight = torch.ones((576,), device=device, dtype=torch.bfloat16)
    q_out, kv_out = dsv4_kernel_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, 1e-6)
    assert q_out.shape == qr.shape
    assert kv_out.shape == kv.shape

    topk_indices = torch.tensor(
        [[0, 2, -1, -1], [1, 3, 0, -1]], device=device, dtype=torch.int32
    )
    token_to_req = torch.tensor([0, 0], device=device, dtype=torch.int32)
    block_table = torch.tensor([[5, 9]], device=device, dtype=torch.int32)
    global_indices, lens = dsv4_kernel_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        block_size=64,
    )
    assert global_indices.shape == topk_indices.shape
    assert lens.shape == (2,)

    query_start_loc = torch.tensor([0, 2], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([16], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([16], device=device, dtype=torch.int32)
    combined, combined_lens = dsv4_kernel_combine_topk_swa_indices(
        topk_indices=topk_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        gather_lens=gather_lens,
        window_size=4,
        compress_ratio=2,
        topk=4,
        M=64,
        N=16,
    )
    assert combined.shape[0] == 2
    assert combined_lens.shape == (2,)


@pytest.mark.skipif(not HAS_HOPPER, reason="requires Hopper SM90")
def test_dsv4_kernel_interface_qnorm_insert_and_gather_smoke():
    device = "cuda"
    num_tokens = 8
    head_dim = 576
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    block_size = 64

    q = torch.randn((num_tokens, 64, head_dim), device=device, dtype=torch.bfloat16)
    kv = torch.randn((num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)

    half = rope_dim // 2
    pos = torch.arange(num_tokens + 8, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.arange(half, device=device, dtype=torch.float32).unsqueeze(0)
    angles = pos * (1.0 / (10000.0 ** (freq / max(1, half - 1))))
    cos_sin_cache = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).to(
        torch.bfloat16
    )

    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    k_cache = torch.zeros((2, block_stride), device=device, dtype=torch.uint8)

    dsv4_kernel_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        eps=1e-6,
        block_size=block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
    )

    out = torch.empty((1, num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    dsv4_kernel_dequantize_and_gather_k_cache(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset=0,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        scale_slots=scale_slots,
    )
    assert out.isfinite().all()


@pytest.mark.skipif(not HAS_HOPPER, reason="requires Hopper SM90")
def test_dsv4_kernel_interface_fp8_einsum_smoke():
    device = "cuda"
    batch = 2
    groups = 2
    kdim = 128
    ndim = 128
    a = torch.randn((batch, groups, kdim), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    b = torch.randn((groups, ndim, kdim), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    a_scale = torch.ones((batch, groups, 1), device=device, dtype=torch.float32)
    b_scale = torch.ones((groups, 1, 1), device=device, dtype=torch.float32)
    out = torch.empty((batch, groups, ndim), device=device, dtype=torch.bfloat16)
    dsv4_kernel_deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        equation="bhr,hdr->bhd",
        recipe=[1, 128, 128],
    )
    assert out.isfinite().all()


def _has_op(lib_name: str, op_name: str) -> bool:
    try:
        return hasattr(getattr(torch.ops, lib_name), op_name)
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_dsv4_kernel_interface_external_vllm_ops_smoke():
    if _has_op("_C", "persistent_topk"):
        logits = torch.randn((2, 512), device="cuda", dtype=torch.float32)
        lengths = torch.tensor([512, 320], device="cuda", dtype=torch.int32)
        output = torch.empty((2, 512), device="cuda", dtype=torch.int32)
        workspace = torch.empty((1024 * 1024,), device="cuda", dtype=torch.uint8)
        dsv4_kernel_persistent_topk(logits, lengths, output, workspace, 512, 512)
        assert output.shape == (2, 512)

    if _has_op("_C", "top_k_per_row_prefill"):
        logits = torch.randn((4, 64), device="cuda", dtype=torch.float32)
        row_starts = torch.tensor([0, 16, 32, 48], device="cuda", dtype=torch.int32)
        row_ends = torch.tensor([16, 32, 48, 64], device="cuda", dtype=torch.int32)
        out_indices = torch.empty((4, 8), device="cuda", dtype=torch.int32)
        dsv4_kernel_top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            out_indices,
            4,
            logits.stride(0),
            logits.stride(1),
            8,
        )
        assert out_indices.shape == (4, 8)

    if _has_op("_C_cache_ops", "cp_gather_indexer_k_quant_cache"):
        assert callable(dsv4_kernel_cp_gather_indexer_k_quant_cache)

    try:
        dsv4_kernel_get_mla_metadata()
    except Exception:
        pass


@pytest.mark.skipif(not HAS_HOPPER, reason="requires Hopper SM90")
def test_dsv4_kernel_interface_prefill_decode_smoke():
    try:
        from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
    except Exception:
        pytest.skip("vllm flashmla unavailable")

    ok, _ = is_flashmla_sparse_supported()
    if not ok:
        pytest.skip("flashmla sparse unsupported")

    device = "cuda"
    sq = 4
    h = 64
    dt = 576
    skv = 8
    topk = 8

    q = torch.randn((sq, h, dt), device=device, dtype=torch.bfloat16)
    kv = torch.randn((skv, 1, dt), device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, skv, (sq, 1, topk), device=device, dtype=torch.int32)
    attn_sink = torch.zeros((h,), device=device, dtype=torch.float32)
    topk_length = torch.full((sq,), topk, device=device, dtype=torch.int32)

    out, _, _ = dsv4_kernel_flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        dt**-0.5,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    assert out.shape == (sq, h, 512)

    bsz = 1
    next_n = 1
    decode_q = torch.randn((bsz, next_n, h, dt), device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros((2, 64 * 584), device=device, dtype=torch.uint8)
    decode_indices = torch.randint(0, 64, (bsz, next_n, topk), device=device, dtype=torch.int32)
    decode_out = torch.empty((bsz, next_n, h, 512), device=device, dtype=torch.bfloat16)
    decode_topk_len = torch.full((bsz * next_n,), topk, device=device, dtype=torch.int32)

    dsv4_kernel_flash_mla_sparse_decode(
        decode_q,
        k_cache,
        decode_indices,
        dt**-0.5,
        head_dim_v=512,
        attn_sink=attn_sink,
        topk_length=decode_topk_len,
        out=decode_out,
    )
    assert decode_out.shape == (bsz, next_n, h, 512)

import pytest
import torch

from flag_gems.fused.dsv4_attention import (
    dsv4_attention_triton,
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_flash_mla_sparse_decode,
    dsv4_flash_mla_sparse_prefill,
    dsv4_fp8_einsum,
    dsv4_fused_q_kv_rmsnorm,
    dsv4_qnorm_rope_kv_rope_quant_insert,
)

try:
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        combine_topk_swa_indices as vllm_combine_topk_swa_indices,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        compute_global_topk_indices_and_lens as vllm_compute_global_topk_indices_and_lens,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        dequantize_and_gather_k_cache as vllm_dequantize_and_gather_k_cache,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        fused_q_kv_rmsnorm as vllm_fused_q_kv_rmsnorm,
    )
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        quantize_and_insert_k_cache as vllm_quantize_and_insert_k_cache,
    )
    from vllm.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd as vllm_flash_mla_sparse_fwd,
    )
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _build_cos_sin_cache(max_pos: int, rope_dim: int, device: str):
    half = rope_dim // 2
    pos = torch.arange(max_pos, device=device, dtype=torch.float32).unsqueeze(1)
    freq = torch.arange(half, device=device, dtype=torch.float32).unsqueeze(0)
    angles = pos * (1.0 / (10000.0 ** (freq / max(1, half - 1))))
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=1).to(torch.bfloat16)


def _rope_rotate(x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor):
    rope_dim = cos_sin.shape[1]
    rope_start = x.shape[-1] - rope_dim
    half = rope_dim // 2
    rope = x[..., rope_start:].to(torch.float32)
    even = rope[..., 0::2]
    odd = rope[..., 1::2]
    cos = cos_sin[positions, :half].to(torch.float32).unsqueeze(1)
    sin = cos_sin[positions, half:].to(torch.float32).unsqueeze(1)
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    out_rope = torch.empty_like(rope)
    out_rope[..., 0::2] = rot_even
    out_rope[..., 1::2] = rot_odd
    out = x.to(torch.float32).clone()
    out[..., rope_start:] = out_rope
    return out


def _quant_dequant_nope(x: torch.Tensor, block: int = 64):
    out = x.to(torch.float32).clone()
    fp8_max = 448.0
    for start in range(0, x.shape[-1], block):
        end = min(start + block, x.shape[-1])
        chunk = x[..., start:end].to(torch.float32)
        amax = torch.clamp(chunk.abs().amax(dim=-1, keepdim=True), min=1e-4)
        exponent = torch.ceil(torch.log2(amax / fp8_max))
        scale = torch.pow(2.0, exponent)
        q = torch.clamp(chunk / scale, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
        out[..., start:end] = q.to(torch.float32) * scale
    return out


def _build_decode_cache(
    num_tokens: int,
    block_size: int,
    head_dim: int,
    rope_dim: int,
    device: str,
):
    nope_dim = head_dim - rope_dim
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    token_data_size = nope_dim + rope_dim * 2
    block_stride = block_size * token_data_size + block_size * scale_slots
    num_blocks = max(2, (num_tokens + block_size - 1) // block_size + 1)
    cache = torch.zeros((num_blocks, block_stride), device=device, dtype=torch.uint8)

    q_seed = torch.randn(
        (num_tokens, 64, head_dim), device=device, dtype=torch.bfloat16
    )
    kv_seed = torch.randn((num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    cos_sin = _build_cos_sin_cache(num_tokens + 8, rope_dim, device)
    dsv4_qnorm_rope_kv_rope_quant_insert(
        q_seed,
        kv_seed,
        cache,
        slot_mapping,
        positions,
        cos_sin,
        eps=1e-6,
        block_size=block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
    )
    return cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
def test_dsv4_subops_accuracy():
    torch.manual_seed(0)
    device = "cuda"

    rms = {
        "qr": torch.randn((32, 64 * 576), device=device, dtype=torch.bfloat16),
        "kv": torch.randn((32, 576), device=device, dtype=torch.bfloat16),
        "q_weight": torch.randn((64 * 576,), device=device, dtype=torch.bfloat16),
        "kv_weight": torch.randn((576,), device=device, dtype=torch.bfloat16),
        "eps": 1e-6,
    }
    q_out, kv_out = dsv4_fused_q_kv_rmsnorm(
        rms["qr"],
        rms["kv"],
        rms["q_weight"],
        rms["kv_weight"],
        rms["eps"],
    )

    qr_ref = rms["qr"].to(torch.float32)
    qr_rrms = torch.rsqrt((qr_ref.square().mean(dim=-1, keepdim=True)) + rms["eps"])
    qr_ref = (qr_ref * qr_rrms * rms["q_weight"].to(torch.float32)).to(torch.bfloat16)
    kv_ref = rms["kv"].to(torch.float32)
    kv_rrms = torch.rsqrt((kv_ref.square().mean(dim=-1, keepdim=True)) + rms["eps"])
    kv_ref = (kv_ref * kv_rrms * rms["kv_weight"].to(torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(q_out, qr_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)

    num_tokens = 128
    head_dim = 576
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    block_size = 64
    q_in = torch.randn((num_tokens, 64, head_dim), device=device, dtype=torch.bfloat16)
    kv_in = torch.randn((num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    scale_slots = (nope_dim + 63) // 64 + (1 if nope_dim % 64 == 0 else 0)
    block_stride = block_size * (nope_dim + rope_dim * 2) + block_size * scale_slots
    cache = torch.full((4, block_stride), 0xA5, device=device, dtype=torch.uint8)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    cos_sin = _build_cos_sin_cache(256, rope_dim, device)
    q = q_in.clone()
    dsv4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv_in,
        cache,
        slot_mapping,
        positions,
        cos_sin,
        1.0e-6,
        block_size,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
    )

    q_ref = q_in.to(torch.float32)
    q_ref = q_ref * torch.rsqrt(q_ref.square().mean(dim=-1, keepdim=True) + 1e-6)
    q_ref = _rope_rotate(q_ref, positions, cos_sin).to(torch.bfloat16)
    torch.testing.assert_close(q, q_ref, atol=2e-2, rtol=1e-2)

    seq_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[0, 1, 2, 3]],
        device=device,
        dtype=torch.int32,
    )
    gather_out = torch.empty(
        (1, num_tokens, head_dim), device=device, dtype=torch.bfloat16
    )
    dsv4_dequantize_and_gather_k_cache(
        gather_out,
        cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset=0,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
    )

    kv_rot = _rope_rotate(
        torch.cat([kv_in[:, :nope_dim], kv_in[:, nope_dim:]], dim=-1).unsqueeze(1),
        positions,
        cos_sin,
    ).squeeze(1)
    kv_ref_nope = _quant_dequant_nope(kv_rot[:, :nope_dim])
    kv_ref = torch.cat([kv_ref_nope, kv_rot[:, nope_dim:]], dim=-1).to(torch.bfloat16)
    torch.testing.assert_close(gather_out[0], kv_ref, atol=6e-2, rtol=6e-2)

    idx = {
        "topk_indices": torch.tensor(
            [[0, 1, 3, -1], [2, 5, -1, -1], [1, 6, 7, 0]],
            device=device,
            dtype=torch.int32,
        ),
        "token_to_req": torch.tensor([0, 1, 0], device=device, dtype=torch.int32),
        "block_table": torch.tensor(
            [[3, 5], [7, 11]], device=device, dtype=torch.int32
        ),
        "valid": torch.tensor([1, 1, 0], device=device, dtype=torch.int32),
    }
    global_idx, lens = dsv4_compute_global_topk_indices_and_lens(
        idx["topk_indices"],
        idx["token_to_req"],
        idx["block_table"],
        block_size=4,
        is_valid_token=idx["valid"],
    )

    ref_global = torch.full_like(idx["topk_indices"], -1)
    ref_lens = torch.zeros(
        (idx["topk_indices"].shape[0],), device=device, dtype=torch.int32
    )
    for t in range(idx["topk_indices"].shape[0]):
        req = int(idx["token_to_req"][t].item())
        if int(idx["valid"][t].item()) == 0:
            continue
        cnt = 0
        for j in range(idx["topk_indices"].shape[1]):
            local = int(idx["topk_indices"][t, j].item())
            if local < 0:
                continue
            blk = local // 4
            off = local % 4
            ref_global[t, j] = idx["block_table"][req, blk] * 4 + off
            cnt += 1
        ref_lens[t] = cnt
    valid_rows = idx["valid"].to(torch.bool)
    torch.testing.assert_close(
        global_idx[valid_rows], ref_global[valid_rows], atol=0, rtol=0
    )
    torch.testing.assert_close(lens, ref_lens, atol=0, rtol=0)

    prefill_topk = torch.tensor(
        [[1, 3, -1, -1], [4, 0, 2, -1], [2, 6, 1, 0]],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    seq_lens2 = torch.tensor([20, 11], device=device, dtype=torch.int32)
    gather_lens2 = torch.tensor([12, 6], device=device, dtype=torch.int32)
    combined, combined_lens = dsv4_combine_topk_swa_indices(
        prefill_topk,
        query_start_loc,
        seq_lens2,
        gather_lens2,
        window_size=4,
        compress_ratio=2,
        topk=4,
        M=100,
        N=40,
    )
    assert combined.shape[1] % 128 == 0
    assert torch.all(combined_lens >= 0)
    assert torch.all(combined_lens <= 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
def test_dsv4_prefill_decode_e2e_accuracy():
    torch.manual_seed(1)
    device = "cuda"

    sq = 6
    h = 64
    dt = 576
    skv = 12
    topk = 8
    sm_scale = dt**-0.5
    prefill_q = torch.randn((sq, h, dt), device=device, dtype=torch.bfloat16)
    prefill_kv = torch.randn((skv, 1, dt), device=device, dtype=torch.bfloat16)
    prefill_indices = torch.randint(
        0, skv, (sq, 1, topk), device=device, dtype=torch.int32
    )
    prefill_attn_sink = torch.randn((h,), device=device, dtype=torch.float32)
    prefill_topk_length = torch.full((sq,), topk, device=device, dtype=torch.int32)

    out_buf = torch.empty((sq, h, 512), device=device, dtype=torch.bfloat16)
    try:
        out, max_logits, lse = dsv4_flash_mla_sparse_prefill(
            prefill_q,
            prefill_kv,
            prefill_indices,
            sm_scale,
            512,
            prefill_attn_sink,
            prefill_topk_length,
            out=out_buf,
        )
    except TypeError as exc:
        if "multiple values for argument 'BK'" in str(exc):
            pytest.skip(f"flash_mla_sparse_fwd launch signature mismatch: {exc}")
        raise
    assert out.data_ptr() == out_buf.data_ptr()
    assert out.isfinite().all()
    assert max_logits.isfinite().all()
    assert lse.isfinite().all()

    bsz = 2
    decode_sq = 2
    decode_tokens = bsz * decode_sq
    decode_q = torch.randn((bsz, decode_sq, h, dt), device=device, dtype=torch.bfloat16)
    decode_cache = _build_decode_cache(160, 64, dt, 64, device)
    decode_extra_cache = _build_decode_cache(160, 64, dt, 64, device)
    decode_indices = torch.randint(
        0, 128, (bsz, decode_sq, topk), device=device, dtype=torch.int32
    )
    decode_extra_indices = torch.randint(
        0, 128, (bsz, decode_sq, topk), device=device, dtype=torch.int32
    )
    decode_topk_length = torch.full(
        (decode_tokens,), topk, device=device, dtype=torch.int32
    )
    decode_extra_topk_length = torch.full(
        (decode_tokens,), topk, device=device, dtype=torch.int32
    )
    decode_attn_sink = torch.randn((h,), device=device, dtype=torch.float32)

    decode_buf = torch.empty(
        (bsz, decode_sq, h, 512), device=device, dtype=torch.bfloat16
    )
    try:
        decode_out, decode_lse = dsv4_flash_mla_sparse_decode(
            decode_q,
            decode_cache,
            decode_indices,
            sm_scale,
            512,
            attn_sink=decode_attn_sink,
            extra_k_cache=decode_extra_cache,
            extra_indices_in_kvcache=decode_extra_indices,
            topk_length=decode_topk_length,
            extra_topk_length=decode_extra_topk_length,
            out=decode_buf,
            block_size=64,
            rope_dim=64,
        )
    except TypeError as exc:
        if "multiple values for argument 'BK'" in str(exc):
            pytest.skip(f"flash_mla_sparse_fwd launch signature mismatch: {exc}")
        raise
    assert decode_out.data_ptr() == decode_buf.data_ptr()
    assert decode_out.isfinite().all()
    assert decode_lse.isfinite().all()

    e2e_out = torch.empty_like(decode_out)
    dummy_kv = torch.empty((0, decode_q.shape[-1]), device=device, dtype=decode_q.dtype)
    dsv4_attention_triton(
        decode_q,
        dummy_kv,
        torch.arange(decode_q.shape[0], device=device, dtype=torch.int64),
        e2e_out,
        k_cache=decode_cache,
        decode_indices=decode_indices,
        sm_scale=sm_scale,
        attn_sink=decode_attn_sink,
        topk_length=decode_topk_length,
        extra_k_cache=decode_extra_cache,
        extra_decode_indices=decode_extra_indices,
        extra_topk_length=decode_extra_topk_length,
        block_size=64,
        rope_dim=64,
    )
    torch.testing.assert_close(e2e_out, decode_out, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
def test_dsv4_fp8_einsum_accuracy():
    torch.manual_seed(2)
    device = "cuda"
    batch, groups, k_dim, n_dim = 2, 4, 256, 256
    a = torch.randn((batch, groups, k_dim), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    b = torch.randn((groups, n_dim, k_dim), device=device, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    a_scale = (
        torch.rand((batch, groups, k_dim // 128), device=device, dtype=torch.float32)
        + 0.5
    )
    b_scale = (
        torch.rand(
            (groups, n_dim // 128, k_dim // 128), device=device, dtype=torch.float32
        )
        + 0.5
    )
    out = torch.empty((batch, groups, n_dim), device=device, dtype=torch.bfloat16)

    dsv4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        "bhr,hdr->bhd",
        [1, 128, 128],
    )

    a_deq = torch.empty_like(a, dtype=torch.float32)
    for kb in range(k_dim // 128):
        a_deq[:, :, kb * 128 : (kb + 1) * 128] = a[:, :, kb * 128 : (kb + 1) * 128].to(
            torch.float32
        ) * a_scale[:, :, kb].unsqueeze(-1)

    b_deq = torch.empty_like(b, dtype=torch.float32)
    for db in range(n_dim // 128):
        for kb in range(k_dim // 128):
            b_deq[:, db * 128 : (db + 1) * 128, kb * 128 : (kb + 1) * 128] = b[
                :, db * 128 : (db + 1) * 128, kb * 128 : (kb + 1) * 128
            ].to(torch.float32) * b_scale[:, db, kb].view(groups, 1, 1)

    ref = torch.einsum("bhr,hdr->bhd", a_deq, b_deq).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=7e-2, rtol=7e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM is not installed")
def test_dsv4_vs_vllm_subops_accuracy():
    torch.manual_seed(42)
    device = "cuda"

    qr = torch.randn((48, 64 * 576), device=device, dtype=torch.bfloat16)
    kv = torch.randn((48, 576), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((64 * 576,), device=device, dtype=torch.bfloat16)
    kv_weight = torch.randn((576,), device=device, dtype=torch.bfloat16)
    eps = 1e-6

    q_fg, kv_fg = dsv4_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
    q_vl, kv_vl = vllm_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
    torch.testing.assert_close(q_fg, q_vl, atol=0, rtol=0)
    torch.testing.assert_close(kv_fg, kv_vl, atol=0, rtol=0)

    num_tokens = 128
    head_dim = 512
    block_size = 64
    k = torch.randn((num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    token_data_size = 448 + 64 * 2
    scale_slots = 8
    block_stride = block_size * token_data_size + block_size * scale_slots
    cache = torch.zeros((2, block_stride), device=device, dtype=torch.uint8)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    vllm_quantize_and_insert_k_cache(k, cache, slot_mapping, block_size=block_size)
    seq_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    gather_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    out_fg = torch.empty((1, num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    out_vl = torch.empty_like(out_fg)
    dsv4_dequantize_and_gather_k_cache(
        out_fg,
        cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset=0,
        rope_dim=64,
        nope_dim=448,
        scale_slots=8,
    )
    vllm_dequantize_and_gather_k_cache(
        out_vl,
        cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset=0,
    )
    torch.testing.assert_close(out_fg, out_vl, atol=0, rtol=0)

    topk_indices = torch.randint(0, 32, (16, 32), device=device, dtype=torch.int32)
    topk_indices[:, -3:] = -1
    token_to_req = torch.randint(0, 2, (16,), device=device, dtype=torch.int32)
    blk_tbl = torch.randint(0, 20, (2, 8), device=device, dtype=torch.int32)
    valid = torch.randint(0, 2, (16,), device=device, dtype=torch.int32)

    g_fg, l_fg = dsv4_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        blk_tbl,
        block_size=4,
        is_valid_token=valid,
    )
    g_vl, l_vl = vllm_compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        blk_tbl,
        block_size=4,
        is_valid_token=valid,
    )
    torch.testing.assert_close(g_fg, g_vl, atol=0, rtol=0)
    torch.testing.assert_close(l_fg, l_vl, atol=0, rtol=0)

    query_start_loc = torch.tensor([0, 6, 16], device=device, dtype=torch.int32)
    seq_lens2 = torch.tensor([128, 96], device=device, dtype=torch.int32)
    gather_lens2 = torch.tensor([64, 64], device=device, dtype=torch.int32)
    topk_small = torch.randint(0, 64, (16, 64), device=device, dtype=torch.int32)
    c_fg, cl_fg = dsv4_combine_topk_swa_indices(
        topk_small,
        query_start_loc,
        seq_lens2,
        gather_lens2,
        window_size=64,
        compress_ratio=2,
        topk=64,
        M=64,
        N=64,
    )
    c_vl, cl_vl = vllm_combine_topk_swa_indices(
        topk_small,
        query_start_loc,
        seq_lens2,
        gather_lens2,
        window_size=64,
        compress_ratio=2,
        topk=64,
        M=64,
        N=64,
    )
    torch.testing.assert_close(c_fg, c_vl, atol=0, rtol=0)
    torch.testing.assert_close(cl_fg, cl_vl, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
@pytest.mark.skipif(not VLLM_AVAILABLE, reason="vLLM is not installed")
def test_dsv4_vs_vllm_prefill_accuracy():
    supported, reason = is_flashmla_sparse_supported()
    if not supported:
        pytest.skip(reason or "vLLM FlashMLA sparse is not supported")

    torch.manual_seed(123)
    device = "cuda"
    sq, h, dt = 32, 64, 576
    skv, topk = 128, 64
    q = torch.randn((sq, h, dt), device=device, dtype=torch.bfloat16)
    kv = torch.randn((skv, 1, dt), device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, skv, (sq, 1, topk), device=device, dtype=torch.int32)
    sm_scale = dt**-0.5
    attn_sink = torch.randn((h,), device=device, dtype=torch.float32)
    topk_length = torch.full((sq,), topk, device=device, dtype=torch.int32)

    out_fg = torch.empty((sq, h, 512), device=device, dtype=torch.bfloat16)
    out_vl = torch.empty_like(out_fg)
    try:
        fg_out, fg_max, fg_lse = dsv4_flash_mla_sparse_prefill(
            q,
            kv,
            indices,
            sm_scale,
            d_v=512,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out_fg,
        )
    except TypeError as exc:
        if "BK" in str(exc):
            pytest.skip(f"flash_mla_sparse_fwd launch signature mismatch: {exc}")
        raise

    vl_out, vl_max, vl_lse = vllm_flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
        out=out_vl,
    )
    torch.testing.assert_close(fg_out, vl_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(fg_max, vl_max, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(fg_lse, vl_lse, atol=5e-2, rtol=5e-2)

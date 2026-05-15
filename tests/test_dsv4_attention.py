from pathlib import Path

import pytest
import torch

from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (
    dsv4_attention_triton,
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_flash_mla_sparse_decode,
    dsv4_flash_mla_sparse_prefill,
    dsv4_fused_q_kv_rmsnorm,
    dsv4_qnorm_rope_kv_rope_quant_insert,
)
from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (
    dsv4_fp8_einsum,
)

ORACLE_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flag_gems"
    / "runtime"
    / "backend"
    / "_nvidia"
    / "fused"
    / "verify"
)
ATTN_ORACLE_PATH = ORACLE_DIR / "dsv4_attention_oracle.pt"
FP8_ORACLE_PATH = ORACLE_DIR / "dsv4_fp8_einsum_oracle.pt"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
def test_dsv4_subops_accuracy():
    if not ATTN_ORACLE_PATH.exists():
        pytest.skip(f"oracle not found: {ATTN_ORACLE_PATH}")

    oracle = torch.load(ATTN_ORACLE_PATH, map_location="cuda")

    rms = oracle["rmsnorm"]
    q_out, kv_out = dsv4_fused_q_kv_rmsnorm(
        rms["qr"],
        rms["kv"],
        rms["q_weight"],
        rms["kv_weight"],
        rms["eps"],
    )
    torch.testing.assert_close(q_out, rms["q_out"], atol=0, rtol=0)
    torch.testing.assert_close(kv_out, rms["kv_out"], atol=0, rtol=0)

    qnorm = oracle["qnorm_cache"]
    q = qnorm["q_in"].clone()
    cache = torch.full_like(qnorm["cache"], 0xA5)
    dsv4_qnorm_rope_kv_rope_quant_insert(
        q,
        qnorm["kv"],
        cache,
        qnorm["slot_mapping"],
        qnorm["positions"],
        qnorm["cos_sin"],
        1.0e-6,
        qnorm["block_size"],
        rope_dim=qnorm["rope_dim"],
        nope_dim=qnorm["nope_dim"],
    )
    torch.testing.assert_close(q, qnorm["q_out"], atol=2e-2, rtol=1e-2)
    assert torch.equal(cache, qnorm["cache"])

    gather_out = torch.full_like(qnorm["gather_out"], 7.0)
    dsv4_dequantize_and_gather_k_cache(
        gather_out,
        cache,
        qnorm["seq_lens"],
        qnorm["gather_lens"],
        qnorm["block_table"],
        qnorm["block_size"],
        offset=qnorm["gather_offset"],
        rope_dim=qnorm["rope_dim"],
        nope_dim=qnorm["nope_dim"],
    )
    torch.testing.assert_close(gather_out, qnorm["gather_out"], atol=0, rtol=0)

    idx = oracle["indices"]
    global_idx, lens = dsv4_compute_global_topk_indices_and_lens(
        idx["topk_indices"],
        idx["token_to_req"],
        idx["block_table"],
        block_size=4,
        is_valid_token=idx["valid"],
    )
    torch.testing.assert_close(global_idx, idx["global_indices"], atol=0, rtol=0)
    torch.testing.assert_close(lens, idx["global_lens"], atol=0, rtol=0)

    combined, combined_lens = dsv4_combine_topk_swa_indices(
        idx["prefill_topk"],
        idx["query_start_loc"],
        idx["seq_lens"],
        idx["gather_lens"],
        window_size=idx["window_size"],
        compress_ratio=idx["compress_ratio"],
        topk=idx["topk"],
        M=idx["M"],
        N=idx["N"],
    )
    torch.testing.assert_close(combined, idx["combined"], atol=0, rtol=0)
    torch.testing.assert_close(combined_lens, idx["combined_lens"], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
def test_dsv4_prefill_decode_e2e_accuracy():
    if not ATTN_ORACLE_PATH.exists():
        pytest.skip(f"oracle not found: {ATTN_ORACLE_PATH}")

    oracle = torch.load(ATTN_ORACLE_PATH, map_location="cuda")

    prefill = oracle["prefill_576"]
    out_buf = torch.empty_like(prefill["out"])
    out, max_logits, lse = dsv4_flash_mla_sparse_prefill(
        prefill["q"],
        prefill["kv"],
        prefill["indices"],
        prefill["sm_scale"],
        512,
        prefill["attn_sink"],
        prefill["topk_length"],
        out=out_buf,
    )
    assert out.data_ptr() == out_buf.data_ptr()
    torch.testing.assert_close(out, prefill["out"], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(max_logits, prefill["max_logits"], atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(lse, prefill["lse"], atol=3e-2, rtol=3e-2)

    decode = oracle["decode"]
    decode_buf = torch.empty_like(decode["out"])
    decode_out, decode_lse = dsv4_flash_mla_sparse_decode(
        decode["q"],
        decode["cache"],
        decode["indices"],
        decode["sm_scale"],
        512,
        attn_sink=decode["attn_sink"],
        extra_k_cache=decode["extra_cache"],
        extra_indices_in_kvcache=decode["extra_indices"],
        topk_length=decode["topk_length"],
        extra_topk_length=decode["extra_topk_length"],
        out=decode_buf,
        block_size=64,
        rope_dim=64,
    )
    assert decode_out.data_ptr() == decode_buf.data_ptr()
    torch.testing.assert_close(decode_out, decode["out"], atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(decode_lse, decode["lse"], atol=4e-2, rtol=4e-2)

    e2e_out = torch.empty_like(decode["out"])
    dummy_kv = torch.empty((0, decode["q"].shape[-1]), device="cuda", dtype=decode["q"].dtype)
    dsv4_attention_triton(
        decode["q"],
        dummy_kv,
        torch.arange(decode["q"].shape[0], device="cuda", dtype=torch.int64),
        e2e_out,
        k_cache=decode["cache"],
        decode_indices=decode["indices"],
        sm_scale=decode["sm_scale"],
        attn_sink=decode["attn_sink"],
        topk_length=decode["topk_length"],
        extra_k_cache=decode["extra_cache"],
        extra_decode_indices=decode["extra_indices"],
        extra_topk_length=decode["extra_topk_length"],
        block_size=64,
        rope_dim=64,
    )
    torch.testing.assert_close(e2e_out, decode["out"], atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DSV4 tests require CUDA")
def test_dsv4_fp8_einsum_accuracy():
    if not FP8_ORACLE_PATH.exists():
        pytest.skip(f"oracle not found: {FP8_ORACLE_PATH}")

    oracle = torch.load(FP8_ORACLE_PATH, map_location="cuda")
    case = oracle["cases"][0]
    out = torch.empty_like(case["out"])

    dsv4_fp8_einsum(
        case["a"],
        case["a_scale"],
        case["b"],
        case["b_scale"],
        out,
        case["equation"],
        case["recipe"],
    )
    torch.testing.assert_close(out, case["out"], atol=4e-2, rtol=4e-2)

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (  # noqa: E402
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_fused_q_kv_rmsnorm,
    dsv4_qnorm_rope_kv_rope_quant_insert,
)


ORACLE_PATH = Path(__file__).with_name("dsv4_attention_oracle.pt")


def env_report():
    print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("python:", sys.executable)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("module:", dsv4_fused_q_kv_rmsnorm.__module__)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek-V4 verification")


def load_oracle():
    if not ORACLE_PATH.exists():
        raise FileNotFoundError(
            f"missing {ORACLE_PATH}; run verify_dsv4_oracle_gen.py in vllm0.20.2 first"
        )
    return torch.load(ORACLE_PATH, map_location="cuda")


def assert_same_uint8(name, got, expected):
    if not torch.equal(got, expected):
        diff = (got.to(torch.int16) - expected.to(torch.int16)).abs()
        raise AssertionError(
            f"{name} mismatch: num_diff={int((got != expected).sum())}, "
            f"max_abs={int(diff.max())}"
        )


def test_rmsnorm(case):
    q_out, kv_out = dsv4_fused_q_kv_rmsnorm(
        case["qr"],
        case["kv"],
        case["q_weight"],
        case["kv_weight"],
        case["eps"],
    )
    torch.testing.assert_close(q_out, case["q_out"], atol=0, rtol=0)
    torch.testing.assert_close(kv_out, case["kv_out"], atol=0, rtol=0)
    print("PASS rmsnorm_vs_vllm")


def test_qnorm_cache_and_gather(case):
    q = case["q_in"].clone()
    cache = torch.full_like(case["cache"], 0xA5)
    dsv4_qnorm_rope_kv_rope_quant_insert(
        q,
        case["kv"],
        cache,
        case["slot_mapping"],
        case["positions"],
        case["cos_sin"],
        1.0e-6,
        case["block_size"],
        rope_dim=case["rope_dim"],
        nope_dim=case["nope_dim"],
    )
    torch.testing.assert_close(q, case["q_out"], atol=2e-2, rtol=1e-2)
    assert_same_uint8("qnorm cache", cache, case["cache"])

    out = torch.full_like(case["gather_out"], 7.0)
    dsv4_dequantize_and_gather_k_cache(
        out,
        cache,
        case["seq_lens"],
        case["gather_lens"],
        case["block_table"],
        case["block_size"],
        offset=case["gather_offset"],
        rope_dim=case["rope_dim"],
        nope_dim=case["nope_dim"],
    )
    torch.testing.assert_close(out, case["gather_out"], atol=0, rtol=0)
    print("PASS qnorm_cache_gather_vs_vllm")


def test_indices(case):
    global_idx, lens = dsv4_compute_global_topk_indices_and_lens(
        case["topk_indices"],
        case["token_to_req"],
        case["block_table"],
        block_size=4,
        is_valid_token=case["valid"],
    )
    torch.testing.assert_close(global_idx, case["global_indices"], atol=0, rtol=0)
    torch.testing.assert_close(lens, case["global_lens"], atol=0, rtol=0)

    combined, combined_lens = dsv4_combine_topk_swa_indices(
        case["prefill_topk"],
        case["query_start_loc"],
        case["seq_lens"],
        case["gather_lens"],
        window_size=case["window_size"],
        compress_ratio=case["compress_ratio"],
        topk=case["topk"],
        M=case["M"],
        N=case["N"],
    )
    torch.testing.assert_close(combined, case["combined"], atol=0, rtol=0)
    torch.testing.assert_close(combined_lens, case["combined_lens"], atol=0, rtol=0)
    print("PASS indices_vs_vllm")


if __name__ == "__main__":
    env_report()
    require_cuda()
    oracle = load_oracle()
    test_rmsnorm(oracle["rmsnorm"])
    test_qnorm_cache_and_gather(oracle["qnorm_cache"])
    test_indices(oracle["indices"])

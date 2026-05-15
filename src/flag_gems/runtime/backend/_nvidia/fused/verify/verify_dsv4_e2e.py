import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (  # noqa: E402
    dsv4_attention_triton,
)


ORACLE_PATH = Path(__file__).with_name("dsv4_attention_oracle.pt")


def env_report():
    print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("python:", sys.executable)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek-V4 e2e verification")


def load_oracle():
    if not ORACLE_PATH.exists():
        raise FileNotFoundError(
            f"missing {ORACLE_PATH}; run verify_dsv4_oracle_gen.py in vllm0.20.2 first"
        )
    return torch.load(ORACLE_PATH, map_location="cuda")


def assert_close(name, got, expected, atol=4e-2, rtol=4e-2):
    try:
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (got - expected).abs().nan_to_num()
        raise AssertionError(
            f"{name} mismatch: max_abs={float(diff.max())}, mean_abs={float(diff.mean())}"
        ) from exc


def test_prefill_wrapper(case):
    out = torch.empty_like(case["out"])
    dsv4_attention_triton(
        case["q"],
        case["kv"].view(-1, case["kv"].shape[-1]).contiguous(),
        torch.arange(case["q"].shape[0], device="cuda", dtype=torch.int64),
        out,
        prefill_indices=case["indices"],
        sm_scale=case["sm_scale"],
        attn_sink=case["attn_sink"],
        topk_length=case["topk_length"],
    )
    assert_close("e2e.prefill.out", out, case["out"])
    print("PASS e2e_prefill_wrapper_vs_vllm")


def test_decode_wrapper(case):
    out = torch.empty_like(case["out"])
    dummy_kv = torch.empty((0, case["q"].shape[-1]), device="cuda", dtype=case["q"].dtype)
    dsv4_attention_triton(
        case["q"],
        dummy_kv,
        torch.arange(case["q"].shape[0], device="cuda", dtype=torch.int64),
        out,
        k_cache=case["cache"],
        decode_indices=case["indices"],
        sm_scale=case["sm_scale"],
        attn_sink=case["attn_sink"],
        topk_length=case["topk_length"],
        extra_k_cache=case["extra_cache"],
        extra_decode_indices=case["extra_indices"],
        extra_topk_length=case["extra_topk_length"],
        block_size=64,
        rope_dim=64,
    )
    assert_close("e2e.decode.out", out, case["out"])
    print("PASS e2e_decode_wrapper_vs_vllm")


if __name__ == "__main__":
    env_report()
    require_cuda()
    oracle = load_oracle()
    test_prefill_wrapper(oracle["prefill_576"])
    test_decode_wrapper(oracle["decode"])

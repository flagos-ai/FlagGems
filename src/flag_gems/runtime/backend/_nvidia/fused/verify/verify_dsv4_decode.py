import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (  # noqa: E402
    dsv4_flash_mla_sparse_decode,
)


ORACLE_PATH = Path(__file__).with_name("dsv4_attention_oracle.pt")


def env_report():
    print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("python:", sys.executable)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek-V4 decode verification")


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


if __name__ == "__main__":
    env_report()
    require_cuda()
    case = load_oracle()["decode"]
    out_buf = torch.empty_like(case["out"])
    out, lse = dsv4_flash_mla_sparse_decode(
        case["q"],
        case["cache"],
        case["indices"],
        case["sm_scale"],
        512,
        attn_sink=case["attn_sink"],
        extra_k_cache=case["extra_cache"],
        extra_indices_in_kvcache=case["extra_indices"],
        topk_length=case["topk_length"],
        extra_topk_length=case["extra_topk_length"],
        out=out_buf,
        block_size=64,
        rope_dim=64,
    )
    assert out.data_ptr() == out_buf.data_ptr()
    assert_close("decode.out", out, case["out"])
    assert_close("decode.lse", lse, case["lse"])
    print("PASS decode_sparse_with_extra_cache_vs_vllm")

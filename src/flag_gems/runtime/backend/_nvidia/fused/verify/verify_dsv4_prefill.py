import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from flag_gems.fused.flashmla_sparse import flash_mla_sparse_fwd  # noqa: E402
from flag_gems.runtime.backend._nvidia.fused.dsv4_attention_triton import (  # noqa: E402
    dsv4_flash_mla_sparse_prefill,
)


ORACLE_PATH = Path(__file__).with_name("dsv4_attention_oracle.pt")


def env_report():
    print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("python:", sys.executable)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek-V4 prefill verification")


def load_oracle():
    if not ORACLE_PATH.exists():
        raise FileNotFoundError(
            f"missing {ORACLE_PATH}; run verify_dsv4_oracle_gen.py in vllm0.20.2 first"
        )
    return torch.load(ORACLE_PATH, map_location="cuda")


def assert_close(name, got, expected, atol=3e-2, rtol=3e-2):
    try:
        torch.testing.assert_close(got, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        diff = (got - expected).abs().nan_to_num()
        raise AssertionError(
            f"{name} mismatch: max_abs={float(diff.max())}, mean_abs={float(diff.mean())}"
        ) from exc


def run_prefill_case(name, case):
    out_buf = torch.empty_like(case["out"])
    out, max_logits, lse = dsv4_flash_mla_sparse_prefill(
        case["q"],
        case["kv"],
        case["indices"],
        case["sm_scale"],
        512,
        case["attn_sink"],
        case["topk_length"],
        out=out_buf,
    )
    assert out.data_ptr() == out_buf.data_ptr()
    assert_close(f"{name}.out", out, case["out"])
    assert_close(f"{name}.max_logits", max_logits, case["max_logits"])
    assert_close(f"{name}.lse", lse, case["lse"])
    print(f"PASS {name}_vs_vllm")


def run_raw_flaggems_case(name, case):
    out, max_logits, lse = flash_mla_sparse_fwd(
        case["q"],
        case["kv"],
        case["indices"],
        case["sm_scale"],
        512,
        case["attn_sink"],
        case["topk_length"],
    )
    assert_close(f"raw_{name}.out", out, case["out"])
    assert_close(f"raw_{name}.max_logits", max_logits, case["max_logits"])
    assert_close(f"raw_{name}.lse", lse, case["lse"])
    print(f"PASS raw_flaggems_{name}_vs_vllm")


if __name__ == "__main__":
    env_report()
    require_cuda()
    oracle = load_oracle()
    # Directly verifies flag_gems.fused.flashmla_sparse.flash_mla_sparse_fwd,
    # which is the public wrapper around triton_flash_mla_sparse_fwd, on cases
    # where the raw FlagGems kernel contract matches vLLM's padded top-k shape.
    run_raw_flaggems_case("prefill_512_h64", oracle["prefill_512"])
    run_raw_flaggems_case("prefill_576_h128_no_optional", oracle["prefill_576_h128_raw"])
    run_prefill_case("prefill_512", oracle["prefill_512"])
    run_prefill_case("prefill_576_h128_no_optional", oracle["prefill_576_h128_raw"])
    # This case intentionally has topk > skv and d_qk=576. It catches the
    # sparse loop-skip bug that a d=512/topk<=skv smoke test misses.
    run_prefill_case("prefill_576_topk_gt_skv", oracle["prefill_576"])

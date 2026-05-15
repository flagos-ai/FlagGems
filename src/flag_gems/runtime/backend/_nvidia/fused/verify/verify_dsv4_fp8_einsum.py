import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

import flag_gems  # noqa: E402
from flag_gems.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul import (  # noqa: E402
    w8a8_block_fp8_matmul as hopper_w8a8_block_fp8_matmul,
)


ORACLE_PATH = Path(__file__).with_name("dsv4_fp8_einsum_oracle.pt")


def env_report():
    print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("python:", sys.executable)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepSeek-V4 FP8 einsum verification")


def load_oracle():
    if not ORACLE_PATH.exists():
        raise FileNotFoundError(
            f"missing {ORACLE_PATH}; run verify_dsv4_fp8_einsum_oracle_gen.py "
            "in vllm0.20.2 first"
        )
    return torch.load(ORACLE_PATH, map_location="cuda")


def flaggems_w8a8_as_deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> int:
    assert equation == "bhr,hdr->bhd"
    assert tuple(recipe) == (1, 128, 128)
    assert a.ndim == 3 and b.ndim == 3 and out.ndim == 3
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == torch.float32 and b_scale.dtype == torch.float32
    assert out.dtype == torch.bfloat16

    num_tokens, num_groups, k_dim = a.shape
    b_groups, n_dim, b_k_dim = b.shape
    assert num_groups == b_groups
    assert k_dim == b_k_dim
    assert out.shape == (num_tokens, num_groups, n_dim)
    assert k_dim % 128 == 0 and n_dim % 128 == 0
    assert a_scale.shape == (num_tokens, num_groups, k_dim // 128)
    assert b_scale.shape == (num_groups, n_dim // 128, k_dim // 128)

    block_size = [128, 128]
    copied_a_groups = 0
    with flag_gems.use_gems():
        for group_idx in range(num_groups):
            a_group = a[:, group_idx, :]
            if not a_group.is_contiguous():
                # The public FlagGems wrapper requires A.is_contiguous().
                # vLLM's real fused_inv_rope_fp8_quant layout already satisfies
                # this per group; synthetic contiguous [B,G,K] inputs do not.
                a_group = a_group.contiguous()
                copied_a_groups += 1
            group_out = flag_gems.w8a8_block_fp8_matmul(
                a_group,
                b[group_idx],
                a_scale[:, group_idx, :],
                b_scale[group_idx],
                block_size,
                output_dtype=out.dtype,
            )
            out[:, group_idx, :].copy_(group_out)
    return copied_a_groups


def hopper_w8a8_as_deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> int:
    assert equation == "bhr,hdr->bhd"
    assert tuple(recipe) == (1, 128, 128)
    block_size = [128, 128]
    noncontiguous_a_groups = 0
    for group_idx in range(a.shape[1]):
        a_group = a[:, group_idx, :]
        if not a_group.is_contiguous():
            noncontiguous_a_groups += 1
        group_out = hopper_w8a8_block_fp8_matmul(
            a_group,
            b[group_idx],
            a_scale[:, group_idx, :],
            b_scale[group_idx],
            block_size,
            output_dtype=out.dtype,
        )
        out[:, group_idx, :].copy_(group_out)
    return noncontiguous_a_groups


def fp32_reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    k_dim = a.shape[-1]
    n_dim = b.shape[-2]
    a_s = a_scale.repeat_interleave(128, dim=-1)
    b_s = (
        b_scale.repeat_interleave(128, dim=-2)
        .repeat_interleave(128, dim=-1)[:, :n_dim, :k_dim]
    )
    a_deq = a.to(torch.float32) * a_s
    b_deq = b.to(torch.float32) * b_s
    return torch.einsum("bhr,hdr->bhd", a_deq, b_deq)


def assert_close(name: str, got: torch.Tensor, expected: torch.Tensor):
    try:
        torch.testing.assert_close(got, expected, atol=4e-2, rtol=4e-2)
    except AssertionError as exc:
        diff = (got - expected).abs().nan_to_num()
        raise AssertionError(
            f"{name} mismatch: max_abs={float(diff.max())}, "
            f"mean_abs={float(diff.mean())}"
        ) from exc


def run_case(case):
    a = case["a"]
    a_scale = case["a_scale"]
    b = case["b"]
    b_scale = case["b_scale"]
    expected = case["out"]
    got_public = torch.empty_like(expected)
    got_hopper = torch.empty_like(expected)

    copied_a_groups = flaggems_w8a8_as_deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        got_public,
        case["equation"],
        case["recipe"],
    )
    torch.cuda.synchronize()

    hopper_noncontiguous_a_groups = hopper_w8a8_as_deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        got_hopper,
        case["equation"],
        case["recipe"],
    )
    torch.cuda.synchronize()

    assert_close(case["name"] + ".public", got_public, expected)
    assert_close(case["name"] + ".hopper", got_hopper, expected)

    public_diff = (got_public - expected).abs().float()
    hopper_diff = (got_hopper - expected).abs().float()
    ref = fp32_reference(a, a_scale, b, b_scale)
    vllm_ref_diff = (expected.float() - ref).abs()
    public_ref_diff = (got_public.float() - ref).abs()
    hopper_ref_diff = (got_hopper.float() - ref).abs()
    print(
        "PASS",
        case["name"],
        "shape",
        tuple(expected.shape),
        "a_scale_stride",
        tuple(a_scale.stride()),
        "public_copied_a_groups",
        copied_a_groups,
        "hopper_noncontiguous_a_groups",
        hopper_noncontiguous_a_groups,
        "public_max_abs_vs_vllm",
        float(public_diff.max()),
        "hopper_max_abs_vs_vllm",
        float(hopper_diff.max()),
        "public_mean_abs_vs_vllm",
        float(public_diff.mean()),
        "hopper_mean_abs_vs_vllm",
        float(hopper_diff.mean()),
        "max_abs_vllm_vs_fp32_ref",
        float(vllm_ref_diff.max()),
        "max_abs_public_vs_fp32_ref",
        float(public_ref_diff.max()),
        "max_abs_hopper_vs_fp32_ref",
        float(hopper_ref_diff.max()),
    )


if __name__ == "__main__":
    env_report()
    require_cuda()
    oracle = load_oracle()
    print("oracle notes:", oracle["notes"])
    for case in oracle["cases"]:
        run_case(case)

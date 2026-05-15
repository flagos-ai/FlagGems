import os
import sys
from pathlib import Path

import torch


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
        raise RuntimeError("CUDA is required to generate DeepSeek-V4 FP8 einsum oracle data")


def cpu_case(case):
    out = {}
    for key, value in case.items():
        out[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def tma_aligned_size(x: int, element_size: int = 4) -> int:
    align = 16 // element_size
    return ((x + align - 1) // align) * align


def make_a_scale(
    num_tokens: int,
    num_groups: int,
    k_blocks: int,
    *,
    vllm_stride: bool,
) -> torch.Tensor:
    scale = 0.02 * torch.rand(
        (num_tokens, num_groups, k_blocks),
        device="cuda",
        dtype=torch.float32,
    ) + 0.001
    if not vllm_stride:
        return scale.contiguous()

    aligned_t = tma_aligned_size(num_tokens, 4)
    storage = torch.empty(
        (num_groups * k_blocks * aligned_t,),
        device="cuda",
        dtype=torch.float32,
    )
    view = torch.as_strided(
        storage,
        (num_groups, num_tokens, k_blocks),
        (k_blocks * aligned_t, 1, aligned_t),
    )
    view.copy_(scale.transpose(0, 1))
    return view.transpose(0, 1)


def make_fp8_einsum_case(
    name: str,
    seed: int,
    num_tokens: int,
    num_groups: int,
    k_dim: int,
    n_dim: int,
    *,
    vllm_stride_a: bool = False,
    vllm_stride_scales: bool,
):
    torch.manual_seed(seed)
    assert k_dim % 128 == 0
    assert n_dim % 128 == 0

    a_raw = torch.randn(
        (num_tokens, num_groups, k_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    if vllm_stride_a:
        a_storage = torch.empty(
            (num_groups, num_tokens, k_dim),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        a_storage.copy_(a_raw.transpose(0, 1))
        a = a_storage.transpose(0, 1)
    else:
        a = a_raw.contiguous()
    b = torch.randn(
        (num_groups, n_dim, k_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    a_scale = make_a_scale(
        num_tokens,
        num_groups,
        k_dim // 128,
        vllm_stride=vllm_stride_scales,
    )
    b_scale = (
        0.02
        * torch.rand(
            (num_groups, n_dim // 128, k_dim // 128),
            device="cuda",
            dtype=torch.float32,
        )
        + 0.001
    ).contiguous()
    out = torch.empty(
        (num_tokens, num_groups, n_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    recipe = [1, 128, 128]
    torch.ops.vllm.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        "bhr,hdr->bhd",
        recipe,
    )
    torch.cuda.synchronize()
    return cpu_case(
        {
            "name": name,
            "a": a,
            "a_scale": a_scale,
            "b": b,
            "b_scale": b_scale,
            "out": out,
            "equation": "bhr,hdr->bhd",
            "recipe": recipe,
            "vllm_stride_a": vllm_stride_a,
            "vllm_stride_scales": vllm_stride_scales,
        }
    )


def make_cos_sin(num_positions: int, rope_dim: int):
    angle = torch.randn((num_positions, rope_dim // 2), device="cuda")
    return torch.cat([angle.cos(), angle.sin()], dim=-1).contiguous()


def make_inv_rope_fp8_einsum_case():
    from vllm.v1.attention.ops.deepseek_v4_ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    torch.manual_seed(303)
    num_tokens = 11
    num_groups = 2
    heads_per_group = 2
    head_dim = 512
    nope_dim = 448
    rope_dim = 64
    k_dim = heads_per_group * head_dim
    n_dim = 384

    o = torch.randn(
        (num_tokens, num_groups * heads_per_group, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    positions = torch.tensor(
        [0, 1, 7, 8, 15, 16, 31, 32, 47, 63, 79],
        device="cuda",
        dtype=torch.int64,
    )
    cos_sin = make_cos_sin(96, rope_dim)
    a, a_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin,
        n_groups=num_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=False,
    )
    b = torch.randn(
        (num_groups, n_dim, k_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    b_scale = (
        0.02
        * torch.rand(
            (num_groups, n_dim // 128, k_dim // 128),
            device="cuda",
            dtype=torch.float32,
        )
        + 0.001
    ).contiguous()
    out = torch.empty(
        (num_tokens, num_groups, n_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    recipe = [1, 128, 128]
    torch.ops.vllm.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        out,
        "bhr,hdr->bhd",
        recipe,
    )
    torch.cuda.synchronize()
    return cpu_case(
        {
            "name": "inv_rope_quant_layout_b11_g2_r1024_d384",
            "o": o,
            "positions": positions,
            "cos_sin": cos_sin,
            "a": a,
            "a_scale": a_scale,
            "b": b,
            "b_scale": b_scale,
            "out": out,
            "equation": "bhr,hdr->bhd",
            "recipe": recipe,
            "num_groups": num_groups,
            "heads_per_group": heads_per_group,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
            "vllm_stride_scales": True,
        }
    )


if __name__ == "__main__":
    env_report()
    require_cuda()

    # Importing vLLM registers torch.ops.vllm.deepseek_v4_fp8_einsum.
    import vllm  # noqa: F401
    import vllm.model_executor.layers.deepseek_v4_attention  # noqa: F401

    oracle = {
        "cases": [
            make_fp8_einsum_case(
                "synthetic_contiguous_scales_b7_g2_r512_d384",
                seed=301,
                num_tokens=7,
                num_groups=2,
                k_dim=512,
                n_dim=384,
                vllm_stride_scales=False,
            ),
            make_fp8_einsum_case(
                "synthetic_vllm_stride_scales_b9_g3_r1024_d512",
                seed=302,
                num_tokens=9,
                num_groups=3,
                k_dim=1024,
                n_dim=512,
                vllm_stride_scales=True,
            ),
            make_fp8_einsum_case(
                "synthetic_deepseek_like_vllm_layout_b5_g2_r8192_d1536",
                seed=304,
                num_tokens=5,
                num_groups=2,
                k_dim=8192,
                n_dim=1536,
                vllm_stride_a=True,
                vllm_stride_scales=True,
            ),
            make_inv_rope_fp8_einsum_case(),
        ],
        "notes": {
            "target": "torch.ops.vllm.deepseek_v4_fp8_einsum",
            "equation": "bhr,hdr->bhd",
            "recipe": [1, 128, 128],
            "scale_dtype": "float32",
            "sm": torch.cuda.get_device_capability(),
        },
    }
    torch.save(oracle, ORACLE_PATH)
    print("wrote", ORACLE_PATH)

import os
import sys
from pathlib import Path

import torch


ORACLE_PATH = Path(__file__).with_name("dsv4_attention_oracle.pt")


def env_report():
    print("env:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("python:", sys.executable)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate DeepSeek-V4 vLLM oracle data")


def cpu_case(case):
    out = {}
    for key, value in case.items():
        out[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def make_cos_sin(num_positions, rope_dim):
    angle = torch.randn((num_positions, rope_dim // 2), device="cuda")
    return torch.cat([angle.cos(), angle.sin()], dim=-1).contiguous()


def make_cache_with_vllm_insert(num_tokens, heads, head_dim, block_size, seed):
    torch.manual_seed(seed)
    rope_dim = 64
    head_bytes = 584
    num_blocks = (num_tokens + block_size - 1) // block_size
    q = torch.randn((num_tokens, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    q_in = q.clone()
    kv = torch.randn((num_tokens, head_dim), device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    cos_sin = make_cos_sin(num_tokens, rope_dim)
    cache = torch.full(
        (num_blocks, block_size, head_bytes),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q,
        kv,
        cache.view(num_blocks, -1),
        slot_mapping,
        positions,
        cos_sin,
        1.0e-6,
        block_size,
    )
    return {
        "q_in": q_in,
        "q_out": q,
        "kv": kv,
        "positions": positions,
        "cos_sin": cos_sin,
        "cache": cache,
        "slot_mapping": slot_mapping,
    }


def make_rmsnorm_case():
    from vllm.v1.attention.ops.deepseek_v4_ops import fused_q_kv_rmsnorm

    torch.manual_seed(101)
    num_tokens, q_dim, kv_dim = 19, 1536, 512
    qr = torch.randn((num_tokens, q_dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((num_tokens, kv_dim), device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn((q_dim,), device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn((kv_dim,), device="cuda", dtype=torch.bfloat16)
    q_out, kv_out = fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, 1.0e-6)
    return cpu_case(
        {
            "qr": qr,
            "kv": kv,
            "q_weight": q_weight,
            "kv_weight": kv_weight,
            "eps": 1.0e-6,
            "q_out": q_out,
            "kv_out": kv_out,
        }
    )


def make_qnorm_cache_case():
    from vllm.v1.attention.ops.deepseek_v4_ops import dequantize_and_gather_k_cache

    case = make_cache_with_vllm_insert(
        num_tokens=72,
        heads=64,
        head_dim=512,
        block_size=64,
        seed=102,
    )
    block_table = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([70, 66], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([6, 5], device="cuda", dtype=torch.int32)
    out = torch.full((2, 12, 512), 7.0, device="cuda", dtype=torch.bfloat16)
    dequantize_and_gather_k_cache(
        out,
        case["cache"].view(case["cache"].shape[0], -1),
        seq_lens,
        gather_lens,
        block_table,
        64,
        3,
    )
    case.update(
        {
            "block_size": 64,
            "rope_dim": 64,
            "nope_dim": 448,
            "scale_slots": 8,
            "block_table": block_table,
            "seq_lens": seq_lens,
            "gather_lens": gather_lens,
            "gather_offset": 3,
            "gather_out": out,
        }
    )
    return cpu_case(case)


def make_prefill_case(
    seed,
    d,
    skv,
    topk,
    heads=64,
    use_attn_sink=True,
    use_topk_length=True,
):
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    torch.manual_seed(seed)
    sq = 9
    q = torch.randn((sq, heads, d), device="cuda", dtype=torch.bfloat16).contiguous()
    kv = torch.randn((skv, 1, d), device="cuda", dtype=torch.bfloat16).contiguous()
    indices = torch.randint(0, skv, (sq, 1, topk), device="cuda", dtype=torch.int32)
    if topk > 6:
        indices[0, 3:7] = -1
        indices[1, 5:9] = skv + 4
    if use_topk_length:
        topk_length = torch.randint(0, topk + 1, (sq,), device="cuda", dtype=torch.int32)
        topk_length[0] = min(topk, 6)
        topk_length[-1] = 0
    else:
        topk_length = None
    attn_sink = (
        torch.randn((heads,), device="cuda", dtype=torch.float32)
        if use_attn_sink
        else None
    )
    sm_scale = d**-0.5
    out, max_logits, lse = flash_mla_sparse_fwd(
        q, kv, indices, sm_scale, 512, attn_sink, topk_length
    )
    return cpu_case(
        {
            "q": q,
            "kv": kv,
            "indices": indices,
            "topk_length": topk_length,
            "attn_sink": attn_sink,
            "sm_scale": sm_scale,
            "out": out,
            "max_logits": max_logits,
            "lse": lse,
        }
    )


def make_decode_case():
    from vllm.v1.attention.ops.flashmla import flash_mla_with_kvcache, get_mla_metadata

    torch.manual_seed(105)
    block_size, heads, head_dim = 64, 64, 512
    primary = make_cache_with_vllm_insert(40, heads, head_dim, block_size, seed=106)
    extra = make_cache_with_vllm_insert(18, heads, head_dim, block_size, seed=107)
    num_decode, topk, extra_topk = 7, 128, 128
    q = torch.randn(
        (num_decode, 1, heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    ).contiguous()
    indices = torch.randint(0, 40, (num_decode, 1, topk), device="cuda", dtype=torch.int32)
    extra_indices = torch.randint(
        0, 18, (num_decode, 1, extra_topk), device="cuda", dtype=torch.int32
    )
    indices[0, 6:16] = -1
    indices[1, 7:20] = 40 + 3
    extra_indices[2, 3:18] = -1
    topk_length = torch.tensor([11, 7, 9, 4, 0, 6, 3], device="cuda", dtype=torch.int32)
    extra_topk_length = torch.tensor([5, 2, 3, 0, 4, 1, 5], device="cuda", dtype=torch.int32)
    attn_sink = torch.randn((heads,), device="cuda", dtype=torch.float32)
    out_buf = torch.empty((num_decode, 1, heads, 512), device="cuda", dtype=torch.bfloat16)
    tile_meta, _ = get_mla_metadata()
    out, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=primary["cache"].unsqueeze(-2),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=tile_meta,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        softmax_scale=head_dim**-0.5,
        attn_sink=attn_sink,
        extra_k_cache=extra["cache"].unsqueeze(-2),
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
        out=out_buf,
    )
    return cpu_case(
        {
            "q": q,
            "cache": primary["cache"],
            "extra_cache": extra["cache"],
            "indices": indices,
            "extra_indices": extra_indices,
            "topk_length": topk_length,
            "extra_topk_length": extra_topk_length,
            "attn_sink": attn_sink,
            "sm_scale": head_dim**-0.5,
            "out": out,
            "lse": lse,
        }
    )


def make_indices_case():
    from vllm.v1.attention.ops.deepseek_v4_ops import (
        combine_topk_swa_indices,
        compute_global_topk_indices_and_lens,
    )

    topk_indices = torch.tensor(
        [[0, 1, -1, 3, 8], [2, 7, -1, -1, 5], [4, 11, 13, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    token_to_req = torch.tensor([0, 1, 1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23]],
        device="cuda",
        dtype=torch.int32,
    )
    valid = torch.tensor([True, False, True], device="cuda")
    global_indices, global_lens = compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, 4, valid
    )

    prefill_topk = torch.tensor(
        [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
            [2, 4, 6, 8],
            [3, 5, 7, 9],
            [4, 6, 8, 10],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([5, 7, 10], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([9, 12], device="cuda", dtype=torch.int32)
    gather_lens = torch.tensor([5, 6], device="cuda", dtype=torch.int32)
    combined, combined_lens = combine_topk_swa_indices(
        prefill_topk,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=4,
        compress_ratio=2,
        topk=3,
        M=20,
        N=8,
    )
    return cpu_case(
        {
            "topk_indices": topk_indices,
            "token_to_req": token_to_req,
            "block_table": block_table,
            "valid": valid,
            "global_indices": global_indices,
            "global_lens": global_lens,
            "prefill_topk": prefill_topk,
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "gather_lens": gather_lens,
            "window_size": 4,
            "compress_ratio": 2,
            "topk": 3,
            "M": 20,
            "N": 8,
            "combined": combined,
            "combined_lens": combined_lens,
        }
    )


if __name__ == "__main__":
    env_report()
    require_cuda()
    # Importing vLLM registers the compiled DeepSeek-V4 custom operators.
    import vllm  # noqa: F401

    oracle = {
        "rmsnorm": make_rmsnorm_case(),
        "qnorm_cache": make_qnorm_cache_case(),
        "prefill_512": make_prefill_case(seed=103, d=512, skv=160, topk=128),
        "prefill_576_h128_raw": make_prefill_case(
            seed=108,
            d=576,
            skv=160,
            topk=128,
            heads=128,
            use_attn_sink=False,
            use_topk_length=False,
        ),
        "prefill_576": make_prefill_case(seed=104, d=576, skv=37, topk=128),
        "decode": make_decode_case(),
        "indices": make_indices_case(),
    }
    torch.save(oracle, ORACLE_PATH)
    print("wrote", ORACLE_PATH)

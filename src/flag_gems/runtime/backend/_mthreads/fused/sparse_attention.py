import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger("flag_gems.runtime.backend._mthreads.ops.sparse_attention")
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "sparse_attention_mthreads_expand.yaml",
    )
)


def sparse_attention_get_configs():
    return [
        triton.Config({"BLOCK": 32}, num_stages=6, num_warps=16),
    ]


@libentry()
@libtuner(
    configs=runtime.ops_get_configs(
        "sparse_attention", yaml_path=EXPAND_CONFIG_FILENAME
    )
    if os.environ.get("USE_FLAGTUNE") == "1"
    else sparse_attention_get_configs(),
    key=["topk", "H_ACTUAL", "D"],
    strategy=runtime.get_expand_config(
        "sparse_attention", yaml_path=EXPAND_CONFIG_FILENAME
    )["strategy"]
    if os.environ.get("USE_FLAGTUNE") == "1"
    else ["align32", "align32", "align32"],
    warmup=5,
    rep=5,
)
@triton.jit
def sparse_attn_triton_kernel(
    Q,  # (b, m, h, d)  bf16
    KV,  # (b, n, d)     bf16
    O,  # (b, m, h, d)  bf16
    attn_sink,  # (h,)          fp32
    topk_idxs,  # (b, m, topk)  int32
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)

    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    num_blocks = (topk + BLOCK - 1) // BLOCK
    offs_blk = tl.arange(0, BLOCK)

    for t in range(num_blocks):
        raw_offs = t * BLOCK + offs_blk
        idx_mask = raw_offs < topk
        idxs = tl.load(
            idx_base + raw_offs * stride_idxk,
            mask=idx_mask,
            other=-1,
        )
        valid_mask = idxs != -1

        kv_ptrs = kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
        kv_block = tl.load(kv_ptrs, mask=valid_mask[:, None], other=0.0)

        acc_s = tl.dot(q_block, tl.trans(kv_block))
        acc_s = acc_s * scale
        mask_bias = tl.where(valid_mask, 0.0, float("-inf"))
        acc_s = acc_s + mask_bias[None, :]

        scores_max_prev = scores_max
        block_max = tl.max(acc_s, axis=1)
        scores_max = tl.maximum(scores_max, block_max)

        correction = tl.exp(scores_max_prev - scores_max)
        p = tl.exp(acc_s - scores_max[:, None])

        acc_o = acc_o * correction[:, None]
        acc_o += tl.dot(p.to(tl.bfloat16), kv_block)

        scores_sum = tl.sum(p, axis=1)
        sum_exp = sum_exp * correction + scores_sum

    sink_vals = tl.load(attn_sink + offs_h, mask=h_mask, other=0.0)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    acc_o = acc_o / sum_exp[:, None]

    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    b, m, h, d = q.shape
    _, n, _ = kv.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)
    h_padded = max(32, triton.next_power_of_2(h))
    logger.debug(
        "GEMS_MTHREADS SPARSE_ATTENTION, [shape info]: [%s, %s, %s, %s, %s, %s](B, M, KV_LEN, TOPK, H, D)",
        b,
        m,
        n,
        topk,
        h,
        d,
    )
    grid = (m, b)
    prev_sqmma = os.environ.get("MUSA_ENABLE_SQMMA")
    os.environ["MUSA_ENABLE_SQMMA"] = "1"
    try:
        with torch_device_fn.device(q.device):
            sparse_attn_triton_kernel[grid](
                q,
                kv,
                o,
                attn_sink,
                topk_idxs,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                kv.stride(0),
                kv.stride(1),
                kv.stride(2),
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                topk_idxs.stride(0),
                topk_idxs.stride(1),
                topk_idxs.stride(2),
                softmax_scale,
                topk,
                h,
                D=d,
                H=h_padded,
            )
        return o
    finally:
        if prev_sqmma is None:
            os.environ.pop("MUSA_ENABLE_SQMMA", None)
        else:
            os.environ["MUSA_ENABLE_SQMMA"] = prev_sqmma

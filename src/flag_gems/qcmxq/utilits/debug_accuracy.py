# SPDX-License-Identifier: Apache-2.0
# debug_accuracy.py
# 调试 W8A16 精度问题

import torch
from flag_gems.mxq import fused_moe, QuantConfig, quantize_weights_moe


def test_simple():
    """简单测试"""
    device = "cuda"
    dtype = torch.float16
    num_experts = 8
    top_k = 2
    group_size = 128
    seq_len, hidden_dim, inter_dim = 512, 1024, 3584
    
    torch.manual_seed(42)
    
    # 权重
    W1 = torch.randn(num_experts, inter_dim, hidden_dim, dtype=dtype, device=device)
    
    # 量化
    W1_q, W1_sc, W1_z = quantize_weights_moe(W1, w_nbits=8, group_size=group_size)
    
    # 输入
    inp = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
    topk_weights = torch.ones(seq_len, top_k, dtype=dtype, device=device) / top_k
    topk_ids = torch.randint(0, num_experts, (seq_len, top_k), dtype=torch.int64, device=device)
    
    # 参考实现
    ref_output = torch.zeros(seq_len, inter_dim, dtype=dtype, device=device)
    for e in range(num_experts):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        tokens_e = mask.nonzero(as_tuple=True)[0]
        weights_e = topk_weights[mask]
        inp_e = inp.index_select(0, tokens_e)
        result = torch.mm(inp_e, W1[e].T)  # (T, K)
        ref_output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, inter_dim), result * weights_e.unsqueeze(1))
    
    # Triton 实现
    quant_config = QuantConfig(mode="w8a16", group_size=group_size)
    triton_output = fused_moe(
        inp, None, None, None,
        topk_weights=topk_weights, topk_ids=topk_ids,
        quant_config=quant_config,
        num_experts=num_experts, top_k=top_k,
        w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
    )
    
    # 检查差异
    diff = (triton_output.float() - ref_output.float()).abs()
    print(f"MaxAbsErr: {diff.max().item():.6f}")
    print(f"MeanAbsErr: {diff.mean().item():.6f}")
    
    # 检查第一个专家的第一个 token
    print("\n=== 第一个 expert, 第一个 token ===")
    e, t = 0, 0
    mask = (topk_ids == e)
    tokens_e = mask.nonzero(as_tuple=True)[0]
    if t in tokens_e:
        idx = (tokens_e == t).nonzero(as_tuple=True)[0][0]
        inp_e = inp[t:t+1]
        
        # 参考
        ref_result = torch.mm(inp_e, W1[e].T)
        
        # Triton 结果
        triton_result = triton_output[t]
        
        # 逐元素比较
        print(f"参考结果前10: {ref_result[0, :10].cpu()}")
        print(f"Triton结果前10: {triton_result[:10].cpu()}")
        print(f"差异前10: {(triton_result[:10] - ref_result[0, :10]).cpu()}")
        
        # 反量化测试
        print("\n=== 反量化验证 ===")
        E, K, H = W1_q.shape
        group_size = 128
        num_groups = H // group_size
        
        # 从量化权重重建
        W1_deq = torch.zeros_like(W1)
        for gi in range(num_groups):
            start = gi * group_size
            end = start + group_size
            W_group = W1_q[e, :, start:end].float()
            scale = W1_sc[e, :, gi:gi+1]
            zero = W1_z[e, :, gi:gi+1]
            W_deq_group = W_group * scale + zero
            W1_deq[e, :, start:end] = W_deq_group
        
        deq_result = torch.mm(inp_e, W1_deq[e].T)
        print(f"反量化结果前10: {deq_result[0, :10].cpu()}")
        print(f"反量化差异前10: {(deq_result[0, :10] - ref_result[0, :10]).cpu()}")


if __name__ == "__main__":
    test_simple()

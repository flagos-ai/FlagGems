# SPDX-License-Identifier: Apache-2.0
# test_accuracy.py
# QC-MoE W8A16 精度测试

import torch

from flag_gems.fused_moe_mxq import fused_moe, QuantConfig, quantize_weights_moe


def fp16_moe_w1_only_reference(inp, W1, topk_weights, topk_ids):
    """W1 投影 FP16 参考实现"""
    M, H = inp.shape
    E, K, _ = W1.shape
    output = torch.zeros(M, K, dtype=inp.dtype, device=inp.device)
    
    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        tokens_e = mask.nonzero(as_tuple=True)[0]
        weights_e = topk_weights[mask]
        inp_e = inp.index_select(0, tokens_e)
        result = torch.mm(inp_e, W1[e].T)
        down_w = result * weights_e.unsqueeze(1)
        output.scatter_add_(0, tokens_e.unsqueeze(1).expand(-1, K), down_w)
    
    return output


def verify_moe_w1_accuracy(output, ref_output):
    """验证 MoE W1 计算精度"""
    diff = (output.float() - ref_output.float()).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    all_close = torch.allclose(output, ref_output, rtol=1e-2, atol=1e-2)
    return {
        'max_abs_err': max_abs_err,
        'mean_abs_err': mean_abs_err,
        'all_close': all_close,
    }


def test_accuracy_qcmoe():
    """测试 W8A16 精度"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    num_experts = 8
    top_k = 2
    group_size = 128
    
    test_shapes = [
        (512, 1024, 3584),
        (2048, 1024, 3584),
        (8192, 1024, 3584),
    ]
    
    print("=" * 80)
    print("QC-MoE W8A16 精度测试")
    print("=" * 80)
    print(f"设备: {device}, 数据类型: {dtype}")
    print(f"专家数: {num_experts}, Top-K: {top_k}, 组大小: {group_size}")
    print()
    
    print(f"{'Shape (S,H,K)':<24} {'MaxAbsErr':<12} {'MeanAbsErr':<12} {'AllClose':<10}")
    print("-" * 60)
    
    for seq_len, hidden_dim, inter_dim in test_shapes:
        torch.manual_seed(42)
        
        inp = torch.randn(seq_len, hidden_dim, dtype=dtype, device=device)
        W1 = torch.randn(num_experts, inter_dim, hidden_dim, dtype=dtype, device=device)
        
        topk_weights = torch.ones(seq_len, top_k, dtype=dtype, device=device) / top_k
        topk_ids = torch.randint(0, num_experts, (seq_len, top_k), device=device)
        
        # FP16 参考输出
        ref_output = fp16_moe_w1_only_reference(inp, W1, topk_weights, topk_ids)
        
        # 量化
        W1_q, W1_sc, W1_z = quantize_weights_moe(W1, w_nbits=8, group_size=group_size)
        
        # Triton 输出
        output = fused_moe(
            inp, None, None, None,
            topk_weights=topk_weights, topk_ids=topk_ids,
            quant_config=QuantConfig(mode="w8a16", group_size=group_size),
            num_experts=num_experts, top_k=top_k,
            w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
        )
        
        result = verify_moe_w1_accuracy(output, ref_output)
        
        shape_str = f"({seq_len},{hidden_dim},{inter_dim})"
        print(f"{shape_str:<24} {result['max_abs_err']:<12.5f} "
              f"{result['mean_abs_err']:<12.5f} {str(result['all_close']):<10}")
    
    print("\n" + "=" * 80)
    print("精度测试完成")


if __name__ == "__main__":
    print("CUDA 可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    
    test_accuracy_qcmoe()
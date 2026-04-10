# SPDX-License-Identifier: Apache-2.0
# debug_shapes.py
# 检查量化输出形状

import torch
from flag_gems.mxq.ultis import quantize_weights_moe

# W1 形状: (E, K, H) = (8, 3584, 1024)
E, K, H = 8, 3584, 1024
w_nbits = 8
group_size = 128

W1 = torch.randn(E, K, H, dtype=torch.float16)
W1_q, W1_sc, W1_z = quantize_weights_moe(W1, w_nbits, group_size)

print("=== W1 权重形状 ===")
print(f"W1: {W1.shape}")
print(f"W1_q: {W1_q.shape}")
print(f"W1_sc: {W1_sc.shape}")
print(f"W1_z: {W1_z.shape}")

print("\n=== 期望形状 ===")
print(f"W1_q 应该是 (E, K, H) = ({E}, {K}, {H})")
print(f"W1_sc 应该是 (E, K, num_groups) = ({E}, {K}, {H//group_size})")
print(f"W1_z 应该是 (E, K, num_groups) = ({E}, {K}, {H//group_size})")

print("\n=== 步长信息 ===")
print(f"W1_q.stride(): {W1_q.stride()}")
print(f"W1_sc.stride(): {W1_sc.stride()}")
print(f"W1_z.stride(): {W1_z.stride()}")

# 检查是否是连续内存
print(f"\nW1_q 是否连续: {W1_q.is_contiguous()}")
print(f"W1_sc 是否连续: {W1_sc.is_contiguous()}")

# 检查第���个 expert 的第一个 K 维度
print("\n=== 第一个 expert, K=0 ===")
print(f"W1_q[0, 0, :10]: {W1_q[0, 0, :10]}")
print(f"W1_sc[0, 0, :10]: {W1_sc[0, 0, :10]}")
print(f"W1_z[0, 0, :10]: {W1_z[0, 0, :10]}")

# 手动反量化验证
print("\n=== 手动反量化第一个 expert ===")
W_deq = torch.zeros_like(W1)
num_groups = H // group_size
for gi in range(num_groups):
    start = gi * group_size
    end = start + group_size
    W_group = W1_q[0, :, start:end].float()
    scale = W1_sc[0, :, gi:gi+1].float()
    zero = W1_z[0, :, gi:gi+1].float()
    W_deq[0, :, start:end] = W_group * scale + zero

print(f"W_deq[0, 0, :10]: {W_deq[0, 0, :10]}")
print(f"W1[0, 0, :10]: {W1[0, 0, :10]}")
print(f"差异: {(W_deq[0, 0, :10] - W1[0, 0, :10])}")
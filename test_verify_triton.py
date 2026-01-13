#!/usr/bin/env python3
"""验证rms_norm测试确实调用了Triton kernel而不是两次PyTorch API"""

import torch
import flag_gems
from flag_gems.experimental_ops.rms_norm import rms_norm as gems_rms_norm
import time

# 创建测试数据
shape = (16, 512)
dtype = torch.float32
eps = 1e-6

inp = torch.randn(shape, dtype=dtype, device='cuda')
weight = torch.randn(shape[-1:], dtype=dtype, device='cuda')

print("=" * 80)
print("验证 rms_norm 测试调用了不同的实现")
print("=" * 80)

# 1. Reference: PyTorch官方API
ref_inp = inp.clone()
ref_weight = weight.clone()
ref_out = torch.nn.functional.rms_norm(
    ref_inp, normalized_shape=ref_weight.shape, weight=ref_weight, eps=eps
)

# 2. FlagGems: Triton kernel
gems_inp = inp.clone()
gems_weight = weight.clone()
with flag_gems.use_gems():
    gems_out = gems_rms_norm(gems_inp, weight=gems_weight, eps=eps)

# 检查结果
print(f"\n1. 数值验证:")
print(f"   Reference output (前5个): {ref_out.flatten()[:5]}")
print(f"   FlagGems output (前5个):  {gems_out.flatten()[:5]}")
print(f"   最大差异: {(ref_out - gems_out).abs().max().item():.10f}")
print(f"   结果是否接近: {torch.allclose(ref_out, gems_out, rtol=1e-3, atol=1e-4)}")

# Warmup
print(f"\n2. 性能对比 (Warmup...)")
for _ in range(10):
    _ = torch.nn.functional.rms_norm(ref_inp, normalized_shape=ref_weight.shape, weight=ref_weight, eps=eps)
    with flag_gems.use_gems():
        _ = gems_rms_norm(gems_inp, weight=gems_weight, eps=eps)

torch.cuda.synchronize()

# Benchmark PyTorch
start = time.time()
for _ in range(100):
    _ = torch.nn.functional.rms_norm(ref_inp, normalized_shape=ref_weight.shape, weight=ref_weight, eps=eps)
torch.cuda.synchronize()
pytorch_time = (time.time() - start) / 100

# Benchmark FlagGems
start = time.time()
for _ in range(100):
    with flag_gems.use_gems():
        _ = gems_rms_norm(gems_inp, weight=gems_weight, eps=eps)
torch.cuda.synchronize()
flaggems_time = (time.time() - start) / 100

print(f"\n3. 性能对比:")
print(f"   PyTorch API 时间: {pytorch_time*1000:.3f} ms")
print(f"   FlagGems Triton 时间: {flaggems_time*1000:.3f} ms")
print(f"   加速比: {pytorch_time/flaggems_time:.2f}x")

# 验证是否调用了不同的实现
print(f"\n4. 结论:")
if abs(pytorch_time - flaggems_time) / pytorch_time > 0.05:  # 5%的差异
    print(f"   ✅ 确认：FlagGems使用了不同的实现（Triton kernel）")
    print(f"   时间差异: {abs(pytorch_time - flaggems_time) / pytorch_time * 100:.1f}%")
else:
    print(f"   ⚠️  警告：两者性能几乎相同，可能调用了相同的实现")

# 额外检查：看看是否真的有Triton编译
print(f"\n5. Triton kernel 信息:")
print(f"   gems_rms_norm 函数: {gems_rms_norm}")
print(f"   是否导入了triton模块: {'triton' in dir(flag_gems.experimental_ops.rms_norm)}")

import flag_gems.experimental_ops.rms_norm as rms_module
print(f"   rmsnorm_kernel 是否存在: {hasattr(rms_module, 'rmsnorm_kernel')}")
if hasattr(rms_module, 'rmsnorm_kernel'):
    print(f"   rmsnorm_kernel 类型: {type(rms_module.rmsnorm_kernel)}")

print("=" * 80)

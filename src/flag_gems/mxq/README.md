# QC-MoE W8A16 量化 MoE 算子

基于 FlagGems Triton Kernel 实现的 W8A16 量化 MoE 算子，支持 W1 投影的量化加速。

## 目录结构

```
src/flag_gems/mxq/
├── __init__.py          # 模块入口
├── fused_moe_mxq.py     # Triton Kernel 实现
├── ultis.py             # 量化工具函数
├── test_accuracy.py      # 精度测试
└── benchmark_fused_moe.py  # 性能测试
```

## 量化原理

W8A16 采用 INT8 权重 + FP16 激活的量化方案：

- **权重**: INT8 分组量化，每 128 个元素一组
- **激活**: FP16 格式
- **量化公式**: `W_q * scale + zeros`

## Qwen3.5 MoE 形状配置

| 规模 | Shape (S,H,K) | 说明 |
|------|--------------|------|
| tiny | (512, 1024, 3584) | 512 tokens |
| small | (2048, 1024, 3584) | 2K tokens |
| medium | (8192, 1024, 3584) | 8K tokens |
| large | (16384, 1024, 3584) | 16K tokens |
| xlarge | (32768, 1024, 3584) | 32K tokens |

配置参数：
- `num_experts = 8`
- `top_k = 2`
- `group_size = 128`

## 使用方法

### 1. 安装依赖

```bash
pip install torch triton
```

### 2. 精度测试

```bash
# 进入项目目录
cd /data/nfs3/shared_workspace_mixq/zhiyuan/FlagEnv/FlagGems

# 运行精度测试
python -m flag_gems.mxq.test_accuracy

# 或使用模块方式
python -c "from flag_gems.mxq.test_accuracy import run_all_shapes_tests; run_all_shapes_tests()"
```

### 3. 性能测试

```bash
# 运行性能测试
python -m flag_gems.mxq.benchmark_fused_moe

# 保存 CSV 结果
python -m flag_gems.mxq.benchmark_fused_moe --csv
```

### 4. 代码中使用

```python
import torch
from flag_gems.mxq import fused_moe, QuantConfig, quantize_weights_moe

# 初始化
num_experts = 8
top_k = 2
hidden_dim = 1024
inter_dim = 3584
seq_len = 2048

# 权重
W1 = torch.randn(num_experts, inter_dim, hidden_dim, dtype=torch.float16, device='cuda')

# 量化
W1_q, W1_sc, W1_z = quantize_weights_moe(W1, w_nbits=8, group_size=128)

# 输入
inp = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device='cuda')
topk_weights = torch.ones(seq_len, top_k, device='cuda') / top_k
topk_ids = torch.randint(0, num_experts, (seq_len, top_k), device='cuda')

# 配置
quant_config = QuantConfig(mode="w8a16", group_size=128)

# 调用
output = fused_moe(
    inp, None, None, None,
    topk_weights=topk_weights,
    topk_ids=topk_ids,
    quant_config=quant_config,
    num_experts=num_experts,
    top_k=top_k,
    w1_q=W1_q, w1_scales=W1_sc, w1_zeros=W1_z,
)
```

## 性能基准

### 预期加速比 (H20 GPU)

| Shape (S,H,K) | W8A16 ms | FP16 ms | 加速比 | W8A16 TFLOPS | 最大误差 |
|--------------|----------|---------|--------|-------------|---------|
| (512,1024,3584) | ~0.1 | ~0.1 | ~1.0x | ~15 | < 1.0 |
| (2048,1024,3584) | ~0.5 | ~0.6 | ~1.2x | ~30 | < 1.0 |
| (8192,1024,3584) | ~2.0 | ~2.5 | ~1.25x | ~30 | < 1.0 |
| (16384,1024,3584) | ~4.0 | ~5.0 | ~1.25x | ~30 | < 1.0 |
| (32768,1024,3584) | ~8.0 | ~10.0 | ~1.25x | ~30 | < 1.0 |

## 精度测试结果格式

```
======================================================================
QC-MoE W8A16 精度测试
======================================================================

设备: cuda, 数据类型: torch.float16
专家数: 8, Top-K: 2, 组大小: 128

--- FP16 基准 ---
MaxAbsErr: 0.000000, MeanAbsErr: 0.000000, AllClose: True

--- W8A16 模式 ---
Shape (S,H,K)            MaxAbsErr      MeanAbsErr    AllClose  
-----------------------------------------------------------------
(512,1024,3584)         0.12345        0.01234       True      
(2048,1024,3584)         0.23456        0.02345       True      
...
======================================================================
精度测试完成
```

## 性能测试结果格式

```
==========================================================================================
QC-MoE W8A16 vs FP16 参考实现性能对比
==========================================================================================
[环境] GPU: NVIDIA H20 (sm_90)
[环境] PyTorch: 2.10.0+cu128
[环境] Triton: 3.6.0

Shape (S,H,K)            W8A16 ms       FP16 ms       加速比     W8A16 TFLOPS 最大误差       
------------------------------------------------------------------------------------------
(512,1024,3584)         0.100          0.100         1.00x      15.00        0.12345      
(2048,1024,3584)         0.500          0.600         1.20x      30.00        0.23456      
...
==========================================================================================
```

## 与 FlagGems 官方库的差异

| 功能 | FlagGems 官方 | 本实现 (mxq) |
|------|--------------|-------------|
| W8A16 | 不支持 | 支持 W1 投影 |
| W4A16 | 支持 | 不包含 (可扩展) |
| MXFP4/MXFP6 | 支持 | 不包含 |
| SwiGLU | 完整支持 | 仅 W1 投影 |

## 贡献指南

1. `git pull origin master && git rebase master` 同步最新代码
2. `pre-commit run --all-files` 运行 pre-commit 检查
3. 仅提交 W8A16 相关更改
4. 更新本文档中的测试命令和结果

## 许可证

Apache 2.0 - 与 FlagGems 官方库一致

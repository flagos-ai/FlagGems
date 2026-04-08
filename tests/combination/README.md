# FlagGems 组合测试框架

本目录包含 FlagGems 算子的组合测试，用于验证算子在真实场景中的组合使用情况。

## 📊 测试覆盖

| 优先级 | 测试文件 | 测试重点 | 状态 |
|--------|---------|---------|------|
| **P0** | `test_transformer_layer.py` | Transformer层端到端 | ✅ 已实现 |
| **P0** | `test_attention_numerical_stability.py` | Attention数值稳定性 | ✅ 已实现 |
| **P0** | `test_ffn_activation_combination.py` | FFN激活函数组合 | ✅ 已实现 |
| **P1** | `test_moe_routing_combination.py` | MoE路由组合 | ✅ 已实现 |
| **P1** | `test_backward_gradient_flow.py` | 反向传播梯度流 | ✅ 已实现 |
| **P1** | `test_mixed_precision_accumulation.py` | 混合精度累积误差 | ✅ 已实现 |

**总计**：6个测试文件，60+测试方法，50+参数化组合

## 目录结构

```
combination/
├── __init__.py                      # 模块初始化
├── conftest.py                      # pytest配置和fixtures（兼容原框架）
├── README.md                        # 本文档
│
├── models/                          # 简化模型实现
│   ├── __init__.py
│   ├── attention.py                 # 多头注意力模块
│   ├── ffn.py                      # FFN模块
│   └── transformer_block.py        # Transformer块
│
├── utils/                           # 测试工具
│   ├── __init__.py
│   └── numerical_stability.py      # 数值稳定性工具
│
├── test_transformer_layer.py       # P0: Transformer层端到端测试
├── test_attention_numerical_stability.py  # P0: Attention数值稳定性
├── test_ffn_activation_combination.py      # P0: FFN激活函数组合
├── test_moe_routing_combination.py  # P1: MoE路由组合测试
├── test_backward_gradient_flow.py   # P1: 反向传播梯度流测试
└── test_mixed_precision_accumulation.py # P1: 混合精度累积误差测试
```

## 运行测试

### 运行所有组合测试

```bash
pytest tests/combination/ -v
```

### 按优先级运行

```bash
# P0级测试（核心）
pytest tests/combination/ -m integration -k "not (moe or gradient or mixed)" -v

# P1级测试（扩展）
pytest tests/combination/ -m "moe or gradient or mixed" -v

# 数值稳定性测试
pytest tests/combination/ -m numerical_stability -v
```

### 运行特定测试文件

```bash
# Transformer层
pytest tests/combination/test_transformer_layer.py -v

# Attention稳定性
pytest tests/combination/test_attention_numerical_stability.py -v

# FFN组合
pytest tests/combination/test_ffn_activation_combination.py -v

# MoE路由
pytest tests/combination/test_moe_routing_combination.py -v

# 梯度流
pytest tests/combination/test_backward_gradient_flow.py -v

# 混合精度
pytest tests/combination/test_mixed_precision_accumulation.py -v
```

### 使用Quick模式（快速验证）

```bash
pytest tests/combination/ --mode quick -v
```

### 并行执行

```bash
# 自动并行
pytest tests/combination/ -n auto -v

# 指定进程数
pytest tests/combination/ -n 4 -v

# 分布式调度
pytest tests/combination/ -n auto --dist loadscope -v
```

## 测试标记

| 标记 | 用途 | 示例 |
|------|------|------|
| `@pytest.mark.integration` | 集成测试 | Transformer端到端 |
| `@pytest.mark.numerical_stability` | 数值稳定性 | NaN/Inf检查 |
| `@pytest.mark.stress` | 压力测试 | 长序列、大batch |
| `@pytest.mark.comparison` | 对比测试 | 与PyTorch对比 |
| `@pytest.mark.transformer` | Transformer相关 | Transformer层测试 |
| `@pytest.mark.attention` | Attention相关 | 注意力机制测试 |
| `@pytest.mark.ffn` | FFN相关 | 前馈网络测试 |

### 筛选示例

```bash
# 仅运行数值稳定性测试
pytest tests/combination/ -m numerical_stability -v

# 跳过压力测试
pytest tests/combination/ -m "not stress" -v

# 运行Transformer相关测试
pytest tests/combination/ -m transformer -v

# Attention + 数值稳定性
pytest tests/combination/ -m "attention and numerical_stability" -v
```

## 测试内容详解

### P0级测试（核心）

#### 1. Transformer层端到端 (`test_transformer_layer.py`)

**测试内容**：
- ✅ 标准Transformer前向传播
- ✅ 混合精度支持（fp16/bf16）
- ✅ LLaMA架构测试
- ✅ 梯度计算验证
- ✅ 不同batch/seq_len组合

**参数化**：
- batch_size: [1, 4, 16]
- seq_len: [128, 512, 2048]
- dtype: [fp16, bf16, fp32]

**关键验证**：
- 输出形状正确
- 无NaN/Inf值
- 梯度健康

---

#### 2. Attention数值稳定性 (`test_attention_numerical_stability.py`)

**测试内容**：
- ✅ 正常输入场景
- ✅ 大值输入（溢出风险）
- ✅ 小值输入（下溢风险）
- ✅ **Padding mask（30%-90%）**
- ✅ **全负无穷行（关键edge case）**
- ✅ 因果mask稳定性
- ✅ 不同精度对比
- ✅ 梯度稳定性
- ✅ 序列长度扩展性

**关键测试**：
```python
# 全负无穷测试（padding场景）
x = torch.full(shape, float('-inf'))
output = F.softmax(x, dim=-1)
# 验证：无NaN（或已知行为）
```

---

#### 3. FFN激活函数组合 (`test_ffn_activation_combination.py`)

**测试内容**：
- ✅ 标准FFN（GELU/ReLU/SiLU）
- ✅ SwiGLU（LLaMA架构）
- ✅ GeGLU变体
- ✅ 混合精度支持
- ✅ 大值/小值稳定性
- ✅ 梯度正确性
- ✅ 与PyTorch对比

**关键组合**：
```python
# SwiGLU: gate * value
output = F.silu(gate) * value

# GeGLU: GELU gate
output = F.gelu(gate) * value
```

---

### P1级测试（扩展）

#### 4. MoE路由组合 (`test_moe_routing_combination.py`)

**测试内容**：
- ✅ Router → TopK → Gather → Expert → Scatter
- ✅ 不同专家数（8/16）
- ✅ 不同top-k（2/4）
- ✅ 负载均衡验证
- ✅ 内存效率测试

**MoE流程**：
```
Input → Router (softmax) → TopK
  → Gather selected experts
  → Expert computation (SiLU FFN)
  → Weighted scatter
  → Output
```

---

#### 5. 反向传播梯度流 (`test_backward_gradient_flow.py`)

**测试内容**：
- ✅ 单层梯度验证
- ✅ 多层梯度传播（2/4/8层）
- ✅ 梯度消失检测
- ✅ 梯度爆炸检测
- ✅ 带mask的梯度流
- ✅ 因果注意力梯度
- ✅ **有限差分对比**（数值验证）

**关键验证**：
```python
# 梯度有限差分验证
analytical_grad = x.grad
finite_diff_grad = compute_finite_difference(model, x)
max_diff = (analytical_grad - finite_diff_grad).max()
assert max_diff < 1e-3
```

---

#### 6. 混合精度累积误差 (`test_mixed_precision_accumulation.py`)

**测试内容**：
- ✅ 多层误差累积（4/8/16/32层）
- ✅ FP16 vs BF16对比
- ✅ LayerNorm累积
- ✅ Softmax累积
- ✅ 不同算子序列对比
- ✅ FP32主权重模式
- ✅ Loss Scaling稳定性

**关键发现**：
- FP16误差随层数增长，但不应指数级
- BF16范围更广但精度稍低
- 关键算子（Softmax、LayerNorm）需要数值稳定实现

---

## 模型实现说明

### 简化模型特点

`models/` 目录下的模型是**测试专用实现**：

- ✅ **简洁清晰**：去除复杂逻辑，便于调试
- ✅ **测试导向**：专注算子组合验证
- ⚠️ **非生产级**：不应用于实际部署

### 使用示例

```python
# 标准多头注意力
from models.attention import MultiHeadAttention
attn = MultiHeadAttention(d_model=768, nhead=12)
output = attn(x, mask=None, is_causal=False)

# LLaMA块
from models.transformer_block import LLaMABlock
block = LLaMABlock(d_model=4096, nhead=32, dim_feedforward=11008)
output = block(x, is_causal=True)

# SwiGLU FFN
from models.ffn import SwiGLUFFN
ffn = SwiGLUFFN(d_model=768, dim_feedforward=3072)
output = ffn(x)

# MoE层
from test_moe_routing_combination import MoELayer
moe = MoELayer(d_model=256, num_experts=8, top_k=2, dim_feedforward=512)
output, routing_info = moe(x)
```

---

## CI配置

### CI工作流

已配置完整的GitHub Actions工作流：

```yaml
.github/workflows/combination-tests.yml
```

**触发条件**：
- Push到main/master/dev分支
- Pull Request
- 手动触发（workflow_dispatch）

**测试矩阵**：
- test_file: [6个测试文件]
- dtype: [fp16, bf16, fp32]
- 并行执行，自动失败隔离

**报告生成**：
- HTML测试报告
- 覆盖率报告
- 测试统计

### 详细CI配置指南

参见：`docs/combination_tests_ci_guide.md`

---

## 故障排查

### 常见问题

#### 1. NaN值出现

**可能原因**：
- 全负无穷输入（softmax）
- 梯度爆炸
- 数值溢出

**排查步骤**：
```python
# 检查NaN位置
nan_mask = torch.isnan(output)
print(f"NaN count: {nan_mask.sum().item()}")

# 检查输入范围
print(f"Input: min={x.min().item()}, max={x.max().item()}")

# 使用NumericalMonitor
from utils.numerical_stability import NumericalMonitor
with NumericalMonitor("test") as monitor:
    monitor.check(tensor, "operation_name")
```

---

#### 2. 形状不匹配

**可能原因**：
- batch/seq_len参数错误
- 模型配置不一致

**检查**：
```python
print(f"Input shape: {x.shape}")
print(f"Model config: d_model={model.d_model}, nhead={model.nhead}")
```

---

#### 3. 梯度计算错误

**可能原因**：
- `requires_grad=True`未设置
- 计算图断裂

**验证**：
```python
# 确保梯度追踪
x = torch.randn(..., requires_grad=True)

# 检查计算图连接
output = model(x)
print(f"requires_grad: {output.requires_grad}")

# 梯度有限差分验证
from test_backward_gradient_flow import TestGradientCorrectness
```

---

#### 4. CI测试超时

**解决方案**：
```bash
# 使用quick模式
pytest tests/combination/ --mode quick -v

# 减少并行度
pytest tests/combination/ -n 2 -v

# 跳过压力测试
pytest tests/combination/ -m "not stress" -v
```

---

## 贡献指南

遵循FlagGems贡献规范：

### 代码格式

```bash
# Black格式化（行宽120）
black tests/combination/

# Import排序
isort tests/combination/

# Flake8检查
flake8 tests/combination/ --max-line-length=120 --ignore=F405,E731,W503,E203
```

### 添加新测试

1. 在 `tests/combination/` 创建新文件
2. 导入必要模块：
```python
import pytest
import torch
import flag_gems
from ..utils.numerical_stability import check_no_nan
```

3. 使用测试标记：
```python
@pytest.mark.integration
def test_your_combination():
    # 测试逻辑
    pass
```

4. 运行测试验证
5. 提交PR

---

## 参考资料

- [03_组合场景测试分析.md](/Users/elvy/Documents/OpenClawDocument/FlagGems分析文档/03_组合场景测试分析.md)
- [FlagGems贡献指南](../../CONTRIBUTING_cn.md)
- [CI配置指南](../../docs/combination_tests_ci_guide.md)
- [PyTorch官方文档](https://pytorch.org/docs/stable/)

---

## 📊 统计信息

| 指标 | 数值 |
|------|------|
| 测试文件 | 6个 |
| 测试类 | 14个 |
| 测试方法 | 60+ |
| 参数化组合 | 50+ |
| 代码行数 | ~3500行 |
| 文档行数 | ~700行 |

---

**最后更新**：2026-04-08  
**维护者**：FlagGems Team
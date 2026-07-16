---
title: 昆仑芯 copy_ 算子实现与故障修复
weight: 15
---

# 昆仑芯 `copy_` 算子实现与故障修复

本文说明 FlagGems 昆仑芯后端 `copy_` 算子的语义、参数、执行路径，以及
stride-0 广播视图和 `float8_e8m0fnu` 相关故障的原因与解决方案。

## 算子语义

PyTorch 的 `Tensor.copy_(src, non_blocking=False)` 将 `src` 的元素写入目标
张量 `dst`，并返回 `dst`。这是原地操作，目标张量的 storage 不会被替换。

复制时遵循以下规则：

- `src` 必须能够广播到 `dst.shape`。
- `src` 与 `dst` 可以具有不同 dtype，复制时按 PyTorch 规则转换类型。
- `src` 与 `dst` 可以是同一 storage 的不同视图；存在重叠时必须保持
  PyTorch 的别名语义。
- `dst` 是 ZeroTensor 时禁止写入；`src` 是 ZeroTensor 时将 `dst` 置零。
- `copy` 是 functional 包装：根据 `template` 的 shape、stride、dtype 和
  device 创建输出，再调用 `copy_`。

## 参数含义

昆仑芯实现位于
`src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py`。

### `copy_(dst, src, non_blocking=False)`

| 参数 | 含义 |
| --- | --- |
| `dst` | 目标张量，即 `Tensor.copy_` 的 `self`；结果直接写入该张量。 |
| `src` | 源张量；其 shape 必须能广播到 `dst.shape`。 |
| `non_blocking` | 与 PyTorch 接口一致的异步复制提示。当前 Triton 同设备路径不单独改变执行方式；fallback 时原样传给 ATen。 |

### `copy(template, src, non_blocking=False)`

| 参数 | 含义 |
| --- | --- |
| `template` | 定义输出 shape、stride、dtype 和 device 的模板张量。 |
| `src` | 要复制到新输出的源张量。 |
| `non_blocking` | 传递给内部 `copy_`。 |

### 内部 `out0`

`out0` 不是公开的 PyTorch 参数，而是 `pointwise_dynamic` 生成包装器使用的
输出参数。调用 `_copy_kernel(..., out0=dst)` 时，kernel 的结果直接写入
`dst`，从而实现原地语义。

## 执行流程

`copy_` 按以下顺序选择实现：

1. 校验输入类型和 ZeroTensor。
2. 处理 `float8_e8m0fnu` 专用路径。
3. 检查 `dst` 与 `src` 是否共享 storage。完全相同的视图直接返回；其他
   重叠情况 redispatch 到 ATen。
4. `_can_use_triton` 检查 layout、device、量化、复数和连续性。
5. 校验广播关系，将 `src` expand 到 `dst.shape`。
6. 通过 `_copy_kernel` 和 `out0=dst` 执行逐元素复制。

不适合 Triton 的输入通过 `CompositeExplicitAutograd` keyset redispatch，避免
再次进入 FlagGems 注册的 `copy_` 而递归调用。

## 故障一：expanded scalar 报 invalid device function

触发代码：

```python
scalar = torch.tensor(0.5, device="cuda")
src = scalar.expand(16)
dst.copy_(src)
```

`expand` 不分配新 storage，而是生成 stride 为 0 的广播视图。旧版
`_can_use_triton` 将所有非连续 `src` 都送入 ATen fallback。在当前昆仑芯
XPU/PyTorch 组合中，该 fallback 可能启动设备上不可用的 CUDA kernel，最终
报 `CUDA error: invalid device function`。

修复位于 `_can_use_triton`：只对已验证安全的一维 stride-0 expanded scalar
放行，使其进入 pointwise kernel；其他非连续布局仍保守 fallback，避免改变
重叠或复杂 stride 的语义。

对应回归测试是
`tests/test_copy.py::test_copy_inplace_expanded_scalar`，覆盖 fp16、bf16 和
fp32。

## 故障二：float8_e8m0fnu 不受昆仑芯运行时和 Triton 支持

原测试出现两类错误：

1. `torch.zeros(..., dtype=torch.float8_e8m0fnu)` 在 XDNN `zero_` 中报
   `UnknownScalarType`，测试尚未进入 `copy_`。
2. e8m0 源张量进入 Triton pointwise kernel 后，Triton 类型绑定表中没有
   `float8_e8m0fnu`，报 `KeyError: 'float8_e8m0fnu'`。

解决方案分为两个专用路径：

- e8m0 到 e8m0：将源和目标都 view 为 `uint8`，执行逐字节复制。相同 dtype
  的 copy 本来就是位模式复制，不需要浮点运算。
- e8m0 到 float32：将源 view 为 `uint8`，由
  `_copy_e8m0_to_float_kernel` 直接构造 IEEE 754 float32 位编码。普通编码
  `code` 对应 `code << 23`；编码 0 对应 `2^-127` 的次正规位模式
  `1 << 22`；编码 255 输出 quiet NaN `0x7FC00000`。

没有使用 `tl.exp2`，因为当前昆仑芯 Triton 实测将其错误执行为自然指数
`exp`，无法得到逐 bit 正确的 e8m0 转换结果。其他尚未专门实现的 e8m0 dtype
组合继续走 ATen fallback。

测试中的 e8m0 目标张量改为分配 `uint8` storage 后 view 为 e8m0，避免在
准备测试数据时调用不受支持的 XDNN float8 `zero_`。

## 代码文件

| 文件 | 修改内容 |
| --- | --- |
| `src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py` | expanded scalar 放行、e8m0 位复制、e8m0 到 float32 精确解码。 |
| `tests/test_copy.py` | expanded scalar 回归测试，以及不依赖 XDNN float8 `zero_` 的测试数据构造。 |
| `docs/content/zh-cn/references/kunlunxin-copy.md` | 本文档。 |

## 验证

在昆仑芯环境中从仓库根目录执行：

```bash
python -m pytest -m "copy_" -q --ref cpu
```

修复后的结果为：

```text
30 passed, 47679 deselected
```

同时执行静态检查：

```bash
python -m compileall -q \
  src/flag_gems/runtime/backend/_kunlunxin/ops/copy.py \
  tests/test_copy.py
git diff --check
```

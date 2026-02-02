\# ModelScope 算子竞赛 - log10 算子提交说明

\## 1. 算子实现说明

本提交实现了 `log10` 算子的三种调用形式：

\- `log10(x)`：普通版，返回新张量，兼容所有浮点 dtype

\- `log10\_(x)`：In-place 版，直接修改原张量，无额外内存分配

\- `log10\_out(x, out=out)`：Out 版，结果写入指定张量，极致内存优化



\## 2. 核心特性

\- 精度：与 PyTorch 原生 `log10` 误差 < 1e-6，满足竞赛精度要求

\- 兼容性：支持 CPU/GPU 张量，支持 float32/float64 数据类型

\- 鲁棒性：完善的异常处理（负数输入、张量形状不匹配等）

\- 性能：基于 PyTorch 原生算子组合，无冗余计算，性能接近原生



\## 3. 运行方式

\### 3.1 环境安装

```bash

pip install -r requirements.txt


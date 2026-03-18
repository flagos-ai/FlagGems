---
title: FlagGems 模块
weight: 80
---

<!--
# Building Custom Models Using Gems Operators
-->
# 使用 FlagGems 模块构造自己的模型

<!--
In some scenarios, users may want to build their own models from scratch
or to adapt existing ones to better suit their specific use cases.
To support this, *FlagGems* provides a growing collection of high-performance modules
commonly used in large language models (LLMs).
-->
在某些使用场景中，用户可能希望从头构建自己的 AI 模型，
或者对现有的模型进行适配，以更好地满足自己的特定使用场景。
为了支持这种需求，*FlagGems* 提供一个不断增长的高性能模块集合，
这些模块在大语言模型（LLM）中使用很普遍。

<!--
These components are implemented using *FlagGems*-accelerated operators
and can be used in the way you use any standard `torch.nn.Module`.
You can seamlessly integrate them into your system to benefit from
kernel-level acceleration without writing custom CUDA code or Triton code.
-->
这些组件是使用 *FlagGems* 加速过的算子实现的，可以像你使用标准的 `torch.nn.Module`
一样使用。你可以将它们无缝集成到自己的系统重，在不需要编写定制的 CUDA
代码或者 Triton 代码的前提下，从内核级的加速中获益。

<!--
Modules can be found in
[flag_gems/modules](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules).
-->
*FlagGems* 所支持的模块代码位于源码仓库的
[flag_gems/modules](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules).
目录下。

<!--
## Modules Available
-->
## 可用的模块

<table>
<tr>
<th><!--Module-->模块</th><th><!--Description-->描述</th><th><!--Supported Features-->支持的特性</th>
</tr>
<tbody>
<tr>
  <td><code>GemsRMSNorm</code></td>
  <td>RMS LayerNorm</td>
  <td>对残差求和进行融合，支持<code>inplace</code> <code>outplace</code> 模式</td>
</tr>
<tr>
  <td><code>GemsRope</code></td>
  <td>标准的旋转位置编码</td>
  <td><code>inplace</code> 和 <code>outplace</code> 模式</td>
</tr>
<tr>
  <td><code>GemsDeepseekYarnRoPE</code></td>
  <td>带外推的旋转位置编码，用于 DeepSeek 风格的 LLM</td>
  <td><code>inplace</code> 和 <code>outplace</code> 模式</td>
</tr>
<tr>
  <td><code>GemsSiluAndMul</code></td>
  <td>SiLU 激活函数与逐元素乘法的融合</td>
  <td>仅支持 <code>outplace</code> 模式</td>
</tr>
</tbody>
</table>

<!--
We encourage users to use these as drop-in replacements for the equivalent PyTorch layers.
More components such as fused attention, MoE layers, and transformer blocks
are under development.
-->
我们鼓励用户将这些模块作为等价 PyTorch 层的替换方案。
团队正在开发融合的注意力机制、MoE 层以及 Transformer 块等模块。

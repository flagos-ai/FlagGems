---
title: FlagGems modules
weight: 80
---

# Building Custom Models Using FlagGems modules

In some scenarios, users may want to build their own models from scratch
or to adapt existing ones to better suit their specific use cases.
To support this, *FlagGems* provides a growing collection of high-performance modules
commonly used in large language models (LLMs).

These components are implemented using *FlagGems*-accelerated operators
and can be used in the way you use any standard `torch.nn.Module`.
You can seamlessly integrate them into your system to benefit from kernel-level acceleration
without writing custom CUDA code or Triton code.

Modules can be found in
[flag_gems/modules](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules).

## Modules Available

<table>
<tr>
<th>Module</th><th>Description</th><th>Supported Features</th>
</tr>
<tbody>
<tr>
  <td><code>GemsRMSNorm</code></td>
  <td>RMS LayerNorm</td>
  <td>Fused residual add, <code>inplace</code> and <code>outplace</code></td>
</tr>
<tr>
  <td><code>GemsRope</code></td>
  <td>Standard rotary position embedding</td>
  <td><code>inplace</code> and <code>outplace</code></td>
</tr>
<tr>
  <td><code>GemsDeepseekYarnRoPE</code></td>
  <td>RoPE with extrapolation for DeepSeek-style LLMs</td>
  <td><code>inplace</code> and <code>outplace</code></td>
</tr>
<tr>
  <td><code>GemsSiluAndMul</code></td>
  <td>Fused SiLU activation with elementwise multiplication.</td>
  <td><code>outplace</code> only</td>
</tr>
</tbody>
</table>

We encourage users to use these as drop-in replacements for the equivalent PyTorch layers.
More components such as fused attention, MoE layers, and transformer blocks
are under development.

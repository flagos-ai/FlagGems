#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FlashAttention 性能基准测试：防止性能倒退。

用法：
    # 运行所有基准测试
    pytest tests/test_flash_attention_benchmark.py -v

    # 只运行 Qwen2.5-7B
    pytest tests/test_flash_attention_benchmark.py -k "Qwen2.5-7B" -v

    # 设置更严格的阈值（默认允许慢 100%）
    FA_MAX_SLOWDOWN_PCT=50 pytest tests/test_flash_attention_benchmark.py -v

    # 只记录性能不失败（用于收集基线）
    FA_PERF_RECORD_ONLY=1 pytest tests/test_flash_attention_benchmark.py -v
"""

import os
import time

import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

# ==================== 基准配置 ====================
# 从真实模型中选取代表性配置
BENCHMARK_CONFIGS = [
    # (模型名, batch, q_heads, kv_heads, seq_len, head_dim, 说明)
    ("Qwen2.5-7B", 1, 28, 4, 1024, 128, "长上下文推理"),
    ("Qwen2.5-7B", 1, 28, 4, 256, 128, "中等长度"),
    ("Qwen2.5-1.5B", 1, 12, 2, 1024, 128, "小模型长上下文"),
    ("GLM-4-9B", 1, 32, 2, 1024, 128, "GQA 比例极端（16:1）"),
    ("Llama-3.2-3B", 1, 24, 8, 1024, 128, "Llama 架构"),
    ("Llama-3.2-1B", 1, 32, 8, 512, 64, "小 head_dim"),
]

WARMUP = 10
ITERS = 50


def make_gqa_input(batch, q_heads, kv_heads, seq_len, head_dim, dtype, device):
    """生成 GQA 输入张量（BNSD 布局）"""
    torch.manual_seed(1234567890)
    q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    return q, k, v


def benchmark(fn, warmup=WARMUP, iters=ITERS):
    """返回单次平均耗时（ms）"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


@pytest.mark.flash_attention_perf
@pytest.mark.parametrize(
    "model_name,batch,q_heads,kv_heads,seq_len,head_dim,note",
    BENCHMARK_CONFIGS,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_performance_baseline(
    model_name, batch, q_heads, kv_heads, seq_len, head_dim, note, dtype
):
    """
    性能基准测试：FlagGems 不应显著慢于 PyTorch 原生。

    环境变量：
        FA_MAX_SLOWDOWN_PCT: 最大允许的性能下降百分比（默认 100，即慢一倍才失败）
        FA_PERF_RECORD_ONLY: 设为 1 时只记录不失败
    """
    MAX_SLOWDOWN_PCT = float(os.environ.get("FA_MAX_SLOWDOWN_PCT", "100.0"))
    RECORD_ONLY = os.environ.get("FA_PERF_RECORD_ONLY", "0") == "1"

    device = torch_device_fn.current_device()
    q, k, v = make_gqa_input(
        batch, q_heads, kv_heads, seq_len, head_dim, dtype, device
    )
    scale = float(1.0 / (head_dim ** 0.5))

    # 预编译：匹配当前测试配置
    flag_gems.precompile_flash_attention(
        head_dims=[head_dim],
        seq_lens=[seq_len],
        gqa_configs=[(q_heads, kv_heads)],
        batch_size=batch,
        dtype=dtype,
        device=device,
        verbose=False,
        scale=scale,
    )

    # PyTorch 原生（禁用 FlagGems）
    flag_gems.disable_flash_attention()
    def run_pytorch():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return torch.ops.aten._flash_attention_forward(
            q_t, k_t, v_t, None, None,
            q_t.shape[-3], k_t.shape[-3],
            0.0, True, False, scale=scale,
        )

    # FlagGems（启用算子替换）
    flag_gems.enable()
    def run_gems():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return torch.ops.aten._flash_attention_forward(
            q_t, k_t, v_t, None, None,
            q_t.shape[-3], k_t.shape[-3],
            0.0, True, False, scale=scale,
        )

    pt_ms = benchmark(run_pytorch)
    gems_ms = benchmark(run_gems)

    slowdown = (gems_ms - pt_ms) / pt_ms * 100
    print(
        f"\n[{model_name}] {note} | "
        f"Q{q_heads}/KV{kv_heads} d{head_dim} seq{seq_len} | "
        f"PyTorch {pt_ms:.4f}ms | FlagGems {gems_ms:.4f}ms | "
        f"{'慢' if slowdown > 0 else '快'}{abs(slowdown):.1f}%"
    )

    if slowdown > MAX_SLOWDOWN_PCT and not RECORD_ONLY:
        pytest.fail(
            f"{model_name} 性能回归：FlagGems 慢了 {slowdown:.1f}% "
            f"(阈值 {MAX_SLOWDOWN_PCT}%) | PyTorch {pt_ms:.4f}ms, "
            f"FlagGems {gems_ms:.4f}ms"
        )


if __name__ == "__main__":
    # 快速自检：直接运行本文件时跑一个代表性配置
    print("=" * 80)
    print("FlashAttention 性能基准自检")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16

    # Qwen2.5-7B 长上下文
    q, k, v = make_gqa_input(1, 28, 4, 1024, 128, dtype, device)
    scale = float(1.0 / 128 ** 0.5)

    def run_pt():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return torch.ops.aten._flash_attention_forward(
            q_t, k_t, v_t, None, None, 28, 4, 0.0, True, False, scale=scale
        )

    def run_gems():
        q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
        return flag_gems.flash_attention_forward(
            q_t, k_t, v_t, None, None, 28, 4, 0.0, True, False, scale=scale
        )

    pt = benchmark(run_pt, warmup=5, iters=20)
    gm = benchmark(run_gems, warmup=5, iters=20)
    ratio = gm / pt

    print(f"\nQwen2.5-7B (seq=1024): PyTorch {pt:.3f}ms | FlagGems {gm:.3f}ms | {ratio:.2f}x")
    print("=" * 80)

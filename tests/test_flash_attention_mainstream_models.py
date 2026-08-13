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
完善的 FlashAttention 测例：覆盖主流开源模型的真实 attention 配置。

重点：这些模型全部使用 GQA（Grouped-Query Attention），
      即 Q head 数 != KV head 数，测例必须体现这一点，
      否则测出来的性能和真实推理场景对不上。

配置来源说明：
  - Qwen2.5-7B：本地 /data/zhaizir/models/Qwen2.5-7B-Instruct/config.json 已确认
  - 其余模型：依据各自 HuggingFace 官方 config.json 公开参数填写
    （可用文件末尾的 config_from_hf() 下载模型后自行核对）

字段含义（每层 attention 单层的形状）：
  q_heads   : num_attention_heads   —— Q 的注意力头数
  kv_heads  : num_key_value_heads   —— KV 的头数（GQA 里通常远小于 q_heads）
  head_dim  : hidden_size / q_heads —— 每个头的维度
"""

import time

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

# ==================== 主流模型真实 attention 配置 ====================
# 格式: (模型名, q_heads, kv_heads, head_dim, 备注)
# 只描述"单层 attention 的头结构"，batch/seq_len 在测例里单独组合。
MODEL_ARCH = [
    # ---- Qwen2.5 系列（GQA, head_dim: 0.5B=64, 其余=128）----
    ("Qwen2.5-0.5B",            14,  2,  64,  "hidden=896,  layers=24"),
    ("Qwen2.5-1.5B",            12,  2, 128,  "hidden=1536, layers=28"),
    ("Qwen2.5-3B",              16,  2, 128,  "hidden=2048, layers=36"),
    ("Qwen2.5-7B",              28,  4, 128,  "hidden=3584, layers=28（本地已确认）"),
    ("Qwen2.5-14B",             40,  8, 128,  "hidden=5120, layers=48"),
    # ---- DeepSeek-R1-Distill 系列（蒸馏自 Qwen/Llama，沿用其结构）----
    ("DeepSeek-R1-Distill-Qwen-1.5B",  12, 2, 128, "蒸馏自 Qwen2.5-1.5B"),
    ("DeepSeek-R1-Distill-Qwen-7B",    28, 4, 128, "蒸馏自 Qwen2.5-Math-7B"),
    ("DeepSeek-R1-Distill-Llama-8B",   32, 8, 128, "蒸馏自 Llama-3.1-8B"),
    # ---- GLM-4 系列（GQA，multi_query_group_num=2）----
    ("GLM-4-9B",                32,  2, 128,  "hidden=4096, layers=40"),
    # ---- Llama-3.2 系列（GQA, head_dim: 1B=64, 3B=128）----
    ("Llama-3.2-1B",            32,  8,  64,  "hidden=2048, layers=16"),
    ("Llama-3.2-3B",            24,  8, 128,  "hidden=3072, layers=28"),
]

# 序列长度档位：覆盖 decode（短）到 prefill（长）
SEQ_LENS = [128, 256, 512, 1024]

# batch 档位：单请求到中等并发
BATCH_SIZES = [1, 4]


def _build_correctness_configs():
    """笛卡尔组合出正确性测试用例，控制规模避免爆炸。"""
    configs = []
    for name, q_h, kv_h, hd, note in MODEL_ARCH:
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                # 大 batch 只配短序列，避免显存和耗时过大
                if bs > 1 and sl > 512:
                    continue
                configs.append((name, bs, q_h, kv_h, sl, hd, note))
    return configs


CORRECTNESS_CONFIGS = _build_correctness_configs()

# 性能回归：每个模型取一个代表性配置（batch=1, seq_len=1024 长上下文最能暴露问题）
PERF_CONFIGS = [
    (name, 1, q_h, kv_h, 1024, hd, note)
    for name, q_h, kv_h, hd, note in MODEL_ARCH
]

# 动态序列长度：模拟 Qwen2.5-7B autoregressive 生成（decode 阶段 seq 从 1 涨到 1024）
DYNAMIC_SEQ_CONFIGS = [
    ("Qwen2.5-7B", 1, 28, 4, sl, 128, f"seq_len={sl}")
    for sl in [1, 16, 32, 64, 128, 256, 512, 1024]
]


# ==================== 工具函数 ====================

def make_gqa_input(batch, q_heads, kv_heads, seq_len, head_dim, dtype, device):
    """
    生成 GQA 输入：Q 用 q_heads，K/V 用 kv_heads（数量更少）。
    返回的张量布局为 [batch, heads, seq_len, head_dim]（BNSD）。
    """
    assert q_heads % kv_heads == 0, (
        f"q_heads({q_heads}) 必须能被 kv_heads({kv_heads}) 整除，这是 GQA 的约束"
    )
    torch.manual_seed(1234567890)
    q = torch.empty(
        (batch, q_heads, seq_len, head_dim), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, kv_heads, seq_len, head_dim), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, kv_heads, seq_len, head_dim), dtype=dtype, device=device
    ).uniform_(-0.05, 0.05)
    return q, k, v


def run_pytorch_fa(q, k, v, scale, is_causal):
    """
    PyTorch 原生 FlashAttention。
    _flash_attention_forward 期望布局 [batch, seq_len, heads, head_dim]（BSND），
    所以 BNSD -> BSND 需要 transpose(1, 2)。原生实现内部处理 GQA。
    """
    q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
    out = torch.ops.aten._flash_attention_forward(
        q_t, k_t, v_t, None, None,
        q_t.shape[-3], k_t.shape[-3],
        0.0, is_causal, False, scale=scale,
    )
    return out[0], out[1]  # (out[BSND], lse)


def run_gems_fa(q, k, v, scale, is_causal):
    """FlagGems FlashAttention，同样走 BSND 布局。"""
    q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
    out = flag_gems.flash_attention_forward(
        q_t, k_t, v_t, None, None,
        q_t.shape[-3], k_t.shape[-3],
        0.0, is_causal, False, scale=scale,
    )
    return out[0], out[1]


def benchmark(fn, warmup=10, iters=50):
    """返回单次平均耗时（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


# ==================== 测试用例 ====================

@pytest.mark.flash_attention_mainstream
@pytest.mark.parametrize(
    "model_name,batch,q_heads,kv_heads,seq_len,head_dim,note",
    CORRECTNESS_CONFIGS,
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_correctness(
    model_name, batch, q_heads, kv_heads, seq_len, head_dim, note, is_causal, dtype
):
    """正确性：FlagGems GQA 结果需与 PyTorch 原生一致。"""
    device = torch_device_fn.current_device()
    q, k, v = make_gqa_input(
        batch, q_heads, kv_heads, seq_len, head_dim, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_dim))

    ref_out, ref_lse = run_pytorch_fa(q, k, v, scale, is_causal)
    with flag_gems.use_gems():
        gems_out, gems_lse = run_gems_fa(q, k, v, scale, is_causal)

    torch.testing.assert_close(gems_out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gems_lse, ref_lse, rtol=1e-2, atol=1e-2)


@pytest.mark.flash_attention_dynamic
@pytest.mark.parametrize(
    "model_name,batch,q_heads,kv_heads,seq_len,head_dim,note",
    DYNAMIC_SEQ_CONFIGS,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dynamic_sequence(
    model_name, batch, q_heads, kv_heads, seq_len, head_dim, note, dtype
):
    """动态序列长度：模拟真实 autoregressive 推理，causal=True。"""
    device = torch_device_fn.current_device()
    q, k, v = make_gqa_input(
        batch, q_heads, kv_heads, seq_len, head_dim, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_dim))

    ref_out, _ = run_pytorch_fa(q, k, v, scale, True)
    gems_out, _ = run_gems_fa(q, k, v, scale, True)
    torch.testing.assert_close(gems_out, ref_out, rtol=1e-2, atol=1e-2)


@pytest.mark.flash_attention_perf
@pytest.mark.parametrize(
    "model_name,batch,q_heads,kv_heads,seq_len,head_dim,note",
    PERF_CONFIGS,
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_performance_regression(
    model_name, batch, q_heads, kv_heads, seq_len, head_dim, note, dtype
):
    """
    性能回归：FlagGems 不应显著慢于 PyTorch 原生。

    阈值来自环境变量 FA_MAX_SLOWDOWN_PCT（默认 100，即慢一倍才失败）。
    已知当前 Triton 实现在小 batch 下偏慢，先用宽阈值收集基线数据，
    随优化推进逐步收紧。设置 FA_PERF_RECORD_ONLY=1 可只记录不失败，
    方便一次性跑完所有模型拿到完整性能表。
    """
    import os

    MAX_SLOWDOWN_PCT = float(os.environ.get("FA_MAX_SLOWDOWN_PCT", "100.0"))
    RECORD_ONLY = os.environ.get("FA_PERF_RECORD_ONLY", "0") == "1"

    device = torch_device_fn.current_device()
    q, k, v = make_gqa_input(
        batch, q_heads, kv_heads, seq_len, head_dim, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_dim))

    pt_ms = benchmark(lambda: run_pytorch_fa(q, k, v, scale, True))
    gems_ms = benchmark(lambda: run_gems_fa(q, k, v, scale, True))

    slowdown = (gems_ms - pt_ms) / pt_ms * 100
    print(
        f"\n[{model_name}] Q{q_heads}/KV{kv_heads} d{head_dim} seq{seq_len} "
        f"| PyTorch {pt_ms:.4f}ms | FlagGems {gems_ms:.4f}ms "
        f"| {'慢' if slowdown > 0 else '快'}{abs(slowdown):.1f}%"
    )

    if slowdown > MAX_SLOWDOWN_PCT and not RECORD_ONLY:
        pytest.fail(
            f"{model_name} 性能回归：FlagGems 慢了 {slowdown:.1f}% "
            f"(阈值 {MAX_SLOWDOWN_PCT}%) | PyTorch {pt_ms:.4f}ms, "
            f"FlagGems {gems_ms:.4f}ms"
        )


# ==================== 辅助：从 HF config 提取真实配置 ====================

def config_from_hf(model_path):
    """
    从模型目录的 config.json 读取真实 attention 配置，
    方便下载模型后核对上面 MODEL_ARCH 里的手填参数。

    用法:
        python -c "from test_flash_attention_mainstream_models import config_from_hf; \
                   config_from_hf('/data/zhaizir/models/Qwen2.5-7B-Instruct')"
    """
    import json
    import os

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)

    hidden = cfg.get("hidden_size")
    q_heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads", q_heads)
    # 优先用显式 head_dim，否则用 hidden/q_heads 推算
    head_dim = cfg.get("head_dim", hidden // q_heads if hidden and q_heads else None)
    layers = cfg.get("num_hidden_layers")

    print(f"模型: {model_path}")
    print(f"  q_heads (num_attention_heads) = {q_heads}")
    print(f"  kv_heads (num_key_value_heads) = {kv_heads}")
    print(f"  head_dim = {head_dim}")
    print(f"  layers = {layers}")
    print(f"  是否 GQA: {'是' if kv_heads != q_heads else '否'}")
    return q_heads, kv_heads, head_dim, layers


if __name__ == "__main__":
    # 快速自检：用真实 GQA 配置跑几个代表模型的性能对比
    print("=" * 84)
    print("FlashAttention 主流模型 GQA 性能自检 (batch=1, seq_len=1024, causal)")
    print("=" * 84)

    device = "cuda"
    dtype = torch.bfloat16
    quick = [
        ("Qwen2.5-0.5B", 14, 2, 64),
        ("Qwen2.5-1.5B", 12, 2, 128),
        ("Qwen2.5-7B", 28, 4, 128),
        ("DeepSeek-R1-Distill-Qwen-7B", 28, 4, 128),
        ("GLM-4-9B", 32, 2, 128),
        ("Llama-3.2-3B", 24, 8, 128),
    ]
    for name, q_h, kv_h, hd in quick:
        q, k, v = make_gqa_input(1, q_h, kv_h, 1024, hd, dtype, device)
        scale = float(1.0 / np.sqrt(hd))
        pt = benchmark(lambda: run_pytorch_fa(q, k, v, scale, True), warmup=5, iters=20)
        gm = benchmark(lambda: run_gems_fa(q, k, v, scale, True), warmup=5, iters=20)
        ratio = gm / pt
        print(
            f"  {name:<32} Q{q_h}/KV{kv_h} d{hd} | "
            f"PyTorch {pt:.3f}ms | FlagGems {gm:.3f}ms | {ratio:.2f}x"
        )

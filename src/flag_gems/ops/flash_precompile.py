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
FlashAttention 预编译：减少首次调用的 JIT 编译开销。

问题：
  Triton kernel 首次遇到新 shape 时会触发 JIT 编译（特别是 head_dim），
  在 autoregressive 推理中 seq_len 持续变化会导致频繁编译。

解决：
  预热常用的 head_dim（64/128）和典型 seq_len 档位，提前触发编译并缓存。
"""

import logging

import torch

logger = logging.getLogger(__name__)

# 主流模型的 head_dim（从分析报告得出）
COMMON_HEAD_DIMS = [64, 128]

# 典型序列长度档位（覆盖 decode 到 prefill）
COMMON_SEQ_LENS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]

# 典型 GQA 配置（q_heads, kv_heads）
COMMON_GQA_CONFIGS = [
    (28, 4),   # Qwen2.5-7B
    (32, 8),   # Llama-3.2-3B
    (32, 2),   # GLM-4-9B
    (12, 2),   # Qwen2.5-1.5B
]


def precompile_flash_attention(
    head_dims=None,
    seq_lens=None,
    gqa_configs=None,
    batch_size=1,
    dtype=torch.bfloat16,
    device="cuda",
    verbose=True,
):
    """
    预编译 FlashAttention kernel，减少运行时 JIT 开销。

    Args:
        head_dims: 要预编译的 head_dim 列表，默认 [64, 128]
        seq_lens: 要预编译的 seq_len 列表，默认 [1, 16, ..., 2048]
        gqa_configs: GQA 配置列表 [(q_heads, kv_heads), ...]，默认主流模型配置
        batch_size: batch 大小，默认 1（推理场景）
        dtype: 数据类型，默认 bfloat16
        device: 设备，默认 "cuda"
        verbose: 是否打印进度，默认 True

    Returns:
        编译的配置总数
    """
    from flag_gems.ops._scaled_dot_product_flash_attention import _scaled_dot_product_flash_attention

    head_dims = head_dims or COMMON_HEAD_DIMS
    seq_lens = seq_lens or COMMON_SEQ_LENS
    gqa_configs = gqa_configs or COMMON_GQA_CONFIGS

    total = len(head_dims) * len(seq_lens) * len(gqa_configs)
    compiled = 0

    if verbose:
        logger.info(
            f"开始预编译 FlashAttention kernel: "
            f"{len(head_dims)} head_dims × {len(seq_lens)} seq_lens × "
            f"{len(gqa_configs)} GQA configs = {total} 个配置"
        )

    for head_dim in head_dims:
        for q_heads, kv_heads in gqa_configs:
            for seq_len in seq_lens:
                try:
                    # 创建输入张量（BNSD 布局）
                    q = torch.randn(
                        batch_size, q_heads, seq_len, head_dim,
                        dtype=dtype, device=device
                    )
                    k = torch.randn(
                        batch_size, kv_heads, seq_len, head_dim,
                        dtype=dtype, device=device
                    )
                    v = torch.randn(
                        batch_size, kv_heads, seq_len, head_dim,
                        dtype=dtype, device=device
                    )

                    # 触发编译（不需要保存结果）
                    # 使用 _scaled_dot_product_flash_attention，它会调用内部 kernel
                    _ = _scaled_dot_product_flash_attention(
                        q, k, v,
                        dropout_p=0.0,
                        is_causal=True,
                        return_debug_mask=False,
                        scale=None,
                    )

                    compiled += 1

                    if verbose and compiled % 10 == 0:
                        logger.info(f"  已编译 {compiled}/{total} 个配置...")

                except Exception as e:
                    logger.warning(
                        f"  跳过配置 head_dim={head_dim}, q_heads={q_heads}, "
                        f"kv_heads={kv_heads}, seq_len={seq_len}: {e}"
                    )

    if verbose:
        logger.info(f"✓ 预编译完成：成功 {compiled}/{total} 个配置")

    return compiled


def precompile_for_model(model_name, dtype=torch.bfloat16, device="cuda"):
    """
    为特定模型预编译 FlashAttention kernel。

    Args:
        model_name: 模型名称，支持：
            - "qwen2.5-7b"
            - "qwen2.5-1.5b"
            - "llama-3.2-3b"
            - "glm-4-9b"
        dtype: 数据类型
        device: 设备

    Returns:
        编译的配置总数
    """
    # 模型特定配置
    MODEL_CONFIGS = {
        "qwen2.5-7b": {
            "head_dims": [128],
            "gqa_configs": [(28, 4)],
        },
        "qwen2.5-1.5b": {
            "head_dims": [128],
            "gqa_configs": [(12, 2)],
        },
        "llama-3.2-3b": {
            "head_dims": [128],
            "gqa_configs": [(24, 8)],
        },
        "llama-3.2-1b": {
            "head_dims": [64],
            "gqa_configs": [(32, 8)],
        },
        "glm-4-9b": {
            "head_dims": [128],
            "gqa_configs": [(32, 2)],
        },
    }

    model_key = model_name.lower()
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"不支持的模型 '{model_name}'。支持的模型: "
            f"{list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_key]
    logger.info(f"为模型 {model_name} 预编译 FlashAttention kernel...")

    return precompile_flash_attention(
        head_dims=config["head_dims"],
        gqa_configs=config["gqa_configs"],
        dtype=dtype,
        device=device,
        verbose=True,
    )

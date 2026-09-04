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
FlashAttention precompilation: reduce the JIT compilation cost of the first call.

Problem:
  A Triton kernel triggers JIT compilation the first time it sees a new shape
  (head_dim in particular). During autoregressive inference the seq_len keeps
  changing, which leads to frequent recompilation.

Solution:
  Warm up the common head_dims (64/128) and typical seq_len buckets ahead of
  time so the kernels are compiled and cached before they are actually needed.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Common head_dims for mainstream models (from the analysis report).
COMMON_HEAD_DIMS = [64, 128]

# Typical sequence-length buckets (covering decode through prefill).
COMMON_SEQ_LENS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]

# Typical GQA configs (q_heads, kv_heads).
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
    scale=None,
):
    """
    Precompile FlashAttention kernels to reduce runtime JIT overhead.

    Args:
        head_dims: list of head_dim values to precompile, defaults to [64, 128]
        seq_lens: list of seq_len values to precompile, defaults to [1, 16, ..., 2048]
        gqa_configs: list of GQA configs [(q_heads, kv_heads), ...], defaults to
            the mainstream-model configs
        batch_size: batch size, defaults to 1 (inference scenario)
        dtype: data type, defaults to bfloat16
        device: device, defaults to "cuda"
        verbose: whether to print progress, defaults to True
        scale: scaling factor, defaults to None (computed as 1/sqrt(head_dim))

    Returns:
        the total number of compiled configs
    """
    head_dims = head_dims or COMMON_HEAD_DIMS
    seq_lens = seq_lens or COMMON_SEQ_LENS
    gqa_configs = gqa_configs or COMMON_GQA_CONFIGS

    total = len(head_dims) * len(seq_lens) * len(gqa_configs)
    compiled = 0

    if verbose:
        logger.info(
            f"Start precompiling FlashAttention kernels: "
            f"{len(head_dims)} head_dims x {len(seq_lens)} seq_lens x "
            f"{len(gqa_configs)} GQA configs = {total} configs"
        )

    for head_dim in head_dims:
        for q_heads, kv_heads in gqa_configs:
            for seq_len in seq_lens:
                try:
                    # Compute scale if it was not provided.
                    actual_scale = scale if scale is not None else float(1.0 / (head_dim ** 0.5))

                    # Create input tensors in BNSD layout to match the PyTorch API.
                    # PyTorch _flash_attention_forward expects the BNSD format.
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

                    # Trigger compilation through the registered operator entry.
                    # This matches the actual replacement path exactly.
                    from flag_gems.ops._flash_attention_forward import _flash_attention_forward
                    _ = _flash_attention_forward(
                        q, k, v,
                        cumulative_sequence_length_q=None,
                        cumulative_sequence_length_k=None,
                        max_q=q_heads,
                        max_k=kv_heads,
                        dropout_p=0.0,
                        is_causal=True,
                        return_debug_mask=False,
                        scale=actual_scale,
                        window_size_left=None,
                        window_size_right=None,
                        seqused_k=None,
                        alibi_slopes=None,
                    )

                    compiled += 1

                    if verbose and compiled % 10 == 0:
                        logger.info(f"  compiled {compiled}/{total} configs...")

                except Exception as e:
                    logger.warning(
                        f"  skipped config head_dim={head_dim}, q_heads={q_heads}, "
                        f"kv_heads={kv_heads}, seq_len={seq_len}: {e}"
                    )

    if verbose:
        logger.info(f"Precompilation done: {compiled}/{total} configs succeeded")

    return compiled


def precompile_for_model(model_name, dtype=torch.bfloat16, device="cuda"):
    """
    Precompile FlashAttention kernels for a specific model.

    Args:
        model_name: model name, supported values:
            - "qwen2.5-7b"
            - "qwen2.5-1.5b"
            - "llama-3.2-3b"
            - "glm-4-9b"
        dtype: data type
        device: device

    Returns:
        the total number of compiled configs
    """
    # Model-specific configs.
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
            f"Unsupported model '{model_name}'. Supported models: "
            f"{list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_key]
    logger.info(f"Precompiling FlashAttention kernels for model {model_name}...")

    return precompile_flash_attention(
        head_dims=config["head_dims"],
        gqa_configs=config["gqa_configs"],
        dtype=dtype,
        device=device,
        verbose=True,
    )

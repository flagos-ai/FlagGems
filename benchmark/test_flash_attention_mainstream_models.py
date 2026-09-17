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
Comprehensive FlashAttention tests: cover the real attention configs of
mainstream open-source models.

Key point: these models all use GQA (Grouped-Query Attention), i.e. the number
of Q heads != the number of KV heads. The tests must reflect this, otherwise the
measured performance will not match real inference scenarios.

Where the configs come from:
  - Qwen2.5-7B: confirmed from the local
    /data/zhaizir/models/Qwen2.5-7B-Instruct/config.json
  - Other models: filled in from each model's official HuggingFace config.json
    public parameters (use config_from_hf() at the end of this file to verify
    after downloading a model).

Field meaning (shape of a single attention layer):
  q_heads   : num_attention_heads   -- number of Q attention heads
  kv_heads  : num_key_value_heads   -- number of KV heads (usually far fewer
                                       than q_heads under GQA)
  head_dim  : hidden_size / q_heads -- dimension of each head
"""

import time

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime import torch_device_fn

# ==================== Real attention configs of mainstream models ====================
# Format: (model_name, q_heads, kv_heads, head_dim, note)
# Only describes the "head structure of a single attention layer"; batch/seq_len
# are combined separately in the tests.
MODEL_ARCH = [
    # ---- Qwen2.5 series (GQA, head_dim: 0.5B=64, others=128) ----
    ("Qwen2.5-0.5B",            14,  2,  64,  "hidden=896,  layers=24"),
    ("Qwen2.5-1.5B",            12,  2, 128,  "hidden=1536, layers=28"),
    ("Qwen2.5-3B",              16,  2, 128,  "hidden=2048, layers=36"),
    ("Qwen2.5-7B",              28,  4, 128,  "hidden=3584, layers=28 (confirmed locally)"),
    ("Qwen2.5-14B",             40,  8, 128,  "hidden=5120, layers=48"),
    # ---- DeepSeek-R1-Distill series (distilled from Qwen/Llama, same structure) ----
    ("DeepSeek-R1-Distill-Qwen-1.5B",  12, 2, 128, "distilled from Qwen2.5-1.5B"),
    ("DeepSeek-R1-Distill-Qwen-7B",    28, 4, 128, "distilled from Qwen2.5-Math-7B"),
    ("DeepSeek-R1-Distill-Llama-8B",   32, 8, 128, "distilled from Llama-3.1-8B"),
    # ---- GLM-4 series (GQA, multi_query_group_num=2) ----
    ("GLM-4-9B",                32,  2, 128,  "hidden=4096, layers=40"),
    # ---- Llama-3.2 series (GQA, head_dim: 3B=128) ----
    ("Llama-3.2-3B",            24,  8, 128,  "hidden=3072, layers=28"),
]

# Sequence-length buckets: cover decode (short) through prefill (long).
SEQ_LENS = [128, 256, 512, 1024]

# Batch buckets: single request through moderate concurrency.
BATCH_SIZES = [1, 4]


def _build_correctness_configs():
    """Build the Cartesian product of correctness test cases, keeping the size bounded."""
    configs = []
    for name, q_h, kv_h, hd, note in MODEL_ARCH:
        for bs in BATCH_SIZES:
            for sl in SEQ_LENS:
                # Only pair large batch with short sequences to avoid excessive
                # memory use and runtime.
                if bs > 1 and sl > 512:
                    continue
                configs.append((name, bs, q_h, kv_h, sl, hd, note))
    return configs


CORRECTNESS_CONFIGS = _build_correctness_configs()

# Performance regression: pick one representative config per model
# (batch=1, seq_len=1024; long context best exposes problems).
PERF_CONFIGS = [
    (name, 1, q_h, kv_h, 1024, hd, note)
    for name, q_h, kv_h, hd, note in MODEL_ARCH
]

# Dynamic sequence length: simulate Qwen2.5-7B autoregressive generation
# (during decode the seq grows from 1 up to 1024).
DYNAMIC_SEQ_CONFIGS = [
    ("Qwen2.5-7B", 1, 28, 4, sl, 128, f"seq_len={sl}")
    for sl in [1, 16, 32, 64, 128, 256, 512, 1024]
]


# ==================== Utility functions ====================

def make_gqa_input(batch, q_heads, kv_heads, seq_len, head_dim, dtype, device):
    """
    Generate GQA inputs: Q uses q_heads, K/V use kv_heads (fewer heads).
    The returned tensors use the [batch, heads, seq_len, head_dim] (BNSD) layout.
    """
    assert q_heads % kv_heads == 0, (
        f"q_heads({q_heads}) must be divisible by kv_heads({kv_heads}); "
        f"this is the GQA constraint"
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
    Native PyTorch FlashAttention.
    _flash_attention_forward expects the [batch, seq_len, heads, head_dim] (BSND)
    layout, so BNSD -> BSND requires transpose(1, 2). The native implementation
    handles GQA internally.
    """
    q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
    out = torch.ops.aten._flash_attention_forward(
        q_t, k_t, v_t, None, None,
        q_t.shape[-3], k_t.shape[-3],
        0.0, is_causal, False, scale=scale,
    )
    return out[0], out[1]  # (out[BSND], lse)


def run_gems_fa(q, k, v, scale, is_causal):
    """FlagGems FlashAttention, also using the BSND layout."""
    q_t, k_t, v_t = (x.transpose(1, 2) for x in (q, k, v))
    out = flag_gems.flash_attention_forward(
        q_t, k_t, v_t, None, None,
        q_t.shape[-3], k_t.shape[-3],
        0.0, is_causal, False, scale=scale,
    )
    return out[0], out[1]


def benchmark(fn, warmup=10, iters=50):
    """Return the average time per call (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / iters * 1000


# ==================== Test cases ====================

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
    """Correctness: FlagGems GQA results must match native PyTorch."""
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
    """Dynamic sequence length: simulate real autoregressive inference, causal=True."""
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
    Performance regression: measure FlagGems against native PyTorch honestly.

    The comparison must isolate the two implementations:
      - native PyTorch: measured outside any use_gems() context, so the aten op
        dispatches to the native kernel;
      - FlagGems: measured inside `with flag_gems.use_gems()`, which registers
        the Triton kernels and truly unregisters them on exit.

    Do NOT rely on disable_flash_attention()/enable() toggling here: those only
    return name lists / register globally and cannot restore the native kernel,
    which silently makes both sides measure the same kernel.

    The threshold comes from the FA_MAX_SLOWDOWN_PCT environment variable. The
    current Triton implementation is known to be significantly slower than the
    native fused kernel at small batch / short sequence lengths, so the default
    threshold is intentionally wide to record a baseline rather than gate CI.
    Set FA_PERF_RECORD_ONLY=1 to record only without failing.
    """
    import os

    MAX_SLOWDOWN_PCT = float(os.environ.get("FA_MAX_SLOWDOWN_PCT", "2000.0"))
    RECORD_ONLY = os.environ.get("FA_PERF_RECORD_ONLY", "0") == "1"

    device = torch_device_fn.current_device()
    q, k, v = make_gqa_input(
        batch, q_heads, kv_heads, seq_len, head_dim, dtype, device
    )
    scale = float(1.0 / np.sqrt(head_dim))

    # Native PyTorch: outside any use_gems() context.
    pt_ms = benchmark(lambda: run_pytorch_fa(q, k, v, scale, True))

    # FlagGems: inside the use_gems() context so the Triton kernels are actually
    # registered (and cleanly unregistered on exit). Precompile first so we
    # measure steady-state performance, not one-off JIT compilation.
    with flag_gems.use_gems():
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
        gems_ms = benchmark(lambda: run_pytorch_fa(q, k, v, scale, True))

    slowdown = (gems_ms - pt_ms) / pt_ms * 100
    print(
        f"\n[{model_name}] Q{q_heads}/KV{kv_heads} d{head_dim} seq{seq_len} "
        f"| PyTorch {pt_ms:.4f}ms | FlagGems {gems_ms:.4f}ms "
        f"| {'slower' if slowdown > 0 else 'faster'} {abs(slowdown):.1f}%"
    )

    if slowdown > MAX_SLOWDOWN_PCT and not RECORD_ONLY:
        pytest.fail(
            f"{model_name} performance regression: FlagGems is {slowdown:.1f}% "
            f"slower (threshold {MAX_SLOWDOWN_PCT}%) | PyTorch {pt_ms:.4f}ms, "
            f"FlagGems {gems_ms:.4f}ms"
        )


# ==================== Helper: extract real config from HF config ====================

def config_from_hf(model_path):
    """
    Read the real attention config from a model directory's config.json, to make
    it easy to verify the hand-filled parameters in MODEL_ARCH above after
    downloading a model.

    Usage:
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
    # Prefer the explicit head_dim, otherwise derive it from hidden/q_heads.
    head_dim = cfg.get("head_dim", hidden // q_heads if hidden and q_heads else None)
    layers = cfg.get("num_hidden_layers")

    print(f"Model: {model_path}")
    print(f"  q_heads (num_attention_heads) = {q_heads}")
    print(f"  kv_heads (num_key_value_heads) = {kv_heads}")
    print(f"  head_dim = {head_dim}")
    print(f"  layers = {layers}")
    print(f"  is GQA: {'yes' if kv_heads != q_heads else 'no'}")
    return q_heads, kv_heads, head_dim, layers


if __name__ == "__main__":
    # Quick self-check: run a performance comparison for a few representative
    # models using their real GQA configs.
    print("=" * 84)
    print("FlashAttention mainstream-model GQA self-check (batch=1, seq_len=1024, causal)")
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

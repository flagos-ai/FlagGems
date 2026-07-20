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

"""Accuracy tests for top_k_per_row_decode (DeepSeek V4 decode-phase top-K).

Tests the Triton radix-select kernel against the vLLM CUDA reference.
Uses value-based comparison (sorted selected values must match) to handle
non-deterministic tie-breaking between implementations.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused import top_k_per_row_decode

from . import conftest as cfg

device = flag_gems.device

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


# --- Shape configuration with QUICK_MODE support ---
if cfg.QUICK_MODE:
    BATCH_SIZE_LIST = [1]
    VOCAB_SIZE_LIST = [262144]
    TOP_K_LIST = [512]
else:
    BATCH_SIZE_LIST = [1, 496]
    VOCAB_SIZE_LIST = [4096, 8192, 16384, 32768, 129280, 262144]
    TOP_K_LIST = [64, 128, 256, 512, 1024]

# --- vLLM CUDA reference (optional) ---
try:
    import vllm._custom_ops  # noqa: F401 — loads torch.ops._C

    def _vllm_top_k_per_row_decode(
        logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
    ):
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )

    HAS_VLLM = True
except (ImportError, AttributeError):
    HAS_VLLM = False
    _vllm_top_k_per_row_decode = None


def check_topk_values_match(logits, indices_test, indices_ref, top_k):
    num_rows = logits.shape[0]
    for i in range(num_rows):
        abs_test = indices_test[i].long()
        abs_ref = indices_ref[i].long()

        valid_test = abs_test[abs_test >= 0]
        valid_ref = abs_ref[abs_ref >= 0]

        vals_test = logits[i].gather(0, valid_test)
        vals_ref = logits[i].gather(0, valid_ref)

        vals_test_sorted, _ = vals_test.sort(descending=True)
        vals_ref_sorted, _ = vals_ref.sort(descending=True)

        if not torch.allclose(vals_test_sorted, vals_ref_sorted, atol=1e-6, rtol=1e-6):
            return False
    return True


def _make_inputs(batch_size, vocab_size, top_k, seq_len=None):
    """Generate test inputs matching DeepSeek V4 decode config."""
    if seq_len is None:
        seq_len = vocab_size
    next_n = 1
    num_rows = batch_size * next_n
    logits = torch.randn(num_rows, vocab_size, dtype=torch.float32, device=device)
    seq_lens = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)
    indices = torch.zeros(num_rows, top_k, dtype=torch.int32, device=device)
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)
    return logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k


def _torch_topk_ref(
    logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
):
    """Pure-PyTorch fallback reference using torch.topk."""
    for i in range(num_rows):
        batch_id = i // next_n
        batch_offset = i % next_n
        seq_len = seq_lens[batch_id]
        row_len = seq_len - next_n + batch_offset + 1
        row_slice = logits[i, :row_len]
        k = min(top_k, row_len)
        _, topk_idx = torch.topk(row_slice, k, largest=True, sorted=False)
        indices[i, :k] = topk_idx.to(torch.int32)
        if k < top_k:
            indices[i, k:] = -1


@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize(
    "batch_size, vocab_size, top_k",
    [(b, v, k) for b, v, k in zip(BATCH_SIZE_LIST, VOCAB_SIZE_LIST, TOP_K_LIST)],
    ids=[
        f"B{b}_V{v}_K{k}"
        for b, v, k in zip(BATCH_SIZE_LIST, VOCAB_SIZE_LIST, TOP_K_LIST)
    ],
)
def test_top_k_per_row_decode(batch_size, vocab_size, top_k):
    """Test top-k correctness: selected values must match reference."""
    torch.manual_seed(42)
    ref_fn = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref

    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        batch_size, vocab_size, top_k
    )
    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    ref_fn(logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, k)

    assert check_topk_values_match(logits, indices, indices_ref, top_k)


@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize(
    "vocab_size, top_k, seq_len",
    [
        (129280, 1024, 100000),
        (32768, 256, 16384),
        (8192, 64, 4096),
    ],
    ids=["V129280_K1024_S100000", "V32768_K256_S16384", "V8192_K64_S4096"],
)
def test_top_k_per_row_decode_partial_seqlen(vocab_size, top_k, seq_len):
    """Test with seq_len < vocab_size (partial valid range)."""
    torch.manual_seed(123)
    batch_size = 1
    ref_fn = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref

    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        batch_size, vocab_size, top_k, seq_len=seq_len
    )
    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    ref_fn(logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, k)

    assert check_topk_values_match(logits, indices, indices_ref, top_k)


@pytest.mark.top_k_per_row_decode
def test_topk_greater_than_row_len():
    torch.manual_seed(456)
    batch_size = 1
    vocab_size = 262144
    top_k = 512
    seq_len = 496
    ref_fn = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref

    logits, next_n, seq_lens, indices, num_rows, s0, s1, k = _make_inputs(
        batch_size, vocab_size, top_k, seq_len=seq_len
    )
    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, k)
    ref_fn(logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, k)

    assert check_topk_values_match(logits, indices, indices_ref, top_k)


@pytest.mark.top_k_per_row_decode
def test_logits_diff_in_8LSBits():
    num_rows = 1
    next_n = 1
    vocab_size = 262144
    top_k = 512
    seq_len = vocab_size
    ref_fn = _vllm_top_k_per_row_decode if HAS_VLLM else _torch_topk_ref

    random_8bit = torch.randint(
        0,
        2**8,
        (num_rows, vocab_size),
        dtype=torch.int32,
        device=device,
    )
    logits_bits = 0x3F900000 | (random_8bit & 0xFF)
    logits = logits_bits.view(torch.float32)
    seq_lens = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)
    indices = torch.zeros(num_rows, top_k, dtype=torch.int32, device=device)
    s0 = logits.stride(0)
    s1 = logits.stride(1)

    logits_ref = logits.clone()
    indices_ref = torch.zeros_like(indices)

    top_k_per_row_decode(logits, next_n, seq_lens, indices, num_rows, s0, s1, top_k)
    ref_fn(logits_ref, next_n, seq_lens, indices_ref, num_rows, s0, s1, top_k)

    assert check_topk_values_match(logits, indices, indices_ref, top_k)

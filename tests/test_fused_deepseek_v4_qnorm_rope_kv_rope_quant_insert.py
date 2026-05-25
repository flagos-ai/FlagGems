import pytest
import torch

import flag_gems
from flag_gems.utils.device_info import get_device_capability

from .conftest import QUICK_MODE


def is_support_fp8e4nv():
    major, minor = get_device_capability()
    return major * 10 + minor >= 89


VLLM_REF_AVAILABLE = hasattr(
    torch.ops._C, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
)
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448
QUANT_BLOCK = 64
FP8_MAX = 448.0
TOKEN_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2  # 576
NUM_QUANT_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 7
SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1  # 8
HEAD_BYTES = TOKEN_DATA_BYTES + SCALE_BYTES_PER_TOKEN  # 584


# ─── pytorch reference implementation from vllm ───


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    """Build a cos||sin cache matching DeepseekV4ScalingRotaryEmbedding layout.
    cos_sin_cache[pos, :rope_dim/2] = cos(theta), [rope_dim/2:] = sin(theta).
    """
    if max_pos <= 8192:
        base = 10000.0
    elif max_pos <= 32768:
        base = 20000.0
    elif max_pos <= 65536:
        base = 40000.0
    elif max_pos <= 98304:
        base = 60000.0
    else:
        base = 100000.0

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rope_dim/2]
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim]
    return cache.to(dtype)


def apply_rope_gptj_last_k(
    x: torch.Tensor,
    x_f32: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
):
    """GPT-J-style (interleaved-pair) RoPE on the LAST rope_dim elements.
    x: [..., head_dim] bfloat16, in place
    x_f32: result of q RMSNorm
    positions: [num_tokens] int64 (positions[i] corresponds to x[i, ...])
    cos_sin_cache: [max_pos, rope_dim] float (cos|sin layout)
    """
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    # Gather cos/sin for each token position: [num_tokens, rope_dim]
    cs = cos_sin_cache[positions].to(torch.float32)  # [N, rope_dim]
    cos = cs[..., :half]  # [N, half]
    sin = cs[..., half:]  # [N, half]

    # Reshape leading dims so we can broadcast: x shape [..., head_dim].
    # Bring token dim to front; assume x is [num_tokens, ..., head_dim].
    # We rely on positions being per-token and all other dims sharing the same pos.
    if x_f32 is None:
        rope = x[..., nope_dim:].float()  # [..., rope_dim]
    else:
        rope = x_f32[..., nope_dim:]
    # Make rope pairs: reshape last dim to [half, 2]
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]  # [..., half]
    odd = rope[..., 1]

    # Broadcast cos/sin over any heads dim in between.  cos/sin are [N, half].
    # Add singleton dims for intermediate axes.
    for _ in range(rope.ndim - 3):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    x[..., nope_dim:] = rope_rotated.to(x.dtype)
    if x_f32 is not None:
        x[..., :nope_dim] = x_f32[..., :nope_dim].to(x.dtype)


def rmsnorm_no_weight_f32(x: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm with no learnable weight, matching
    `RMSNorm(head_dim, has_weight=False)`."""
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps)


# ─── pytorch reference implementation from vllm end ───


def torch_quantize_and_insert_k_cache(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens_insert] int64
    block_size: int = 64,
):
    """
    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576
    """
    assert k.is_contiguous() and k.dim() == 2 and k.shape[1] == HEAD_DIM
    assert k.dtype == torch.bfloat16
    assert k_cache.dim() == 2
    assert k_cache.dtype == torch.uint8
    assert slot_mapping.dim() == 1
    num_tokens = k.shape[0]
    num_blocks, block_bytes = k_cache.shape
    num_tokens_insert = slot_mapping.shape[0]
    assert num_tokens_insert <= num_tokens

    token_id = torch.arange(num_tokens_insert)
    slot_id = slot_mapping[token_id]
    mask = slot_id >= 0
    num = mask.sum()
    if num == 0:
        return
    if num < num_tokens_insert:
        slot_id = slot_id[mask]
        token_id = token_id[mask]
    block_id = slot_id // block_size
    pos_in_block = slot_id % block_size
    fp8_off = pos_in_block * TOKEN_DATA_BYTES
    bf16_off = fp8_off + NOPE_DIM
    scale_off = block_size * TOKEN_DATA_BYTES + pos_in_block * SCALE_BYTES_PER_TOKEN
    scale_pad_off = scale_off + NUM_QUANT_BLOCKS

    k_direct = (
        k[token_id, NOPE_DIM:].view(torch.uint8).view(num, ROPE_DIM * 2).to(torch.uint8)
    )
    bf16_range = torch.arange(ROPE_DIM * 2, dtype=torch.int64, device=k.device)
    k_cache[block_id[:, None], bf16_off[:, None] + bf16_range[None, :]] = k_direct

    k_quant = k[token_id, :NOPE_DIM]
    kv_quant_blk = k_quant.view(num, NUM_QUANT_BLOCKS, QUANT_BLOCK).to(torch.float32)
    block_max = torch.max(torch.abs(kv_quant_blk), dim=-1).values
    block_max = torch.clamp(block_max, min=1e-4)
    raw_scale = block_max / FP8_MAX
    log_scale = torch.log2(raw_scale)
    exponent = torch.ceil(log_scale)
    scale = torch.exp2(exponent)
    x_scaled = kv_quant_blk / scale[:, :, None]
    x_clamped = torch.clamp(x_scaled, min=-FP8_MAX, max=FP8_MAX)
    x_fp8 = x_clamped.to(torch.float8_e4m3fn)
    x_uint8 = x_fp8.view(torch.uint8).view(num, NOPE_DIM)
    fp8_range = torch.arange(NOPE_DIM, dtype=torch.int64, device=k.device)
    k_cache[block_id[:, None], fp8_off[:, None] + fp8_range[None, :]] = x_uint8
    encoded_scale = exponent + 127.0
    encoded_scale = torch.clamp(encoded_scale, min=0.0, max=255.0).to(torch.uint8)
    scale_range = torch.arange(NUM_QUANT_BLOCKS, dtype=torch.int64, device=k.device)
    k_cache[
        block_id[:, None], scale_off[:, None] + scale_range[None, :]
    ] = encoded_scale
    k_cache[block_id, scale_pad_off] = 0


def k_cache_compare(
    k_cache: torch.Tensor,
    k_cache_ref: torch.Tensor,
    block_size: int,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    """
    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576
    """
    assert k_cache.dim() == 2 and k_cache.shape == k_cache_ref.shape
    num_blocks = k_cache.shape[0]
    scale_start = block_size * TOKEN_DATA_BYTES
    scale_end = block_size * HEAD_BYTES
    token_data = k_cache[:, :scale_start].view(num_blocks, block_size, TOKEN_DATA_BYTES)
    token_data_ref = k_cache_ref[:, :scale_start].view(
        num_blocks, block_size, TOKEN_DATA_BYTES
    )
    # quantization data part, uint8
    torch.testing.assert_close(
        token_data[:, :, :NOPE_DIM], token_data_ref[:, :, :NOPE_DIM], rtol=0, atol=0
    )
    # rope part, bf16
    torch.testing.assert_close(
        token_data[:, :, NOPE_DIM:TOKEN_DATA_BYTES].view(torch.bfloat16),
        token_data_ref[:, :, NOPE_DIM:TOKEN_DATA_BYTES].view(torch.bfloat16),
        rtol=rtol,
        atol=atol,
    )
    # quantization scale part, uint8
    torch.testing.assert_close(
        k_cache[:, scale_start:scale_end],
        k_cache_ref[:, scale_start:scale_end],
        rtol=0,
        atol=0,
    )


def ref_impl(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs):
    if VLLM_REF_AVAILABLE:
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
        )
    else:
        q_norm_f32 = rmsnorm_no_weight_f32(q, eps)
        apply_rope_gptj_last_k(q, q_norm_f32, positions, cos_sin_cache)
        if kv.size(0) > slot_mapping.size(0):
            kv = kv[: slot_mapping.size(0), :]
            positions = positions[: slot_mapping.size(0)]
        apply_rope_gptj_last_k(kv, None, positions, cos_sin_cache)
        torch_quantize_and_insert_k_cache(kv, k_cache, slot_mapping, block_size=bs)


def fused_impl(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs):
    flag_gems.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs
    )


# ── Test 1: Q path numerical parity ──────────────────────────────────────────


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
@pytest.mark.parametrize("n_heads", [8, 64])
def test_q_path_matches_reference(num_tokens: int, n_heads: int):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = max(4096, num_tokens)
    num_blocks = 2
    bs = 16

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.zeros(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    k_cache = torch.zeros(
        num_blocks, bs, HEAD_BYTES, dtype=torch.uint8, device=device
    ).view(num_blocks, -1)
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)
    q_ref = q.clone()
    kv_ref = kv.clone()
    k_cache_ref = k_cache.clone()
    slot_mapping_ref = slot_mapping.clone()
    positions_ref = positions.clone()
    cos_sin_cache_ref = cos_sin_cache.clone()

    ref_impl(
        q_ref,
        kv_ref,
        k_cache_ref,
        slot_mapping_ref,
        positions_ref,
        cos_sin_cache_ref,
        eps,
        bs,
    )

    fused_impl(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, bs)

    torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)


# ── Test 2: KV path round-trip byte/value parity ─────────────────────────────


def _ue8m0_per_block_scales(kv_roped_nope_f32: torch.Tensor, qblock: int):
    """Return per-token per-block max scale (used to bound FP8 error)."""
    n_tok, nope = kv_roped_nope_f32.shape
    n_blocks = nope // qblock
    blocks = kv_roped_nope_f32.view(n_tok, n_blocks, qblock)
    absmax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    raw = absmax / FP8_MAX
    exponent = torch.ceil(torch.log2(raw))
    return torch.pow(2.0, exponent)  # [n_tok, n_blocks]


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize("num_tokens", [1, 4, 17, 64])
@pytest.mark.parametrize("block_size", [16, 64])
def test_kv_path_matches_reference(num_tokens: int, block_size: int):
    torch.manual_seed(1)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = max(4096, num_tokens)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    q = torch.zeros(num_tokens, 1, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    k_cache = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    q_ref = q.clone()
    kv_ref = kv.clone()
    k_cache_ref = k_cache.clone()
    positions_ref = positions.clone()
    cos_sin_cache_ref = cos_sin_cache.clone()
    slot_mapping_ref = slot_mapping.clone()

    ref_impl(
        q_ref,
        kv_ref,
        k_cache_ref,
        slot_mapping_ref,
        positions_ref,
        cos_sin_cache_ref,
        eps,
        block_size,
    )

    fused_impl(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size)

    k_cache_compare(k_cache, k_cache_ref, block_size, rtol=1e-2, atol=1e-2)


# ── Test 2b: DP padding (slot_mapping shorter than q/kv) ─────────────────────


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize("num_tokens", [4, 17])
@pytest.mark.parametrize("pad", [1, 5])
@pytest.mark.parametrize("block_size", [16, 64])
def test_kv_path_with_dp_padding(num_tokens: int, pad: int, block_size: int):
    """slot_mapping.size(0) < q.size(0): the kernel must skip padded
    tokens in the KV branch while still running Q-norm+RoPE on all rows."""
    torch.manual_seed(3)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = max(4096, num_tokens)
    total = num_tokens + pad
    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    q = torch.zeros(total, 1, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(total, HEAD_DIM, dtype=dtype, device=device)
    k_cache = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    positions = torch.arange(total, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    q_ref = q.clone()
    kv_ref = kv.clone()
    k_cache_ref = k_cache.clone()
    positions_ref = positions.clone()
    cos_sin_cache_ref = cos_sin_cache.clone()
    slot_mapping_ref = slot_mapping.clone()

    ref_impl(
        q_ref,
        kv_ref,
        k_cache_ref,
        slot_mapping_ref,
        positions_ref,
        cos_sin_cache_ref,
        eps,
        block_size,
    )

    fused_impl(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size)

    torch.testing.assert_close(k_cache, k_cache_ref, rtol=0, atol=0)


# ── Test 3: combined single-call Q + KV parity ───────────────────────────────


@pytest.mark.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
@pytest.mark.skipif(
    not is_support_fp8e4nv(), reason="Do not support fp8e4nv when capability < 89"
)
@pytest.mark.parametrize(
    "num_tokens",
    [1, 4, 17, 64] if QUICK_MODE else [1, 4, 17, 64, 8192, 32768, 65536, 98304, 131072],
)
@pytest.mark.parametrize("n_heads", [64, 128])
@pytest.mark.parametrize("block_size", [16, 64])
def test_combined_q_and_kv(num_tokens: int, n_heads: int, block_size: int):
    # out of memory for huge shape on H800
    if (num_tokens == 98304 or num_tokens == 131072) and n_heads == 128:
        return

    torch.manual_seed(2)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    max_pos = max(4096, num_tokens)
    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    q = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=device)
    k_cache = torch.zeros(
        num_blocks, block_size * HEAD_BYTES, dtype=torch.uint8, device=device
    )
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cos_sin_cache = make_cos_sin_cache(max_pos, ROPE_DIM, torch.float32, device)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    q_ref = q.clone()
    kv_ref = kv.clone()
    k_cache_ref = k_cache.clone()
    positions_ref = positions.clone()
    cos_sin_cache_ref = cos_sin_cache.clone()
    slot_mapping_ref = slot_mapping.clone()

    ref_impl(
        q_ref,
        kv_ref,
        k_cache_ref,
        slot_mapping_ref,
        positions_ref,
        cos_sin_cache_ref,
        eps,
        block_size,
    )
    fused_impl(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, eps, block_size)

    torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
    k_cache_compare(k_cache, k_cache_ref, block_size, rtol=1e-2, atol=1e-2)

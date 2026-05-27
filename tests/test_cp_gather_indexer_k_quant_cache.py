import math
import os

import pytest
import torch
from packaging.version import InvalidVersion, Version

from flag_gems.fused import cp_gather_indexer_k_quant_cache

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)


def _default_fp8_dtype():
    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("float8_e4m3fn is required for cp_gather_indexer_k_quant_cache")


def _check_target_vllm_version(vllm):
    version = getattr(vllm, "__version__", "0.0.0")
    try:
        parsed = Version(version.split("+", 1)[0])
        if parsed < _TARGET_VLLM_VERSION or parsed >= _NEXT_VLLM_VERSION:
            return False
    except InvalidVersion:
        pass
    return True


def _load_vllm_cuda_op_and_fp8_dtype():
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    if getattr(torch.version, "cuda", None) is None:
        return None, _default_fp8_dtype(), False
    try:
        import vllm
        import vllm._custom_ops as ops
        from vllm.platforms import current_platform
    except Exception:
        return None, _default_fp8_dtype(), False

    if not _check_target_vllm_version(vllm):
        return None, _default_fp8_dtype(), False

    if not hasattr(ops, "cp_gather_indexer_k_quant_cache"):
        return None, _default_fp8_dtype(), False

    def vllm_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache,
            dst_k,
            dst_scale,
            block_table,
            cu_seq_lens,
        )

    return vllm_gather, current_platform.fp8_dtype(), True


def torch_gather(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens):
    block_size = kv_cache.size(1)
    head_dim = dst_k.size(1)
    quant_block_size = head_dim * 4 // dst_scale.size(1)
    num_quant_blocks = head_dim // quant_block_size

    flat_cache = kv_cache.view(kv_cache.shape[0], -1)
    cache_values = flat_cache[:, : block_size * head_dim]
    cache_scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    dst_k_bytes = dst_k.view(torch.uint8)
    dst_scale_float = dst_scale.view(torch.float32)

    for batch_idx in range(block_table.size(0)):
        seq_start = int(cu_seq_lens[batch_idx].item())
        seq_end = int(cu_seq_lens[batch_idx + 1].item())
        for token_idx in range(seq_start, seq_end):
            batch_offset = token_idx - seq_start
            block_table_id = batch_offset // block_size
            block_offset = batch_offset % block_size
            block_id = int(block_table[batch_idx, block_table_id].item())

            src_value_start = block_offset * head_dim
            src_value_end = src_value_start + head_dim
            dst_k_bytes[token_idx].copy_(
                cache_values[block_id, src_value_start:src_value_end]
            )

            src_scale_start = block_offset * num_quant_blocks
            src_scale_end = src_scale_start + num_quant_blocks
            dst_scale_float[token_idx].copy_(
                cache_scales[block_id, src_scale_start:src_scale_end]
            )


def _make_cache(num_blocks, block_size, head_dim, quant_block_size, fp8_dtype, device):
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    k_cache = torch.empty(
        (num_blocks, block_size, cache_stride),
        dtype=torch.uint8,
        device=device,
    )
    num_quant_blocks = head_dim // quant_block_size
    flat_cache = k_cache.view(num_blocks, -1)
    values = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    values.copy_(torch.randn(values.shape, device=device).to(fp8_dtype))
    scales = flat_cache[:, block_size * head_dim :].view(torch.float32)
    scales.copy_(
        torch.rand(
            num_blocks,
            block_size * num_quant_blocks,
            device=device,
            dtype=torch.float32,
        )
        + 0.01
    )
    return k_cache


def _make_gather_metadata(seq_lens, block_size, device):
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    cu_seqlen = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    cu_seqlen[1:] = torch.cumsum(seq_lens_tensor, dim=0)

    max_blocks = max(math.ceil(seq_len / block_size) for seq_len in seq_lens)
    block_table = torch.full(
        (len(seq_lens), max_blocks),
        -1,
        dtype=torch.int32,
        device=device,
    )
    next_block = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        num_seq_blocks = math.ceil(seq_len / block_size)
        blocks = torch.arange(
            next_block,
            next_block + num_seq_blocks,
            dtype=torch.int32,
            device=device,
        )
        block_table[batch_idx, :num_seq_blocks] = blocks
        next_block += num_seq_blocks

    return block_table, cu_seqlen, next_block


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.parametrize(
    "seq_lens,block_size,head_dim,quant_block_size,extra_tokens",
    [
        ([13, 7, 16], 8, 128, 128, 0),
        ([17, 1, 33, 9], 16, 512, 128, 5),
    ],
)
@torch.inference_mode()
def test_cp_gather_indexer_k_quant_cache_matches_reference(
    seq_lens,
    block_size,
    head_dim,
    quant_block_size,
    extra_tokens,
):
    vllm_op, fp8_dtype, has_vllm = _load_vllm_cuda_op_and_fp8_dtype()

    torch.manual_seed(1)
    device = torch.device("cuda")
    block_table, cu_seqlen, num_blocks = _make_gather_metadata(
        seq_lens,
        block_size,
        device,
    )
    valid_tokens = int(cu_seqlen[-1].item())
    allocated_tokens = valid_tokens + extra_tokens

    k_cache = _make_cache(
        num_blocks,
        block_size,
        head_dim,
        quant_block_size,
        fp8_dtype,
        device,
    )

    num_scale_bytes = head_dim * 4 // quant_block_size
    sentinel = 0x7B
    gems_k = torch.empty(
        (allocated_tokens, head_dim),
        dtype=fp8_dtype,
        device=device,
    )
    gems_k.view(torch.uint8).fill_(sentinel)
    reference_k = torch.empty_like(gems_k)
    reference_k.view(torch.uint8).fill_(sentinel)
    gems_scale = torch.full(
        (allocated_tokens, num_scale_bytes),
        sentinel,
        dtype=torch.uint8,
        device=device,
    )
    reference_scale = torch.full_like(gems_scale, sentinel)

    if has_vllm:
        vllm_op(
            k_cache,
            reference_k,
            reference_scale,
            block_table,
            cu_seqlen,
        )
    else:
        torch_gather(
            k_cache,
            reference_k,
            reference_scale,
            block_table,
            cu_seqlen,
        )
    cp_gather_indexer_k_quant_cache(
        k_cache,
        gems_k,
        gems_scale,
        block_table,
        cu_seqlen,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        gems_k.view(torch.uint8),
        reference_k.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(gems_scale, reference_scale, rtol=0, atol=0)
    if extra_tokens:
        assert torch.all(gems_k[valid_tokens:].view(torch.uint8) == sentinel)
        assert torch.all(gems_scale[valid_tokens:] == sentinel)

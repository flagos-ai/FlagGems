import os

import pytest
import torch
from packaging.version import InvalidVersion, Version

from flag_gems.fused import indexer_k_quant_and_cache

_TARGET_VLLM_VERSION = Version("0.20.2")
_NEXT_VLLM_VERSION = Version("0.21.0")


def _is_fp8e4nv_supported():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA device required",
    ),
    pytest.mark.skipif(
        not _is_fp8e4nv_supported(),
        reason="fp8e4nv requires device capability >= 8.9",
    ),
]


def _default_fp8_dtype():
    if getattr(torch.version, "hip", None) is not None and hasattr(
        torch, "float8_e4m3fnuz"
    ):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    pytest.skip("float8_e4m3fn is required for indexer_k_quant_and_cache")


def _is_fp8_fnuz(dtype):
    return hasattr(torch, "float8_e4m3fnuz") and dtype == torch.float8_e4m3fnuz


def _check_target_vllm_version(vllm):
    version = getattr(vllm, "__version__", "0.0.0")
    try:
        parsed = Version(version.split("+", 1)[0])
        if parsed < _TARGET_VLLM_VERSION or parsed >= _NEXT_VLLM_VERSION:
            return False
    except InvalidVersion:
        pass
    return True


def _load_vllm_cuda_op():
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    if getattr(torch.version, "cuda", None) is None:
        return None, False
    try:
        import vllm
        import vllm._custom_ops as ops
    except Exception:
        return None, False

    if not _check_target_vllm_version(vllm):
        return None, False

    if not hasattr(ops, "indexer_k_quant_and_cache"):
        return None, False

    def vllm_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    return vllm_indexer, True


def torch_indexer(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
    num_blocks = kv_cache.shape[0]
    block_size = kv_cache.shape[1]
    head_dim = k.shape[-1]
    num_quant_blocks = head_dim // quant_block_size
    fp8_dtype = _default_fp8_dtype()
    scale_divisor = 224.0 if _is_fp8_fnuz(fp8_dtype) else 448.0

    flat_cache = kv_cache.view(num_blocks, -1)
    cache_values = flat_cache[:, : block_size * head_dim].view(fp8_dtype)
    cache_scales = flat_cache[:, block_size * head_dim :].view(torch.float32)

    for token_idx in range(slot_mapping.numel()):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue

        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        for quant_block_id in range(num_quant_blocks):
            start = quant_block_id * quant_block_size
            end = start + quant_block_size
            val = k[token_idx, start:end]
            amax = val.abs().to(torch.float32).amax()

            scale = (
                torch.maximum(
                    amax,
                    torch.tensor(1e-4, dtype=torch.float32, device=k.device),
                )
                / scale_divisor
            )
            if scale_fmt == "ue8m0":
                scale = torch.exp2(torch.ceil(torch.log2(scale)))

            value_start = block_offset * head_dim + start
            value_end = value_start + quant_block_size
            cache_values[block_id, value_start:value_end].copy_(
                (val.to(torch.float32) / scale).to(fp8_dtype)
            )
            cache_scales[
                block_id,
                block_offset * num_quant_blocks + quant_block_id,
            ] = scale


def _make_cache(num_blocks, block_size, head_dim, quant_block_size, device):
    cache_stride = head_dim + head_dim * 4 // quant_block_size
    fp8_dtype = _default_fp8_dtype()
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


def _make_slot_mapping(num_tokens, num_blocks, block_size, device):
    slot_mapping = torch.randperm(num_blocks * block_size, device=device)[:num_tokens]
    slot_mapping = slot_mapping.to(torch.long)
    slot_mapping[-1] = -1
    return slot_mapping


@pytest.mark.indexer_k_quant_and_cache
@pytest.mark.parametrize(
    "dtype,num_tokens,num_blocks,block_size,head_dim,quant_block_size,scale_fmt",
    [
        (torch.bfloat16, 19, 4, 16, 128, 128, "ue8m0"),
        (torch.float16, 23, 5, 16, 512, 128, "ue8m0"),
        (torch.bfloat16, 17, 4, 64, 512, 128, "ue8m0"),
    ],
)
@torch.inference_mode()
def test_indexer_k_quant_and_cache_matches_reference(
    dtype,
    num_tokens,
    num_blocks,
    block_size,
    head_dim,
    quant_block_size,
    scale_fmt,
):
    vllm_op, has_vllm = _load_vllm_cuda_op()

    torch.manual_seed(0)
    device = torch.device("cuda")
    k = torch.randn(num_tokens, head_dim, device=device, dtype=dtype)
    slot_mapping = _make_slot_mapping(num_tokens, num_blocks, block_size, device)

    gems_cache = _make_cache(
        num_blocks,
        block_size,
        head_dim,
        quant_block_size,
        device,
    )
    reference_cache = gems_cache.clone()

    if has_vllm and dtype != torch.float16:
        vllm_op(k, reference_cache, slot_mapping, quant_block_size, scale_fmt)
    else:
        torch_indexer(k, reference_cache, slot_mapping, quant_block_size, scale_fmt)
    indexer_k_quant_and_cache(
        k,
        gems_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(gems_cache, reference_cache, rtol=0, atol=0)

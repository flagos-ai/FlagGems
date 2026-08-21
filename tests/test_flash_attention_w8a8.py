import math

import pytest
import torch
import triton

import flag_gems
from flag_gems.ops.flash_api_w8a8 import mha_fwd
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import random_utils

from . import conftest as cfg

if cfg.QUICK_MODE:
    W8A8_CONFIGS = [(1, 16, 512, 512)]
else:
    W8A8_CONFIGS = [
        (1, 16, 512, 512),
        (2, 16, 1024, 1024),
        (4, 16, 2048, 2048),
        (8, 32, 512, 512),
    ]


def _supports_hopper_fp8() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _get_fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is not available")
    return dtype


def _hadamard_matrix(dim, device):
    assert dim > 0 and dim & (dim - 1) == 0, "head_size must be a power of two"
    h = torch.tensor([[1.0]], device=device)
    while h.shape[0] < dim:
        h = torch.cat(
            (
                torch.cat((h, h), dim=1),
                torch.cat((h, -h), dim=1),
            ),
            dim=0,
        )
    return h / math.sqrt(dim)


def _apply_incoherent_qk(x):
    h = _hadamard_matrix(x.shape[-1], x.device).to(torch.float32)
    return torch.matmul(x.float(), h).to(x.dtype)


def _quantize_per_block_fp8(x, fp8_dtype, block_size=128):
    batch, seq_len, num_head, _ = x.shape
    fp8_max = float(torch.finfo(fp8_dtype).max)
    nblocks = triton.cdiv(seq_len, block_size)

    out = torch.empty_like(x, dtype=fp8_dtype)
    descale = torch.empty(
        (batch, num_head, nblocks),
        device=x.device,
        dtype=torch.float32,
    )

    for block_idx in range(nblocks):
        lo = block_idx * block_size
        hi = min(seq_len, lo + block_size)
        tile = x[:, lo:hi, :, :].float()
        scale = (tile.abs().amax(dim=(1, 3)) / fp8_max).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        out[:, lo:hi, :, :] = torch.clamp(
            tile / scale[:, None, :, None],
            -fp8_max,
            fp8_max,
        ).to(fp8_dtype)
        descale[:, :, block_idx] = scale

    return out.contiguous(), descale.contiguous()


def _quantize_qkv_w8a8(q, k, v):
    fp8_dtype = _get_fp8_dtype()
    q_fp8, q_descale = _quantize_per_block_fp8(_apply_incoherent_qk(q), fp8_dtype)
    k_fp8, k_descale = _quantize_per_block_fp8(_apply_incoherent_qk(k), fp8_dtype)
    v_fp8, v_descale = _quantize_per_block_fp8(v, fp8_dtype)
    return (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        float(torch.finfo(fp8_dtype).max),
    )


def make_input(
    batch,
    num_head,
    num_head_k,
    q_seq_len,
    kv_seq_len,
    head_size,
    dtype,
    device,
):
    random_utils.set_philox_state(1234567890, 0, device)
    q = torch.empty(
        (batch, num_head, q_seq_len, head_size),
        dtype=dtype,
        device=device,
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, num_head_k, kv_seq_len, head_size),
        dtype=dtype,
        device=device,
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, num_head_k, kv_seq_len, head_size),
        dtype=dtype,
        device=device,
    ).uniform_(-0.05, 0.05)
    return q, k, v


def torch_flash_fwd(q, k, v, scale, is_causal):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    return torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        0.0,
        is_causal,
        False,
        scale=scale,
    )


def gems_flash_fwd_w8a8(q, k, v, scale, is_causal):
    (
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        fp8_p_max,
    ) = _quantize_qkv_w8a8(q, k, v)

    out = torch.empty_like(q, dtype=torch.float16)
    result = mha_fwd(
        q_fp8,
        k_fp8,
        v_fp8,
        out,
        None,
        0.0,
        scale,
        is_causal,
        -1,
        -1,
        0.0,
        False,
        disable_splitkv=False,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        fp8_p_max=fp8_p_max,
    )
    return result[0]


def _assert_w8a8_attention_close(actual, expected):
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f

    mse = torch.mean(diff * diff)
    rel_mse = mse / torch.mean(expected_f * expected_f).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )

    assert mse.item() < 1.0e-4, f"mse={mse.item():.6e}"
    assert rel_mse.item() < 2.0e-2, f"rel_mse={rel_mse.item():.6e}"
    assert cosine.item() > 0.99, f"cosine={cosine.item():.6f}"


@pytest.mark.flash_attention_forward
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.vendor_name != "nvidia", reason="NVIDIA-only path")
@pytest.mark.skipif(
    not _supports_hopper_fp8(), reason="Requires NVIDIA Hopper or newer"
)
@pytest.mark.skipif(
    getattr(torch, "float8_e4m3fn", None) is None,
    reason="FP8 is not available",
)
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    W8A8_CONFIGS,
)
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_flash_attention_forward_w8a8(
    batch,
    num_head,
    q_seq_len,
    kv_seq_len,
    head_size,
    is_causal,
    dtype,
):
    device = torch_device_fn.current_device()
    q, k, v = make_input(
        batch,
        num_head,
        num_head,
        q_seq_len,
        kv_seq_len,
        head_size,
        dtype,
        device,
    )
    scale = 1.0 / math.sqrt(head_size)

    torch_out, _, _, _, _ = torch_flash_fwd(q, k, v, scale, is_causal)
    gems_out = gems_flash_fwd_w8a8(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        scale,
        is_causal,
    )

    _assert_w8a8_attention_close(gems_out, torch_out)

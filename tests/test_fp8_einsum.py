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

import math

import pytest
import torch
import triton

import flag_gems

from .conftest import QUICK_MODE

# The Gluon fp8_einsum kernel requires Triton >= 3.6.0 and specific TLE features.
fp8_einsum = None
if triton.__version__ >= "3.6.0":
    try:
        from flag_gems.runtime.backend._nvidia.hopper.ops.fp8_einsum import fp8_einsum
    except (AttributeError, ImportError):
        pass

DEFAULT_BLOCK_SHAPE = [128, 128]


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()
TRITON_VERSION_OK = triton.__version__ >= "3.6.0"


# (h, r, d) groups -- r and d must be divisible by the 128 block grid.
_HRD_GROUPS = {
    "flash": (8, 4096, 1024),
    "pro": (16, 7168, 1024),
}
_BATCH_SIZES = (1, 4, 8, 16, 32, 64, 128)
if not QUICK_MODE:
    _BATCH_SIZES += (4096, 8192, 16384, 32768)

# (b, h, r, d)
FP8_EINSUM_CONFIGS = [
    (b, h, r, d) for (h, r, d) in _HRD_GROUPS.values() for b in _BATCH_SIZES
]


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 scales up to the nearest power-of-two (UE8M0 grid)."""
    bits = x.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def per_token_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    return x_fp8, sf


def per_block_cast_to_fp8(x: torch.Tensor, use_ue8m0: bool = True, gran_k: int = 128):
    assert x.dim() == 2
    m, n = x.shape
    padded_m = math.ceil(m / gran_k) * gran_k
    padded_n = math.ceil(n / gran_k) * gran_k
    x_padded = torch.zeros((padded_m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, gran_k, x_padded.size(1) // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return (
        x_scaled.view_as(x_padded)[:m, :n].contiguous(),
        sf.view(x_view.size(0), x_view.size(2)),
    )


def _make_fp8_einsum_inputs(b, h, r, d, block_shape, device, seed=0):
    """Build block-wise FP8 ``bhr,hdr->bhd`` inputs (per-token x, per-block y)."""
    block_n, block_k = block_shape
    torch.manual_seed(seed)
    x = torch.randn((b, h, r), device=device, dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device=device, dtype=torch.bfloat16)

    x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=True, gran_k=block_k)
    x_data = x_fp8[0].view(b, h, r)
    x_scale = x_fp8[1].view(b, h, math.ceil(r / block_k))

    y_data = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    y_scale = torch.empty(
        (h, math.ceil(d / block_n), math.ceil(r / block_k)),
        device=device,
        dtype=torch.float32,
    )
    for i in range(h):
        y_data[i], y_scale[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)

    return x_data, x_scale, y_data, y_scale


def torch_fp8_block_einsum_reference(x_data, x_scale, y_data, y_scale, block_shape):
    """Pure-PyTorch reference: dequantize the block-wise FP8 inputs, then einsum.

    Mirrors the kernel math (FP8 data scaled by the block grid, FP32 accumulation):
        out[b,h,d] = sum_r x[b,h,r] * y[h,d,r]
    """
    block_n, block_k = block_shape
    b, h, r = x_data.shape
    d = y_data.shape[1]

    x_f = x_data.to(torch.float32)
    y_f = y_data.to(torch.float32)

    k_tiles = x_scale.shape[-1]
    n_tiles = y_scale.shape[1]

    x_deq = torch.empty_like(x_f)
    for kt in range(k_tiles):
        ks, ke = kt * block_k, min((kt + 1) * block_k, r)
        x_deq[:, :, ks:ke] = x_f[:, :, ks:ke] * x_scale[:, :, kt : kt + 1]

    y_deq = torch.empty_like(y_f)
    for nt in range(n_tiles):
        ns, ne = nt * block_n, min((nt + 1) * block_n, d)
        for kt in range(k_tiles):
            ks, ke = kt * block_k, min((kt + 1) * block_k, r)
            y_deq[:, ns:ne, ks:ke] = y_f[:, ns:ne, ks:ke] * y_scale[:, nt, kt].view(
                h, 1, 1
            )

    return torch.einsum("bhr,hdr->bhd", x_deq, y_deq)


@pytest.mark.fp8_einsum
@pytest.mark.parametrize("config", FP8_EINSUM_CONFIGS)
@pytest.mark.parametrize("block_shape", [[128, 128]])
@pytest.mark.skipif(
    not (CUDA_AVAILABLE and TRITON_VERSION_OK and fp8_einsum is not None),
    reason="requires NVIDIA Hopper GPU, Triton >= 3.6.0 with TLE support",
)
def test_accuracy_fp8_einsum(config, block_shape):
    """Validate FlagGems fp8_einsum against a dequantized PyTorch reference."""
    b, h, r, d = config
    device = flag_gems.device

    x_data, x_scale, y_data, y_scale = _make_fp8_einsum_inputs(
        b, h, r, d, block_shape, device
    )

    result = fp8_einsum(
        "bhr,hdr->bhd",
        x_data,
        x_scale,
        y_data,
        y_scale,
        block_size=block_shape,
    )

    ref = torch_fp8_block_einsum_reference(
        x_data, x_scale, y_data, y_scale, block_shape
    )

    torch.cuda.synchronize()

    assert result.shape == (b, h, d)
    # FP8 block-wise quantization + bf16 output accumulate rounding error.
    rtol = 2e-1
    atol = max(5e-2, ref.abs().max().item() * 5e-2)
    torch.testing.assert_close(result, ref.to(result.dtype), rtol=rtol, atol=atol)

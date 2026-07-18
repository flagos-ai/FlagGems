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

import pytest
import torch

import flag_gems
from flag_gems.fused import pack_seq_triton

from . import accuracy_utils as utils
from . import conftest as cfg

# =============================================================================
# CUDA / Hopper check for FP8
# =============================================================================


def is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = is_cuda_available()


def _ref_pack_seq(x, lengths, pad_value=-float("inf")):
    """Pure PyTorch reference for pack_seq_triton."""
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    B = len(lengths)
    Lmax = max(lengths)
    original_shape = x.shape
    if len(original_shape) > 2:
        D_flat = original_shape[1:].numel()
        x_flat = x.reshape(original_shape[0], -1)
    else:
        D_flat = x.shape[1]
        x_flat = x

    # Always use float32 to hold the fill value so that values like -inf
    # are representable even when the target dtype is fp8 or uint8.
    out_reshaped = torch.full(
        (B, Lmax, D_flat), pad_value, device=x.device, dtype=torch.float32
    )
    offset = 0
    for b in range(B):
        seq_len = lengths[b]
        out_reshaped[b, :seq_len] = x_flat[offset : offset + seq_len]
        offset += seq_len

    out_reshaped = out_reshaped.to(x.dtype)

    if len(original_shape) > 2:
        return out_reshaped.reshape((B, Lmax) + original_shape[1:])
    return out_reshaped


# =============================================================================
# Test shapes
# =============================================================================

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    PACK_SHAPES_2D = [(32, 64, [4, 7, 1, 8, 12])]
    PACK_SHAPES_3D = [(6, 8, 4, [3, 3])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    PACK_SHAPES_2D = [
        (32, 64, [4, 7, 1, 8, 12]),
        (128, 256, [32, 64, 32]),
        (16, 32, [1] * 16),
        (200, 128, [100, 50, 30, 20]),
    ]
    PACK_SHAPES_3D = [
        (6, 8, 4, [3, 3]),
        (10, 4, 8, [2, 4, 4]),
        (20, 16, 32, [5, 5, 5, 5]),
        (15, 8, 16, [7, 5, 3]),
    ]


@pytest.mark.pack_seq_triton
@pytest.mark.parametrize("N, D, lengths_list", PACK_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pack_seq_accuracy_2d(N, D, lengths_list, dtype):
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, D, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = _ref_pack_seq(ref_x, lengths_list)
    res_out = pack_seq_triton(x, lengths)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pack_seq_triton
@pytest.mark.parametrize("N, H, D, lengths_list", PACK_SHAPES_3D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pack_seq_accuracy_3d(N, H, D, lengths_list, dtype):
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    B = len(lengths_list)
    Lmax = max(lengths_list)
    x = torch.randn(N, H, D, dtype=dtype, device=flag_gems.device)

    result = pack_seq_triton(x, lengths)

    expected_shape = (B, Lmax, H, D)
    assert result.shape == expected_shape, f"{result.shape} != {expected_shape}"
    assert result.dtype == dtype
    assert result.device.type == flag_gems.device

    ref_x = utils.to_reference(x, True)
    ref_out = _ref_pack_seq(ref_x, lengths_list)
    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.pack_seq_triton
@pytest.mark.parametrize(
    "N, H, D, lengths_list",
    (
        [(6, 8, 4, [3, 3])]
        if cfg.QUICK_MODE
        else [(20, 8, 16, [10, 10]), (15, 4, 8, [5, 7, 3])]
    ),
)
def test_pack_seq_shape_consistency(N, H, D, lengths_list):
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    B = len(lengths_list)
    x = torch.randn(N, H, D, dtype=torch.float32, device=flag_gems.device)

    result = pack_seq_triton(x, lengths)
    assert result.shape[0] == B
    assert result.shape[1] == max(lengths_list)
    assert result.shape[2:] == x.shape[1:]


@pytest.mark.pack_seq_triton
@pytest.mark.parametrize("pad_value", [-100.0, -10.0, 0.0, 10.0, 100.0])
def test_pack_seq_custom_padding(pad_value):
    N, D = 20, 16
    lengths = torch.tensor([10, 10], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, D, dtype=torch.float32, device=flag_gems.device)

    result = pack_seq_triton(x, lengths, pad_value=pad_value)

    ref_x = utils.to_reference(x, True)
    ref_out = _ref_pack_seq(ref_x, [10, 10], pad_value=pad_value)
    utils.gems_assert_close(result, ref_out, torch.float32)

    padded_data = result[:, 10:].to(torch.float32)
    assert torch.allclose(
        padded_data, torch.full_like(padded_data, pad_value), atol=1e-5
    )


@pytest.mark.pack_seq_triton
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pack_seq_default_inf_padding(dtype):
    N, D = 20, 16
    lengths = torch.tensor([10, 10], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, D, dtype=dtype, device=flag_gems.device)

    result = pack_seq_triton(x, lengths)

    padded_data = result[:, 10:].to(torch.float32)
    assert torch.all(torch.isinf(padded_data) & (padded_data < 0))


@pytest.mark.pack_seq_triton
def test_pack_seq_single_batch():
    lengths = torch.tensor([10], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(10, 16, dtype=torch.float32, device=flag_gems.device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (1, 10, 16)


@pytest.mark.pack_seq_triton
def test_pack_seq_short_sequences():
    lengths = torch.tensor([1, 1, 1], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(20, 4, dtype=torch.float32, device=flag_gems.device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (3, 1, 4)


@pytest.mark.pack_seq_triton
def test_pack_seq_3d_single_batch():
    lengths = torch.tensor([10], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(10, 8, 16, dtype=torch.float32, device=flag_gems.device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (1, 10, 8, 16)


@pytest.mark.pack_seq_triton
def test_pack_seq_3d_short_sequences():
    lengths = torch.tensor([1, 1, 1], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(20, 4, 8, dtype=torch.float32, device=flag_gems.device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (3, 1, 4, 8)


@pytest.mark.pack_seq_triton
@pytest.mark.parametrize("block_t, block_d", [(32, 32), (64, 64), (128, 128)])
def test_pack_seq_block_sizes(block_t, block_d):
    N, D = 100, 32
    lengths_list = [25, 25, 25, 25]
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, D, dtype=torch.float32, device=flag_gems.device)

    result = pack_seq_triton(x, lengths, block_t=block_t, block_d=block_d)
    assert result.shape == (4, 25, 32)

    ref_x = utils.to_reference(x, True)
    ref_out = _ref_pack_seq(ref_x, lengths_list)
    utils.gems_assert_close(result, ref_out, torch.float32)


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires NVIDIA Hopper architecture for FP8",
)
@pytest.mark.parametrize(
    "N, H, D, lengths_list",
    [(6, 8, 4, [3, 3]), (10, 4, 8, [2, 4, 4]), (20, 16, 32, [5, 5, 5, 5])],
)
def test_pack_seq_fp8_basic(N, H, D, lengths_list):
    FP8 = torch.float8_e4m3fn
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    B = len(lengths_list)
    Lmax = max(lengths_list)
    x = torch.randn(N, H, D, dtype=torch.float32, device=flag_gems.device) * 0.1
    x_fp8 = x.to(FP8)
    packed = pack_seq_triton(x_fp8, lengths)
    assert packed.shape == (B, Lmax, H, D)
    assert packed.dtype == FP8
    for b in range(B):
        start_idx = sum(lengths_list[:b])
        seq_len = lengths_list[b]
        expected = x_fp8[start_idx : start_idx + seq_len].to(torch.float32)
        actual = packed[b, :seq_len].to(torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-2)


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires NVIDIA Hopper architecture for FP8",
)
def test_pack_seq_fp8_custom_padding():
    FP8 = torch.float8_e4m3fn
    N, H, D = 20, 8, 16
    lengths = torch.tensor([10, 10], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, H, D, dtype=torch.float32, device=flag_gems.device) * 0.1
    x_fp8 = x.to(FP8)
    for pad_value in [-100.0, 0.0, 100.0]:
        result = pack_seq_triton(x_fp8, lengths, pad_value=pad_value)
        padded_data = result[:, 10:].to(torch.float32)
        if pad_value < 0:
            assert torch.all(padded_data < -50)
        elif pad_value > 0:
            assert torch.all(padded_data > 50)
        else:
            assert torch.allclose(padded_data, torch.zeros_like(padded_data), atol=1e-2)


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires NVIDIA Hopper architecture for FP8",
)
def test_pack_seq_fp8_default_inf_padding():
    FP8 = torch.float8_e4m3fn
    N, H, D = 20, 8, 16
    lengths = torch.tensor([10, 10], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, H, D, dtype=torch.float32, device=flag_gems.device) * 0.1
    x_fp8 = x.to(FP8)
    result = pack_seq_triton(x_fp8, lengths)
    padded_data = result[:, 10:].to(torch.float32)
    assert torch.all(padded_data < -100)


@pytest.mark.pack_seq_triton
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires NVIDIA Hopper architecture for FP8",
)
@pytest.mark.parametrize("block_t, block_d", [(32, 32), (64, 64), (128, 128)])
def test_pack_seq_fp8_block_sizes(block_t, block_d):
    FP8 = torch.float8_e4m3fn
    N, H, D = 100, 16, 32
    lengths = torch.tensor([25, 25, 25, 25], dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, H, D, dtype=torch.float32, device=flag_gems.device) * 0.1
    x_fp8 = x.to(FP8)
    result = pack_seq_triton(x_fp8, lengths, block_t=block_t, block_d=block_d)
    assert result.shape == (4, 25, 16, 32)
    for b in range(4):
        expected = x_fp8[b * 25 : b * 25 + 25].to(torch.float32)
        actual = result[b, :25].to(torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-1, atol=1e-2)

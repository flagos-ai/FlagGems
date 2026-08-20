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
from flag_gems.fused import unpack_seq_triton

from . import accuracy_utils as utils
from . import conftest as cfg

# =============================================================================
# CUDA available check for FP8
# =============================================================================


def _is_cuda_available():
    if flag_gems.device != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


CUDA_AVAILABLE = _is_cuda_available()


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


def _ref_unpack_seq(packed_tensor, lengths):
    """Pure PyTorch reference for unpack_seq_triton."""
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.tolist()
    original_shape = packed_tensor.shape
    if len(original_shape) > 3:
        B, Lmax = original_shape[:2]
        D_flat = original_shape[2:].numel()
        packed_flat = packed_tensor.reshape(B, Lmax, -1)
    else:
        B, Lmax, D_flat = packed_tensor.shape
        packed_flat = packed_tensor

    N = sum(lengths)
    out_reshaped = torch.empty(
        (N, D_flat), device=packed_tensor.device, dtype=packed_tensor.dtype
    )
    offset = 0
    for b in range(B):
        seq_len = lengths[b]
        out_reshaped[offset : offset + seq_len] = packed_flat[b, :seq_len]
        offset += seq_len

    if len(original_shape) > 3:
        return out_reshaped.reshape((N,) + original_shape[2:])
    return out_reshaped


# =============================================================================
# Test shapes
# =============================================================================

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    UNPACK_SHAPES_2D = [(32, 64, [4, 7, 1, 8, 12])]
    UNPACK_SHAPES_3D = [(6, 8, 4, [3, 3])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    UNPACK_SHAPES_2D = [
        (32, 64, [4, 7, 1, 8, 12]),
        (128, 256, [32, 64, 32]),
        (16, 32, [1] * 16),
        (200, 128, [100, 50, 30, 20]),
    ]
    UNPACK_SHAPES_3D = [
        (6, 8, 4, [3, 3]),
        (10, 4, 8, [2, 4, 4]),
        (20, 16, 32, [5, 5, 5, 5]),
        (15, 8, 16, [7, 5, 3]),
    ]


@pytest.mark.unpack_seq_triton
@pytest.mark.parametrize("N, D, lengths_list", UNPACK_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unpack_seq_accuracy_2d(N, D, lengths_list, dtype):
    B = len(lengths_list)
    Lmax = max(lengths_list)
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)

    packed = torch.randn(B, Lmax, D, dtype=dtype, device=flag_gems.device)
    ref_packed = utils.to_reference(packed, True)

    ref_out = _ref_unpack_seq(ref_packed, lengths_list)
    res_out = unpack_seq_triton(packed, lengths)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.unpack_seq_triton
@pytest.mark.parametrize("N, D, lengths_list", UNPACK_SHAPES_2D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pack_unpack_roundtrip_2d(N, D, lengths_list, dtype):
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, D, dtype=dtype, device=flag_gems.device)

    packed = _ref_pack_seq(x, lengths_list)
    unpacked = unpack_seq_triton(packed, lengths)

    assert unpacked.shape == x.shape
    ref_x = utils.to_reference(x, True)
    utils.gems_assert_close(unpacked, ref_x, dtype)


@pytest.mark.unpack_seq_triton
@pytest.mark.parametrize("N, H, D, lengths_list", UNPACK_SHAPES_3D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_pack_unpack_roundtrip_3d(N, H, D, lengths_list, dtype):
    lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
    x = torch.randn(N, H, D, dtype=dtype, device=flag_gems.device)

    packed = _ref_pack_seq(x, lengths_list)
    unpacked = unpack_seq_triton(packed, lengths)

    assert unpacked.shape == x.shape
    ref_x = utils.to_reference(x, True)
    utils.gems_assert_close(unpacked, ref_x, dtype)


@pytest.mark.unpack_seq_triton
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unpack_seq_single_batch(dtype):
    x = torch.randn(10, 16, dtype=dtype, device=flag_gems.device)
    lengths = torch.tensor([10], dtype=torch.int32, device=flag_gems.device)
    packed = _ref_pack_seq(x, [10])
    unpacked = unpack_seq_triton(packed, lengths)
    assert unpacked.shape == x.shape

    ref_x = utils.to_reference(x, True)
    utils.gems_assert_close(unpacked, ref_x, dtype)


@pytest.mark.unpack_seq_triton
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unpack_seq_short_sequences(dtype):
    x = torch.randn(20, 4, dtype=dtype, device=flag_gems.device)
    lengths = torch.tensor([1, 1, 1], dtype=torch.int32, device=flag_gems.device)
    packed = _ref_pack_seq(x, [1, 1, 1])
    unpacked = unpack_seq_triton(packed, lengths)

    ref_x = utils.to_reference(x[:3], True)
    utils.gems_assert_close(unpacked, ref_x, dtype)


@pytest.mark.unpack_seq_triton
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unpack_seq_3d_edge_cases(dtype):
    x = torch.randn(15, 8, 16, dtype=dtype, device=flag_gems.device)
    lengths = torch.tensor([5, 7, 3], dtype=torch.int32, device=flag_gems.device)
    packed = _ref_pack_seq(x, [5, 7, 3])
    unpacked = unpack_seq_triton(packed, lengths)
    assert unpacked.shape == x.shape

    ref_x = utils.to_reference(x, True)
    utils.gems_assert_close(unpacked, ref_x, dtype)


@pytest.mark.unpack_seq_triton
@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="requires NVIDIA Hopper architecture for FP8",
)
def test_pack_unpack_fp8_roundtrip():
    FP8 = torch.float8_e4m3fn
    for N, H, D, lengths_list in [
        (6, 8, 4, [3, 3]),
        (10, 4, 8, [2, 4, 4]),
        (15, 8, 16, [7, 5, 3]),
    ]:
        lengths = torch.tensor(lengths_list, dtype=torch.int32, device=flag_gems.device)
        x = torch.randn(N, H, D, dtype=torch.float32, device=flag_gems.device) * 0.1
        x_fp8 = x.to(FP8)
        packed = _ref_pack_seq(x_fp8, lengths_list)
        unpacked = unpack_seq_triton(packed, lengths)
        assert unpacked.shape == x_fp8.shape
        torch.testing.assert_close(
            x_fp8.to(torch.float32),
            unpacked.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
        )

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

from . import accuracy_utils as utils


def _reference_weight_int4pack_mm(A, mat2_packed, qGroupSize, qScaleAndZeros):
    """Hand-written Python reference for _weight_int4pack_mm.

    Uses the same byte-pair int4 packing format as _convert_weight_to_int4pack:
      - mat2_packed: uint8 tensor of shape (N, K//2)
      - Each byte: low nibble = even column, high nibble = odd column
      - Dequantize: w = (int4_value - zero) * scale
      - Compute: C = A @ W_dequant.T
    """
    M, K = A.shape
    N = mat2_packed.shape[0]
    assert mat2_packed.dtype == torch.uint8
    assert mat2_packed.shape[1] * 2 == K

    W_dequant = torch.empty((N, K), dtype=A.dtype, device=A.device)

    for n in range(N):
        for k in range(K):
            packed_byte = mat2_packed[n, k // 2].item()
            if k % 2 == 0:
                # Even column: low nibble
                q = packed_byte & 0xF
            else:
                # Odd column: high nibble
                q = (packed_byte >> 4) & 0xF
            g = k // qGroupSize
            scale = qScaleAndZeros[g, n, 0].item()
            zero = qScaleAndZeros[g, n, 1].item()
            W_dequant[n, k] = (q - zero) * scale

    return A.float() @ W_dequant.T.float()


def _create_int4_packed_weights(weight_int4, N, K):
    """Pack int4 weights (int32, shape (N, K)) into uint8 byte-pair format.

    Packing: (odd_val << 4) | even_val  (matches _convert_weight_to_int4pack).
    """
    packed = torch.empty((N, K // 2), dtype=torch.uint8, device=weight_int4.device)
    for n in range(N):
        for k_half in range(K // 2):
            even = weight_int4[n, 2 * k_half].item() & 0xF
            odd = weight_int4[n, 2 * k_half + 1].item() & 0xF
            packed[n, k_half] = (odd << 4) | even
    return packed


def _create_scale_and_zeros(K, N, qGroupSize, dtype, device):
    """Create random scale and zero tensors for testing.

    scales are in range (0.5, 2.0), zeros are in range 2..13 (int4 range 0..15).
    """
    num_groups = K // qGroupSize
    scales = torch.rand((num_groups, N), dtype=dtype, device=device) * 1.5 + 0.5
    # zero points: small integer offset from center of int4 range
    zeros = torch.randint(4, 11, (num_groups, N), dtype=dtype, device=device)
    qScaleAndZeros = torch.stack([scales, zeros], dim=-1)
    return qScaleAndZeros


@pytest.mark.weight_int4pack_mm
@pytest.mark.parametrize(
    "M,N,K,qGroupSize",
    [
        (4, 8, 32, 16),
        (8, 16, 64, 32),
        (16, 16, 128, 32),
        (16, 32, 128, 64),
        (32, 64, 256, 64),
        (64, 32, 128, 64),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weight_int4pack_mm(M, N, K, qGroupSize, dtype):
    """Test _weight_int4pack_mm accuracy against hand-written reference."""
    device = flag_gems.device

    # Create activation tensor
    A = torch.randn((M, K), dtype=dtype, device=device)

    # Create int4 weights (values 0..15)
    weight_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)

    # Pack weights into byte-pair format
    mat2_packed = _create_int4_packed_weights(weight_int4, N, K)

    # Create scales and zeros
    qScaleAndZeros = _create_scale_and_zeros(K, N, qGroupSize, dtype, device)

    # Golden reference inputs (mat2_packed keeps uint8 for bit-level unpacking).
    ref_A = utils.to_reference(A)
    ref_mat2 = utils.to_reference(mat2_packed)
    ref_qsz = utils.to_reference(qScaleAndZeros)

    # Reference computation (using Float32 matmul inside the reference).
    ref_out = _reference_weight_int4pack_mm(ref_A, ref_mat2, qGroupSize, ref_qsz).to(
        dtype
    )

    # GEMS computation
    res_out = flag_gems._weight_int4pack_mm(A, mat2_packed, qGroupSize, qScaleAndZeros)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=K)
